"""One CSV row binds one image, numeric vector, label and patient group.

Inspection stays Torch-free. Referenced images alone define this adapter's
image set; no directory scanning or missing-row/modality skipping is performed.
"""
import hashlib
import math
from pathlib import Path

from .classification_data import (MAX_CLASSES, MAX_CSV_BYTES, MAX_FLOAT32, MAX_IMAGE_BYTES, MAX_IMAGE_TOTAL_BYTES,
                                 IMAGE_EXTENSIONS, SPLITS, _cancel, _check_partition, _csv_records, _decode_image,
                                 _float32_digest, _group_split, _json_digest, _manifest_digest, _prepare_image,
                                 _read_source, _source_name, _spec)
from .experiment_spec import validate_classification_config


def _paired_spec(data_spec):
    effective = validate_classification_config({"profile": "classification-v2", "data": data_spec})["data"]
    csv_keys = {"kind", "path", "target_column", "feature_columns", "group_column", "split_column", "standardize", "validation_fraction", "test_fraction"}
    legacy = _spec(dict({key: value for key, value in effective.items() if key in csv_keys}, kind="csv"))
    effective.update(legacy)
    effective["kind"] = "paired_csv"
    csv_parent = Path(effective["path"]).parent
    configured_root = Path(effective["image_root"]).expanduser() if effective["image_root"] else csv_parent
    if not configured_root.is_absolute():
        configured_root = csv_parent / configured_root
    try:
        image_root = configured_root.resolve(strict=True)
        if not image_root.is_dir():
            raise ValueError("image_root must be a directory: " + str(image_root))
    except OSError as exc:
        raise ValueError("Image root is unavailable: " + str(configured_root)) from exc
    effective["image_root"] = str(image_root)
    return effective


def _referenced_image(reference, spec, record, cancelled, cache):
    source = _source_name(record)
    if not reference or len(reference) > 4096:
        raise ValueError("Missing or oversized image reference in " + source)
    root = Path(spec["image_root"])
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Image reference escapes image_root: " + str(candidate))
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image reference: " + str(resolved))
        key = str(resolved)
        if key not in cache:
            raw = _read_source(resolved, MAX_IMAGE_BYTES, cancelled)
            image, decoded_digest = _decode_image(raw, resolved)
            effective = _prepare_image(image, spec["image_size"], spec["channels"])
            digest = hashlib.sha256(effective.mode.encode("ascii") + str(effective.size).encode("ascii") + effective.tobytes()).hexdigest()
            cache[key] = {"source_path": key, "source_digest": hashlib.sha256(raw).hexdigest(), "content_digest": digest,
                          "decoded_digest": decoded_digest, "source_size": list(image.size), "source_bytes": len(raw),
                          "relative_path": resolved.relative_to(root).as_posix()}
        return dict(cache[key], reference=reference)
    except (InterruptedError, TimeoutError):
        raise
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid image in " + source + " -> " + str(candidate) + ": " + str(exc)) from exc


def _paired_partition(records, class_names):
    seen_paths, seen_contents = {}, {}
    for record in records:
        for category, seen, identity in (("image path", seen_paths, record["image"]["source_path"]),
                                         ("image content", seen_contents, record["image"]["content_digest"])):
            previous = seen.setdefault(identity, record)
            if previous["split"] != record["split"]:
                raise ValueError("Cross-split " + category + " leakage: " + _source_name(record) + " -> " + record["image"]["source_path"] +
                                 " conflicts with " + _source_name(previous) + " -> " + previous["image"]["source_path"])
    return _check_partition(records, class_names)


def inspect_paired_dataset(data_spec, seed=42, cancelled=None):
    _cancel(cancelled)
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be an integer in 0..2^31-1")
    spec = _paired_spec(data_spec)
    records = _csv_records(spec, cancelled, extra_columns=(spec["image_column"], spec["id_column"]))
    seen_ids, images, image_bytes = {}, {}, 0
    for record in records:
        _cancel(cancelled)
        extras = record.pop("extra_values")
        identifier = extras[spec["id_column"]].strip()
        if not identifier or len(identifier) > 256:
            raise ValueError("Missing or oversized sample ID in " + _source_name(record))
        if identifier in seen_ids:
            raise ValueError("Duplicate sample ID " + identifier + " in " + _source_name(record) + " conflicts with " + _source_name(seen_ids[identifier]))
        seen_ids[identifier] = record
        record["id"] = identifier
        before = len(images)
        record["image"] = _referenced_image(extras[spec["image_column"]].strip(), spec, record, cancelled, images)
        if len(images) > before:
            image_bytes += record["image"]["source_bytes"]
            if image_bytes > MAX_IMAGE_TOTAL_BYTES:
                raise ValueError("Total image source size limit exceeded in " + _source_name(record))
    class_names = sorted({record["class_name"] for record in records})
    if not 2 <= len(class_names) <= MAX_CLASSES:
        raise ValueError("Classification requires 2..128 classes")
    labels = {name: index for index, name in enumerate(class_names)}
    for record in records:
        record["label"] = labels[record["class_name"]]
    if spec["split_column"] is None:
        _group_split(records, spec, seed, cancelled)
    train = [record["features"] for record in records if record["split"] == "train"]
    if not train:
        raise ValueError("Paired classification requires a nonempty train partition")
    width = len(spec["feature_columns"])
    mean = [math.fsum(row[column] / len(train) for row in train) for column in range(width)] if spec["standardize"] else [0.0] * width
    std = [math.sqrt(math.fsum((row[column] - mean[column]) ** 2 / len(train) for row in train)) or 1.0 for column in range(width)] if spec["standardize"] else [1.0] * width
    if any(not math.isfinite(value) for value in mean + std):
        raise ValueError("Non-finite train-only paired standardization")
    for record in records:
        _cancel(cancelled)
        values = [(value - mean[index]) / std[index] for index, value in enumerate(record["features"])]
        if any(not math.isfinite(value) or abs(value) > MAX_FLOAT32 for value in values):
            raise ValueError("Preprocessed float32 value out of range at " + _source_name(record))
        record["raw_content_digest"] = record["content_digest"]
        record["tabular_content_digest"] = _float32_digest(values)
        record["content_digest"] = _json_digest({"image": record["image"]["content_digest"], "tabular": record["tabular_content_digest"]})
    counts, class_counts = _paired_partition(records, class_names)
    preprocessing = {"tabular": {"kind": "standardize" if spec["standardize"] else "identity", "fit_split": "train", "dtype": "float32",
                                  "feature_columns": list(spec["feature_columns"]), "mean": mean, "std": std},
                     "image": {"kind": "resize_scale", "image_size": list(spec["image_size"]), "channels": spec["channels"], "resampling": "bilinear", "scale": 1.0 / 255.0}}
    manifest = {"version": 3, "kind": "paired_csv", "data_spec": spec, "seed": seed, "class_names": class_names,
                "input_shapes": {"image": [None, spec["channels"], *spec["image_size"]], "tabular": [None, width]},
                "counts": counts, "class_counts": class_counts, "records": records, "preprocessing": preprocessing, "ignored_files": [], "warnings": []}
    manifest["dataset_digest"] = _manifest_digest(manifest)
    _cancel(cancelled)
    return manifest


def build_paired_datasets(manifest, cancelled=None):
    if manifest.get("version") != 3 or manifest.get("kind") != "paired_csv":
        raise ValueError("Unsupported paired manifest")
    if _manifest_digest(manifest) != manifest.get("dataset_digest"):
        raise ValueError("Dataset manifest digest does not match its contents")
    seen = {}
    for record in manifest["records"]:
        for source, maximum in ((record, MAX_CSV_BYTES), (record["image"], MAX_IMAGE_BYTES)):
            _cancel(cancelled)
            path = source["source_path"]
            if path not in seen:
                seen[path] = hashlib.sha256(_read_source(path, maximum, cancelled)).hexdigest()
            if seen[path] != source["source_digest"]:
                raise ValueError("Dataset source changed since inspection: " + path)
    import torch
    from torch.utils.data import Dataset

    class PairedDataset(Dataset):
        def __init__(self, records):
            self.records = records
            numeric = manifest["preprocessing"]["tabular"]
            self.matrix = torch.tensor([[(value - numeric["mean"][column]) / numeric["std"][column] for column, value in enumerate(record["features"])] for record in records], dtype=torch.float32)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            _cancel(cancelled)
            record = self.records[index]
            source = record["image"]
            raw = _read_source(source["source_path"], MAX_IMAGE_BYTES, cancelled)
            if hashlib.sha256(raw).hexdigest() != source["source_digest"]:
                raise ValueError("Dataset image source changed since inspection: " + source["source_path"])
            image, _ = _decode_image(raw, source["source_path"])
            preprocess = manifest["preprocessing"]["image"]
            image = _prepare_image(image, preprocess["image_size"], preprocess["channels"])
            height, width = preprocess["image_size"]
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, preprocess["channels"]).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
            return {"image": tensor, "tabular": self.matrix[index]}, record["label"], record["id"]

    return {split: PairedDataset([record for record in manifest["records"] if record["split"] == split]) for split in SPLITS}
