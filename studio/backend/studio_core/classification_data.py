"""Bounded, content-addressed CSV and 2D image classification datasets.

Inspection deliberately uses no Torch. CSV matrices are frozen in the manifest;
image contents are rechecked on every lazy read, before decoding those bytes.
"""
import copy
import csv
import hashlib
import io
import json
import math
import random
import struct
from collections import Counter, defaultdict
from pathlib import Path

MAX_SAMPLES = 20_000
MAX_FEATURES = 512
MAX_CLASSES = 128
MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_IGNORED_FILES = 40_000
MAX_FLOAT32 = 3.4028234663852886e38
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
SYSTEM_SIDECARS = {"thumbs.db": "windows-thumbnail-cache", ".ds_store": "macos-folder-metadata", "desktop.ini": "windows-folder-metadata"}


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Dataset inspection cancelled")


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _manifest_digest(manifest):
    # Descriptive audit metadata is not model input. Synchronizers can create,
    # rewrite or remove known sidecars between UI inspection and worker startup.
    return _json_digest({key: value for key, value in manifest.items() if key not in {"dataset_digest", "ignored_files", "warnings"}})


def _float32_digest(values):
    """Hash precisely the numeric representation consumed by Torch, without it."""
    packed = struct.pack("<" + "f" * len(values), *values)
    # Positive/negative zero compare identically in tensors, including values
    # that underflow while converting otherwise finite doubles to float32.
    words = [0 if word == 0x80000000 else word for (word,) in struct.iter_unpack("<I", packed)]
    return hashlib.sha256(struct.pack("<" + "I" * len(words), *words)).hexdigest()


def _read_source(path, maximum, cancelled=None):
    path = Path(path)
    try:
        before = path.stat()
        if not path.is_file() or before.st_size > maximum:
            raise ValueError("Source size/type limit exceeded: " + str(path))
        chunks, size = [], 0
        with path.open("rb") as stream:
            while True:
                _cancel(cancelled)
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum:
                    raise ValueError("Source size limit exceeded: " + str(path))
                chunks.append(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
            raise ValueError("Source changed during reading: " + str(path))
        return b"".join(chunks)
    except (InterruptedError, TimeoutError):
        raise
    except OSError as exc:
        raise ValueError("Source is unavailable: " + str(path) + ": " + str(exc)) from exc


def _spec(data_spec):
    if not isinstance(data_spec, dict):
        raise ValueError("data must be an object")
    data_spec = copy.deepcopy(data_spec)
    kind = data_spec.get("kind")
    common = {"kind", "path", "validation_fraction", "test_fraction"}
    csv_keys = {"target_column", "feature_columns", "group_column", "split_column", "standardize"}
    image_keys = {"split_mode", "image_size", "channels"}
    if kind not in ("csv", "image_folder") or not data_spec.keys() <= common | (csv_keys if kind == "csv" else image_keys):
        raise ValueError("Unsupported dataset kind or data setting")
    result = {"kind": kind, "validation_fraction": 0.2, "test_fraction": 0.2}
    result.update({"group_column": None, "split_column": None, "standardize": True} if kind == "csv" else {"split_mode": "stratified", "image_size": [32, 32], "channels": 3})
    result.update(data_spec)
    if not isinstance(result.get("path"), str) or not result["path"].strip():
        raise ValueError("Dataset path must be a nonempty string")
    try:
        result["path"] = str(Path(result["path"]).expanduser().resolve(strict=True))
    except OSError as exc:
        raise ValueError("Dataset source is unavailable: " + result["path"]) from exc
    for key in ("validation_fraction", "test_fraction"):
        value = result[key]
        if type(value) not in (float, int) or not 0 < value < 1 or not math.isfinite(value):
            raise ValueError(key + " must be finite and between zero and one")
    if result["validation_fraction"] + result["test_fraction"] >= 1:
        raise ValueError("Validation and test fractions must leave a training partition")
    if kind == "csv":
        if not isinstance(result.get("target_column"), str) or not result["target_column"]:
            raise ValueError("target_column must be a column name")
        features = result.get("feature_columns")
        if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES or any(not isinstance(name, str) or not name for name in features) or len(set(features)) != len(features):
            raise ValueError("feature_columns must contain 1..512 unique column names")
        if type(result["standardize"]) is not bool:
            raise ValueError("standardize must be boolean")
        for key in ("group_column", "split_column"):
            if result[key] is not None and (not isinstance(result[key], str) or not result[key]):
                raise ValueError(key + " must be null or a column name")
        reserved = [result[key] for key in ("target_column", "group_column", "split_column") if result[key] is not None]
        if len(set(reserved)) != len(reserved) or set(features) & set(reserved):
            raise ValueError("Features, target, group and split columns must be distinct")
    else:
        size = result["image_size"]
        if result["split_mode"] not in ("stratified", "directories") or type(result["channels"]) is not int or result["channels"] not in (1, 3):
            raise ValueError("Invalid image split_mode or channels")
        if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or not 8 <= n <= 512 for n in size):
            raise ValueError("image_size must be [height,width], each 8..512")
    return result


def _csv_records(spec, cancelled, extra_columns=()):
    path = Path(spec["path"])
    raw = _read_source(path, MAX_CSV_BYTES, cancelled)
    source_digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8: " + str(path)) from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames or []
    requested = spec["feature_columns"] + [spec[key] for key in ("target_column", "group_column", "split_column") if spec[key] is not None] + list(extra_columns)
    if len(fields) != len(set(fields)) or any(key not in fields for key in requested):
        raise ValueError("Missing or duplicate CSV columns in " + str(path))
    records = []
    try:
        for row_number, row in enumerate(reader, 2):
            _cancel(cancelled)
            source = str(path) + " row " + str(row_number)
            if len(records) >= MAX_SAMPLES:
                raise ValueError("Dataset sample limit exceeded (20000): " + source)
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV fields at " + source)
            features = []
            for name in spec["feature_columns"]:
                try:
                    value = float(row[name])
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError("Invalid numeric feature in " + source + " column " + name) from exc
                if not math.isfinite(value) or abs(value) > MAX_FLOAT32:
                    raise ValueError("Non-finite or out-of-range feature in " + source + " column " + name)
                features.append(value)
            label = row[spec["target_column"]].strip()
            if not label:
                raise ValueError("Missing target label in " + source)
            if len(label) > 256:
                raise ValueError("Target label length limit exceeded (256): " + source)
            group = row[spec["group_column"]].strip() if spec["group_column"] is not None else None
            if spec["group_column"] is not None and not group:
                raise ValueError("Missing group in " + source)
            split = row[spec["split_column"]] if spec["split_column"] is not None else None
            if split is not None and split not in SPLITS:
                raise ValueError("Invalid explicit split in " + source + ": " + split)
            records.append({"id": "csv:" + _json_digest([str(path), row_number])[:24], "source_path": str(path), "row_number": row_number,
                            "source_digest": source_digest, "content_digest": _json_digest(features), "features": features,
                            "class_name": label, "group": group, "split": split})
            if extra_columns:
                records[-1]["extra_values"] = {name: row[name] for name in extra_columns}
    except csv.Error as exc:
        raise ValueError("Malformed CSV at " + str(path) + " row " + str(reader.line_num)) from exc
    return records


def _decode_image(raw, path):
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Image pixel/frame limit exceeded: " + str(path))
            image.load()
            converted = image.convert("RGB")
            digest = hashlib.sha256(str(converted.size).encode("ascii") + converted.tobytes()).hexdigest()
            return converted, digest
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
        raise ValueError("Cannot decode supported 2D image " + str(path) + ": " + str(exc)) from exc


def _prepare_image(image, size, channels):
    from PIL import Image
    height, width = size
    return image.convert("L" if channels == 1 else "RGB").resize((width, height), resample=Image.Resampling.BILINEAR)


def _tree_files(root, cancelled):
    pending, visited = [(root, 0)], 0
    while pending:
        directory, depth = pending.pop()
        if depth > 16:
            raise ValueError("Image directory depth limit exceeded: " + str(directory))
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            raise ValueError("Cannot read image directory: " + str(directory)) from exc
        for entry in entries:
            _cancel(cancelled)
            visited += 1
            if visited > 60_000:
                raise ValueError("Image directory entry limit exceeded: " + str(root))
            if entry.is_dir():
                if entry.is_symlink():
                    raise ValueError("Symbolic image directories are unsupported: " + str(entry))
                pending.append((entry, depth + 1))
            else:
                yield entry


def _ignore_sidecar(path, root, ignored_files):
    name = path.name.casefold()
    reason = SYSTEM_SIDECARS.get(name)
    if name.endswith(".baiduyun.uploading.cfg"):
        reason = "baiduyun-upload-metadata"
    if reason is None or path.is_dir():
        return False
    if len(ignored_files) >= MAX_IGNORED_FILES:
        raise ValueError("Ignored sidecar file limit exceeded (40000): " + str(path))
    ignored_files.append({"relative_path": path.relative_to(root).as_posix(), "source_path": str(path), "reason": reason})
    return True


def _image_entries(directory, root, ignored_files, cancelled):
    entries = []
    try:
        for index, path in enumerate(directory.iterdir(), 1):
            _cancel(cancelled)
            if index > 60_000:
                raise ValueError("Image directory entry limit exceeded: " + str(directory))
            if not _ignore_sidecar(path, root, ignored_files):
                entries.append(path)
    except (InterruptedError, TimeoutError):
        raise
    except OSError as exc:
        raise ValueError("Cannot read image directory: " + str(directory)) from exc
    return sorted(entries, key=lambda path: path.name)


def _image_records(spec, cancelled, ignored_files):
    root = Path(spec["path"])
    if not root.is_dir():
        raise ValueError("ImageFolder source must be a directory: " + str(root))
    partitions = [(split, root / split) for split in SPLITS] if spec["split_mode"] == "directories" else [(None, root)]
    records, total_bytes = [], 0
    if spec["split_mode"] == "directories":
        entries = _image_entries(root, root, ignored_files, cancelled)
        for path in entries:
            if path.name not in SPLITS or not path.is_dir():
                raise ValueError("Unexpected entry alongside train/val/test directories: " + str(path))
        if {path.name for path in entries} != set(SPLITS):
            raise ValueError("Directory split mode requires exactly train/val/test directories: " + str(root))
    for split, partition in partitions:
        if not partition.is_dir():
            raise ValueError("Missing split directory: " + str(partition))
        classes = _image_entries(partition, root, ignored_files, cancelled)
        for path in classes:
            if not path.is_dir() or path.is_symlink():
                raise ValueError("Unexpected entry alongside image class directories: " + str(path))
        if not 2 <= len(classes) <= MAX_CLASSES:
            raise ValueError("ImageFolder requires 2..128 class directories: " + str(partition))
        for class_path in classes:
            class_samples = 0
            for source in _tree_files(class_path, cancelled):
                if _ignore_sidecar(source, root, ignored_files):
                    continue
                if source.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError("Unsupported image source: " + str(source))
                if len(records) >= MAX_SAMPLES:
                    raise ValueError("Dataset sample limit exceeded (20000): " + str(source))
                path = source.resolve()
                raw = _read_source(path, MAX_IMAGE_BYTES, cancelled)
                total_bytes += len(raw)
                if total_bytes > MAX_IMAGE_TOTAL_BYTES:
                    raise ValueError("Total image source size limit exceeded: " + str(path))
                image, decoded_digest = _decode_image(raw, source)
                effective = _prepare_image(image, spec["image_size"], spec["channels"])
                content_digest = hashlib.sha256(effective.mode.encode("ascii") + str(effective.size).encode("ascii") + effective.tobytes()).hexdigest()
                relative = source.relative_to(root).as_posix()
                records.append({"id": "image:" + _json_digest(relative)[:24], "source_path": str(path), "relative_path": relative,
                                "source_digest": hashlib.sha256(raw).hexdigest(), "content_digest": content_digest,
                                "decoded_digest": decoded_digest,
                                "source_size": list(image.size), "class_name": class_path.name, "group": None, "split": split})
                class_samples += 1
            if not class_samples:
                raise ValueError("Empty image class directory: " + str(class_path))
    return sorted(records, key=lambda record: record["relative_path"])


def _stratify(records, spec, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for record in records:
        by_class[record["label"]].append(record)
    for label, items in sorted(by_class.items()):
        if len(items) < 3:
            raise ValueError("Class " + items[0]["class_name"] + " requires at least three samples for train/val/test")
        rng.shuffle(items)
        val_count = max(1, round(len(items) * spec["validation_fraction"]))
        test_count = max(1, round(len(items) * spec["test_fraction"]))
        while val_count + test_count >= len(items):
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("No usable train/val/test partition for class " + str(label))
        for index, record in enumerate(items):
            record["split"] = "val" if index < val_count else "test" if index < val_count + test_count else "train"


def _group_split(records, spec, seed, cancelled):
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    totals = Counter(record["label"] for record in records)
    label_groups = defaultdict(set)
    for group, items in groups.items():
        for label in {record["label"] for record in items}:
            label_groups[label].add(group)
    if any(len(group_set) < 3 for group_set in label_groups.values()):
        raise ValueError("Group isolation requires each class in at least three distinct groups")
    ratios = {"train": 1 - spec["validation_fraction"] - spec["test_fraction"], "val": spec["validation_fraction"], "test": spec["test_fraction"]}
    best, best_score = None, float("inf")
    # A bounded deterministic search handles mixed-class groups without splitting
    # subjects. Failure is explicit if no all-class partition is found.
    for attempt in range(32):
        _cancel(cancelled)
        rng = random.Random(seed + attempt * 104729)
        ordered = sorted(groups)
        rng.shuffle(ordered)
        ordered.sort(key=lambda group: -len(groups[group]))
        counts = {split: Counter() for split in SPLITS}
        assignments = {}
        for group in ordered:
            histogram = Counter(record["label"] for record in groups[group])
            options = list(SPLITS)
            rng.shuffle(options)
            def cost(split):
                return sum(((counts[split][label] + amount - totals[label] * ratios[split]) ** 2 - (counts[split][label] - totals[label] * ratios[split]) ** 2) / max(1, totals[label] * ratios[split]) for label, amount in histogram.items())
            selected = min(options, key=cost)
            assignments[group] = selected
            counts[selected].update(histogram)
        missing = sum(counts[split][label] == 0 for split in SPLITS for label in totals)
        score = missing * 1_000_000 + sum((counts[split][label] - totals[label] * ratios[split]) ** 2 / max(1, totals[label] * ratios[split]) for split in SPLITS for label in totals)
        if score < best_score:
            best, best_score = assignments, score
        if missing == 0:
            break
    if best_score >= 1_000_000:
        raise ValueError("Cannot form group-isolated train/val/test partitions containing every class")
    for record in records:
        record["split"] = best[record["group"]]


def _source_name(record):
    return record["source_path"] + (" row " + str(record["row_number"]) if "row_number" in record else "")


def _check_partition(records, class_names):
    seen = {"id": {}, "path": {}, "content": {}, "group": {}}
    counts = Counter()
    class_counts = {split: {name: 0 for name in class_names} for split in SPLITS}
    for record in records:
        split = record["split"]
        counts[split] += 1
        class_counts[split][record["class_name"]] += 1
        identities = {"id": record["id"], "content": record["content_digest"], "group": record.get("group")}
        if "row_number" not in record:
            identities["path"] = record["source_path"]
        for kind, identity in identities.items():
            if identity is None:
                continue
            previous = seen[kind].setdefault(identity, record)
            if previous["split"] != split:
                raise ValueError("Cross-split " + kind + " leakage " + str(identity) + ": " + _source_name(record) + " conflicts with " + _source_name(previous))
    for split in SPLITS:
        missing = [name for name, amount in class_counts[split].items() if not amount]
        if missing:
            raise ValueError("Split " + split + " is missing classes: " + ", ".join(missing))
    return {split: counts[split] for split in SPLITS}, class_counts


def inspect_dataset(data_spec, seed=42, cancelled=None):
    """Inspect/freeze actual file contents and return a finite manifest v2."""
    _cancel(cancelled)
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be an integer in 0..2^31-1")
    if isinstance(data_spec, dict) and data_spec.get("kind") == "paired_csv":
        from .paired_data import inspect_paired_dataset
        return inspect_paired_dataset(data_spec, seed=seed, cancelled=cancelled)
    spec = _spec(data_spec)
    ignored_files = []
    records = _csv_records(spec, cancelled) if spec["kind"] == "csv" else _image_records(spec, cancelled, ignored_files)
    class_names = sorted({record["class_name"] for record in records})
    if not 2 <= len(class_names) <= MAX_CLASSES:
        raise ValueError("Classification requires 2..128 classes")
    labels = {name: index for index, name in enumerate(class_names)}
    for record in records:
        record["label"] = labels[record["class_name"]]
    if records[0]["split"] is None:
        if spec.get("group_column") is not None:
            _group_split(records, spec, seed, cancelled)
        else:
            _stratify(records, spec, seed)
    counts, class_counts = _check_partition(records, class_names)
    if spec["kind"] == "csv":
        train = [record["features"] for record in records if record["split"] == "train"]
        width = len(spec["feature_columns"])
        mean = [math.fsum(row[column] / len(train) for row in train) for column in range(width)] if spec["standardize"] else [0.0] * width
        std = [math.sqrt(math.fsum(((row[column] - mean[column]) ** 2) / len(train) for row in train)) or 1.0 for column in range(width)] if spec["standardize"] else [1.0] * width
        if any(not math.isfinite(value) for value in mean + std):
            raise ValueError("Non-finite train-only standardization")
        for record in records:
            processed = [(value - mean[index]) / std[index] for index, value in enumerate(record["features"])]
            if any(not math.isfinite(value) or abs(value) > MAX_FLOAT32 for value in processed):
                raise ValueError("Preprocessed float32 value out of range at " + _source_name(record))
            record["raw_content_digest"] = record["content_digest"]
            record["content_digest"] = _float32_digest(processed)
        # Raw double precision rows may be distinct while the actual model
        # inputs collide after train-only scaling and float32 conversion.
        _check_partition(records, class_names)
        preprocessing = {"kind": "standardize" if spec["standardize"] else "identity", "fit_split": "train", "dtype": "float32", "feature_columns": spec["feature_columns"], "mean": mean, "std": std}
        input_shape = [None, width]
    else:
        preprocessing = {"kind": "resize_scale", "image_size": spec["image_size"], "channels": spec["channels"], "resampling": "bilinear", "scale": 1.0 / 255.0}
        input_shape = [None, spec["channels"], *spec["image_size"]]
    ignored_files.sort(key=lambda item: item["relative_path"])
    warnings = ["Ignored " + str(len(ignored_files)) + " known system/synchronization sidecar files; see ignored_files"] if ignored_files else []
    manifest = {"version": 2, "kind": spec["kind"], "data_spec": spec, "seed": seed, "class_names": class_names,
                "input_shape": input_shape, "counts": counts, "class_counts": class_counts, "records": records, "preprocessing": preprocessing,
                "ignored_files": ignored_files, "warnings": warnings}
    manifest["dataset_digest"] = _manifest_digest(manifest)
    _cancel(cancelled)
    return manifest


def build_datasets(manifest, cancelled=None):
    """Return Torch datasets; enforce the inspected source hashes before use."""
    if manifest.get("kind") == "paired_csv":
        from .paired_data import build_paired_datasets
        return build_paired_datasets(manifest, cancelled=cancelled)
    if manifest.get("version") != 2 or manifest.get("kind") not in ("csv", "image_folder"):
        raise ValueError("Unsupported classification manifest")
    if _manifest_digest(manifest) != manifest.get("dataset_digest"):
        raise ValueError("Dataset manifest digest does not match its contents")
    checked = {}
    for record in manifest["records"]:
        _cancel(cancelled)
        source = record["source_path"]
        expected = record["source_digest"]
        if source not in checked:
            raw = _read_source(source, MAX_CSV_BYTES if manifest["kind"] == "csv" else MAX_IMAGE_BYTES, cancelled)
            checked[source] = hashlib.sha256(raw).hexdigest()
        if checked[source] != expected:
            raise ValueError("Dataset source changed since inspection: " + source)

    import torch
    from torch.utils.data import Dataset

    class FrozenDataset(Dataset):
        def __init__(self, records):
            self.records = records
            self.preprocessing = manifest["preprocessing"]
            self.matrix = None
            if manifest["kind"] == "csv":
                mean, std = self.preprocessing["mean"], self.preprocessing["std"]
                self.matrix = torch.tensor([[(value - mean[index]) / std[index] for index, value in enumerate(record["features"])] for record in records], dtype=torch.float32)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            _cancel(cancelled)
            record = self.records[index]
            if self.matrix is not None:
                tensor = self.matrix[index]
            else:
                raw = _read_source(record["source_path"], MAX_IMAGE_BYTES, cancelled)
                if hashlib.sha256(raw).hexdigest() != record["source_digest"]:
                    raise ValueError("Dataset source changed since inspection: " + record["source_path"])
                image, _ = _decode_image(raw, record["source_path"])
                channels = self.preprocessing["channels"]
                height, width = self.preprocessing["image_size"]
                image = _prepare_image(image, self.preprocessing["image_size"], channels)
                tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, channels).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
            return tensor, record["label"], record["id"]

    return {split: FrozenDataset([record for record in manifest["records"] if record["split"] == split]) for split in SPLITS}
