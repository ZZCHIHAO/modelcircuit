"""Headless S3 acceptance using only generated paired CSV/PNG test fixtures.

Fixture/document helpers import neither Torch nor DPG and are shared by native
render checks. RunManager ownership and worker shutdown remain with the caller.
"""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile


def _require(condition, message):
    if not condition:
        raise RuntimeError("S3 acceptance: " + message)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def create_paired_fixture(directory):
    """Return {fixture_directory,data_spec}; preserve generated samples on disk."""
    from PIL import Image
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="s3-paired-fixtures-", dir=directory))
    images = root / "images"; images.mkdir()
    (root / "README.txt").write_text("Generated local paired CSV/PNG test fixtures only. No external, clinical or benchmark data; metrics are diagnostic evidence only.\n", encoding="utf-8")
    csv_path = root / "paired.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "image_path", "f", "constant", "label", "patient_id", "split"])
        for index in range(24):
            identifier = "sample_" + str(index).zfill(3)
            reference = "images/" + identifier + ".png"
            Image.new("RGB", (10, 11), (20 + index * 5, index, 10 + (index % 2) * 100)).save(root / reference)
            writer.writerow([identifier, reference, index % 3, 1, "class_" + str(index % 2), "fixture_patient_" + str(index // 2),
                             "train" if index < 12 else "val" if index < 18 else "test"])
    data = {"kind": "paired_csv", "path": str(csv_path), "image_root": "", "image_column": "image_path", "id_column": "sample_id",
            "target_column": "label", "feature_columns": ["f", "constant"], "group_column": "patient_id", "split_column": "split",
            "standardize": True, "image_size": [8, 8], "channels": 3, "validation_fraction": 0.2, "test_fraction": 0.2}
    return {"fixture_directory": str(root), "data_spec": data}


def build_s3_document(manifest):
    """Return (Document, metadata): native starter -> named fusion -> sharing."""
    from .document import Document
    from .workspace_state import starter_graph
    document = Document(starter_graph(manifest))
    _require(document.validation()["valid"], "paired native starter is invalid")
    definition_id = "paired_fusion_chip"
    reference = document.encapsulate(["encode_image", "encode_table", "fuse"], "Fixture fusion chip", definition_id,
                                     outputs={"fused": "call:fuse:out", "image_features": "call:encode_image:features"})
    call_id = reference.split(":", 1)[1]
    document.set_input("classify", "x", reference + ":fused")
    document.set_output("features", reference + ":image_features")
    before = document.validation()["parameter_count"]
    shared = document.share(reference)
    shared_call_id = shared.split(":", 1)[1]
    document.add_output("shared_features", shared + ":image_features")
    report = document.validation()
    _require(report["valid"], "encapsulated/shared paired model is invalid: " + str(report["errors"]))
    _require(report["parameter_count"] == before, "sharing unexpectedly created extra parameters")
    calls = {call["id"]: call for call in document.graph["calls"]}
    _require(calls[call_id]["instance_id"] == calls[shared_call_id]["instance_id"], "shared call points at a different instance")
    metadata = {"definition_id": definition_id, "call_id": call_id, "shared_call_id": shared_call_id, "instance_id": calls[call_id]["instance_id"],
                "input_sources": deepcopy(calls[call_id]["inputs"]), "output_ports": document.output_ports(call_id),
                "parameter_count_before_share": before, "parameter_count_after_share": report["parameter_count"]}
    return document, metadata


def _model_inputs(modalities, mapping, device):
    return {mapping[modality]: tensor.to(device) for modality, tensor in modalities.items()}


def _independent_inference(graph, result, config, device):
    import torch
    from torch.utils.data import DataLoader
    from studio_core.classification_data import build_datasets
    from studio_core.runtime import build_model
    checkpoint = torch.load(result["artifacts"]["best_checkpoint"], weights_only=True, map_location="cpu")
    _require(checkpoint["completed_epochs"] == result["best_epoch"], "best checkpoint epoch differs from the reported selection")
    model = build_model(graph, device=device, capture=False)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8"))
    datasets = build_datasets(manifest)
    with Path(result["artifacts"]["predictions"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == len(datasets["test"]) == result["metrics"]["test"]["count"], "test prediction count disagrees with the manifest")
    total_loss, count, correct, maximum_error = 0.0, 0, 0, 0.0
    loader = DataLoader(datasets["test"], batch_size=config["batch_size"], shuffle=False, num_workers=0)
    with torch.no_grad():
        for modalities, labels, identifiers in loader:
            logits = model(**_model_inputs(modalities, config["input_ports"], device))["logits"]
            total_loss += float(torch.nn.functional.cross_entropy(logits, labels.to(device))) * len(labels)
            predicted, probability = logits.argmax(1).cpu(), logits.softmax(1).cpu()
            correct += int((predicted == labels).sum())
            for offset, identifier in enumerate(identifiers):
                saved = rows[count + offset]
                _require(saved["sample_id"] == identifier and int(saved["label"]) == int(labels[offset]), "saved row lost the paired sample identity")
                _require(int(saved["prediction"]) == int(predicted[offset]), "best checkpoint did not reproduce the saved prediction")
                expected = torch.tensor(json.loads(saved["probabilities"]))
                maximum_error = max(maximum_error, float((expected - probability[offset]).abs().max()))
                _require(bool(torch.allclose(expected, probability[offset], rtol=1e-5, atol=1e-6)), "best checkpoint probabilities differ")
            count += len(labels)
    loss, accuracy = total_loss / count, correct / count
    _require(math.isclose(loss, result["metrics"]["test"]["loss"], rel_tol=1e-6, abs_tol=1e-6), "independent test loss differs")
    _require(accuracy == result["metrics"]["test"]["accuracy"], "independent test accuracy differs")
    return {"predictions_equal": True, "probabilities_close": True, "count": count, "loss": loss, "accuracy": accuracy,
            "maximum_probability_error": maximum_error, "device": str(device), "checkpoint_load": "weights_only=True, map_location=cpu"}, model, datasets


def _compare_outputs(reference, observed, device, label):
    import torch
    _require(reference.keys() == observed.keys(), label + " output names differ")
    close = all(bool(torch.allclose(reference[key], observed[key], rtol=1e-5, atol=1e-6)) for key in reference)
    exact = all(torch.equal(reference[key], observed[key]) for key in reference)
    _require(exact if device == "cpu" else close, label + " forward differs with identical weights")
    error = max(float((reference[key] - observed[key]).abs().max()) for key in reference)
    return {"close": close, "bit_equal": exact, "maximum_absolute_error": error}


def _python_round_trip(document, model, modalities, config, directory, device):
    import torch
    from studio_core.python_io import import_python, python_profile
    from studio_core.runtime import build_model
    source = document.export_python()
    no_metadata = "\n".join(line for line in source.splitlines() if not line.startswith("STUDIO_VISUAL_METADATA = ")) + "\n"
    source_path, clean_path = directory / "paired-model.py", directory / "paired-model-no-metadata.py"
    source_path.write_text(source, encoding="utf-8"); clean_path.write_text(no_metadata, encoding="utf-8")
    imported = import_python(source); metadata_free = import_python(no_metadata)
    _write_json(directory / "paired-imported.json", imported)
    _write_json(directory / "paired-imported-no-metadata.json", metadata_free)
    candidates = {"imported": build_model(imported, device=device, capture=False), "metadata_free": build_model(metadata_free, device=device, capture=False)}
    namespace = {"__name__": "generated_s3_acceptance_model"}
    # Generated from the fixed native fixture graph above, never user source.
    exec(compile(source, str(source_path), "exec"), namespace)
    candidates["standalone"] = namespace["Model"]().to(device)
    state = model.state_dict()
    for candidate in candidates.values(): candidate.load_state_dict(state, strict=True); candidate.eval()
    inputs = _model_inputs(modalities, config["input_ports"], device)
    with torch.no_grad():
        reference = model(**inputs)
        comparisons = {label: _compare_outputs(reference, candidate(**inputs), device, label) for label, candidate in candidates.items()}
    _require(python_profile(document.graph) == "ports-pytorch-v3", "named fusion export did not select the ports profile")
    return {"profile": python_profile(document.graph), "device": str(device), "comparison": "bit_exact" if device == "cpu" else "allclose",
            "metadata_free_close": comparisons["metadata_free"]["close"], "standalone_close": comparisons["standalone"]["close"], "comparisons": comparisons}, {
                "python_source": str(source_path), "python_no_metadata": str(clean_path), "imported_graph": str(directory / "paired-imported.json"),
                "metadata_free_graph": str(directory / "paired-imported-no-metadata.json")}


def _library_forward(document, chip_path, metadata, model, modalities, config, device):
    import torch
    from .document import Document
    from studio_core.runtime import build_model
    graph = {"schema_version": 1, "resource_profile": "classification-v2", "id": "chip_library_replay", "revision": 1,
             "inputs": deepcopy(document.graph["inputs"]), "instances": [], "calls": [], "outputs": [{"id": "fused"}], "definitions": []}
    fresh = Document(graph)
    definition_id = fresh.import_chip(chip_path)
    reference = fresh.add("chip", definition_id)
    for port, source in metadata["input_sources"].items(): fresh.set_input(reference, port, source)
    fresh.set_output("fused", reference + ":fused")
    fresh.add_output("image_features", reference + ":image_features")
    _require(fresh.validation()["valid"], "fresh library import is not a runnable named chip")
    replay = build_model(fresh.graph, device=device, capture=False).eval()
    fresh_call = next(call for call in fresh.graph["calls"] if "call:" + call["id"] == reference)
    selected = model.components[metadata["instance_id"]]
    replay.components[fresh_call["instance_id"]].load_state_dict(selected.state_dict(), strict=True)
    inputs = _model_inputs(modalities, config["input_ports"], device)
    boundaries = {port: inputs[source.split(":", 1)[1]] for port, source in metadata["input_sources"].items()}
    with torch.no_grad():
        comparison = _compare_outputs(selected(**boundaries), replay(**inputs), device, "fresh chip library")
    return comparison


def check_s3_runtime(manager, directory, device="cpu"):
    """Perform the native S3 workflow; caller closes its supplied RunManager."""
    from .classification_acceptance import _resume_check, _wait
    from .workspace_state import compatibility_issue, read_workspace, save_workspace
    from studio_core.classification_data import inspect_dataset
    from studio_core.devices import device_capabilities
    from studio_core.experiment_spec import validate_classification_config
    import torch
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "s3-acceptance.json"
    report = {"status": "running", "kind": "native-s3-runtime", "profile": "classification-v2", "test_fixtures": True,
              "fixture_description": "Generated paired CSV/PNG diagnostic fixtures only; no scientific performance claim",
              "device": device, "frozen": bool(getattr(sys, "frozen", False)), "worker_cleanup": "caller",
              "gpu_execution": {"executed": False, "reason": "No CUDA worker has completed" if device != "cpu" else "Explicit CPU-only diagnostic"},
              "report_path": str(report_path)}
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report["capabilities"] = device_capabilities()
        fixture = create_paired_fixture(directory); report["fixture_directory"] = fixture["fixture_directory"]
        manifest = inspect_dataset(fixture["data_spec"], seed=77)
        document, metadata = build_s3_document(manifest)
        config = validate_classification_config({"profile": "classification-v2", "data": fixture["data_spec"], "epochs": 3, "batch_size": 4,
                                                "optimizer": "adamw", "lr": 0.02, "seed": 77, "device": device, "max_runtime_seconds": 120,
                                                "expected_dataset_digest": manifest["dataset_digest"]})
        _require(not compatibility_issue(document, manifest, input_ports=config["input_ports"]), "native paired compatibility check failed")
        chip_path = document.save_chip(metadata["definition_id"], "1.0.0", directory / "chips")
        workspace_path = directory / "paired-workspace.json"
        save_workspace(workspace_path, document, config)
        restored_document, restored_config = read_workspace(workspace_path)
        _require(restored_document.graph == document.graph and restored_config == config, "workspace reopen changed graph or experiment")
        before = deepcopy(restored_document.graph)
        imported_id = restored_document.import_chip(chip_path)
        reused = imported_id == metadata["definition_id"] and restored_document.graph == before
        _require(reused, "reimport of the saved immutable chip did not reuse its definitions")
        document = restored_document
        run_id, result = _wait(manager, document.graph, config)
        if device != "cpu": report["gpu_execution"] = {"executed": True, "device": result["device"], "reason": "Actual paired classification workers completed"}
        report["training"] = {"run_id": run_id, "completed_epochs": result["completed_epochs"], "best_epoch": result["best_epoch"], "global_step": result["global_step"],
                              "metrics": result["metrics"], "dataset": result["dataset"], "parameter_count": result["parameter_count"], "probe_context": result["probe_context"]}
        shared = next((item for item in result["shared_instances"] if item["instance_id"] == metadata["instance_id"]), None)
        _require(shared is not None and set(shared["call_ids"]) == {metadata["call_id"], metadata["shared_call_id"]}, "real worker did not preserve the shared fusion instance")
        _require("input:image" in result["probes"] and "input:clinical" in result["probes"], "both actual input probes were not captured")
        _require("call:" + metadata["call_id"] + ":fused" in result["probes"], "named fusion output probe is missing")
        report["sharing"] = {"same_instance": True, **shared, "parameter_count_before_share": metadata["parameter_count_before_share"], "parameter_count_after_share": metadata["parameter_count_after_share"]}
        inference, model, datasets = _independent_inference(document.graph, result, config, device)
        report["checkpoint_inference"] = inference
        first_run, first = _wait(manager, document.graph, dict(config, epochs=1))
        resumed_run, resumed = _wait(manager, document.graph, dict(config, resume_checkpoint=first["artifacts"]["last_checkpoint"]))
        report["resume"] = _resume_check(result, resumed, first, first_run, resumed_run, device=device)
        modalities = {key: torch.stack([datasets["val"][index][0][key] for index in range(2)]) for key in ("image", "tabular")}
        python_report, python_artifacts = _python_round_trip(document, model, modalities, config, directory, device)
        report["python_round_trip"] = python_report
        library = _library_forward(document, chip_path, metadata, model, modalities, config, device)
        report["storage"] = {"workspace_equal": True, "chip_reimport_reused": reused, "fresh_library_forward_close": library["close"],
                             "fresh_library_comparison": library, "definition_id": metadata["definition_id"], "output_ports": metadata["output_ports"]}
        report["artifacts"] = {**result["artifacts"], **python_artifacts, "workspace": str(workspace_path), "chip_library": str(chip_path), "report": str(report_path)}
        report["status"] = "passed"
        return report
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        torch.set_num_threads(prior_threads)
        _write_json(report_path, report)
