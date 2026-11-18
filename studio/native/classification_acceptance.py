"""DPG-independent source/frozen acceptance on generated local test fixtures.

The supplied RunManager owns every training subprocess. This diagnostic runs
the explicitly selected device and preserves fixture files, checkpoints and its
JSON report for replay; its measurements are never presented as real data.
"""
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise RuntimeError("Classification acceptance: " + message)


def _wait(manager, graph, config):
    submitted = manager.submit(graph, config)
    run_id = submitted["run_id"]
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        finished = manager.get(run_id)
        if finished["status"] in {"succeeded", "failed", "cancelled"}:
            _require(finished["status"] == "succeeded", "worker " + run_id + " ended " + finished["status"] + ": " + str(finished.get("error")))
            result = finished["result"]
            _require(result["profile"] == "classification-v2" and result["device"] == config["device"], "worker did not execute classification on the requested device")
            _require(result["completed_epochs"] == config["epochs"], "worker did not finish all requested epochs")
            _require(all(Path(path).is_file() for path in result["artifacts"].values()), "a reported training artifact is missing")
            _require(any(value["update_norm"] > 0 for value in result["parameters"]), "training did not update parameters")
            _require(result["probe_context"]["split"] == "val", "probe did not use an actual validation batch")
            _require(result["probe_context"]["epoch"] == result["best_epoch"], "probe checkpoint provenance does not match selected epoch")
            return run_id, result
        time.sleep(0.05)
    manager.cancel(run_id)
    raise TimeoutError("Classification acceptance worker did not finish within 150 seconds: " + run_id)


def _fixtures(directory, device):
    from PIL import Image
    fixture_root = Path(tempfile.mkdtemp(prefix="classification-fixtures-", dir=directory))
    (fixture_root / "README.txt").write_text("Generated local CSV/PNG test fixtures only. No external, clinical or benchmark data. Explicit diagnostic device: " + device + ".\n", encoding="utf-8")
    csv_path = fixture_root / "classification.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature_a", "feature_b", "label", "subject", "split"])
        for index in range(24):
            split = "train" if index < 12 else "val" if index < 18 else "test"
            label = index % 2
            writer.writerow([(1 if label else -1) + index / 100, index / 30 + ((index * 5) % 7) / 100,
                             "class_" + str(label), "fixture_subject_" + str(index // 2), split])
    image_root = fixture_root / "images"
    for split_index, split in enumerate(("train", "val", "test")):
        for label in range(2):
            for index in range(4 if split == "train" else 2):
                path = image_root / split / ("class_" + str(label)) / (str(index) + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (10, 12), (30 + label * 120 + split_index * 12 + index, 20 + index, index * 7)).save(path)
    return fixture_root, {"kind": "csv", "path": str(csv_path), "target_column": "label", "feature_columns": ["feature_a", "feature_b"],
                          "group_column": "subject", "split_column": "split", "standardize": True}, {
                              "kind": "image_folder", "path": str(image_root), "split_mode": "directories", "image_size": [8, 8], "channels": 3}


def _graph(kind):
    if kind == "csv":
        shape = [None, 2]
        stages = [("dense", "Linear", {"in_features": 2, "out_features": 8}), ("activation", "ReLU", {}),
                  ("drop", "Dropout", {"p": 0.25}), ("head", "Linear", {"in_features": 8, "out_features": 2})]
    else:
        shape = [None, 3, 8, 8]
        stages = [("conv", "Conv2d", {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
                  ("activation", "ReLU", {}), ("max_pool", "MaxPool2d", {"kernel_size": 2}),
                  ("average_pool", "AdaptiveAvgPool2d", {"output_size": [1, 1]}), ("flatten", "Flatten", {}),
                  ("head", "Linear", {"in_features": 4, "out_features": 2})]
    graph = {"schema_version": 1, "resource_profile": "classification-v2", "id": "acceptance_" + kind, "name": "Generated test fixture classifier", "revision": 1,
             "inputs": [{"id": "x", "shape": shape, "dtype": "float32"}], "instances": [], "calls": [], "outputs": [], "definitions": []}
    source = "input:x"
    for identifier, primitive, params in stages:
        graph["instances"].append({"id": identifier, "kind": primitive, "params": params})
        graph["calls"].append({"id": identifier + "_call", "instance_id": identifier, "inputs": {"x": source}})
        source = "call:" + identifier + "_call"
    graph["outputs"].append({"id": "logits", "source": source})
    return graph


def _best_inference(graph, result, device):
    import torch
    from studio_core.classification_data import build_datasets
    from studio_core.runtime import build_model
    checkpoint = torch.load(result["artifacts"]["best_checkpoint"], weights_only=True, map_location="cpu")
    last = torch.load(result["artifacts"]["last_checkpoint"], weights_only=True, map_location="cpu")
    _require(checkpoint["completed_epochs"] == result["best_epoch"], "best checkpoint epoch disagrees with result")
    _require(last["completed_epochs"] == result["completed_epochs"], "last checkpoint epoch disagrees with result")
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8"))
    dataset = build_datasets(manifest)["test"]
    with Path(result["artifacts"]["predictions"]).open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    _require(len(predictions) == len(dataset) == result["metrics"]["test"]["count"], "prediction count differs from held-out test count")
    model = build_model(graph, device=device, capture=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    count, correct, total_loss, maximum_error = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(dataset), 4):
            samples = [dataset[index] for index in range(start, min(start + 4, len(dataset)))]
            features = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long, device=device)
            logits = model(x=features)["logits"]
            probabilities = logits.softmax(1)
            predicted = logits.argmax(1)
            total_loss += float(torch.nn.functional.cross_entropy(logits, labels)) * len(samples)
            correct += int((predicted == labels).sum())
            count += len(samples)
            for offset, (_, label, identifier) in enumerate(samples):
                saved = predictions[start + offset]
                _require(saved["sample_id"] == identifier and int(saved["label"]) == label, "saved prediction sample/label does not match manifest")
                _require(int(saved["prediction"]) == int(predicted[offset]), "independent best-weight prediction differs from saved prediction")
                expected = torch.tensor(json.loads(saved["probabilities"]))
                actual_probability = probabilities[offset].detach().cpu()
                error = float((expected - actual_probability).abs().max())
                maximum_error = max(maximum_error, error)
                _require(bool(torch.allclose(expected, actual_probability, rtol=1e-5, atol=1e-6)), "independent best-weight probabilities differ")
    loss, accuracy = total_loss / count, correct / count
    _require(math.isclose(loss, result["metrics"]["test"]["loss"], rel_tol=1e-6, abs_tol=1e-6), "independent held-out test loss differs")
    _require(accuracy == result["metrics"]["test"]["accuracy"], "independent held-out test accuracy differs")
    return {"predictions_equal": True, "probabilities_close": True, "count": count, "loss": loss, "accuracy": accuracy,
            "maximum_probability_error": maximum_error, "checkpoint_load": "weights_only=True, map_location=cpu", "inference_device": str(device),
            "probability_rtol": 1e-5, "probability_atol": 1e-6}


def _tree_equal(first, second):
    import torch
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(_tree_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(_tree_equal(a, b) for a, b in zip(first, second))
    return first == second


def _resume_check(continuous, resumed, first_result, first_run, resumed_run, device="cpu"):
    import torch
    actual = torch.load(resumed["artifacts"]["last_checkpoint"], weights_only=True, map_location="cpu")
    partial = torch.load(first_result["artifacts"]["last_checkpoint"], weights_only=True, map_location="cpu")
    _require(actual["config"]["resume_checkpoint"] == first_result["artifacts"]["last_checkpoint"], "continuation checkpoint origin was not recorded")
    _require(actual["config"]["device"] == partial["config"]["device"] == str(device) == resumed["device"], "continuation device provenance differs")
    _require(actual["completed_epochs"] == resumed["completed_epochs"] > partial["completed_epochs"], "continuation did not advance completed epochs")
    _require(actual["global_step"] == resumed["global_step"] > partial["global_step"], "continuation did not advance optimization steps")
    _require(actual["history"][:len(partial["history"])] == partial["history"], "continuation discarded the committed epoch history")
    provenance = {"continuation_succeeded": True, "origin_device": partial["config"]["device"], "target_device": str(device),
                  "completed_epochs": resumed["completed_epochs"], "global_step": resumed["global_step"], "partial_run_id": first_run,
                  "resumed_run_id": resumed_run, "source_checkpoint": first_result["artifacts"]["last_checkpoint"],
                  "checkpoint_path": resumed["artifacts"]["last_checkpoint"]}
    if str(device) != "cpu":
        _require(all(bool(torch.isfinite(value).all()) for value in actual["model_state"].values()), "CUDA continuation saved non-finite weights")
        return {**provenance, "bit_exact": None, "bit_exact_required": False, "comparison": "completed-epoch continuation; CUDA bit identity is not asserted"}
    expected = torch.load(continuous["artifacts"]["last_checkpoint"], weights_only=True, map_location="cpu")
    comparisons = {"model_state_equal": _tree_equal(expected["model_state"], actual["model_state"]),
                   "optimizer_state_equal": _tree_equal(expected["optimizer_state"], actual["optimizer_state"]),
                   "rng_equal": _tree_equal(expected["rng_state"], actual["rng_state"]),
                   "shuffle_equal": _tree_equal(expected["shuffle_state"], actual["shuffle_state"]),
                   "history_equal": continuous["history"] == resumed["history"], "metrics_equal": continuous["metrics"] == resumed["metrics"]}
    _require(all(comparisons.values()), "CPU completed-epoch continuation is not bit exact: " + str(comparisons))
    return {**provenance, "bit_exact": True, "bit_exact_required": True, "comparison": "bit_exact", **comparisons}


def _cnn_round_trip(graph, result, directory, device):
    import torch
    from studio_core.classification_data import build_datasets
    from studio_core.python_io import export_python, import_python
    from studio_core.runtime import build_model
    source = export_python(graph)
    imported = import_python(source)
    source_path, graph_path = directory / "cnn-export.py", directory / "cnn-imported.json"
    source_path.write_text(source, encoding="utf-8")
    graph_path.write_text(json.dumps(imported, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    state = torch.load(result["artifacts"]["best_checkpoint"], weights_only=True, map_location="cpu")["model_state"]
    original, restored = build_model(graph, device=device, capture=False), build_model(imported, device=device, capture=False)
    original.load_state_dict(state, strict=True)
    restored.load_state_dict(state, strict=True)
    namespace = {"__name__": "generated_acceptance_cnn"}
    # This source was generated above from our fixed fixture graph. Arbitrary
    # supplied Python remains parsed statically by the production importer.
    exec(compile(source, str(source_path), "exec"), namespace)
    standalone = namespace["Model"]().to(device)
    standalone.load_state_dict(state, strict=True)
    original.eval(); restored.eval(); standalone.eval()
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8"))
    dataset = build_datasets(manifest)["val"]
    features = torch.stack([dataset[index][0] for index in range(min(2, len(dataset)))]).to(device)
    with torch.no_grad():
        expected = original(x=features)["logits"]
        restored_output = restored(x=features)["logits"]
        standalone_output = standalone(x=features)["logits"]
    imported_equal, standalone_equal = torch.equal(expected, restored_output), torch.equal(expected, standalone_output)
    forward_close = bool(torch.allclose(expected, restored_output, rtol=1e-5, atol=1e-6)) and bool(torch.allclose(expected, standalone_output, rtol=1e-5, atol=1e-6))
    _require((imported_equal and standalone_equal) if str(device) == "cpu" else forward_close, "CNN exported/parsed/standalone forwards differ with identical selected weights")
    error = max(float((expected - restored_output).abs().max()), float((expected - standalone_output).abs().max()))
    return {"forward_equal": imported_equal and standalone_equal, "forward_close": forward_close, "imported_forward_equal": imported_equal, "standalone_forward_equal": standalone_equal,
            "input_shape": list(features.shape), "output_shape": list(expected.shape), "maximum_absolute_error": error, "device": str(device),
            "comparison": "bit_exact" if str(device) == "cpu" else "allclose", "rtol": 1e-5, "atol": 1e-6,
            "profile": "classification-pytorch-v2", "source_path": str(source_path), "imported_graph_path": str(graph_path)}


def check_classification_runtime(manager, directory, device="cpu"):
    """Run CSV MLP, resume, and CNN on one explicit CPU/CUDA device."""
    from studio_core.classification_data import inspect_dataset
    from studio_core.devices import device_capabilities
    from studio_core.experiment_spec import validate_classification_config
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "classification-acceptance.json"
    report = {"status": "running", "kind": "native-classification-runtime", "profile": "classification-v2", "test_fixtures": True,
              "fixture_description": "Generated local CSV/PNG test fixtures; no external, clinical or benchmark data", "device": device,
              "frozen": bool(getattr(sys, "frozen", False)), "gpu_execution": {"executed": False, "reason": "No CUDA worker has completed in this diagnostic" if device != "cpu" else "This diagnostic explicitly runs CPU only"},
              "tasks": {}, "report_path": str(report_path)}
    try:
        report["capabilities"] = device_capabilities()
        fixture_root, csv_spec, image_spec = _fixtures(directory, device)
        report["fixture_directory"] = str(fixture_root)
        base = {"profile": "classification-v2", "epochs": 3, "batch_size": 4, "lr": 0.02, "optimizer": "adamw", "seed": 77,
                "device": device, "max_runtime_seconds": 120}
        csv_manifest = inspect_dataset(csv_spec, seed=77)
        csv_config = validate_classification_config(dict(base, data=csv_spec, expected_dataset_digest=csv_manifest["dataset_digest"]))
        csv_graph = _graph("csv")
        csv_run, csv_result = _wait(manager, csv_graph, csv_config)
        if device != "cpu":
            report["gpu_execution"] = {"executed": True, "device": csv_result["device"], "reason": "Actual classification worker completed on the requested CUDA device"}
        report["tasks"]["csv"] = {"run_id": csv_run, "test_fixtures": True, "completed_epochs": csv_result["completed_epochs"], "best_epoch": csv_result["best_epoch"],
                                   "global_step": csv_result["global_step"], "metrics": csv_result["metrics"], "dataset": csv_result["dataset"], "artifacts": csv_result["artifacts"],
                                   "checkpoint_inference": _best_inference(csv_graph, csv_result, device)}
        first_run, first_result = _wait(manager, csv_graph, dict(csv_config, epochs=1))
        resumed_run, resumed = _wait(manager, csv_graph, dict(csv_config, resume_checkpoint=first_result["artifacts"]["last_checkpoint"]))
        report["resume"] = _resume_check(csv_result, resumed, first_result, first_run, resumed_run, device=device)
        image_manifest = inspect_dataset(image_spec, seed=77)
        image_config = validate_classification_config(dict(base, data=image_spec, optimizer="sgd", lr=0.05, expected_dataset_digest=image_manifest["dataset_digest"]))
        image_graph = _graph("image_folder")
        image_run, image_result = _wait(manager, image_graph, image_config)
        report["tasks"]["image_folder"] = {"run_id": image_run, "test_fixtures": True, "completed_epochs": image_result["completed_epochs"], "best_epoch": image_result["best_epoch"],
                                            "global_step": image_result["global_step"], "metrics": image_result["metrics"], "dataset": image_result["dataset"], "artifacts": image_result["artifacts"],
                                            "checkpoint_inference": _best_inference(image_graph, image_result, device)}
        report["cnn_python_round_trip"] = _cnn_round_trip(image_graph, image_result, directory, device)
        report["status"] = "passed"
        return report
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
