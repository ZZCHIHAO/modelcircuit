"""Real, resumable supervised classification on a single selected device."""
import csv
import io
import json
import math
import os
import platform
import random
import tempfile
import time
from pathlib import Path

from .classification_data import build_datasets, inspect_dataset
from .execution_policy import hardware_memory_errors

MAX_RESULT_BYTES = 2_000_000


def _atomic_write(path, writer):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path, value):
    contents = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(contents))


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def _probe(value, limit):
    import torch
    tensor = value.detach().cpu()
    flat = tensor.reshape(-1)
    finite = flat[torch.isfinite(flat)].double()
    count = min(limit, flat.numel())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."), "count": tensor.numel(),
            "sample_size": count, "finite_count": finite.numel(), "nan_count": int(torch.isnan(flat).sum()), "inf_count": int(torch.isinf(flat).sum()),
            "min": float(finite.min()) if finite.numel() else None, "max": float(finite.max()) if finite.numel() else None,
            "mean": float(finite.mean()) if finite.numel() else None, "std": float(finite.std(unbiased=False)) if finite.numel() else None,
            "values": flat[:count].tolist(), "source": "measured"}


def _bounded_entries(entries, byte_budget):
    selected, used = [], 2
    for entry in entries:
        size = len(json.dumps(entry, allow_nan=False).encode("utf-8")) + 1
        if used + size <= byte_budget:
            selected.append(entry)
            used += size
    return selected


def _result_bytes(result):
    # Include the worker envelope, whose full serialized size is what the
    # coordinator validates. A result alone just below 2MB can still overflow.
    return len(json.dumps({"type": "result", "result": result}, ensure_ascii=False, allow_nan=False).encode("utf-8"))


@hardware_memory_errors
def run_classification(graph, config, artifact_directory, progress=None, cancelled=None):
    """Train complete epochs, select by validation loss, then test best once.

    CPU continuation restores model/optimizer, Torch/Python RNG, and the train
    DataLoader generator at the last fully committed epoch boundary. Probe
    gradients describe that selected epoch's last optimization minibatch.
    """
    from .experiment_spec import checkpoint_signature, model_digest, validate_classification_config
    from .task_adapters import get_task_adapter
    config = validate_classification_config(config)
    typed = isinstance(graph, dict) and graph.get('schema_version') == 2
    if not typed and 'attention_probe_requests' in config:
        raise ValueError('Attention probes require tensor-v1')
    task = get_task_adapter('classification', 1)(config)
    checkpoint_version = 3 if typed else 2
    adapter_info = {'adapter': {'id': task.id, 'version': task.version}} if typed else {}
    from .execution_policy import check_config_batch
    check_config_batch(graph, config["batch_size"])
    started = time.monotonic()

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("Classification experiment cancelled")
        if time.monotonic() - started > config["max_runtime_seconds"]:
            raise TimeoutError("Classification experiment exceeded max_runtime_seconds")

    total_steps = 0

    def emit(phase, epoch=0, batch=0, batches=0, step=0, **values):
        if progress is not None:
            progress({"phase": phase, "epoch": epoch, "epochs": config["epochs"], "batch": batch, "batches": batches,
                      "step": step, "total": total_steps, **values})

    # The callback used during scanning also enforces the explicit run deadline.
    def scanning_cancelled():
        check()
        return False

    check()
    emit("inspect")
    manifest = inspect_dataset(config["data"], seed=config["seed"], cancelled=scanning_cancelled)
    if config["expected_dataset_digest"] is not None and manifest["dataset_digest"] != config["expected_dataset_digest"]:
        raise ValueError("Dataset digest changed since inspection")
    from .schema import prepare_graph
    resolved, shapes, parameter_count = prepare_graph(graph)
    paired = task.paired
    output_sources = task.validate_contract(resolved, shapes, manifest)
    if paired:
        input_ports = config["input_ports"]
        input_metadata = {"input_shapes": manifest["input_shapes"], "input_ports": dict(input_ports)}
    else:
        input_metadata = {"input_shape": manifest["input_shape"]}
    from .devices import resolve_device
    device = resolve_device(config["device"])
    memory_report = None
    if graph.get('resource_profile') == 'hardware-v5':
        from .memory_admission import admit_memory
        host_dataset_bytes = 0
        if manifest['kind'] in ('csv', 'paired_csv'):
            feature_shape = manifest['input_shapes']['tabular'] if paired else manifest['input_shape']
            host_dataset_bytes = sum(manifest['counts'].values()) * math.prod(feature_shape[1:]) * 4
        # drop_last=False caps each loader at its inspected partition size.
        # The final probe is capped at 16 and cannot exceed the val loader.
        # Keep the configured batch in the experiment/checkpoint contract, but
        # do not call an unattainable full batch a mandatory storage minimum.
        execution_batch_size = max(min(config['batch_size'], manifest['counts'][split])
                                   for split in ('train', 'val', 'test'))
        minimum_batch_size = execution_batch_size
        if config['resume_checkpoint'] is not None:
            # Before loading the checkpoint, epochs may already be complete.
            # Only test and the capped val probe are guaranteed to execute;
            # retain possible training/epoch-val batches in the projection.
            minimum_batch_size = max(min(config['batch_size'], manifest['counts']['test']),
                                     min(config['batch_size'], 16, manifest['counts']['val']))
        memory_report = admit_memory(resolved, device, batch_size=execution_batch_size,
                                     minimum_batch_size=minimum_batch_size, training=True,
                                     optimizer=config['optimizer'], capture=False, workflow='classification',
                                     resume=config['resume_checkpoint'] is not None, host_dataset_bytes=host_dataset_bytes)
    import torch
    from torch.utils.data import DataLoader
    from .runtime import _shared_instances, build_model
    datasets = build_datasets(manifest, cancelled=scanning_cancelled)
    check()
    train_batches = math.ceil(len(datasets["train"]) / config["batch_size"])
    total_steps = config["epochs"] * train_batches
    identity = checkpoint_signature(graph, config, manifest["dataset_digest"])
    artifact_directory = Path(artifact_directory).resolve()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    paths = {"last_checkpoint": artifact_directory / "last.pt", "best_checkpoint": artifact_directory / "best.pt",
             "predictions": artifact_directory / "predictions.csv", "manifest": artifact_directory / "manifest.json",
             "run_spec": artifact_directory / "run_spec.json", "metrics": artifact_directory / "metrics.json",
             "parameter_measurements": artifact_directory / "parameters.json"}
    if memory_report is not None:
        paths['memory_report'] = artifact_directory / 'memory_report.json'
        _write_json(paths['memory_report'], memory_report)
    _write_json(paths["manifest"], manifest)
    from PIL import __version__ as pillow_version
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "pillow": pillow_version,
                   "platform": platform.platform(), "machine": platform.machine(), "cuda": None, "cudnn": None}
    if device.type == "cuda":
        environment["cuda"] = torch.version.cuda
        environment["cudnn"] = torch.backends.cudnn.version()
    _write_json(paths["run_spec"], {"version": checkpoint_version, "profile": "classification-v2", **adapter_info, "graph": graph, "config": config, "environment": environment,
                                  "model_digest": model_digest(graph), "dataset_digest": manifest["dataset_digest"], "checkpoint_signature": identity,
                                  "resolved_device": str(device), "data_spec": manifest["data_spec"],
                                  **({'memory_report': memory_report} if memory_report is not None else {})})
    python_rng = random.getstate()
    cuda_indices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_indices):
            random.seed(config["seed"])
            # torch.manual_seed also seeds every CUDA device, including devices
            # outside this run's RNG fork/reservation. Seed only selected RNGs.
            torch.random.default_generator.manual_seed(config["seed"])
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(config["seed"])
            capture_options = {'capture_policy': {'requests': config.get('attention_probe_requests', [])}} if typed else {}
            model = build_model(graph, device=device, capture=False, **capture_options,
                                **({'memory_report': memory_report} if memory_report is not None else {}))
            parameters = dict(model.named_parameters())
            trainable = [parameter for parameter in parameters.values() if parameter.requires_grad]
            if not trainable:
                raise ValueError("Classification logits require at least one trainable parameter")
            optimizer_class = torch.optim.SGD if config["optimizer"] == "sgd" else torch.optim.AdamW
            optimizer = optimizer_class(trainable, lr=config["lr"])
            initial_state = _cpu_tree(model.state_dict())
            generator = torch.Generator(device="cpu").manual_seed(config["seed"])
            train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
            history, global_step, completed_epochs, best = [], 0, 0, None
            if config["resume_checkpoint"] is not None:
                check()
                checkpoint_path = Path(config["resume_checkpoint"])
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ValueError("Cannot load local classification checkpoint: " + str(checkpoint_path) + ": " + str(exc)) from exc
                if (not isinstance(checkpoint, dict) or checkpoint.get('version') != checkpoint_version
                        or checkpoint.get('profile') != 'classification-v2' or checkpoint.get('checkpoint_signature') != identity
                        or (typed and checkpoint.get('adapter') != adapter_info['adapter'])):
                    raise ValueError("Classification checkpoint identity does not match graph, data, preprocessing or optimizer")
                completed_epochs = checkpoint["completed_epochs"]
                if type(completed_epochs) is not int or completed_epochs < 1 or completed_epochs > config["epochs"]:
                    raise ValueError("Checkpoint completed epochs exceed requested epochs or are invalid")
                model.load_state_dict(checkpoint["model_state"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                initial_state = checkpoint["initial_state"]
                history = checkpoint["history"]
                global_step = checkpoint["global_step"]
                best = checkpoint["best"]
                generator.set_state(checkpoint["shuffle_state"].cpu())
                torch.set_rng_state(checkpoint["rng_state"]["cpu"].cpu())
                random.setstate(checkpoint["rng_state"]["python"])
                if device.type == "cuda":
                    cuda_state = checkpoint["rng_state"].get("cuda")
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state.cpu(), device)
                _atomic_write(paths["last_checkpoint"], lambda stream: torch.save(checkpoint, stream))
                _atomic_write(paths["best_checkpoint"], lambda stream: torch.save(dict(best, best=best), stream))

            def checked_forward(batch):
                outputs = model(**batch['features'])
                for name, value in outputs.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Non-finite classification output " + name + " from " + output_sources[name])
                logits = outputs["logits"]
                if logits.ndim != 2 or logits.shape != (len(batch['targets']), len(manifest["class_names"])):
                    raise ValueError("Runtime logits shape does not match batch/class count")
                return outputs

            def evaluate(split, epoch, prediction_rows=None):
                model.eval()
                # An independent generator keeps validation/test iteration from
                # consuming dropout RNG or changing training shuffle order.
                loader = DataLoader(datasets[split], batch_size=config["batch_size"], shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(config["seed"] + 1))
                total_loss, correct, count = 0.0, 0, 0
                with torch.no_grad():
                    for batch_index, (features, labels, identifiers) in enumerate(loader, 1):
                        check()
                        batch = task.bind_batch((features, labels, identifiers), device)
                        labels = batch['targets']
                        outputs = checked_forward(batch)
                        logits = outputs['logits']
                        loss = task.loss(outputs, batch)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("Non-finite " + split + " loss")
                        predicted = logits.argmax(dim=1)
                        stats = task.summarize(outputs, batch, loss)
                        total_loss += stats['loss_sum']
                        correct += stats['correct']
                        count += stats['count']
                        if prediction_rows is not None:
                            probabilities = logits.softmax(1).detach().cpu().tolist()
                            for identifier, label, prediction, probability in zip(identifiers, labels.cpu().tolist(), predicted.cpu().tolist(), probabilities):
                                prediction_rows.append({"sample_id": identifier, "label": label, "label_name": manifest["class_names"][label],
                                                        "prediction": prediction, "prediction_name": manifest["class_names"][prediction],
                                                        "probabilities": json.dumps(probability, separators=(",", ":"), allow_nan=False)})
                        emit(split, epoch, batch_index, len(loader), global_step, loss=float(loss), accuracy=correct / count)
                check()
                return task.finalize({'loss_sum': total_loss, 'correct': correct, 'count': count})

            for epoch in range(completed_epochs + 1, config["epochs"] + 1):
                check()
                model.train()
                model.set_capture(False)
                total_loss, correct, count = 0.0, 0, 0
                for batch_index, raw_batch in enumerate(train_loader, 1):
                    check()
                    batch = task.bind_batch(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = checked_forward(batch)
                    loss = task.loss(outputs, batch)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Non-finite training loss")
                    if not loss.requires_grad:
                        raise ValueError("Classification logits must depend on trainable parameters")
                    loss.backward()
                    for name, parameter in parameters.items():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                            raise ValueError("Non-finite training gradient for " + name)
                    optimizer.step()
                    for name, parameter in parameters.items():
                        if not bool(torch.isfinite(parameter).all()):
                            raise ValueError("Non-finite training parameter after optimizer step: " + name)
                    global_step += 1
                    stats = task.summarize(outputs, batch, loss)
                    total_loss += stats['loss_sum']
                    correct += stats['correct']
                    count += stats['count']
                    emit("train", epoch, batch_index, train_batches, global_step, loss=float(loss.detach()), accuracy=correct / count)
                train_metrics = task.finalize({'loss_sum': total_loss, 'correct': correct, 'count': count})
                val_metrics = evaluate("val", epoch)
                measurements = [{"name": name, "shape": list(parameter.shape), "trainable": parameter.requires_grad,
                                 "grad_norm": float(parameter.grad.detach().double().norm()) if parameter.grad is not None else None,
                                 "update_norm": float((parameter.detach().cpu().double() - initial_state[name].double()).norm())}
                                for name, parameter in parameters.items()]
                history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                                "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "step": global_step, "loss": train_metrics["loss"]})
                completed_epochs = epoch
                snapshot = {"version": checkpoint_version, "profile": "classification-v2", **adapter_info, "checkpoint_signature": identity, "model_digest": model_digest(graph),
                            "dataset_digest": manifest["dataset_digest"], "preprocessing": manifest["preprocessing"], "class_names": manifest["class_names"],
                            **input_metadata, "config": config, "completed_epochs": completed_epochs, "global_step": global_step,
                            "model_state": _cpu_tree(model.state_dict()), "optimizer_state": _cpu_tree(optimizer.state_dict()), "initial_state": initial_state,
                            "history": list(history), "metrics": {"train": train_metrics, "val": val_metrics}, "parameters": measurements,
                            "shuffle_state": generator.get_state().cpu().clone(),
                            "rng_state": {"cpu": torch.get_rng_state().cpu().clone(), "python": random.getstate(),
                                          "cuda": torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None}}
                check()
                if best is None or val_metrics["loss"] < best["metrics"]["val"]["loss"]:
                    best = snapshot
                    _atomic_write(paths["best_checkpoint"], lambda stream: torch.save(dict(best, best=best), stream))
                _atomic_write(paths["last_checkpoint"], lambda stream: torch.save(dict(snapshot, best=best), stream))
                _write_json(paths["metrics"], {"history": history, "best_epoch": best["completed_epochs"], "completed_epochs": completed_epochs,
                                               "global_step": global_step, "metrics": snapshot["metrics"]})
                emit("epoch", epoch, train_batches, train_batches, global_step, loss=train_metrics["loss"],
                     train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
                     train_accuracy=train_metrics["accuracy"], val_accuracy=val_metrics["accuracy"])

            check()
            # Reload the actual atomic best artifact using the restricted loader.
            best_checkpoint = torch.load(paths["best_checkpoint"], weights_only=True, map_location="cpu")
            model.load_state_dict(best_checkpoint["model_state"], strict=True)
            best = best_checkpoint["best"]
            predictions = []
            test_metrics = evaluate("test", best["completed_epochs"], predictions)
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=["sample_id", "label", "label_name", "prediction", "prediction_name", "probabilities"])
            writer.writeheader()
            writer.writerows(predictions)
            _atomic_write(paths["predictions"], lambda stream: stream.write(output.getvalue().encode("utf-8")))
            check()
            probe_loader = DataLoader(datasets["val"], batch_size=min(config["batch_size"], 16), shuffle=False, num_workers=0,
                                      generator=torch.Generator().manual_seed(config["seed"] + 2))
            features, probe_labels, identifiers = next(iter(probe_loader))
            model.eval()
            model.set_capture(True)
            with torch.no_grad():
                outputs = checked_forward(task.bind_batch((features, probe_labels, identifiers), device))
            check()
            warnings = list(memory_report['warnings']) if memory_report is not None else []
            observations = list(model.last_values.items())
            if len(observations) > 128:
                observations = observations[:128]
                warnings.append("Tensor probes are limited to the first 128 observed sources")
            probe_limit = min(64, 4096 // max(1, len(observations)))
            probes = model.last_probes if typed else {source: _probe(value, probe_limit) for source, value in observations}
            limit = min(256, 4096 // max(1, len(outputs)))
            output_previews, output_metadata = {}, {}
            for name, value in outputs.items():
                value = value.detach().cpu()
                count = min(limit, value.numel())
                output_previews[name] = value.tolist() if value.numel() <= count else value.reshape(-1)[:count].tolist()
                output_metadata[name] = {"shape": list(value.shape), "count": value.numel(), "preview_count": count, "truncated": count < value.numel(),
                                         "source": "measured", "tensor_source": output_sources[name]}
            metrics = dict(best["metrics"], test=test_metrics)
            _write_json(paths["metrics"], {"history": history, "best_epoch": best["completed_epochs"], "completed_epochs": completed_epochs,
                                           "global_step": global_step, "metrics": metrics})
            _write_json(paths["parameter_measurements"], best["parameters"])
            measurements = best["parameters"]
            if len(measurements) > 1024:
                warnings.append("Parameter summary is limited to 1024 entries; full measurements are in parameters.json")
                measurements = measurements[:1024]
            result = {"status": "succeeded", "device": str(device), "seed": config["seed"], "profile": "classification-v2", "data_source": manifest["kind"],
                      "history": history, "metrics": metrics, "best_epoch": best["completed_epochs"], "completed_epochs": completed_epochs, "global_step": global_step,
                      "probes": probes, "parameters": measurements, "parameter_count": parameter_count, "shared_instances": _shared_instances(model),
                      "outputs": output_previews, "output_metadata": output_metadata, "warnings": warnings,
                      "dataset": {"dataset_digest": manifest["dataset_digest"], "class_names": manifest["class_names"], "counts": manifest["counts"], "manifest_path": str(paths["manifest"])},
                      "artifacts": {key: str(path) for key, path in paths.items()},
                      "probe_context": {"split": "val", "sample_ids": list(identifiers), "epoch": best["completed_epochs"], "mode": "eval", "device": str(device),
                                        "gradient_source": "last training minibatch of selected checkpoint epoch", "parameter_update_source": "selected checkpoint minus initial parameters"}}
            if typed:
                result.update(adapter_info, capture_policy=capture_options['capture_policy'])
            if memory_report is not None: result['memory_report'] = memory_report
            if paired:
                result["probe_context"].update(input_metadata)
                result["dataset"]["input_shapes"] = manifest["input_shapes"]
            if _result_bytes(result) >= MAX_RESULT_BYTES:
                instrument_path = artifact_directory / "instruments.json"
                _write_json(instrument_path, {key: result[key] for key in ("probes", "parameters", "shared_instances", "outputs", "output_metadata")})
                result["artifacts"]["instrument_summaries"] = str(instrument_path)
                result["probes"] = dict(_bounded_entries(list(probes.items()), 256 * 1024))
                result["parameters"] = _bounded_entries(measurements, 256 * 1024)
                result["shared_instances"] = _bounded_entries(result["shared_instances"], 128 * 1024)
                names = ["logits"] + [name for name in output_previews if name != "logits"]
                selected = _bounded_entries([({name: output_previews[name]}, {name: output_metadata[name]}) for name in names], 256 * 1024)
                result["outputs"] = {name: value for preview, _ in selected for name, value in preview.items()}
                result["output_metadata"] = {name: value for _, metadata in selected for name, value in metadata.items()}
                warnings.append("Instrument entries exceeding the control-message byte budget are saved in instruments.json")
            if _result_bytes(result) >= MAX_RESULT_BYTES:
                raise ValueError("Classification result summary exceeds the 2MB limit")
            return result
    finally:
        random.setstate(python_rng)
