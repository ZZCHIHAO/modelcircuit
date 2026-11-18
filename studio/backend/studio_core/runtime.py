"""PyTorch execution with shared modules, explicit devices and bounded probes."""
import math

import torch
from torch import nn

from .schema import prepare_graph, resource_limits, graph_resource_limits
from .execution_policy import check_input_batches, check_config_batch, hardware_memory_errors
from .shape_rules import CNN_PARAMETERS, cnn_output_shape
from .operator_runtime import make_operator, _Add, _Concat, _Multiply

MAX_OUTPUT_PREVIEW_ELEMENTS = 4096
MAX_OUTPUT_PREVIEW_PER_TENSOR = 256


def _check_runtime_budget(elements, total=None, limits=None):
    limits = limits if limits is not None else resource_limits()
    if limits.max_tensor_elements is not None and elements > limits.max_tensor_elements:
        raise ValueError("Runtime tensor element budget exceeded")
    if total is not None and limits.max_inferred_elements is not None and total > limits.max_inferred_elements:
        raise ValueError("Runtime cumulative tensor budget exceeded")


def _output_elements(component, arguments, instance=None):
    """Check concrete dynamic output size before a primitive allocates it."""
    if isinstance(component, _GraphModule): return None
    if isinstance(component, (_Add, _Concat, _Multiply)):
        shape = list(arguments["a"].shape)
        if isinstance(component, _Concat):
            other = list(arguments["b"].shape)
            dim = component.dim % len(shape)
            if len(shape) != len(other) or any(a != b for i, (a, b) in enumerate(zip(shape, other)) if i != dim):
                raise ValueError("Concat runtime dimensions must match outside its concatenation axis")
            shape[dim] += other[dim]
        elif isinstance(component, _Multiply) and arguments["a"].shape != arguments["b"].shape:
            raise ValueError("Multiply requires equal runtime shapes; broadcasting is not implicit")
    else:
        shape = list(arguments["x"].shape)
        if instance is not None and instance["kind"] in CNN_PARAMETERS:
            shape = cnn_output_shape(instance["kind"], instance["params"], shape)
        elif isinstance(component, nn.Linear): shape[-1] = component.out_features
    return math.prod(shape)


class _ExecutionBudget:
    def __init__(self, limits):
        self.limits, self.total = limits, 0

    def check(self, elements):
        _check_runtime_budget(elements, self.total + elements, self.limits)

    def add(self, elements):
        self.check(elements)
        self.total += elements


class _GraphModule(nn.Module):
    def __init__(self, graph, limits=None, capture=True):
        super().__init__()
        self.graph = graph
        self.limits = limits if limits is not None else graph_resource_limits(graph)
        self.capture = capture
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.components = nn.ModuleDict()
        self.last_values = {}
        for instance in graph["instances"]:
            kind, params = instance["kind"], instance.get("params", {})
            if kind == "chip":
                component = _GraphModule(instance["_graph"], self.limits, capture)
            else: component = make_operator(kind, params)
            if instance.get("frozen", False): component.requires_grad_(False)
            self.components[instance["id"]] = component

    def set_capture(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("capture must be a boolean")
        self.capture = enabled
        self.last_values = {}
        for component in self.components.values():
            if isinstance(component, _GraphModule): component.set_capture(enabled)
        return self

    def forward(self, **inputs):
        check_input_batches(inputs, self.limits)
        return self._execute(inputs, _ExecutionBudget(self.limits))

    def _execute(self, inputs, budget):
        self.last_values = {}
        expected = {p["id"] for p in self.graph["inputs"]}
        if inputs.keys() != expected:
            raise ValueError("Input names must be exactly: " + ", ".join(sorted(expected)))
        for port in self.graph["inputs"]:
            value, shape = inputs[port["id"]], port["shape"]
            if not isinstance(value, torch.Tensor) or value.device != self._device_anchor.device or value.dtype != torch.float32:
                raise ValueError("Input " + port["id"] + " must be a float32 tensor on " + str(self._device_anchor.device))
            if value.ndim != len(shape) or any(dim is not None and value.shape[i] != dim for i, dim in enumerate(shape)):
                raise ValueError("Input shape mismatch for " + port["id"] + ": expected " + str(shape) + ", received " + str(list(value.shape)))
            budget.add(value.numel())
        values = {"input:" + key: value for key, value in inputs.items()}
        observed = {key: value.detach().clone() for key, value in values.items()} if self.capture else {}
        instances = {instance["id"]: instance for instance in self.graph["instances"]}
        for call in self.graph["_ordered_calls"]:
            component = self.components[call["instance_id"]]
            arguments = {key: values[source] for key, source in call["inputs"].items()}
            estimated_elements = _output_elements(component, arguments, instances[call["instance_id"]])
            if estimated_elements is not None:
                budget.check(estimated_elements)
            if isinstance(component, _GraphModule):
                value = component._execute(arguments, budget)
            elif "a" in arguments:
                value = component(**arguments)
            else:
                try:
                    value = component(arguments["x"])
                except ValueError as error:
                    if isinstance(component, nn.BatchNorm2d):
                        raise ValueError("call:" + call["id"] + ": " + str(error)) from error
                    raise
            source = "call:" + call["id"]
            outputs = value if isinstance(component, _GraphModule) else {"out": value}
            if self.capture and isinstance(component, _GraphModule):
                observed.update({source + "/" + key: child for key, child in component.last_values.items()})
            child_sources = {output["id"]: output["source"] for output in component.graph["outputs"]} if isinstance(component, _GraphModule) else {}
            measured, snapshots = set(), {}
            for port, tensor in outputs.items():
                charge_key = id(tensor)
                if self.limits.enforce_batch_bound and isinstance(component, _GraphModule):
                    # Only source aliases of this call share a charge. Distinct
                    # Identity calls can return the same Tensor object and must
                    # retain their conservative execution-path charges.
                    charge_key = component.graph['_output_aliases'][port]
                if charge_key not in measured:
                    budget.add(tensor.numel())
                    measured.add(charge_key)
                if self.capture and id(tensor) not in snapshots:
                    child_snapshot = component.last_values.get(child_sources[port]) if port in child_sources else None
                    snapshots[id(tensor)] = child_snapshot if child_snapshot is not None else tensor.detach().clone()
                refs = ([source] if len(outputs) == 1 else []) + [source + ":" + port]
                for ref in refs:
                    values[ref] = tensor
                    if self.capture: observed[ref] = snapshots[id(tensor)]
        self.last_values = observed
        return {output["id"]: values[output["source"]] for output in self.graph["outputs"]}


def build_model(graph, device="cpu", capture=True, *, memory_report=None):
    """Create one module per instance on an explicit CPU or single CUDA device."""
    if type(capture) is not bool:
        raise ValueError("capture must be a boolean")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Graph execution supports CPU or a single CUDA device")
    resolved, _, _ = prepare_graph(graph)
    if graph.get('resource_profile') == 'hardware-v5' and memory_report is None:
        from .memory_admission import admit_memory
        memory_report = admit_memory(resolved, device, capture=capture)
    try:
        model = _GraphModule(resolved, graph_resource_limits(graph), capture).to(device)
    except (MemoryError, torch.OutOfMemoryError) as error:
        if graph.get('resource_profile') == 'hardware-v5':
            raise ValueError('Hardware memory allocation failed while constructing model on ' + str(device) + ': ' + str(error)) from error
        raise
    if memory_report is not None: model.memory_report = memory_report
    return model


def _config(config):
    if not isinstance(config, dict): raise ValueError("Experiment config must be an object")
    values = {"steps": 20, "lr": 0.05, "seed": 42, "batch_size": 16}
    if not config.keys() <= values.keys(): raise ValueError("Unsupported experiment configuration key")
    values.update(config)
    for key, low, high in (("steps", 1, 200), ("seed", 0, 2**31 - 1), ("batch_size", 2, 64)):
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise ValueError(key + " must be an integer between " + str(low) + " and " + str(high))
    if type(values["lr"]) not in (int, float) or not 0.001 <= values["lr"] <= 1 or not math.isfinite(values["lr"]):
        raise ValueError("lr must be between 0.001 and 1")
    return values


def _probe(value, limit=256):
    value = value.detach().cpu()
    flat = value.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    sample = flat[:limit]
    sampled = value if value.numel() <= limit else sample

    def json_values(tensor):
        if tensor.ndim == 0:
            number = tensor.item()
            return number if math.isfinite(number) else None
        return [json_values(child) for child in tensor]

    return {
        "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
        "count": value.numel(), "sample_size": sample.numel(),
        "finite_count": finite.numel(), "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "min": float(finite.min()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
        "mean": float(finite.mean()) if finite.numel() else None,
        "std": float(finite.std(unbiased=False)) if finite.numel() else None,
        "values": json_values(sampled), "source": "measured",
    }


def _shared_instances(model, prefix=""):
    shared = []
    by_instance = {}
    for call in model.graph["calls"]:
        by_instance.setdefault(call["instance_id"], []).append(call["id"])
    for instance_id, call_ids in by_instance.items():
        component = model.components[instance_id]
        # Resolve each call through the actual ModuleDict and check identity.
        references = [model.components[c["instance_id"]] for c in model.graph["calls"] if c["id"] in call_ids]
        if any(reference is not component for reference in references):
            raise RuntimeError("Shared module identity was not preserved")
        if len(call_ids) > 1:
            shared.append({"instance_id": prefix + instance_id, "call_ids": [prefix + name for name in call_ids]})
        if isinstance(component, _GraphModule): shared.extend(_shared_instances(component, prefix + instance_id + "/"))
    return shared


def _check_finite_observations(model, outputs, output_sources):
    # Report the public output name first; it points users to the failing view.
    for name, value in outputs.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Non-finite output '" + name + "' from " + output_sources[name])
    for source, value in model.last_values.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Non-finite tensor source " + source)


def _output_previews(outputs, output_sources):
    # Share the global budget fairly so every declared output has visible data.
    limit = min(MAX_OUTPUT_PREVIEW_PER_TENSOR, MAX_OUTPUT_PREVIEW_ELEMENTS // len(outputs))
    previews, metadata = {}, {}
    for name, value in outputs.items():
        value = value.detach()
        count = value.numel()
        preview_count = min(count, limit)
        truncated = preview_count < count
        preview = value.reshape(-1)[:preview_count] if truncated else value
        previews[name] = preview.tolist()
        metadata[name] = {
            "shape": list(value.shape), "count": count,
            "preview_count": preview_count, "truncated": truncated,
            "source": "measured", "tensor_source": output_sources[name],
        }
    return previews, metadata


@hardware_memory_errors
def run_experiment(graph, config, progress=None, cancelled=None):
    """Train a two-class synthetic CPU task and report observed tensor values.

    Inputs are one fixed seeded synthetic batch. Labels are the sign of the sum
    of each input's feature mean. History is pre-update training loss; before,
    after, accuracy and probes are evaluated on that same batch in eval mode.
    Parameter gradients are from the last optimization step and update norms
    compare final parameters with their initial values. JSON output values are
    previews bounded to 256 scalars per tensor and 4096 in total; accompanying
    metadata records original shapes/counts and any truncation. Model forward
    itself always returns complete tensors.
    """
    config = _config(config)
    check_config_batch(graph, config['batch_size'])

    def check_cancelled():
        if cancelled is not None and cancelled(): raise InterruptedError("Experiment cancelled")

    check_cancelled()
    resolved, shapes, count = prepare_graph(graph)
    ports = resolved["inputs"]
    if any(len(p["shape"]) != 2 or p["shape"][0] not in (None, config["batch_size"]) for p in ports):
        raise ValueError("Synthetic S1 training requires rank-two inputs with batch dimension null or batch_size")
    output_sources = {o["id"]: o["source"] for o in resolved["outputs"]}
    if "logits" not in output_sources or shapes[output_sources["logits"]] != [ports[0]["shape"][0], 2]:
        raise ValueError("Synthetic S1 training requires named logits output with shape [batch, 2]")
    memory_report = None
    if graph.get('resource_profile') == 'hardware-v5':
        from .memory_admission import admit_memory
        memory_report = admit_memory(resolved, 'cpu', batch_size=config['batch_size'], training=True,
                                     optimizer='sgd', capture=True, workflow='synthetic')
    # fork_rng restores the caller's CPU RNG state after a run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = _GraphModule(resolved).cpu()
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        inputs = {p["id"]: torch.randn(config["batch_size"], p["shape"][1], generator=generator) for p in ports}
        labels = (sum(tensor.mean(dim=1) for tensor in inputs.values()) > 0).long()
        parameters = dict(model.named_parameters())
        trainable = [p for p in parameters.values() if p.requires_grad]
        if not trainable: raise ValueError("Synthetic S1 training requires at least one trainable parameter")
        originals = {name: p.detach().clone() for name, p in parameters.items()}
        optimizer = torch.optim.SGD(trainable, lr=config["lr"])
        model.eval()
        with torch.no_grad():
            initial_outputs = model(**inputs)
            _check_finite_observations(model, initial_outputs, output_sources)
            loss_before = float(nn.functional.cross_entropy(initial_outputs["logits"], labels))
        history = []
        model.train()
        for step in range(1, config["steps"] + 1):
            check_cancelled()
            optimizer.zero_grad(set_to_none=True)
            training_outputs = model(**inputs)
            _check_finite_observations(model, training_outputs, output_sources)
            loss = nn.functional.cross_entropy(training_outputs["logits"], labels)
            if not torch.isfinite(loss): raise ValueError("Training produced a non-finite loss")
            if not loss.requires_grad:
                raise ValueError("Synthetic S1 logits must depend on trainable parameters")
            loss.backward()
            optimizer.step()
            observation = {"step": step, "loss": float(loss.detach())}
            history.append(observation)
            if progress is not None: progress({"step": step, "total": config["steps"], "loss": observation["loss"]})
        check_cancelled()
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            _check_finite_observations(model, outputs, output_sources)
            loss_after = float(nn.functional.cross_entropy(outputs["logits"], labels))
            accuracy = float((outputs["logits"].argmax(dim=1) == labels).float().mean())
        parameter_measurements = [{
            "name": name, "shape": list(parameter.shape), "trainable": parameter.requires_grad,
            "grad_norm": float(parameter.grad.norm()) if parameter.grad is not None else None,
            "update_norm": float((parameter.detach() - originals[name]).norm()),
        } for name, parameter in parameters.items()]
        output_previews, output_metadata = _output_previews(outputs, output_sources)
        result = {
            "status": "succeeded", "device": "cpu", "seed": config["seed"], "steps": config["steps"],
            "data_source": "synthetic_s1", "loss_before": loss_before, "loss_after": loss_after,
            "accuracy": accuracy, "history": history,
            "probes": {source: _probe(value) for source, value in model.last_values.items()},
            "parameters": parameter_measurements, "parameter_count": count,
            "shared_instances": _shared_instances(model),
            "outputs": output_previews, "output_metadata": output_metadata, "warnings": memory_report['warnings'] if memory_report else [],
        }
        if memory_report is not None: result['memory_report'] = memory_report
        return result
