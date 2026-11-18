"""Pure arithmetic D0 fixture. Never imported by production code.

This creates a sizing ledger, not a runnable graph or a component factory.
All wiring in the storage-sizing proxy is deliberately the string AUDIT_ONLY.
No torch, production shape helper, model, allocation, training or forward pass
is used. The candidate changes only the null-batch substitution from 64 to an
explicit bound; existing hard limits and nesting charges are retained.
"""
from dataclasses import dataclass
import json
import math


LIMITS = {"max_parameters": 20_000_000, "max_tensor_elements": 16_777_216,
          "max_inferred_elements": 134_217_728}


def candidate_elements(shape, batch_bound=2):
    if type(batch_bound) is not int or not 1 <= batch_bound <= 64:
        raise ValueError("Candidate batch bound must be an integer in 1..64")
    if not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 8:
        raise ValueError("Candidate shape must have rank 1..8")
    axes = []
    for index, value in enumerate(shape):
        if index == 0 and value is None:
            axes.append(batch_bound)
        elif type(value) is int and value > 0:
            axes.append(value)
        else:
            raise ValueError("Only a null batch or positive integer axis is allowed")
    return math.prod(axes)


@dataclass
class Leaf:
    name: str
    kind: str
    shape: tuple
    parameters: int = 0
    config: dict = None


@dataclass
class Scope:
    name: str
    input_shape: tuple
    output_shape: tuple
    items: list


def _conv(name, source, channels, kernel, stride=1, padding=0):
    batch, incoming, height, width = source
    output = (batch, channels, (height + 2 * padding - kernel) // stride + 1,
              (width + 2 * padding - kernel) // stride + 1)
    return Leaf(name, "Conv2d", output, incoming * channels * kernel * kernel,
                {"in_channels": incoming, "out_channels": channels,
                 "kernel_size": kernel, "stride": stride, "padding": padding,
                 "groups": 1, "dilation": 1, "bias": False})


def _bn(name, shape):
    return Leaf(name, "BatchNorm2d", shape, 2 * shape[1],
                {"num_features": shape[1], "eps": 1e-5, "momentum": 0.1,
                 "affine": True, "track_running_stats": True})


def _block(name, source, channels, stride):
    first = _conv("conv1", source, channels, 3, stride, 1)
    output = first.shape
    items = [first, _bn("bn1", output), Leaf("relu1", "ReLU", output),
             _conv("conv2", output, channels, 3, 1, 1), _bn("bn2", output)]
    if source[1] != channels or stride != 1:
        projection = _conv("conv", source, channels, 1, stride)
        items.append(Scope("projection", source, output, [projection, _bn("bn", output)]))
    # Identity residual uses the block's input source, with no extra allocation.
    items += [Leaf("add", "Add", output), Leaf("relu2", "ReLU", output)]
    return Scope(name, source, output, items)


def _walk(scope, path=""):
    for item in scope.items:
        child_path = path + "." + item.name if path else item.name
        yield child_path, item
        if isinstance(item, Scope):
            yield from _walk(item, child_path)


def _measure(scope):
    """Count a single _infer traversal, runtime traversal, and resolve traversal.

    Each scope has one input/output; each instance has exactly one call.
    _infer visits a child once while counting instances and once for its call.
    Runtime visits it once, charging child input and parent call output too.
    Declared output aliases themselves add no extra tensor charge.
    """
    input_elements = math.prod(scope.input_shape)
    local_nodes = 2 + 2 * len(scope.items)
    static = runtime = input_elements
    infer_nodes = resolve_nodes = local_nodes
    parameters = 0
    for item in scope.items:
        output_elements = math.prod(item.output_shape if isinstance(item, Scope) else item.shape)
        if isinstance(item, Scope):
            child = _measure(item)
            static += 2 * child["static_elements"]
            runtime += child["runtime_elements"]
            infer_nodes += 2 * child["inference_nodes"]
            resolve_nodes += child["resolved_nodes"]
            parameters += child["parameters"]
        else:
            parameters += item.parameters
        static += output_elements
        runtime += output_elements
    return {"static_elements": static, "runtime_elements": runtime,
            "inference_nodes": infer_nodes, "resolved_nodes": resolve_nodes,
            "parameters": parameters, "local_nodes": local_nodes,
            "input_elements": input_elements}


def _storage_proxy(scopes, batch_bound):
    """Finite sizing proxy with graph-like fields; deliberately invalid wiring.

    Actual closure fields, layout, public bindings, selected outputs and real
    sources do not exist yet. Counts are estimates of this declared proxy only.
    """
    records = []
    for path, scope in scopes:
        instances, calls = [], []
        for item in scope.items:
            instance = {"id": item.name, "kind": "chip" if isinstance(item, Scope) else item.kind,
                        "params": {} if isinstance(item, Scope) else (item.config or {})}
            if isinstance(item, Scope):
                instance["definition_id"] = path + "." + item.name
            instances.append(instance)
            calls.append({"id": item.name, "instance_id": item.name,
                          "inputs": {key: "AUDIT_ONLY" for key in
                                     (("a", "b") if instance["kind"] == "Add" else ("x",))}})
        records.append({"id": path, "version": "1.0.0", "public_parameters": [], "bindings": {},
                        "graph": {"inputs": [{"id": "x", "shape": [None, *scope.input_shape[1:]], "dtype": "float32"}],
                                  "instances": instances, "calls": calls,
                                  "outputs": [{"id": "out", "source": "AUDIT_ONLY"}]}})
    proxy = {"audit_only": True, "resource_profile": "vision-v4", "resource_batch_bound": batch_bound,
             "definition_sizing_records": records}
    pending = [(proxy, 0)]
    values = including_keys = fields = max_depth = 0
    while pending:
        value, depth = pending.pop()
        values += 1
        including_keys += 1
        max_depth = max(max_depth, depth)
        if isinstance(value, dict):
            fields += len(value)
            including_keys += len(value)
            if value:
                max_depth = max(max_depth, depth + 1)
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
    return {"kind": "AUDIT_ONLY finite sizing proxy, not generated production closure",
            "dictionary_fields": fields, "document_values_excluding_keys": values,
            "schema_values_including_keys": including_keys, "max_json_depth": max_depth,
            "utf8_compact_bytes": len(json.dumps(proxy, separators=(",", ":")).encode("utf-8")),
            "utf8_pretty_bytes": len(json.dumps(proxy, indent=2).encode("utf-8"))}


def audit_resnet18(batch_bound=2, input_channels=3, num_classes=1000):
    """Return conditional arithmetic for 17 explicit, independently sized defs.

    No reuse of stage1's equal block definitions is assumed. Definitions are
    not weight instances; reusing one later must not share distinct BN state.
    """
    candidate_elements([None, 3, 224, 224], batch_bound)
    if type(input_channels) is not int or input_channels not in (1, 3):
        raise ValueError("Only declared one/three-channel audit cases are supported")
    if num_classes is not None and (type(num_classes) is not int or num_classes < 1):
        raise ValueError("num_classes must be a positive integer or None")
    source = (batch_bound, input_channels, 224, 224)
    conv = _conv("conv", source, 64, 7, 2, 3)
    stem_output = (batch_bound, 64, 56, 56)
    stem = Scope("stem", source, stem_output, [conv, _bn("bn", conv.shape),
                 Leaf("relu", "ReLU", conv.shape),
                 Leaf("pool", "MaxPool2d", stem_output, config={"kernel_size": 3, "stride": 2, "padding": 1})])
    stages = []
    previous = stem_output
    for index, channels in enumerate((64, 128, 256, 512), 1):
        first = _block("block0", previous, channels, 1 if index == 1 else 2)
        second = _block("block1", first.output_shape, channels, 1)
        stage = Scope("stage" + str(index), previous, second.output_shape, [first, second])
        stages.append(stage)
        previous = stage.output_shape
    features = (batch_bound, 512)
    items = [stem, *stages, Leaf("pool", "AdaptiveAvgPool2d", (batch_bound, 512, 1, 1), config={"output_size": [1, 1]}),
             Leaf("flatten", "Flatten", features, config={"start_dim": 1, "end_dim": -1})]
    logits = None if num_classes is None else (batch_bound, num_classes)
    if logits is not None:
        items.append(Leaf("fc", "Linear", logits, 512 * num_classes + num_classes,
                          {"in_features": 512, "out_features": num_classes}))
    entry = Scope("entry", source, logits or features, items)
    root = Scope("root", source, entry.output_shape, [entry])
    walked = list(_walk(root))
    scopes = [(path, item) for path, item in walked if isinstance(item, Scope)]
    leaves = [(path, item) for path, item in walked if isinstance(item, Leaf)]
    root_metrics = _measure(root)
    defaults = [_measure(scope) for _, scope in scopes]
    bn_leaves = [(path, leaf) for path, leaf in leaves if leaf.kind == "BatchNorm2d"]
    bn_channels = sum(leaf.shape[1] for _, leaf in bn_leaves)
    parameters = root_metrics["parameters"]
    total_static = root_metrics["static_elements"] + sum(row["static_elements"] for row in defaults)
    largest = max([math.prod(source)] + [math.prod(leaf.shape) for _, leaf in leaves])
    ledger = []
    for (path, scope), metrics in zip(scopes, defaults):
        ledger.append({"path": path, "input_shape": list(scope.input_shape),
                       "output_shape": list(scope.output_shape), "instance_count": len(scope.items),
                       **metrics})
    return {
        "evidence_kind": "conditional independent arithmetic; not platform validation or execution",
        "topology_assumptions": "root->entry->stem/stage->block->projection; all 17 definitions explicit; one call per instance; default logits only; no Identity primitive",
        "batch_bound": batch_bound, "input_channels": input_channels, "num_classes": num_classes,
        "limits": dict(LIMITS), "input_shape": list(source), "stem_conv_shape": list(conv.shape),
        "stage_shapes": [list(stage.output_shape) for stage in stages],
        "features_shape": list(features), "logits_shape": list(logits) if logits else None,
        "stem_parameters": _measure(stem)["parameters"],
        "stage_parameters": [_measure(stage)["parameters"] for stage in stages],
        "parameters": parameters, "backbone_parameters": parameters - (512 * num_classes + num_classes if num_classes else 0),
        "bn_instances": len(bn_leaves), "bn_affine_parameters": 2 * bn_channels,
        "bn_float_buffer_elements": 2 * bn_channels, "bn_int64_buffer_elements": len(bn_leaves),
        "parameter_and_persistent_buffer_bytes": 4 * (parameters + 2 * bn_channels) + 8 * len(bn_leaves),
        "bn_state_instances": [path for path, _ in bn_leaves],
        "block_instances": [path for path, _ in scopes if path.split(".")[-1] in ("block0", "block1")],
        "definitions": len(scopes), "primitive_instances": len(leaves), "chip_instances": len(scopes),
        "chip_depth": max(path.count(".") + 1 for path, _ in scopes),
        "root_output_elements": math.prod(entry.output_shape),
        "flat_leaf_elements": math.prod(source) + sum(math.prod(leaf.shape) for _, leaf in leaves),
        "runtime_counting_estimate_elements": root_metrics["runtime_elements"],
        "root_inference_elements": root_metrics["static_elements"],
        "definition_inference_elements": sum(row["static_elements"] for row in defaults),
        "root_repeat_inference_increment": root_metrics["static_elements"] - root_metrics["runtime_elements"],
        "total_inference_elements": total_static,
        "largest_tensor_elements": largest,
        "scope_elements": {scope.name: _measure(scope)["static_elements"] for scope in [stem, *stages]},
        "resolved_nodes_including_definition_checks": root_metrics["resolved_nodes"] + sum(row["resolved_nodes"] + 3 for row in defaults),
        "inference_nodes_including_definition_checks": root_metrics["inference_nodes"] + sum(row["inference_nodes"] for row in defaults),
        "parameters_fit": parameters <= LIMITS["max_parameters"],
        "single_tensor_fits": largest <= LIMITS["max_tensor_elements"],
        "cumulative_fits": total_static <= LIMITS["max_inferred_elements"],
        "estimated_storage": _storage_proxy([("root", root), *scopes], batch_bound),
        "scope_ledger": ledger,
        "leaf_ledger": [{"path": path, "kind": leaf.kind, "output_shape": list(leaf.shape),
                         "output_elements": math.prod(leaf.shape), "parameters": leaf.parameters,
                         "config": leaf.config or {}} for path, leaf in leaves],
    }


if __name__ == "__main__":
    print(json.dumps({"batch1": audit_resnet18(1), "batch2": audit_resnet18(2)}, indent=2))
