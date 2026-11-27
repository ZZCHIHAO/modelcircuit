"""Standard-library validation, chip resolution and static tensor shapes.

The resolved graph is a private copy. Public bindings never mutate definitions.
No PyTorch or GUI dependency is needed to validate a saved document.
"""
import copy
import math
import re
from dataclasses import dataclass

from .operators import CLASSICAL_KINDS, operator_kinds, operator_spec
from .operator_rules import normalize_operator, infer_operator, _parameter_count, _buffer_shapes
from .ports import parse_source, output_port_ids, output_mode
from .tensor_ops import TENSOR_KINDS, infer_tensor_operator, normalize_tensor_operator, tensor_input_ports, tensor_parameter_count
from .tensor_specs import SPEC_FIELDS, TensorSpecError, normalize_tensor_spec

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PRIMITIVES = set(operator_kinds())
PARAMETERS = {kind: (set(operator_spec(kind).required), set(operator_spec(kind).defaults) - set(operator_spec(kind).required)) for kind in operator_kinds()}

# S1 is a local CPU workbench. These budgets are checked before model/tensor
# allocation and reject oversized documents explicitly instead of rewriting them.
MAX_DOCUMENT_VALUES = 20_000
MAX_DOCUMENT_DEPTH = 32
MAX_STRING_LENGTH = 4096
MAX_ITEMS = 512
MAX_DEFINITIONS = 32
MAX_CHIP_DEPTH = 6
MAX_RESOLVED_NODES = 2048
MAX_INFERENCE_NODES = 16_384
MAX_PARAMETERS = 2_000_000
MAX_TENSOR_ELEMENTS = 1_000_000
MAX_INFERRED_ELEMENTS = 16_000_000
MAX_RANK = 8
PUBLIC_TYPES = ("auto", "int", "float", "bool", "shape")


@dataclass(frozen=True)
class ResourceLimits:
    max_parameters: int | None = MAX_PARAMETERS
    max_tensor_elements: int | None = MAX_TENSOR_ELEMENTS
    max_inferred_elements: int | None = MAX_INFERRED_ELEMENTS
    batch_bound: int = 64
    enforce_batch_bound: bool = False


def resource_limits(profile=None, *, batch_bound=None):
    if profile in ("vision-v4", "hardware-v5"):
        _require(type(batch_bound) is int and 1 <= batch_bound <= 64,
                 profile + " requires resource_batch_bound as an integer between 1 and 64", field="resource_batch_bound")
        if profile == "hardware-v5":
            return ResourceLimits(None, None, None, batch_bound, True)
        return ResourceLimits(20_000_000, 16_777_216, 134_217_728, batch_bound, True)
    _require(batch_bound is None, "resource_batch_bound requires vision-v4", field="resource_batch_bound")
    if profile is None:
        return ResourceLimits()
    if profile == "classification-v2":
        return ResourceLimits(20_000_000, 16_777_216, 134_217_728)
    raise GraphError("Unsupported graph resource_profile", field="resource_profile")


def graph_resource_limits(graph):
    """Read the root policy, rejecting even null bound fields on legacy graphs."""
    _require(isinstance(graph, dict), "Graph document must be an object")
    profile = graph.get("resource_profile")
    _require(profile in ("vision-v4", "hardware-v5") or "resource_batch_bound" not in graph,
             "resource_batch_bound requires vision-v4", field="resource_batch_bound")
    return resource_limits(profile, batch_bound=graph.get("resource_batch_bound"))


def bounded_elements(shape, limits):
    _require(isinstance(shape, (list, tuple)) and 1 <= len(shape) <= MAX_RANK and
             all(_positive(axis) or (index == 0 and axis is None) for index, axis in enumerate(shape)),
             "shape dimensions must be positive integers; only batch dimension may be null", field="shape")
    return math.prod(limits.batch_bound if dimension is None else dimension for dimension in shape)



class GraphError(ValueError):
    def __init__(self, message, node_id=None, field=None):
        super().__init__(message)
        self.diagnostic = {"message": message}
        if node_id is not None: self.diagnostic["node_id"] = node_id
        if field is not None: self.diagnostic["field"] = field


def _require(condition, message, node_id=None, field=None):
    if not condition: raise GraphError(message, node_id, field)


def _positive(value):
    return type(value) is int and value > 0


def _finite_number(value):
    if type(value) not in (int, float): return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _public_value(value, declared_type, identifier, field):
    _require(declared_type in PUBLIC_TYPES, "Unsupported public parameter type: " + str(declared_type), identifier, "type")
    if declared_type == "auto":
        return
    if declared_type == "int": valid = type(value) is int
    elif declared_type == "float": valid = _finite_number(value)
    elif declared_type == "bool": valid = type(value) is bool
    else:
        dimensions = value if isinstance(value, list) else [value]
        valid = 0 < len(dimensions) <= MAX_RANK and all(_positive(axis) for axis in dimensions)
    _require(valid, "Public parameter " + field + " must match declared type " + declared_type, identifier, field)


def _preflight(document):
    """Bound raw JSON work before deepcopy or recursive chip expansion."""
    pending = [(document, 0)]
    visited = 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        _require(visited <= MAX_DOCUMENT_VALUES, "Document value budget exceeded", field="resource_limits")
        _require(depth <= MAX_DOCUMENT_DEPTH, "Document nesting depth limit exceeded", field="resource_limits")
        if isinstance(value, float):
            _require(math.isfinite(value), "Graph JSON contains a non-finite number", field="document")
        elif isinstance(value, str):
            _require(len(value) <= MAX_STRING_LENGTH, "Document string length limit exceeded", field="resource_limits")
        elif isinstance(value, dict):
            _require(len(value) <= MAX_DOCUMENT_VALUES, "Document object size limit exceeded", field="resource_limits")
            for key, child in value.items():
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))
        elif isinstance(value, list):
            _require(len(value) <= MAX_DOCUMENT_VALUES, "Document array size limit exceeded", field="resource_limits")
            pending.extend((child, depth + 1) for child in value)


def _elements(shape):
    return math.prod(64 if dimension is None else dimension for dimension in shape)


class _Budget:
    def __init__(self, limits=None, *, tensor_contracts=False):
        self.limits = limits if limits is not None else resource_limits()
        self.tensor_contracts = tensor_contracts
        self.resolved_nodes = 0
        self.inference_nodes = 0
        self.inferred_elements = 0

    def resolve(self, graph):
        self.resolved_nodes += sum(len(graph.get(key, [])) for key in ("inputs", "instances", "calls", "outputs"))
        _require(self.resolved_nodes <= MAX_RESOLVED_NODES, "Expanded chip node budget exceeded", field="resource_limits")

    def infer(self, graph):
        self.inference_nodes += sum(len(graph[key]) for key in ("inputs", "instances", "calls", "outputs"))
        _require(self.inference_nodes <= MAX_INFERENCE_NODES, "Shape inference work budget exceeded", field="resource_limits")

    def tensor(self, shape, node_id):
        _shape(shape, node_id, self.limits)
        self.inferred_elements += bounded_elements(shape, self.limits)
        if not self.limits.enforce_batch_bound and self.limits.max_inferred_elements is not None:
            _require(self.inferred_elements <= self.limits.max_inferred_elements, "Cumulative inferred tensor budget exceeded", node_id, "resource_limits")


def _items(graph, key, allow_empty=False):
    items = graph.get(key)
    _require(isinstance(items, list) and (allow_empty or len(items) > 0), key + " must be a " + ("" if allow_empty else "nonempty ") + "list", field=key)
    limit = MAX_DEFINITIONS if key == "definitions" else MAX_ITEMS
    _require(len(items) <= limit, key + " item limit exceeded (maximum " + str(limit) + ")", field="resource_limits")
    seen = set()
    for item in items:
        _require(isinstance(item, dict), key + " entries must be objects", field=key)
        identifier = item.get("id")
        _require(isinstance(identifier, str) and IDENTIFIER.fullmatch(identifier), "Invalid identifier in " + key, field="id")
        _require(identifier not in seen, "Duplicate identifier: " + identifier, identifier, "id")
        seen.add(identifier)
    return items


def _shape(value, node_id, limits=None):
    limits = limits if limits is not None else resource_limits()
    _require(isinstance(value, list) and len(value) > 0, "shape must be a nonempty list", node_id, "shape")
    _require(len(value) <= MAX_RANK, "Tensor rank limit exceeded", node_id, "resource_limits")
    _require(all(_positive(d) or (i == 0 and d is None) for i, d in enumerate(value)), "shape dimensions must be positive integers; only batch dimension may be null", node_id, "shape")
    if limits.max_tensor_elements is not None:
        _require(all(d is None or d <= limits.max_tensor_elements for d in value) and bounded_elements(value, limits) <= limits.max_tensor_elements, "Tensor element budget exceeded (maximum " + str(limits.max_tensor_elements) + " at batch " + str(limits.batch_bound) + ")", node_id, "resource_limits")
    return list(value)


def _primitive(instance, *, tensor_contracts=False):
    kind, params, identifier = instance["kind"], instance.get("params", {}), instance["id"]
    try:
        normalize = normalize_tensor_operator if tensor_contracts else normalize_operator
        normalized = normalize(kind, params)
    except ValueError as error:
        raise GraphError(str(error), identifier, getattr(error, "field", None) or "params") from None
    # Old non-CNN graphs preserve omitted params and defaults in resolved JSON.
    if normalized != params or "params" in instance:
        instance["params"] = normalized


def _port_spec(port, budget):
    identifier = port['id']
    try:
        spec = normalize_tensor_spec({key: port[key] for key in SPEC_FIELDS if key in port})
    except TensorSpecError as error:
        raise GraphError(str(error), identifier, error.field) from None
    _require(spec['batch_axis'] is None or spec['shape'][0] is None or spec['shape'][0] <= budget.limits.batch_bound,
             'Fixed batch exceeds resource_batch_bound', identifier, 'shape')
    return spec


def _resolve_scope(graph, definitions, stack=(), budget=None):
    _require(isinstance(graph, dict), "graph must be an object")
    budget = budget if budget is not None else _Budget()
    _items(graph, "inputs", allow_empty=budget.tensor_contracts)
    instances = _items(graph, "instances", allow_empty=True)
    _items(graph, "calls", allow_empty=True)
    _items(graph, "outputs")
    budget.resolve(graph)
    graph = copy.deepcopy(graph)
    inputs, instances = graph["inputs"], graph["instances"]
    for port in inputs:
        if budget.tensor_contracts:
            port.update(_port_spec(port, budget))
        else:
            _shape(port.get("shape"), port["id"], budget.limits)
            _require(port.get("dtype") == "float32", "S1 supports float32 inputs only", port["id"], "dtype")
    # ModuleDict rejects keys which shadow module attributes.
    reserved = set(dir(object)) | set("""
        training forward modules parameters children items keys values update clear pop get
        to cpu cuda eval train state_dict load_state_dict add_module register_parameter
        register_buffer apply named_parameters named_buffers buffers named_children named_modules
        get_submodule set_submodule get_parameter get_buffer register_module requires_grad_
        zero_grad type float double half bfloat16 share_memory extra_repr to_empty compile
        xpu ipu mtia register_backward_hook register_full_backward_hook register_forward_hook
        register_forward_pre_hook register_full_backward_pre_hook register_state_dict_pre_hook
        register_state_dict_post_hook register_load_state_dict_pre_hook register_load_state_dict_post_hook
        get_extra_state set_extra_state append insert extend fromkeys setdefault copy
        _modules _parameters _buffers _non_persistent_buffers_set _backward_hooks
        _backward_pre_hooks _forward_hooks _forward_pre_hooks _forward_hooks_with_kwargs
        _forward_hooks_always_called _forward_pre_hooks_with_kwargs _state_dict_hooks
        _state_dict_pre_hooks _load_state_dict_pre_hooks _load_state_dict_post_hooks
        _is_full_backward_hook _version _compiled_call_impl call_super_init dump_patches
        __dict__ __weakref__ __annotations__ __module__ __doc__ __call__ __contains__
        __getitem__ __setitem__ __delitem__ __iter__ __len__ __getattr__ __dir__
    """.split())
    allowed_kinds = (PRIMITIVES - CLASSICAL_KINDS if budget.tensor_contracts else PRIMITIVES) | {'chip'} | (TENSOR_KINDS.keys() if budget.tensor_contracts else set())
    for instance in instances:
        identifier, kind = instance["id"], instance.get("kind")
        _require(identifier not in reserved, "Instance identifier conflicts with a PyTorch module attribute", identifier, "id")
        _require(isinstance(kind, str) and kind in allowed_kinds, "Unsupported instance kind: " + str(kind), identifier, "kind")
        _require(type(instance.get("frozen", False)) is bool, "frozen must be a boolean", identifier, "frozen")
        if kind != "chip":
            _primitive(instance, tensor_contracts=budget.tensor_contracts)
            continue
        definition_id = instance.get("definition_id")
        _require(isinstance(definition_id, str) and definition_id in definitions, "Unknown chip definition: " + str(definition_id), identifier, "definition_id")
        _require(definition_id not in stack, "Recursive chip definition: " + definition_id, identifier, "definition_id")
        _require(len(stack) < MAX_CHIP_DEPTH, "Chip nesting depth limit exceeded (maximum " + str(MAX_CHIP_DEPTH) + ")", identifier, "resource_limits")
        definition = definitions[definition_id]
        public = _items({"public_parameters": definition.get("public_parameters", [])}, "public_parameters", allow_empty=True)
        defaults = {}
        for parameter in public:
            _require("default" in parameter, "Public parameter requires a default", parameter["id"], "default")
            _public_value(parameter["default"], parameter.get("type", "auto"), parameter["id"], "default")
            defaults[parameter["id"]] = parameter["default"]
        params = instance.get("params", {})
        _require(isinstance(params, dict), "params must be an object", identifier, "params")
        _require(params.keys() <= defaults.keys(), "Unknown public chip parameter", identifier, "params")
        defaults.update(params)
        for parameter in public:
            _public_value(defaults[parameter["id"]], parameter.get("type", "auto"), identifier, parameter["id"])
        bindings = definition.get("bindings", {})
        _require(isinstance(bindings, dict) and bindings.keys() <= defaults.keys(), "bindings must map declared public parameters", identifier, "bindings")
        chip_graph = copy.deepcopy(definition.get("graph"))
        _require(isinstance(chip_graph, dict), "Chip graph must be an object", identifier, "graph")
        child_instances = _items(chip_graph, "instances", allow_empty=True)
        targets = {child["id"]: child for child in child_instances}
        for parameter, paths in bindings.items():
            _require(isinstance(paths, list) and bool(paths), "Each public binding requires target paths", identifier, "bindings")
            for path in paths:
                _require(isinstance(path, str) and len(path.split(".")) == 2, "Binding target must be instance.parameter", identifier, "bindings")
                target_id, parameter_id = path.split(".")
                _require(target_id in targets, "Unknown binding target: " + path, identifier, "bindings")
                target = targets[target_id]
                _require(isinstance(target.get("params", {}), dict), "Binding target params must be an object", identifier, "bindings")
                # Unknown names are validated after overrides, never ignored.
                target.setdefault("params", {})[parameter_id] = copy.deepcopy(defaults[parameter])
        if "in_features" in defaults:
            ports = _items(chip_graph, "inputs", allow_empty=budget.tensor_contracts)
            if len(ports) == 1:
                _shape(ports[0].get("shape"), ports[0]["id"], budget.limits)
                ports[0]["shape"][-1] = defaults["in_features"]
        instance["_graph"] = _resolve_scope(chip_graph, definitions, stack + (definition_id,), budget)
        try:
            instance["_output_mode"] = output_mode(definition)
        except ValueError as error:
            raise GraphError(str(error), identifier, "outputs") from None
    return graph


def _source(source, shapes, node_id):
    _require(isinstance(source, str) and source in shapes, "Unknown or unavailable tensor source: " + str(source), node_id, "inputs")
    return shapes[source]


def _check_source_reference(source, inputs, calls, instances, node_id):
    try:
        kind, identifier, port = parse_source(source)
    except ValueError as error:
        raise GraphError(str(error), node_id, "inputs") from None
    if kind == "input":
        _require("input:" + identifier in inputs, "Unknown tensor source: " + str(source), node_id, "inputs")
        return
    _require(identifier in calls, "Unknown tensor source: " + str(source), node_id, "inputs")
    ports = output_port_ids(instances[calls[identifier]["instance_id"]])
    if port is None:
        _require(len(ports) == 1, "Ambiguous output port for " + source + "; select a named port", node_id, "inputs")
    else:
        _require(port in ports, "Unknown output port in tensor source: " + source, node_id, "inputs")


def _check_mask_routes(graph, calls, instances):
    """The internal -inf score type cannot leave this scope or feed other ops."""
    masked = {identifier for identifier, call in calls.items()
              if instances[call['instance_id']]['kind'] == 'MaskApply'}
    if not masked:
        return
    consumed = set()
    for call in calls.values():
        for port, reference in call['inputs'].items():
            source, identifier, _ = parse_source(reference)
            if source == 'call' and identifier in masked:
                instance = instances[call['instance_id']]
                _require(port == 'x' and instance['kind'] == 'Softmax' and instance['params']['dim'] == -1,
                         'A masked score must go directly into Softmax(dim=-1)', call['id'], 'inputs')
                consumed.add(identifier)
    for output in graph['outputs']:
        source, identifier, _ = parse_source(output['source'])
        _require(source != 'call' or identifier not in masked,
                 'A masked score cannot be a graph or chip output', output['id'], 'source')
    for identifier in calls:
        if identifier in masked and identifier not in consumed:
            raise GraphError('MaskApply must feed a Softmax(dim=-1) call', identifier, 'inputs')


def _infer(graph, budget=None):
    budget = budget if budget is not None else _Budget()
    budget.infer(graph)
    shapes = {"input:" + p["id"]: list(p["shape"]) for p in graph["inputs"]}
    specs = ({'input:' + port['id']: _port_spec(port, budget) for port in graph['inputs']}
             if budget.tensor_contracts else None)
    if specs is not None:
        fixed_batches = {spec['shape'][0] for spec in specs.values()
                         if spec['batch_axis'] == 0 and spec['shape'][0] is not None}
        _require(len(fixed_batches) <= 1, 'Batch-bearing ports disagree on the fixed batch size', field='inputs')
    origins = {source: ('input', source) for source in shapes} if budget.limits.enforce_batch_bound else {}
    for source, shape in shapes.items(): budget.tensor(shape, source)
    execution_elements = sum(bounded_elements(shape, budget.limits) for shape in shapes.values())
    instances = {i["id"]: i for i in graph["instances"]}
    calls = {c["id"]: c for c in graph["calls"]}
    pending = list(graph["calls"])
    ordered = []
    count = 0
    for instance in instances.values():
        kind, params = instance["kind"], instance.get("params", {})
        if kind == "chip":
            _, subtotal = _infer(instance["_graph"], budget)
            count += subtotal
        else:
            count += (tensor_parameter_count if budget.tensor_contracts else _parameter_count)(kind, params)
        if budget.limits.max_parameters is not None:
            _require(count <= budget.limits.max_parameters, "Model parameter budget exceeded (maximum " + str(budget.limits.max_parameters) + ")", instance["id"], "resource_limits")
    for call in pending:
        identifier = call["id"]
        _require(isinstance(call.get("instance_id"), str) and call["instance_id"] in instances, "Call refers to an unknown instance", identifier, "instance_id")
        args = call.get("inputs")
        _require(isinstance(args, dict), "Call inputs must be an object", identifier, "inputs")
        instance = instances[call["instance_id"]]
        kind = instance['kind']
        if kind == 'chip':
            names = {port['id'] for port in instance['_graph']['inputs']}
        else:
            names = set(tensor_input_ports(kind) if budget.tensor_contracts else operator_spec(kind).input_ports)
        _require(args.keys() == names, "Call arguments must be exactly: " + ", ".join(sorted(names)), identifier, "inputs")
        position = call.get("position")
        if position is not None:
            _require(isinstance(position, dict) and all(_finite_number(position.get(k)) for k in ("x", "y")), "position requires finite x and y", identifier, "position")
    for call in pending:
        for source in call["inputs"].values():
            _check_source_reference(source, shapes, calls, instances, call["id"])
    for item in graph["outputs"]:
        _check_source_reference(item.get("source"), shapes, calls, instances, item["id"])
    if budget.tensor_contracts:
        _check_mask_routes(graph, calls, instances)
    while pending:
        ready = [c for c in pending if all(s in shapes for s in c["inputs"].values())]
        _require(bool(ready), "Graph contains a call dependency cycle", pending[0]["id"], "inputs")
        # A newly-ready earlier declaration must run before a later ready call:
        # batching the ready set changes the RNG/effects of Dropout calls.
        for call in ready[:1]:
            instance = instances[call["instance_id"]]
            kind, params, identifier = instance["kind"], instance.get("params", {}), call["id"]
            args = {k: _source(v, shapes, identifier) for k, v in call["inputs"].items()}
            spec_args = {k: specs[v] for k, v in call['inputs'].items()} if specs is not None else None
            if kind == "chip":
                chip_graph = instance["_graph"]
                for port in chip_graph["inputs"]:
                    actual, expected = args[port["id"]], port["shape"]
                    _require(len(actual) == len(expected) and all(e is None or e == a for a, e in zip(actual, expected)), "Chip input shape mismatch: " + str(actual) + " versus " + str(expected), identifier, "inputs")
                    if spec_args is not None:
                        for key in ('dtype', 'axes', 'batch_axis'):
                            _require(spec_args[port['id']][key] == port[key],
                                     f"Chip input {port['id']} has incompatible {key}", identifier, 'inputs')
                actual_graph = copy.deepcopy(chip_graph)
                for port in actual_graph["inputs"]: port["shape"] = args[port["id"]]
                child_shapes, _ = _infer(actual_graph, budget)
                if budget.limits.enforce_batch_bound:
                    execution_elements += actual_graph["_execution_elements"]
                outputs = {port["id"]: child_shapes[port["source"]] for port in chip_graph["outputs"]}
                if specs is not None:
                    output_specs = {port['id']: actual_graph['_tensor_specs'][port['source']] for port in chip_graph['outputs']}
            else:
                try:
                    if spec_args is None:
                        output, _ = infer_operator(kind, params, args)
                    else:
                        output_spec = infer_tensor_operator(kind, params, spec_args)
                        output_specs = {'out': output_spec}
                        output = output_spec['shape']
                except ValueError as error:
                    raise GraphError(str(error), identifier, getattr(error, "field", None) or "inputs") from None
            if kind != "chip":
                outputs = {"out": list(output)}
            unique_shapes = set()
            for port, shape in outputs.items():
                reference = "call:" + identifier + ":" + port
                shapes[reference] = shape
                if specs is not None:
                    specs[reference] = output_specs[port]
                charge_key = id(shape)
                if budget.limits.enforce_batch_bound:
                    charge_key = ('call', identifier, actual_graph['_output_aliases'][port] if kind == 'chip' else port)
                    origins[reference] = charge_key
                if charge_key not in unique_shapes:
                    budget.tensor(shape, identifier)
                    execution_elements += bounded_elements(shape, budget.limits)
                    unique_shapes.add(charge_key)
            if len(outputs) == 1:
                shapes["call:" + identifier] = next(iter(outputs.values()))
                if specs is not None:
                    specs['call:' + identifier] = next(iter(output_specs.values()))
                if budget.limits.enforce_batch_bound:
                    origins["call:" + identifier] = next(origins['call:' + identifier + ':' + port] for port in outputs)
            ordered.append(call)
            pending.remove(call)
            if budget.limits.enforce_batch_bound and budget.limits.max_inferred_elements is not None:
                _require(execution_elements <= budget.limits.max_inferred_elements,
                         "Cumulative execution tensor budget exceeded", identifier, "resource_limits")
    for output in graph["outputs"]: _source(output.get("source"), shapes, output["id"])
    graph["_ordered_calls"] = ordered
    if specs is not None:
        graph['_tensor_specs'] = specs
    if budget.limits.enforce_batch_bound:
        if budget.limits.max_inferred_elements is not None:
            _require(execution_elements <= budget.limits.max_inferred_elements,
                     "Cumulative execution tensor budget exceeded", field="resource_limits")
        graph["_execution_elements"] = execution_elements
        graph['_output_aliases'] = {output['id']: origins[output['source']] for output in graph['outputs']}
    return shapes, count


def _check_buffer_allocations(graph, limits):
    """Bound persistent buffers by constructed instance, never by call count.

    Each definition validation root and the actual resolved model have their
    own ledger. This does not change activation or parameter budget accounting.
    """
    pending = [(graph, ())]
    total = 0
    while pending:
        scope, path = pending.pop()
        for instance in scope["instances"]:
            instance_path = path + (instance["id"],)
            if instance["kind"] == "chip":
                pending.append((instance["_graph"], instance_path))
                continue
            for shape in _buffer_shapes(instance["kind"], instance.get("params", {})):
                elements = math.prod(shape)
                identifier = "/".join(instance_path)
                if limits.max_tensor_elements is not None:
                    _require(elements <= limits.max_tensor_elements, "Persistent buffer tensor element budget exceeded", identifier, "resource_limits")
                total += elements
                if limits.max_inferred_elements is not None:
                    _require(total <= limits.max_inferred_elements, "Cumulative persistent buffer element budget exceeded", identifier, "resource_limits")


def prepare_graph(graph):
    """Validate and return (resolved copy, source shapes, unique parameter count).

    Raises GraphError for invalid documents before model allocation.
    """
    _require(isinstance(graph, dict), "Graph document must be an object")
    _preflight(graph)
    version = graph.get('schema_version')
    _require(type(version) is int and version in (1, 2), 'schema_version must equal 1 or 2', field='schema_version')
    if version == 2:
        _require(graph.get('capability_profile') == 'tensor-v1', 'schema v2 requires capability_profile tensor-v1', field='capability_profile')
        _require(graph.get('resource_profile') == 'hardware-v5', 'tensor-v1 requires resource_profile hardware-v5', field='resource_profile')
    else:
        _require('capability_profile' not in graph, 'capability_profile requires schema v2', field='capability_profile')
    _require(isinstance(graph.get("id"), str) and bool(IDENTIFIER.fullmatch(graph["id"])), "Root id must be an identifier", field="id")
    _require(type(graph.get("revision")) is int and graph["revision"] >= 1, "revision must be a positive integer", field="revision")
    definitions = _items({"definitions": graph.get("definitions", [])}, "definitions", allow_empty=True)
    registry = {d["id"]: d for d in definitions}
    budget = _Budget(graph_resource_limits(graph), tensor_contracts=version == 2)
    definition_execution = {}
    for definition in definitions:
        # Check even unused definitions, including recursive graphs.
        instance = {"id": "definition_check", "kind": "chip", "definition_id": definition["id"], "params": {}}
        scope = {"inputs": [{"id": "unused", "shape": [None, 1], "dtype": "float32"}], "instances": [instance], "calls": [], "outputs": [{"id": "unused", "source": "input:unused"}]}
        if budget.tensor_contracts:
            scope['inputs'][0].update(axes=['batch', 'feature'], batch_axis=0)
        resolved_definition = _resolve_scope(scope, registry, budget=budget)
        _check_buffer_allocations(resolved_definition["instances"][0]["_graph"], budget.limits)
        _infer(resolved_definition["instances"][0]["_graph"], budget)
        if budget.limits.enforce_batch_bound:
            definition_execution[definition["id"]] = resolved_definition["instances"][0]["_graph"]["_execution_elements"]
    resolved = _resolve_scope(graph, registry, budget=budget)
    _check_buffer_allocations(resolved, budget.limits)
    shapes, count = _infer(resolved, budget)
    if budget.limits.enforce_batch_bound:
        resolved["_resource_ledger"] = {"resolved_nodes": budget.resolved_nodes,
            "inference_nodes": budget.inference_nodes, "compile_tensor_elements": budget.inferred_elements,
            "execution_elements": resolved["_execution_elements"], "definition_execution_elements": definition_execution}
    return resolved, shapes, count


def validate_graph(graph):
    """Return diagnostics, static source shapes and unique parameter count."""
    try:
        resolved, shapes, count = prepare_graph(graph)
        result = {"valid": True, "errors": [], "warnings": [], "shapes": shapes, "parameter_count": count}
        if graph['schema_version'] == 2:
            result['tensor_specs'] = resolved['_tensor_specs']
        return result
    except GraphError as exc:
        return {"valid": False, "errors": [exc.diagnostic], "warnings": [], "shapes": {}, "parameter_count": None}
