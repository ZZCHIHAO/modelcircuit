"""Restricted, static PyTorch bridge for legacy graphs and vision v4.

Only constructor literals/public parameters, ModuleDict components and explicit
forward calls are accepted. Uploaded source is parsed with ast, never executed.
Visual metadata is optional and contains no executable graph representation.
"""
from __future__ import annotations

import ast
import copy
import keyword
import re

from .shape_rules import CNN_DEFAULTS, CNN_PARAMETERS, normalize_cnn_params
from .ports import output_port_ids, output_mode, call_source_refs
from .schema import PUBLIC_TYPES
from .vision_python import VISION_KINDS, HELPER_KINDS, HELPERS as VISION_HELPERS, POLICY_HELPER, parse_policy, exact_policy_helper, remove_policy_guard
from .operators import operator_spec
from .operator_rules import normalize_operator
from . import tensor_python


_PARAMS = {
    "Linear": ("in_features", "out_features"), "ReLU": (), "GELU": (),
    "Add": (), "Concat": ("dim",), "LayerNorm": ("normalized_shape",),
    "Dropout": ("p",),
}
_DEFAULTS = {"Concat": {"dim": -1}, "Dropout": {"p": 0.5}}
_PARAMS.update({kind: required + optional for kind, (required, optional) in CNN_PARAMETERS.items()})
_DEFAULTS.update(CNN_DEFAULTS)
for _kind in VISION_KINDS:
    _spec = operator_spec(_kind)
    _PARAMS[_kind] = _spec.required + tuple(_spec.defaults)
    _DEFAULTS[_kind] = _spec.defaults
_PUBLIC_RESERVED = {"self", "nn", "torch", "super", "Add", "Concat", "Model"}
_V4_RESERVED = {*HELPER_KINDS, 'MLFORGE_EXECUTION_POLICY'}
_HELPERS = '''class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat((a, b), dim=self.dim)
'''


def _fail(node, message):
    line = getattr(node, "lineno", None)
    suffix = f" (line {line})" if line else ""
    raise ValueError(message + suffix)


def _check_active_global_bindings(bindings, active_names):
    """An exact helper AST must also retain its actual Python global binding."""
    for name in active_names:
        declarations = bindings.get(name, [])
        if len(declarations) != 1:
            _fail(declarations[-1] if declarations else None,
                  'Top-level symbol binding conflicts with active helper or execution policy: ' + name)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def _literal(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        _fail(node, "Unsupported expression; use a literal or a public constructor parameter")
    if not isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
        _fail(node, "Unsupported literal")
    def json_sequences(item):
        if isinstance(item, (list, tuple)):
            return [json_sequences(child) for child in item]
        if isinstance(item, dict):
            return {key: json_sequences(child) for key, child in item.items()}
        return item
    return json_sequences(value)


def _path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _path(node.value)
        return base + "." + node.attr if base else None
    return None


def _ordered_calls(graph, definitions=()):
    calls = list(graph.get("calls", []))
    instances = {item["id"]: item for item in graph.get("instances", [])}
    available = {"input:" + i["id"] for i in graph.get("inputs", [])}
    ordered = []
    while calls:
        ready = next((c for c in calls if set(c.get("inputs", {}).values()) <= available), None)
        if ready is None:
            raise ValueError("Graph has a cycle or a missing source")
        calls.remove(ready)
        ordered.append(ready)
        available.update(call_source_refs(ready["id"], output_port_ids(instances[ready["instance_id"]], definitions)))
    return ordered


def _visual_scope(graph):
    return {
        "inputs": {i["id"]: {"name": i.get("name", i["id"]), "position": i.get("position")} for i in graph.get("inputs", [])},
        "instances": {i["id"]: {"name": i.get("name", i["id"]), "kind": i["kind"], "definition_id": i.get("definition_id"), **({"frozen": False} if i.get("frozen") is False else {})} for i in graph.get("instances", [])},
        "calls": {c["id"]: {"instance_id": c["instance_id"], "inputs": c["inputs"], "position": c.get("position")} for c in graph.get("calls", [])},
        "outputs": {o["id"]: {"position": o.get("position")} for o in graph.get("outputs", [])},
    }


def _export_output_mode(definition):
    ports = [item["id"] for item in definition["graph"]["outputs"]]
    return "named" if output_mode(definition) == "named" or ports != ["out"] else "tensor"


def python_profile(graph):
    if graph.get('schema_version') == 2:
        return 'tensor-pytorch-v1'
    scopes = [graph] + [definition['graph'] for definition in graph.get('definitions', [])]
    if graph.get('resource_profile') == 'hardware-v5':
        return 'vision-pytorch-v5'
    if graph.get('resource_profile') == 'vision-v4' or any(instance['kind'] in VISION_KINDS for scope in scopes for instance in scope.get('instances', [])):
        return 'vision-pytorch-v4'
    if any(_export_output_mode(definition) == "named" or any("type" in parameter for parameter in definition.get("public_parameters", [])) for definition in graph.get("definitions", [])) or any(
        isinstance(source, str) and source.count(":") == 2 for scope in scopes
        for source in [value for call in scope.get("calls", []) for value in call.get("inputs", {}).values()] + [item.get("source") for item in scope.get("outputs", [])]):
        return "ports-pytorch-v3"
    expanded = graph.get('resource_profile') == 'classification-v2' or any(
        instance['kind'] in CNN_PARAMETERS for scope in scopes for instance in scope.get('instances', []))
    return 'classification-pytorch-v2' if expanded else 's1-pytorch-v1'


def export_python(graph):
    """Generate a standalone Model plus ordinary reusable chip classes."""
    if not isinstance(graph, dict):
        raise ValueError("Graph must be an object")
    # The shared validator is optional during bootstrap, never a torch dependency.
    try:
        from .schema import validate_graph
    except ImportError:
        validate_graph = None
    if validate_graph is not None:
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError("Invalid graph: " + "; ".join(e["message"] for e in report["errors"]))
    from .classical_task import contains_classical
    if contains_classical(graph):
        from .classical_python import export_classical
        return export_classical(graph)
    definitions = graph.get("definitions", [])
    by_id = {_id(d["id"]): d for d in definitions}
    profile = python_profile(graph)
    typed = profile == 'tensor-pytorch-v1'
    metadata = {
        "profile": profile, "id": graph.get("id", "imported_model"),
        "name": graph.get("name", "Model"), "revision": graph.get("revision", 1),
        "root": _visual_scope(graph),
        "definitions": {d["id"]: {"name": d.get("name", d["id"]), "version": d.get("version", "0.1.0"), "scope": _visual_scope(d["graph"]), "public_names": {p["id"]: p.get("name", p["id"]) for p in d.get("public_parameters", [])}} for d in definitions},
    }
    if graph.get("resource_profile") == "classification-v2":
        metadata["resource_profile"] = "classification-v2"
    lines = ['"""Standalone PyTorch model. Import profile: ' + profile + '."""', "import torch", "from torch import nn", "", "STUDIO_VISUAL_METADATA = " + repr(metadata), "", _HELPERS]
    if profile == 'vision-pytorch-v5':
        lines.insert(1, '# Hardware-v5: actual PyTorch allocation applies; platform memory estimates are not reproduced here.')
    if typed or profile in ('vision-pytorch-v4', 'vision-pytorch-v5'):
        lines.append(VISION_HELPERS)
        policy = {'resource_profile': graph.get('resource_profile')}
        if graph.get('resource_profile') in ('vision-v4', 'hardware-v5'):
            policy['resource_batch_bound'] = graph['resource_batch_bound']
        if typed:
            policy['capability_profile'] = 'tensor-v1'
            lines.append(tensor_python.HELPERS)
        lines.extend(["MLFORGE_EXECUTION_POLICY = " + repr(policy), ''])
    if graph.get('resource_profile') in ('vision-v4', 'hardware-v5'):
        lines.append(tensor_python.GUARD if typed else POLICY_HELPER)

    def emit(class_name, scope, definition=None):
        public = definition.get("public_parameters", []) if definition else []
        public_ids = {_id(p["id"]) for p in public}
        public_defaults = {p["id"]: p["default"] for p in public}
        if any(keyword.iskeyword(p) for p in public_ids):
            raise ValueError("Public parameter names cannot be Python keywords")
        reserved = _PUBLIC_RESERVED | (_V4_RESERVED if typed or profile in ('vision-pytorch-v4', 'vision-pytorch-v5') else set()) | {"Chip_" + identifier for identifier in by_id}
        if typed:
            reserved |= tensor_python.PARAMS.keys()
        if graph.get('resource_profile') in ('vision-v4', 'hardware-v5'): reserved.add('_check_execution_policy')
        if public_ids & reserved:
            raise ValueError("Public parameter uses a reserved exporter name: " + ", ".join(sorted(public_ids & reserved)))
        args = "".join(", " + p["id"] + (": " + repr(p["type"]) if "type" in p else "") + "=" + repr(p["default"]) for p in public)
        lines.extend(["", "class " + class_name + "(nn.Module):", "    def __init__(self" + args + "):", "        super().__init__()", "        self.components = nn.ModuleDict({"])
        targets = {}
        for parameter, paths in (definition.get("bindings", {}) if definition else {}).items():
            for target in paths:
                if target in targets:
                    raise ValueError("Multiple public bindings target " + target)
                targets[target] = parameter
        used_targets = set()
        for instance in scope.get("instances", []):
            instance_id = _id(instance["id"])
            kind = instance["kind"]
            if kind == "chip":
                child = by_id.get(instance.get("definition_id"))
                if child is None:
                    raise ValueError("Unknown chip definition")
                constructor = "Chip_" + child["id"]
                params = dict(instance.get("params", {}))
            elif typed and kind in tensor_python.PARAMS:
                constructor = kind
                params = dict(instance.get('params', {}))
            elif kind in _PARAMS:
                constructor = kind if kind in ("Add", "Concat", *HELPER_KINDS) else "nn." + kind
                params = dict(instance.get("params", {}))
            else:
                raise ValueError("Unsupported component kind: " + str(kind))
            for target, parameter in targets.items():
                if target.startswith(instance_id + ".") and target.count(".") == 1:
                    if parameter not in public_defaults:
                        raise ValueError("Binding refers to an unknown public parameter")
                    target_parameter = target.split(".")[1]
                    params.setdefault(target_parameter, copy.deepcopy(public_defaults[parameter]))
            rendered = []
            for parameter, value in params.items():
                _id(parameter)
                target = instance_id + "." + parameter
                if target in targets:
                    if targets[target] not in public_ids:
                        raise ValueError("Binding refers to an unknown public parameter")
                    rendered.append(parameter + "=" + targets[target])
                    used_targets.add(target)
                else:
                    rendered.append(parameter + "=" + repr(value))
            lines.append("            " + repr(instance_id) + ": " + constructor + "(" + ", ".join(rendered) + "),")
        if used_targets != set(targets):
            raise ValueError("S1 Python export supports bindings to immediate component parameters only")
        lines.append("        })")
        for instance in scope.get("instances", []):
            if instance.get("frozen", False):
                lines.append("        self.components[" + repr(instance["id"]) + "].requires_grad_(False)")
        input_ids = [_id(i["id"]) for i in scope.get("inputs", [])]
        if any(keyword.iskeyword(i) or i == 'self' or (i == '_check_execution_policy' and graph.get('resource_profile') in ('vision-v4', 'hardware-v5') and definition is None) or i.startswith("call_") for i in input_ids):
            raise ValueError("Input IDs must be Python argument names and cannot start with call_")
        arguments = []
        for item in scope.get("inputs", []):
            annotation = item.get("dtype", "float32") + "[" + ",".join("?" if d is None else str(d) for d in item["shape"]) + "]"
            arguments.append(item["id"] + ": " + repr(tensor_python.annotation(item) if typed else annotation))
        lines.extend(["", "    def forward(self" + "".join(", " + i for i in arguments) + "):"])
        if definition is None and graph.get('resource_profile') in ('vision-v4', 'hardware-v5'):
            lines.append('        ' + tensor_python.guard_call(scope['inputs']) if typed else
                         "        _check_execution_policy(" + ', '.join(input_ids) + ")")
        sources = {"input:" + i: i for i in input_ids}
        instances_by_id = {i["id"]: i for i in scope.get("instances", [])}
        for call in _ordered_calls(scope, by_id):
            variable = "call_" + _id(call["id"])
            arguments = []
            instance = instances_by_id[call["instance_id"]]
            kind = instance["kind"]
            for port, source in call["inputs"].items():
                if source not in sources:
                    raise ValueError("Missing call source: " + source)
                argument_name = "input" if kind not in ("chip", "Add", "Concat", "Multiply", *tensor_python.PARAMS) and port == "x" else _id(port)
                arguments.append(argument_name + "=" + sources[source])
            lines.append("        " + variable + " = self.components[" + repr(call["instance_id"]) + "](" + ", ".join(arguments) + ")")
            ports = output_port_ids(instance, by_id)
            named = kind == "chip" and _export_output_mode(by_id[instance["definition_id"]]) == "named"
            for port in ports:
                expression = variable + "[" + repr(port) + "]" if named else variable
                sources["call:" + call["id"] + ":" + port] = expression
                if len(ports) == 1:
                    sources["call:" + call["id"]] = expression
        outputs = scope.get("outputs", [])
        if any(o.get("source") not in sources for o in outputs):
            raise ValueError("Missing output source")
        if definition and _export_output_mode(definition) == "tensor":
            lines.append("        return " + sources[outputs[0]["source"]])
        else:
            lines.append("        return {" + ", ".join(repr(o["id"]) + ": " + sources[o["source"]] for o in outputs) + "}")

    for definition in definitions:
        emit("Chip_" + definition["id"], definition["graph"], definition)
    emit("Model", graph)
    source = "\n".join(lines) + "\n"
    try:
        compile(source, "<studio-export>", "exec")
    except (SyntaxError, ValueError) as error:
        raise ValueError("Cannot generate valid Python: " + str(error)) from None
    return source


class _AliasNormalizer(ast.NodeTransformer):
    def __init__(self, aliases):
        self.aliases = aliases

    def visit_Name(self, node):
        if node.id in self.aliases:
            return ast.copy_location(ast.Name(id=self.aliases[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if _path(node) == "torch.nn":
            return ast.copy_location(ast.Name(id="nn", ctx=node.ctx), node)
        return node


def _is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _component_ref(node):
    if isinstance(node, ast.Subscript) and _path(node.value) == "self.components":
        return _id(_literal(node.slice))
    return None


def _public_annotation(annotation):
    if not isinstance(annotation, ast.Constant) or type(annotation.value) is not str or annotation.value not in PUBLIC_TYPES:
        _fail(annotation, "Unsupported public constructor annotation; expected a literal auto/int/float/bool/shape string")
    return annotation.value


def _function_signature(method, defaults_required=False, *, reserved=None, typed=False):
    args = method.args
    if method.decorator_list or method.returns is not None or args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs:
        _fail(method, "Unsupported function signature")
    names = [arg.arg for arg in args.args]
    if not names or names[0] != "self":
        _fail(method, "Methods must have self as the first argument")
    for index, argument in enumerate(args.args):
        if argument.annotation is not None:
            if defaults_required:
                if index == 0: _fail(argument, "Unsupported self annotation")
                _public_annotation(argument.annotation)
            elif not typed:
                _annotation_spec(argument.annotation)
    names = names[1:]
    if defaults_required and set(names) & (_PUBLIC_RESERVED if reserved is None else reserved):
        _fail(method, "Public parameter uses a reserved exporter name")
    if defaults_required and len(args.defaults) != len(names):
        _fail(method, "Every public constructor parameter needs a literal default")
    if not defaults_required and args.defaults:
        _fail(method, "Unsupported forward defaults")
    return names


def _annotation_spec(annotation):
    if annotation is None or _path(annotation) == "torch.Tensor":
        return {"shape": [None, None], "dtype": "float32"}
    if not isinstance(annotation, ast.Constant) or not isinstance(annotation.value, str):
        _fail(annotation, "Unsupported input annotation; expected a literal shape annotation or torch.Tensor")
    match = re.fullmatch(r"(float32)\[([?0-9,]+)\]", annotation.value)
    if match is None:
        _fail(annotation, "Unsupported input annotation; use 'float32[?,4]' shape notation")
    dimensions = match.group(2).split(",")
    shape = []
    for index, value in enumerate(dimensions):
        if index == 0 and value == "?":
            shape.append(None)
        elif value.isdigit() and int(value) > 0:
            shape.append(int(value))
        else:
            _fail(annotation, "Input dimensions must be positive; only the batch dimension may be ?")
    return {"shape": shape, "dtype": match.group(1)}


def _class_output_contract(node):
    forward = next((method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "forward"), None)
    returned = next((statement for statement in forward.body if isinstance(statement, ast.Return)), None) if forward else None
    if returned is None:
        _fail(node, "Chip forward needs an explicit return")
    if isinstance(returned.value, ast.Dict):
        ports = [_id(_literal(key)) for key in returned.value.keys]
        if not ports or len(set(ports)) != len(ports):
            _fail(returned, "Chip dictionary return needs distinct named output ports")
        return ports, "named"
    return ["out"], "tensor"


def _read_class(node, all_classes, helper_names=None, reserved=None, *, typed=False):
    helper_names = {'Add', 'Concat'} if helper_names is None else helper_names
    methods = {}
    for child in node.body:
        if _is_docstring(child):
            continue
        if not isinstance(child, ast.FunctionDef) or child.name not in ("__init__", "forward") or child.name in methods:
            _fail(child, "Unsupported class member; expected __init__ and forward only")
        methods[child.name] = child
    if set(methods) != {"__init__", "forward"}:
        _fail(node, "A model class must define __init__ and forward")
    constructor = methods["__init__"]
    public_ids = _function_signature(constructor, defaults_required=True, reserved=reserved)
    public = [{"id": name, "name": name, "default": _literal(value)} for name, value in zip(public_ids, constructor.args.defaults)]
    for parameter, argument in zip(public, constructor.args.args[1:]):
        if argument.annotation is not None:
            parameter["type"] = _public_annotation(argument.annotation)
    defaults = {p["id"]: p["default"] for p in public}
    bindings = {p: [] for p in public_ids}
    instances = []
    found_components = False
    found_super = False
    for statement in constructor.body:
        if _is_docstring(statement):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            value = statement.value
            if isinstance(value.func, ast.Attribute) and value.func.attr == "__init__" and isinstance(value.func.value, ast.Call) and _path(value.func.value.func) == "super" and not value.args and not value.keywords and not value.func.value.args and not value.func.value.keywords and not found_super:
                found_super = True
                continue
            if isinstance(value.func, ast.Attribute) and value.func.attr == "requires_grad_" and len(value.args) == 1 and not value.keywords:
                component = _component_ref(value.func.value)
                frozen_value = _literal(value.args[0])
                if component is None or type(frozen_value) is not bool:
                    _fail(statement, "Unsupported requires_grad_ target or value")
                target = next((i for i in instances if i["id"] == component), None)
                if target is None:
                    _fail(statement, "Freeze references a component before its construction")
                target["frozen"] = not frozen_value
                continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or _path(statement.targets[0]) != "self.components" or found_components:
            _fail(statement, "Unsupported constructor statement")
        value = statement.value
        if not isinstance(value, ast.Call) or _path(value.func) != "nn.ModuleDict" or len(value.args) != 1 or value.keywords or not isinstance(value.args[0], ast.Dict):
            _fail(statement, "Expected self.components = nn.ModuleDict({...})")
        found_components = True
        dictionary = value.args[0]
        for key, expression in zip(dictionary.keys, dictionary.values):
            instance_id = _id(_literal(key))
            if any(i["id"] == instance_id for i in instances):
                _fail(key, "Duplicate component ID: " + instance_id)
            if not isinstance(expression, ast.Call):
                _fail(expression, "Unsupported component constructor")
            path = _path(expression.func)
            if path in {"nn." + p for p in _PARAMS if p not in ("Add", "Concat", "Multiply", "Scale", "OneMinus")}:
                kind = path[3:]
                parameter_ids = _PARAMS[kind]
                if typed:
                    parameter_ids += {'Linear': ('bias',), 'LayerNorm': ('eps', 'elementwise_affine')}.get(kind, ())
                instance = {"id": instance_id, "kind": kind, "name": instance_id, "params": dict(_DEFAULTS.get(kind, {})) if kind == "Dropout" else {}}
            elif path in helper_names:
                if path not in all_classes:
                    _fail(expression, "Missing helper class: " + path)
                kind = path
                parameter_ids = tensor_python.PARAMS[kind] if typed and kind in tensor_python.PARAMS else _PARAMS[kind]
                instance = {"id": instance_id, "kind": kind, "name": instance_id, "params": {}}
            elif path in all_classes and path not in helper_names:
                child = all_classes[path]
                child_methods = [m for m in child.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"]
                if len(child_methods) != 1:
                    _fail(expression, "Chip class needs an explicit constructor")
                parameter_ids = tuple(_function_signature(child_methods[0], defaults_required=True, reserved=reserved))
                instance = {"id": instance_id, "kind": "chip", "definition_id": path[5:] if path.startswith("Chip_") else path, "name": instance_id, "params": {}}
            else:
                _fail(expression, "Unsupported component constructor: " + str(path))
            if len(expression.args) > len(parameter_ids):
                _fail(expression, "Too many positional constructor arguments")
            values = dict(zip(parameter_ids, expression.args))
            for kw in expression.keywords:
                if kw.arg is None or kw.arg not in parameter_ids or kw.arg in values:
                    _fail(kw, "Unsupported or duplicate constructor argument")
                values[kw.arg] = kw.value
            for parameter, parameter_value in values.items():
                if isinstance(parameter_value, ast.Name) and parameter_value.id in defaults:
                    instance["params"][parameter] = copy.deepcopy(defaults[parameter_value.id])
                    bindings[parameter_value.id].append(instance_id + "." + parameter)
                else:
                    instance["params"][parameter] = _literal(parameter_value)
            kind = instance["kind"]
            required_ids = operator_spec(kind).required if kind in VISION_KINDS else (CNN_PARAMETERS[kind][0] if kind in CNN_PARAMETERS else (() if kind in ("chip", "Concat") else parameter_ids))
            if typed and kind in ('Linear', 'LayerNorm'):
                required_ids = _PARAMS[kind]
            if any(p not in instance["params"] for p in required_ids):
                _fail(expression, "Missing required constructor argument")
            if typed and kind != 'chip':
                from .tensor_ops import normalize_tensor_operator
                try:
                    normalize_tensor_operator(kind, instance['params'])
                except ValueError as error:
                    _fail(expression, str(error))
            else:
                _validate_params(instance, expression)
            instances.append(instance)
    if not found_components or not found_super:
        _fail(constructor, "Expected super().__init__() and self.components ModuleDict")
    forward = methods["forward"]
    input_ids = _function_signature(forward, typed=typed)
    def parse_annotation(arg):
        try:
            return tensor_python.parse_annotation(arg.annotation) if typed else _annotation_spec(arg.annotation)
        except ValueError as error:
            _fail(arg, str(error))
    inputs = [{"id": _id(arg.arg), "name": arg.arg, **parse_annotation(arg)} for arg in forward.args.args[1:]]
    references = {i: "input:" + i for i in input_ids}
    bundles = {}
    defined_names = {"self", *input_ids}

    def tensor_source(expression):
        if isinstance(expression, ast.Name) and expression.id in references:
            return references[expression.id]
        if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name) and expression.value.id in bundles:
            if not isinstance(expression.slice, ast.Constant) or not isinstance(expression.slice.value, str):
                _fail(expression, "Named chip output selection needs a literal string port")
            port = expression.slice.value
            if port not in bundles[expression.value.id]:
                _fail(expression, "Unknown named chip output port: " + port)
            return bundles[expression.value.id][port]
        _fail(expression, "Tensor expression must be an earlier named tensor or a literal selection from a named chip result")

    calls = []
    outputs = None
    instances_by_id = {i["id"]: i for i in instances}
    for statement in forward.body:
        if _is_docstring(statement):
            continue
        if outputs is not None:
            _fail(statement, "Unsupported statement after return")
        if isinstance(statement, ast.Return):
            value = statement.value
            if isinstance(value, ast.Dict):
                outputs = []
                for key, source in zip(value.keys, value.values):
                    output_id = _id(_literal(key))
                    if any(o["id"] == output_id for o in outputs):
                        _fail(key, "Duplicate output ID")
                    outputs.append({"id": output_id, "source": tensor_source(source)})
            elif isinstance(value, (ast.Name, ast.Subscript)):
                outputs = [{"id": "out", "source": tensor_source(value)}]
            else:
                _fail(statement, "Unsupported return; use a named tensor or a dictionary of named tensors")
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            _fail(statement, "Unsupported forward statement; use explicit named component calls")
        variable = statement.targets[0].id
        if variable in defined_names:
            _fail(statement, "Unsupported reassignment of a tensor or chip result name")
        defined_names.add(variable)
        expression = statement.value
        if isinstance(expression, ast.Name) and expression.id in bundles:
            bundles[variable] = dict(bundles[expression.id])
            continue
        if isinstance(expression, (ast.Name, ast.Subscript)):
            references[variable] = tensor_source(expression)
            continue
        if not isinstance(expression, ast.Call):
            _fail(expression, "Unsupported forward expression")
        instance_id = _component_ref(expression.func)
        if instance_id not in instances_by_id:
            _fail(expression, "Call must target an existing self.components entry")
        instance = instances_by_id[instance_id]
        if instance["kind"] == "chip":
            child_class = "Chip_" + instance["definition_id"]
            if child_class not in all_classes:
                child_class = instance["definition_id"]
            child_forward = next((m for m in all_classes[child_class].body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
            if child_forward is None:
                _fail(expression, "Chip has no forward")
            ports = _function_signature(child_forward, typed=typed)
            result_ports, result_mode = _class_output_contract(all_classes[child_class])
        elif typed and instance['kind'] in tensor_python.PARAMS:
            from .tensor_ops import TENSOR_KINDS
            ports = TENSOR_KINDS[instance['kind']]
            result_ports, result_mode = ['out'], 'tensor'
        else:
            ports = ["a", "b"] if instance["kind"] in ("Add", "Concat", "Multiply") else ["input"]
            result_ports, result_mode = ["out"], "tensor"
        if len(expression.args) > len(ports):
            _fail(expression, "Too many forward arguments")
        arguments = dict(zip(ports, expression.args))
        for kw in expression.keywords:
            if kw.arg is None or kw.arg not in ports or kw.arg in arguments:
                _fail(kw, "Unsupported or duplicate forward argument")
            arguments[kw.arg] = kw.value
        if set(arguments) != set(ports):
            _fail(expression, "Component call does not supply every input port")
        connections = {}
        for port, source in arguments.items():
            graph_port = "x" if port == "input" and instance["kind"] not in ("chip", "Add", "Concat", "Multiply") else port
            connections[graph_port] = tensor_source(source)
        call_id = _id(variable[5:] if variable.startswith("call_") else variable)
        if any(c["id"] == call_id for c in calls):
            _fail(statement, "Duplicate call ID")
        calls.append({"id": call_id, "instance_id": instance_id, "inputs": connections})
        if result_mode == "named":
            bundles[variable] = {port: "call:" + call_id + ":" + port for port in result_ports}
        else:
            references[variable] = "call:" + call_id
    if not outputs:
        _fail(forward, "Forward needs at least one return value")
    return {"inputs": inputs, "instances": instances, "calls": calls, "outputs": outputs}, public, {p: paths for p, paths in bindings.items() if paths}


def _validate_params(instance, node):
    kind, params = instance["kind"], instance["params"]
    if kind in VISION_KINDS:
        try:
            instance['params'] = normalize_operator(kind, params)
        except ValueError as error:
            _fail(node, str(error))
    elif kind in CNN_PARAMETERS:
        try:
            normalize_cnn_params(kind, params)
        except ValueError as error:
            _fail(node, str(error))
    elif kind == "Linear":
        if any(type(params[p]) is not int or params[p] <= 0 for p in ("in_features", "out_features")):
            _fail(node, "Linear dimensions must be positive integers")
    elif kind == "Dropout":
        if type(params["p"]) not in (int, float) or not 0 <= params["p"] <= 1:
            _fail(node, "Dropout p must be between 0 and 1")
    elif kind == "Concat" and type(params.get("dim", -1)) is not int:
        _fail(node, "Concat dim must be an integer")
    elif kind == "LayerNorm":
        normalized = params["normalized_shape"]
        values = normalized if isinstance(normalized, (list, tuple)) else [normalized]
        if not values or any(type(v) is not int or v <= 0 for v in values):
            _fail(node, "LayerNorm normalized_shape must contain positive integers")
        if isinstance(normalized, tuple):
            params["normalized_shape"] = list(normalized)


def _infer_shapes(scope, definitions, active=None):
    """Infer input widths from actual constructor contracts, without executing."""
    active = set() if active is None else active
    instances = {i["id"]: i for i in scope["instances"]}
    widths = {"input:" + i["id"]: None for i in scope["inputs"]}
    constraints = []
    for call in scope["calls"]:
        instance = instances[call["instance_id"]]
        kind, params = instance["kind"], instance["params"]
        output = "call:" + call["id"]
        sources = list(call["inputs"].values())
        port_ids = output_port_ids(instance, definitions)
        for port in port_ids:
            qualified = output + ":" + port
            widths[qualified] = None
            if len(port_ids) == 1:
                widths[output] = None
                constraints.append(("same", qualified, output))
        if kind == "Linear":
            widths[output] = params["out_features"]
            constraints.append(("fixed", sources[0], params["in_features"]))
        elif kind == "LayerNorm":
            normal = params["normalized_shape"]
            size = normal[-1] if isinstance(normal, (list, tuple)) else normal
            widths[output] = size
            constraints.append(("fixed", sources[0], size))
        elif kind == "chip":
            definition_id = instance["definition_id"]
            if definition_id in active:
                raise ValueError("Recursive chip definitions are unsupported")
            definition = definitions[definition_id]
            child = copy.deepcopy(definition["graph"])
            for public, targets in definition["bindings"].items():
                if public not in params:
                    continue
                for target in targets:
                    target_id, target_parameter = target.split(".")
                    next(i for i in child["instances"] if i["id"] == target_id)["params"][target_parameter] = params[public]
            child_widths = _infer_shapes(child, definitions, active | {definition_id})
            for port, source in call["inputs"].items():
                size = child_widths.get("input:" + port)
                if size is not None:
                    constraints.append(("fixed", source, size))
            for port in child["outputs"]:
                size = child_widths.get(port["source"])
                widths[output + ":" + port["id"]] = size
                if len(port_ids) == 1:
                    widths[output] = size
        elif kind == "Concat":
            if params.get("dim", -1) in (-1, 1):
                constraints.append(("sum", output, sources))
            else:
                for source in sources:
                    constraints.append(("same", output, source))
        elif kind in CNN_PARAMETERS or kind == 'BatchNorm2d':
            # Spatial dimensions need the source's explicit rank-four shape;
            # treating pooling/flatten as a width identity invents dimensions.
            continue
        else:
            for source in sources:
                constraints.append(("same", output, source))
    for _ in range(max(2, len(widths) * 2)):
        changed = False
        for kind, target, value in constraints:
            if kind == "fixed":
                if widths.get(target) is None:
                    widths[target] = value
                    changed = True
            elif kind == "same":
                if widths.get(target) is None and widths.get(value) is not None:
                    widths[target] = widths[value]
                    changed = True
                elif widths.get(value) is None and widths.get(target) is not None:
                    widths[value] = widths[target]
                    changed = True
            elif all(widths.get(s) is not None for s in value) and widths.get(target) is None:
                widths[target] = sum(widths[s] for s in value)
                changed = True
        if not changed:
            break
    for item in scope["inputs"]:
        inferred = widths.get("input:" + item["id"])
        if inferred is not None and item["shape"][-1] is None:
            item["shape"][-1] = inferred
    return widths


def _restore_visual(scope, metadata):
    if not isinstance(metadata, dict):
        return
    for group in ("inputs", "instances", "outputs"):
        saved = metadata.get(group, {})
        if not isinstance(saved, dict):
            continue
        for item in scope[group]:
            previous = saved.get(item["id"])
            if not isinstance(previous, dict):
                continue
            if group == "instances" and (previous.get("kind") != item["kind"] or previous.get("definition_id") != item.get("definition_id")):
                continue
            if isinstance(previous.get("name"), str):
                item["name"] = previous["name"]
            if group == "instances" and previous.get("frozen") is False and "frozen" not in item:
                item["frozen"] = False
            if group in ("inputs", "outputs"):
                position = previous.get("position")
                if isinstance(position, dict) and set(position) == {"x", "y"} and all(type(position[p]) in (int, float) for p in position):
                    item["position"] = copy.deepcopy(position)
    saved_calls = metadata.get("calls", {})
    if isinstance(saved_calls, dict):
        for call in scope["calls"]:
            previous = saved_calls.get(call["id"])
            if not isinstance(previous, dict) or previous.get("instance_id") != call["instance_id"] or previous.get("inputs") != call["inputs"]:
                continue
            position = previous.get("position")
            if isinstance(position, dict) and set(position) == {"x", "y"} and all(type(position[p]) in (int, float) for p in position):
                call["position"] = copy.deepcopy(position)


def import_python(code):
    """Parse supported constructor/forward syntax without executing the source."""
    if not isinstance(code, str) or len(code) > 2_000_000:
        raise ValueError("Python source must be text of at most 2 MB")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("Invalid Python syntax: " + str(error)) from None
    if any(isinstance(node, ast.ImportFrom) and node.module == "sklearn.svm" for node in tree.body):
        from .classical_python import import_classical
        return import_classical(tree)
    aliases = {}
    classes = {}
    metadata = {}
    policy = None
    policy_helper = False
    tensor_guard = False
    global_bindings = {}

    def bind_global(name, node):
        global_bindings.setdefault(name, []).append(node)
    for node in tree.body:
        if _is_docstring(node):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ("torch", "torch.nn"):
                    _fail(node, "Unsupported import: " + alias.name)
                bind_global(alias.asname or 'torch', node)
                aliases[alias.asname or "torch"] = "nn" if alias.name == "torch.nn" and alias.asname else "torch"
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "torch" and node.level == 0:
            for alias in node.names:
                if alias.name != "nn":
                    _fail(node, "Unsupported import from torch")
                bind_global(alias.asname or 'nn', node)
                aliases[alias.asname or "nn"] = "nn"
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "STUDIO_VISUAL_METADATA":
            if metadata:
                _fail(node, "Duplicate visual metadata")
            value = _literal(node.value)
            metadata = value if isinstance(value, dict) else {}
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'MLFORGE_EXECUTION_POLICY':
            if policy is not None:
                _fail(node, 'Duplicate execution policy')
            policy = parse_policy(node.value)
            bind_global('MLFORGE_EXECUTION_POLICY', node)
            continue
        if isinstance(node, ast.FunctionDef) and node.name == '_check_execution_policy':
            tensor_form = ast.dump(node) == ast.dump(ast.parse(tensor_python.GUARD).body[0])
            if policy_helper or not (tensor_form or exact_policy_helper(node)):
                _fail(node, 'Unsupported modified execution policy helper')
            policy_helper = True
            tensor_guard = tensor_form
            bind_global('_check_execution_policy', node)
            continue
        if isinstance(node, ast.ClassDef):
            if node.name in classes:
                _fail(node, "Duplicate class: " + node.name)
            classes[node.name] = node
            bind_global(node.name, node)
            continue
        _fail(node, "Unsupported top-level Python statement")
    active_globals = set()
    if policy is not None: active_globals.add('MLFORGE_EXECUTION_POLICY')
    if policy_helper: active_globals.add('_check_execution_policy')
    typed = bool(policy and policy.get('capability_profile') == 'tensor-v1')
    if tensor_guard != typed:
        raise ValueError('Tensor execution helper requires an explicit matching tensor-v1 policy')
    _check_active_global_bindings(global_bindings, active_globals)
    normalizer = _AliasNormalizer(aliases)
    expected_helpers = {node.name: node for node in ast.parse(_HELPERS + '\n' + VISION_HELPERS +
                                                             ('\n' + tensor_python.HELPERS if typed else '')).body}
    helper_names = {'Add', 'Concat'}
    if typed:
        helper_names.update(tensor_python.PARAMS)
    for name, node in list(classes.items()):
        if name in aliases:
            _fail(node, "Class name conflicts with an imported module")
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                if any(argument.arg in aliases for argument in method.args.args[1:]):
                    _fail(method, "Public parameter shadows an imported module alias")
        for reference in ast.walk(node):
            if isinstance(reference, ast.Name) and isinstance(reference.ctx, ast.Load) and reference.id in ("nn", "torch") and reference.id not in aliases:
                _fail(reference, "Missing import for " + reference.id)
        node = normalizer.visit(node)
        if name in HELPER_KINDS and (policy is not None or ast.dump(node, include_attributes=False) == ast.dump(expected_helpers[name], include_attributes=False)):
            helper_names.add(name)
        expected_base = 'nn.BatchNorm2d' if name == 'BatchNorm2d' and name in helper_names else 'nn.Module'
        if node.decorator_list or node.keywords or len(node.bases) != 1 or _path(node.bases[0]) != expected_base:
            _fail(node, "Unsupported class base; expected nn.Module")
        classes[name] = node
    for name in helper_names:
        if name in classes and ast.dump(classes[name], include_attributes=False) != ast.dump(expected_helpers[name], include_attributes=False):
            _fail(classes[name], "Unsupported modified " + name + " helper implementation")
    _check_active_global_bindings(global_bindings, active_globals | (helper_names & classes.keys()))
    if policy_helper and policy is None:
        raise ValueError('Execution policy helper requires MLFORGE_EXECUTION_POLICY')
    graph_classes = {name: node for name, node in classes.items() if name not in helper_names}
    guarded = {name: (tensor_python.strip_guard(node) if typed else remove_policy_guard(node, policy, policy_helper))
               for name, node in graph_classes.items()}
    if not graph_classes:
        raise ValueError("No supported nn.Module model class found")
    reserved = _PUBLIC_RESERVED | helper_names
    if policy is not None: reserved |= {'MLFORGE_EXECUTION_POLICY'}
    if policy_helper: reserved |= {'_check_execution_policy'}
    parsed = {name: _read_class(node, classes, helper_names, reserved, typed=typed) for name, node in graph_classes.items()}
    referenced = {i["definition_id"] for scope, _, _ in parsed.values() for i in scope["instances"] if i["kind"] == "chip"}
    roots = [name for name in parsed if (name[5:] if name.startswith("Chip_") else name) not in referenced]
    if "Model" in roots and all(name == "Model" or name.startswith("Chip_") for name in roots):
        root_name = "Model"
    elif "Model" not in parsed and len(roots) == 1:
        root_name = roots[0]
    else:
        raise ValueError("Expected exactly one root model; recursive or unrelated model classes are unsupported")
    if policy is not None and policy['resource_profile'] in ('vision-v4', 'hardware-v5'):
        if not policy_helper or not guarded[root_name]:
            raise ValueError(policy['resource_profile'] + ' requires the exact root batch guard and execution policy helper')
    if any(present and name != root_name for name, present in guarded.items()):
        raise ValueError('Execution policy guard is supported on the root only; chip/internal guards are unsupported')
    root_scope, root_public, _ = parsed[root_name]
    if root_public:
        raise ValueError("Root constructor parameters are unsupported; use chip public parameters")
    root_forward = next(m for m in graph_classes[root_name].body if isinstance(m, ast.FunctionDef) and m.name == "forward")
    root_return = next((s for s in root_forward.body if isinstance(s, ast.Return)), None)
    if root_return is None or not isinstance(root_return.value, ast.Dict):
        raise ValueError("Root forward must return a dictionary of named outputs")
    definitions = []
    for class_name, (scope, public, bindings) in parsed.items():
        if class_name == root_name:
            continue
        definition_id = class_name[5:] if class_name.startswith("Chip_") else class_name
        _id(definition_id)
        method = next(m for m in graph_classes[class_name].body if isinstance(m, ast.FunctionDef) and m.name == "forward")
        returned = next(s for s in method.body if isinstance(s, ast.Return))
        definition = {"id": definition_id, "name": definition_id, "version": "0.1.0", "public_parameters": public, "bindings": bindings, "graph": scope}
        if isinstance(returned.value, ast.Dict):
            definition["output_mode"] = "named"
        definitions.append(definition)
    by_id = {d["id"]: d for d in definitions}

    def has_frozen_descendant(definition_id, visiting):
        if definition_id in visiting:
            return False
        visiting = visiting | {definition_id}
        for instance in by_id[definition_id]["graph"]["instances"]:
            if instance.get("frozen") is True or (instance["kind"] == "chip" and has_frozen_descendant(instance["definition_id"], visiting)):
                return True
        return False

    # Here frozen=False can only come from an explicit source unfreeze; visual
    # metadata has not yet restored optional False-presence hints.
    for class_name, (scope, _, _) in parsed.items():
        for instance in scope["instances"]:
            if instance["kind"] == "chip" and instance.get("frozen") is False and has_frozen_descendant(instance["definition_id"], set()):
                _fail(graph_classes[class_name], "Recursive unfreeze of chip " + instance["id"] + " over frozen descendants is unsupported")
    if not typed:
        for definition in definitions:
            _infer_shapes(definition["graph"], by_id, {definition["id"]})
        _infer_shapes(root_scope, by_id)
    result = {"schema_version": 2 if typed else 1, "id": "imported_model", "name": root_name, "revision": 1, **root_scope, "definitions": definitions}
    if metadata.get("profile") in ("s1-pytorch-v1", "classification-pytorch-v2", "ports-pytorch-v3", "vision-pytorch-v4", "vision-pytorch-v5", "tensor-pytorch-v1"):
        if isinstance(metadata.get("id"), str):
            result["id"] = metadata["id"]
        if isinstance(metadata.get("name"), str):
            result["name"] = metadata["name"]
        if type(metadata.get("revision")) is int and metadata["revision"] >= 1:
            result["revision"] = metadata["revision"]
        _restore_visual(result, metadata.get("root"))
        saved_definitions = metadata.get("definitions", {})
        if isinstance(saved_definitions, dict):
            for definition in definitions:
                previous = saved_definitions.get(definition["id"])
                if not isinstance(previous, dict):
                    continue
                for field in ("name", "version"):
                    if isinstance(previous.get(field), str):
                        definition[field] = previous[field]
                _restore_visual(definition["graph"], previous.get("scope"))
                names = previous.get("public_names", {})
                if isinstance(names, dict):
                    for parameter in definition["public_parameters"]:
                        if isinstance(names.get(parameter["id"]), str):
                            parameter["name"] = names[parameter["id"]]
    uses_vision = any(instance['kind'] in VISION_KINDS for scope in [root_scope] + [d['graph'] for d in definitions] for instance in scope['instances'])
    if policy is None and not uses_vision and (any(instance["kind"] in CNN_PARAMETERS for scope in [root_scope] + [d["graph"] for d in definitions] for instance in scope["instances"]) or (
        metadata.get("profile") in ("classification-pytorch-v2", "ports-pytorch-v3") and metadata.get("resource_profile") == "classification-v2")):
        result["resource_profile"] = "classification-v2"
    from .schema import validate_graph
    if policy is not None:
        result.update(policy)
        if result.get('resource_profile') is None:
            result.pop('resource_profile', None)
    report = validate_graph(result)
    if not report["valid"]:
        raise ValueError("Invalid imported graph: " + "; ".join(error["message"] for error in report["errors"]))
    return result
