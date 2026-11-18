"""Editable graph documents for the native workbench.

This module imports no GUI or PyTorch library. Semantic edits may leave a graph
incomplete; validation reports that state without rewriting its parameters.
Navigation records concrete call IDs so shared/nested chip probes stay distinct.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from studio_core.examples import example_graph
from studio_core.schema import MAX_CHIP_DEPTH, PRIMITIVES, validate_graph
from studio_core.ports import parse_source
from studio_core.operators import operator_defaults
from .document_editing import EditableCommands, command
from .component_insert import ComponentInsertion
from .persistence import replace_with_retry

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REF = re.compile(r"^(input|call|output):([A-Za-z_][A-Za-z0-9_]*)$")
_SCOPE_UNSET = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_values(value):
    """Reject non-JSON values before copying, hashing or replacing a document."""
    pending, count = [(value, 0)], 0
    while pending:
        current, depth = pending.pop()
        count += 1
        _require(count <= 20_000 and depth <= 32, "Graph JSON exceeds document size or nesting limits")
        if type(current) is dict:
            _require(all(type(key) is str for key in current), "Graph JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in current.values())
        elif type(current) is list:
            _require(len(current) <= 20_000, "Graph JSON array exceeds document limit")
            pending.extend((child, depth + 1) for child in current)
        elif type(current) is float:
            _require(math.isfinite(current), "Graph JSON numbers must be finite")
        else:
            _require(current is None or type(current) in (str, bool, int), "Graph must contain JSON values only")


def _structure(graph):
    """Check storage types, permitting algorithmically invalid editable drafts."""
    _json_values(graph)
    _require(type(graph) is dict, "Graph document must be an object")
    _require(type(graph.get("schema_version")) is int and graph["schema_version"] == 1, "Graph schema_version must equal 1")
    _require(type(graph.get("id")) is str, "Graph id must be a string")
    _require(type(graph.get("revision")) is int and graph["revision"] >= 1, "Graph revision must be a positive integer")
    profile=graph.get('resource_profile')
    if profile in ('vision-v4','hardware-v5'):
        bound=graph.get('resource_batch_bound')
        _require(type(bound) is int and 1<=bound<=64,'resource_batch_bound must be an integer in 1..64 for ' + profile)
    else:
        _require('resource_batch_bound' not in graph,'resource_batch_bound requires vision-v4 or hardware-v5')
    definitions = graph.get("definitions", [])
    _require(type(definitions) is list, "Graph definitions must be a list")
    scopes = [graph]
    for definition in definitions:
        _require(type(definition) is dict and type(definition.get("id")) is str, "Chip definitions must be objects with string IDs")
        _require(type(definition.get("graph")) is dict, "Chip definition graph must be an object")
        _require(type(definition.get("public_parameters", [])) is list and type(definition.get("bindings", {})) is dict, "Chip public parameters and bindings have invalid storage types")
        for parameter in definition.get("public_parameters", []):
            _require(type(parameter) is dict and type(parameter.get("id")) is str and "default" in parameter, "Public parameters must be objects with string IDs and defaults")
        _require(all(type(paths) is list and all(type(path) is str for path in paths) for paths in definition.get("bindings", {}).values()), "Public bindings must contain lists of target strings")
        scopes.append(definition["graph"])
    for scope in scopes:
        for key in ("inputs", "instances", "calls", "outputs"):
            _require(type(scope.get(key)) is list, "Graph " + key + " must be a list")
            for item in scope[key]:
                _require(type(item) is dict and type(item.get("id")) is str, key + " entries must be objects with string IDs")
                if "position" in item:
                    position = item["position"]
                    _require(type(position) is dict and all(type(position.get(axis)) in (int, float) for axis in ("x", "y")), "Node position must contain numeric x and y")
        for port in scope["inputs"]:
            _require("shape" not in port or type(port["shape"]) is list, "Input shape must be a list")
            _require("dtype" not in port or type(port["dtype"]) is str, "Input dtype must be a string")
        for instance in scope["instances"]:
            _require(type(instance.get("kind")) is str and type(instance.get("params", {})) is dict, "Instance kind/params have invalid storage types")
            _require("definition_id" not in instance or type(instance["definition_id"]) is str, "Instance definition_id must be a string")
            _require("frozen" not in instance or type(instance["frozen"]) is bool, "Instance frozen flag must be a boolean")
        for call in scope["calls"]:
            _require(type(call.get("instance_id")) is str and type(call.get("inputs")) is dict, "Call instance_id/inputs have invalid storage types")
            _require(all(type(source) is str for source in call["inputs"].values()), "Call input sources must be strings")
        for output in scope["outputs"]:
            _require("source" not in output or type(output["source"]) is str, "Output source must be a string")


def _find(items, identifier, label):
    matches = [item for item in items if item["id"] == identifier]
    _require(len(matches) == 1, "Unknown or ambiguous " + label + ": " + str(identifier))
    return matches[0]


def _identifier(value, label):
    _require(type(value) is str and bool(_ID.fullmatch(value)), label + " must be an identifier")
    return value


def _call_id(value):
    if type(value) is str and value.startswith("call:"):
        value = value[5:]
    return _identifier(value, "Call ID")


def _source(value):
    parse_source(value)
    return value


def _unique(items, base):
    existing = {item["id"] for item in items}
    candidate, index = base, 2
    while candidate in existing:
        candidate = base + "_" + str(index)
        index += 1
    return candidate


def _same_json(first, second):
    # Python equality conflates 8 with 8.0 and True with 1. Parameter storage
    # must retain those distinctions because the schema validates numeric types.
    options = {"sort_keys": True, "allow_nan": False, "separators": (",", ":")}
    return json.dumps(first, **options) == json.dumps(second, **options)


class Document(ComponentInsertion, EditableCommands):
    """A graph plus explicit view state; public edit commands update revision.

    ``load_json`` retains the loaded file's revision. Python import uses the
    current document revision plus one. Both reset navigation only after a
    successful parse/type check. ``scope`` and selection properties expose the
    declared document dictionaries, including incomplete connections.

    Example::

        doc = Document()
        doc.enter("encode_a")
        doc.select("call:project")
        source = doc.probe_source()  # call:encode_a/call:project
        doc.set_frozen("dense", True)
        doc.root()
        new_call = doc.add("Linear")
        doc.set_input(new_call, "x", "call:fuse")
        report = doc.validation()  # shape errors remain explicit

    Mutating commands increment root revision once for a real JSON value
    change. Movement and navigation never increment it. Calls may be passed as
    plain IDs or ``call:<id>`` references to edit/share/enter methods.
    """
    def __init__(self, graph=None):
        candidate = example_graph() if graph is None else graph
        _structure(candidate)
        self.graph = deepcopy(candidate)
        self.trail = []
        self.selected = None
        self._scope_override = _SCOPE_UNSET
        self._init_edits()

    @property
    def scope_id(self):
        if self._scope_override is not _SCOPE_UNSET:
            return self._scope_override
        return self.trail[-1]["id"] if self.trail else None

    def _resolve_call_path(self, call_path):
        _require(type(call_path) in (list, tuple) and len(call_path) <= MAX_CHIP_DEPTH, 'Invalid call path depth')
        scope, trail = self.graph, []
        for call_id in call_path:
            _identifier(call_id, 'Call ID')
            call = _find(scope['calls'], call_id, 'call')
            instance = _find(scope['instances'], call['instance_id'], 'instance')
            _require(instance['kind'] == 'chip', 'Only concrete chip calls form a call path')
            definition = _find(self.graph.get('definitions', []), instance.get('definition_id'), 'chip definition')
            trail.append({'id': definition['id'], 'call_id': call_id})
            scope = definition['graph']
        return trail

    def _restore_cursor(self, override, trail, selected):
        """Restore IDs against the current graph after rollback/import/undo."""
        try:
            resolved = self._resolve_call_path([entry['call_id'] for entry in trail])
            _require(resolved == trail, 'The original navigation target changed')
            if override is not _SCOPE_UNSET and override is not None:
                _find(self.graph.get('definitions', []), override, 'chip definition')
                _require(not trail or trail[-1]['id'] == override, 'The original scope changed')
        except ValueError:
            override, trail, selected = _SCOPE_UNSET, [], None
        self._scope_override, self.trail, self.selected = override, trail, None
        if selected is not None:
            try:
                self.select(selected)
            except ValueError:
                pass

    @property
    def scope(self):
        if self.scope_id is None:
            return self.graph
        return _find(self.graph.get("definitions", []), self.scope_id, "chip definition")["graph"]

    @property
    def selected_call(self):
        if not self.selected or not self.selected.startswith("call:"):
            return None
        return next((call for call in self.scope["calls"] if call["id"] == self.selected[5:]), None)

    @property
    def selected_instance(self):
        call = self.selected_call
        if call is None:
            return None
        return next((instance for instance in self.scope["instances"] if instance["id"] == call["instance_id"]), None)

    def validation(self):
        return validate_graph(self.graph)

    def _node(self, ref):
        match = _REF.fullmatch(ref) if type(ref) is str else None
        _require(match is not None, "Selection must be input:<id>, call:<id>, or output:<id>")
        collection = {"input": "inputs", "call": "calls", "output": "outputs"}[match[1]]
        return _find(self.scope[collection], match[2], match[1])

    def _call(self, identifier):
        return _find(self.scope["calls"], _call_id(identifier), "call")

    def _instance(self, identifier):
        return _find(self.scope["instances"], identifier, "instance")

    def _changed(self):
        self.graph["revision"] += 1

    def select(self, ref):
        if ref is not None:
            self._node(ref)
        self.selected = ref
        return ref

    def enter(self, call_id):
        call = self._call(call_id)
        instance = self._instance(call["instance_id"])
        _require(instance["kind"] == "chip", "Only a concrete chip call can be opened")
        definition = _find(self.graph.get("definitions", []), instance.get("definition_id"), "chip definition")
        # An unbound definition has no root invocation. Its child definitions
        # remain unbound too; a local call ID cannot become a root probe path.
        if self._scope_override is _SCOPE_UNSET or self.scope_id is None or self.trail:
            self.trail.append({"id": definition["id"], "call_id": call["id"]})
        if self._scope_override is not _SCOPE_UNSET:
            self._scope_override = definition['id']
        self.selected = None
        return self.scope

    def back(self):
        if not self.trail:
            return False
        previous = self.trail.pop()
        if self._scope_override is not _SCOPE_UNSET:
            self._scope_override = self.trail[-1]['id'] if self.trail else None
        self.selected = "call:" + previous["call_id"]
        return True

    def root(self):
        self.trail = []
        self.selected = None
        if self._scope_override is not _SCOPE_UNSET:
            self._scope_override = None
        return self.scope

    def probe_source(self, ref=None):
        if self._scope_override is not _SCOPE_UNSET and self.scope_id is not None:
            if not self.trail:
                return None
            resolved = self._resolve_call_path([entry['call_id'] for entry in self.trail])
            _require(resolved == self.trail and resolved[-1]['id'] == self.scope_id, 'Probe needs a real call context')
        ref = self.selected if ref is None else ref
        if ref is None:
            return None
        if ref.startswith('call:') and len(ref.split(':'))==3:
            _,call_id,port=parse_source(ref)
            _require(port in self.output_ports(call_id),'Unknown chip output port')
            source=ref
        else:
            node = self._node(ref)
            source = node.get("source") if ref.startswith("output:") else ref
            if ref.startswith('call:') and len(self.output_ports(ref))>1:return None
        if source is None:
            return None
        _source(source)
        prefix = "".join("call:" + entry["call_id"] + "/" for entry in self.trail)
        return prefix + source

    def semantic_key(self):
        from studio_core.experiment_spec import model_digest
        return model_digest(self.graph)

    def load_json(self, path):
        path = Path(path)
        _require(path.stat().st_size <= 4 * 1024 * 1024, "Graph JSON file exceeds 4 MiB limit")
        def invalid_constant(value):
            raise ValueError("Graph JSON numbers must be finite: " + value)
        try:
            candidate = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid graph JSON: " + str(exc)) from exc
        _structure(candidate)
        self.graph = candidate
        self.root()
        return self.graph

    def save_json(self, path):
        _structure(self.graph)
        text = json.dumps(self.graph, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            replace_with_retry(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    def export_python(self):
        from studio_core.python_io import export_python
        return export_python(self.graph)

    def import_python(self, code):
        from studio_core.python_io import import_python
        candidate = import_python(code)
        _structure(candidate)
        candidate["revision"] = self.graph["revision"] + 1
        self.graph = candidate
        self.root()
        return self.graph

    def set_param(self, instance_id, name, value):
        _identifier(name, "Parameter name")
        _json_values(value)
        instance = self._instance(instance_id)
        params = instance.setdefault("params", {})
        if name not in params or not _same_json(params[name], value):
            params[name] = deepcopy(value)
            self._changed()

    def set_frozen(self, instance_id, frozen):
        _require(type(frozen) is bool, "Frozen must be a boolean")
        instance = self._instance(instance_id)
        if instance.get("frozen", False) != frozen:
            instance["frozen"] = frozen
            self._changed()

    def set_input(self, call_id, port, source):
        _identifier(port, "Input port")
        call = self._call(call_id)
        if source is None:
            if port in call["inputs"]:
                del call["inputs"][port]
                self._changed()
        else:
            _source(source)
            if call["inputs"].get(port) != source:
                call["inputs"][port] = source
                self._changed()

    def set_output(self, output_id, source):
        if type(output_id) is str and output_id.startswith("output:"):
            output_id = output_id[7:]
        output = _find(self.scope["outputs"], output_id, "output")
        if source is None:
            if "source" in output:
                del output["source"]
                self._changed()
            return
        _source(source)
        if output.get("source") != source:
            output["source"] = source
            self._changed()

    def move(self, ref, x, y):
        _require(type(x) in (int, float) and type(y) in (int, float), "Position coordinates must be numbers")
        try:
            finite = math.isfinite(x) and math.isfinite(y)
        except OverflowError:
            finite = False
        _require(finite, "Position coordinates must be finite")
        self._node(ref)["position"] = {"x": x, "y": y}

    def _append(self, kind, definition_id=None):
        scope = self.scope
        if kind == "chip":
            definition = _find(self.graph.get("definitions", []), definition_id, "chip definition")
            params = {parameter["id"]: deepcopy(parameter["default"]) for parameter in definition.get("public_parameters", [])}
            base, title = "chip_" + definition["id"], definition.get("name", definition["id"])
        else:
            _require(kind in PRIMITIVES, "Unsupported primitive kind: " + str(kind))
            params, base, title = operator_defaults(kind), kind.lower(), kind
        instance_id = _unique(scope["instances"], base)
        call_id = _unique(scope["calls"], instance_id + "_call")
        instance = {"id": instance_id, "kind": kind, "name": title, "params": params, "frozen": False}
        if kind == "chip": instance["definition_id"] = definition_id
        call = {"id": call_id, "instance_id": instance_id, "inputs": {}, "position": {"x": 280, "y": 120 + 30 * len(scope["calls"])}}
        scope["instances"].append(instance)
        scope["calls"].append(call)
        self.selected = "call:" + call_id
        return self.selected

    def add(self, kind, definition_id=None):
        ref = self._append(kind, definition_id)
        self._changed()
        return ref

    def share(self, call_id):
        original = self._call(call_id)
        call = deepcopy(original)
        call["id"] = _unique(self.scope["calls"], original["id"] + "_shared")
        position = call.get("position", {"x": 280, "y": 120})
        call["position"] = {"x": position.get("x", 280) + 40, "y": position.get("y", 120) + 40}
        self.scope["calls"].append(call)
        self.selected = "call:" + call["id"]
        self._changed()
        return self.selected

    def remove(self, ref):
        node = self._node(ref)
        kind = ref.split(":", 1)[0]
        collection = {"input": "inputs", "call": "calls", "output": "outputs"}[kind]
        scope = self.scope
        scope[collection].remove(node)
        if kind in ("input", "call"):
            for call in scope["calls"]:
                call["inputs"] = {port: source for port, source in call["inputs"].items() if source != ref and not (kind=='call' and source.startswith(ref+':'))}
            for output in scope["outputs"]:
                if output.get("source") == ref or (kind=='call' and output.get('source','').startswith(ref+':')): output.pop("source")
        if kind == "call" and not any(call["instance_id"] == node["instance_id"] for call in scope["calls"]):
            scope["instances"] = [instance for instance in scope["instances"] if instance["id"] != node["instance_id"]]
        if self.selected == ref: self.selected = None
        self._changed()

    def _new_definition(self, name, definition_id):
        _identifier(definition_id, "Definition ID")
        _require(type(name) is str and bool(name.strip()), "Chip name must be a nonempty string")
        _require(not any(d["id"] == definition_id for d in self.graph.get("definitions", [])), "Chip definition ID already exists: " + definition_id)

    def encapsulate(self, call_ids, name, definition_id, output_call_id=None, outputs=None):
        """Copy root calls into a chip, preserving originals and sharing.

        A valid graph is required to infer named boundary shapes. External
        sources become inputs in first-declared-use order; one selected call
        supplies the chip's tensor output. The new root instance is independent.
        """
        self._new_definition(name, definition_id)
        _require(type(call_ids) in (list, tuple) and bool(call_ids), "Select at least one call to encapsulate")
        selected_ids = [_call_id(identifier) for identifier in call_ids]
        _require(len(set(selected_ids)) == len(selected_ids), "Selected calls must be distinct")
        for identifier in selected_ids: self._call(identifier)
        if outputs is None:
            output_call_id = _call_id(output_call_id)
            outputs={'out':'call:'+output_call_id}
        _require(type(outputs) is dict and bool(outputs),'Expose at least one chip output')
        for identifier,source in outputs.items():
            _identifier(identifier,'Output ID');kind,owner,_=parse_source(source)
            _require(kind=='call' and owner in selected_ids,'Chip output call must be selected')
        report = self.validation()
        _require(report["valid"], "Fix graph validation errors before inferring chip boundaries")
        selected_ids = set(selected_ids)
        current_scope=self.scope
        calls = [deepcopy(call) for call in current_scope["calls"] if call["id"] in selected_ids]
        used_instances = {call["instance_id"] for call in calls}
        instances = [deepcopy(instance) for instance in current_scope["instances"] if instance["id"] in used_instances]
        inputs, boundary = [], {}
        input_sources = {"input:" + port["id"]: port for port in current_scope["inputs"]}
        if self.scope_id:
            # Use the definition's declared default dimensions, not the concrete
            # outer call's overrides, when inferring reusable boundaries.
            from studio_core.schema import prepare_graph
            definitions=deepcopy(self.graph['definitions'])
            temporary={'schema_version':1,'resource_profile':self.graph.get('resource_profile','s1'),
                       'id':'encapsulation','revision':1,**deepcopy(current_scope),'definitions':definitions}
            if temporary['resource_profile']=='s1':temporary.pop('resource_profile')
            if 'resource_batch_bound' in self.graph:
                temporary['resource_batch_bound']=self.graph['resource_batch_bound']
            _,scope_shapes,_=prepare_graph(temporary)
        else:scope_shapes=report['shapes']
        for call in calls:
            for port, source in list(call["inputs"].items()):
                source_kind,source_owner,_=parse_source(source)
                if source_kind=='call' and source_owner in selected_ids:
                    continue
                if source not in boundary:
                    base = "in_" + source.split(":", 1)[1].replace(':','_')
                    identifier = _unique(inputs, base)
                    boundary[source] = identifier
                    original = input_sources.get(source, {})
                    inputs.append({"id": identifier, "name": original.get("name", source),
                                   "shape": list(scope_shapes[source]), "dtype": original.get("dtype", "float32")})
                call["inputs"][port] = "input:" + boundary[source]
        _require(bool(inputs), "A chip must expose at least one boundary input")
        for source in outputs.values():_require(source in scope_shapes,'Unknown or ambiguous exposed output source')
        definition = {"id": definition_id, "name": name, "version": "0.1.0", "output_mode":"named", "public_parameters": [], "bindings": {},
                      "graph": {"inputs": inputs, "instances": instances, "calls": calls, "outputs": [{"id": identifier, "source": source} for identifier,source in outputs.items()]}}
        # Everything above is preparatory; failed inference never changes state.
        self.graph.setdefault("definitions", []).append(definition)
        ref = self._append("chip", definition_id)
        self.selected_call["inputs"] = {identifier: source for source, identifier in boundary.items()}
        self._changed()
        return ref


    def add_template(self, name, id):
        """Create a reusable Linear→ReLU definition and an unconnected call."""
        self._new_definition(name, id)
        definition = {"id": id, "name": name, "version": "0.1.0", "output_mode":"named",
                      "public_parameters": [{"id": "in_features", "name": "Input features", "default": 4},
                                            {"id": "hidden_features", "name": "Hidden features", "default": 8}],
                      "bindings": {"in_features": ["dense.in_features"], "hidden_features": ["dense.out_features"]},
                      "graph": {"inputs": [{"id": "x", "name": "Input", "shape": [None, 4], "dtype": "float32"}],
                                "instances": [{"id": "dense", "kind": "Linear", "params": {"in_features": 4, "out_features": 8}},
                                              {"id": "activation", "kind": "ReLU", "params": {}}],
                                "calls": [{"id": "project", "instance_id": "dense", "inputs": {"x": "input:x"}, "position": {"x": 220, "y": 140}},
                                          {"id": "activate", "instance_id": "activation", "inputs": {"x": "call:project"}, "position": {"x": 500, "y": 140}}],
                                "outputs": [{"id": "out", "source": "call:activate"}]}}
        self.graph.setdefault("definitions", []).append(definition)
        ref = self._append("chip", id)
        self._changed()
        return ref


# Existing public mutation APIs receive the same transactional history as new
# interface commands. Nested operations commit one history entry.
for _method_name in ('load_json','import_python','set_param','set_frozen','set_input','set_output','add','share','remove','encapsulate','add_template'):
    setattr(Document,_method_name,command(getattr(Document,_method_name)))
