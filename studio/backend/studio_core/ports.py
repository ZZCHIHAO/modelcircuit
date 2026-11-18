"""Scope-local tensor references and declared output ports; no Torch or GUI."""
import re

_REFERENCE = re.compile(r"^(input|call):([A-Za-z_][A-Za-z0-9_]*)(?::([A-Za-z_][A-Za-z0-9_]*))?$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_source(source):
    """Return (kind, node_id, optional_port), rejecting non-source references."""
    match = _REFERENCE.fullmatch(source) if isinstance(source, str) else None
    if match is None or (match[1] == "input" and match[3] is not None):
        raise ValueError("Tensor source must be input:<id>, call:<id>, or call:<id>:<port>")
    return match[1], match[2], match[3]


def _declared_ports(graph):
    outputs = graph.get("outputs") if isinstance(graph, dict) else None
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Chip requires a nonempty outputs list")
    ports = [item.get("id") if isinstance(item, dict) else None for item in outputs]
    if any(not isinstance(port, str) or not _IDENTIFIER.fullmatch(port) for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("Chip output ports must be distinct identifiers")
    return ports


def output_port_ids(instance, definitions=()):
    """List an instance's ports from resolved _graph or a definition dict/list."""
    if not isinstance(instance, dict):
        raise ValueError("Instance must be an object")
    if instance.get("kind") != "chip":
        return ["out"]
    if "_graph" in instance:
        return _declared_ports(instance["_graph"])
    registry = definitions if isinstance(definitions, dict) else {item["id"]: item for item in definitions}
    definition = registry.get(instance.get("definition_id"))
    if not isinstance(definition, dict):
        raise ValueError("Unknown chip definition: " + str(instance.get("definition_id")))
    return _declared_ports(definition.get("graph"))


def output_mode(definition):
    """Infer the legacy tensor/named-dict return contract of a definition."""
    ports = _declared_ports(definition.get("graph"))
    mode = definition.get("output_mode", "tensor" if len(ports) == 1 else "named")
    if mode not in ("tensor", "named") or (mode == "tensor" and len(ports) != 1):
        raise ValueError("output_mode must be named, or tensor with exactly one output")
    return mode


def call_source_refs(call_id, port_ids):
    """Return legacy singleton alias first, followed by qualified port refs."""
    return (["call:" + call_id] if len(port_ids) == 1 else []) + ["call:" + call_id + ":" + port for port in port_ids]
