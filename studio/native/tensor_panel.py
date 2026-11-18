"""Tensor input editor and pure checks for connections in a partially wired graph."""
import json

from studio_core.ports import output_port_ids, parse_source
from studio_core.tensor_specs import SPEC_FIELDS, normalize_tensor_spec


def _contract(port):
    return normalize_tensor_spec({key: port[key] for key in SPEC_FIELDS if key in port})


def check_connection(document, call, port, source):
    from studio_core.tensor_ops import infer_tensor_operator, tensor_input_ports
    from studio_core.schema import _Budget, _infer, _resolve_scope, graph_resource_limits

    scope = document.scope
    by_call = {item['id']: item for item in scope['calls']}
    by_instance = {item['id']: item for item in scope['instances']}
    definitions = {item['id']: item for item in document.graph.get('definitions', [])}
    by_input = {item['id']: item for item in scope['inputs']}
    resolved = {}
    budget = _Budget(graph_resource_limits(document.graph), tensor_contracts=True)

    def spec(ref, active):
        if ref in resolved:
            return resolved[ref]
        kind, identifier, output = parse_source(ref)
        if kind == 'input':
            if identifier not in by_input:
                raise ValueError('Unknown tensor source: ' + ref)
            return _contract(by_input[identifier])
        if identifier in active or identifier == call['id']:
            raise ValueError('Tensor connection would create a call dependency cycle')
        if identifier not in by_call:
            raise ValueError('Unknown tensor source: ' + ref)
        if len(active) >= 64:
            return None
        current = by_call[identifier]
        instance = by_instance[current['instance_id']]
        ports = output_port_ids(instance, definitions)
        if output is None and len(ports) != 1 or output is not None and output not in ports:
            raise ValueError('Choose a valid named tensor output: ' + ref)
        inputs = {key: spec(value, active | {identifier}) for key, value in current['inputs'].items()}
        if instance['kind'] == 'chip':
            definition = definitions[instance['definition_id']]['graph']
            if set(inputs) != {item['id'] for item in definition['inputs']} or any(value is None for value in inputs.values()):
                return None
            wrapper = {'inputs': [dict(id=key, **value) for key, value in inputs.items()],
                       'instances': [instance],
                       'calls': [{**current, 'inputs': {key: 'input:' + key for key in inputs}}],
                       'outputs': [{'id': 'out', 'source': f'call:{identifier}:{output or ports[0]}'}]}
            graph = _resolve_scope(wrapper, definitions, budget=budget)
            _infer(graph, budget)
            value = graph['_tensor_specs'][wrapper['outputs'][0]['source']]
            resolved[ref] = value
            return value
        if set(inputs) != set(tensor_input_ports(instance['kind'])) or any(value is None for value in inputs.values()):
            return None
        value = infer_tensor_operator(instance['kind'], instance.get('params', {}), inputs)
        resolved[ref] = value
        return value

    instance = by_instance[call['instance_id']]
    actual = spec(source, set())
    kind = instance['kind']
    if kind == 'chip':
        target = next((item for item in definitions[instance['definition_id']]['graph']['inputs'] if item['id'] == port), None)
        if target is None:
            raise ValueError('Unknown chip input port: ' + port)
        if actual is None:
            return
        expected = _contract(target)
        for key in ('dtype', 'axes', 'batch_axis'):
            if actual[key] != expected[key]:
                raise ValueError(f'Chip input {port} has incompatible {key}')
        if any(wanted is not None and existing != wanted for existing, wanted in zip(actual['shape'], expected['shape'])):
            raise ValueError(f'Chip input {port} has incompatible shape')
        return
    ports = tensor_input_ports(kind)
    if port not in ports:
        raise ValueError('Unknown operator input port: ' + port)
    if actual is None:
        return
    if kind not in ('Reshape', 'Permute', 'AxisRename', 'BroadcastTo', 'Select'):
        expected_dtype = 'bool' if kind == 'MaskApply' and port == 'keep' else 'float32'
        if actual['dtype'] != expected_dtype:
            raise ValueError(f'{kind} input {port} requires dtype {expected_dtype}')
    inputs = {key: spec(value, set()) for key, value in call['inputs'].items() if key != port}
    inputs[port] = actual
    if set(inputs) == set(ports) and all(value is not None for value in inputs.values()):
        infer_tensor_operator(kind, instance.get('params', {}), inputs)


class TensorPanel:
    def __init__(self, ui):
        self.ui = ui

    def fields(self, item, *, parent=0):
        import dearpygui.dearpygui as dpg

        return {'shape': dpg.add_input_text(label='形状', width=82, parent=parent,
                                            default_value=json.dumps(item.get('shape', []))),
                'dtype': dpg.add_combo(('float32', 'bool'), label='类型', width=72, parent=parent,
                                      default_value=item.get('dtype', 'float32')),
                'axes': dpg.add_input_text(label='轴', width=124, parent=parent,
                                           default_value=', '.join(str(axis) for axis in item.get('axes', []))),
                'batch_axis': dpg.add_combo(('0', 'null'), label='批次轴', width=54, parent=parent,
                                            default_value='0' if item.get('batch_axis') == 0 else 'null')}

    @staticmethod
    def read(fields):
        import dearpygui.dearpygui as dpg

        values = {key: dpg.get_value(widget) for key, widget in fields.items()}
        return {'shape': json.loads(values['shape']), 'dtype': values['dtype'],
                'axes': [axis.strip() for axis in values['axes'].split(',')],
                'batch_axis': json.loads(values['batch_axis'])}

    def apply(self, identifier, fields, **_):
        self.ui.document.set_tensor_input(identifier, self.read(fields))
        self.ui._refresh()

    def render(self, parent, item):
        import dearpygui.dearpygui as dpg

        fields = self.fields(item, parent=parent)
        dpg.add_button(label='应用输入契约', parent=parent,
                       callback=self.ui.guarded(self.apply, item['id'], fields))
