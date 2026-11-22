"""Read port contracts without requiring every branch of the canvas to be wired."""
from studio_core.operator_rules import infer_operator
from studio_core.operators import operator_spec
from studio_core.ports import output_port_ids, parse_source
from studio_core.schema import _Budget, _infer, _resolve_scope, graph_resource_limits
from studio_core.tensor_specs import SPEC_FIELDS, normalize_tensor_spec


class PortContracts:
    def __init__(self, document):
        self.typed = document.graph['schema_version'] == 2
        self.limits = graph_resource_limits(document.graph)
        self.inputs = {value['id']: value for value in document.scope['inputs']}
        self.instances = {value['id']: value for value in document.scope['instances']}
        self.calls = {value['id']: value for value in document.scope['calls']}
        self.definitions = {value['id']: value for value in document.graph.get('definitions', [])}
        self.cache = {}

    def declared(self, item):
        if self.typed:
            return normalize_tensor_spec({key: item[key] for key in SPEC_FIELDS if key in item})
        return {'shape': list(item['shape']), 'dtype': item['dtype'], 'axes': None}

    def source(self, ref, active=()):
        kind, identifier, port = parse_source(ref)
        if kind == 'input':
            return self.declared(self.inputs[identifier])
        if identifier in active:
            raise ValueError('连线会形成循环依赖')
        if len(active) >= 64:
            return None
        if (ref, active[:1]) in self.cache:
            return self.cache[ref, active[:1]]
        call = self.calls[identifier]
        instance = self.instances[call['instance_id']]
        ports = output_port_ids(instance, self.definitions)
        if port is None and len(ports) == 1:
            port = ports[0]
        if port not in ports:
            raise ValueError('请选择一个具名输出端口')
        arguments = {name: self.source(value, (*active, identifier)) for name, value in call['inputs'].items()}
        result = self.infer(call, instance, arguments, port)
        self.cache[ref, active[:1]] = result
        return result

    def infer(self, call, instance, arguments, port):
        kind, params = instance['kind'], instance.get('params', {})
        if kind == 'chip':
            child = self.definitions[instance['definition_id']]['graph']
            required = {value['id'] for value in child['inputs']}
            if arguments.keys() != required or any(value is None for value in arguments.values()):
                return None
            inputs = [{**value, 'id': name} if self.typed else
                      {'id': name, 'shape': value['shape'], 'dtype': value['dtype']} for name, value in arguments.items()]
            source = f"call:{call['id']}:{port}"
            wrapper = {'inputs': inputs, 'instances': [instance],
                       'calls': [{**call, 'inputs': {name: 'input:' + name for name in arguments}}],
                       'outputs': [{'id': 'out', 'source': source}]}
            budget = _Budget(self.limits, tensor_contracts=self.typed)
            graph = _resolve_scope(wrapper, self.definitions, budget=budget)
            shapes, _ = _infer(graph, budget)
            if self.typed:
                return graph['_tensor_specs'][source]
            return {'shape': shapes[source], 'dtype': 'float32', 'axes': None}
        if self.typed:
            from studio_core.tensor_ops import tensor_input_ports, infer_tensor_operator
            required = set(tensor_input_ports(kind))
            if arguments.keys() != required or any(value is None for value in arguments.values()):
                return None
            return infer_tensor_operator(kind, params, arguments)
        if arguments.keys() != set(operator_spec(kind).input_ports) or any(value is None for value in arguments.values()):
            return None
        shape, _ = infer_operator(kind, params, {name: value['shape'] for name, value in arguments.items()})
        return {'shape': shape, 'dtype': 'float32', 'axes': None}

    def expected(self, call, port, actual, arguments):
        instance = self.instances[call['instance_id']]
        kind = instance['kind']
        if kind == 'chip':
            child = self.definitions[instance['definition_id']]['graph']
            return self.declared(next(value for value in child['inputs'] if value['id'] == port))
        if kind in ('Add', 'Multiply'):
            return arguments.get('b' if port == 'a' else 'a')
        if actual and kind == 'Linear' and actual['shape']:
            return {**actual, 'dtype': 'float32',
                    'shape': [*actual['shape'][:-1], instance['params']['in_features']]}
        return None


def source_contract(document, source):
    try:
        return PortContracts(document).source(source)
    except (ValueError, KeyError, StopIteration, TypeError):
        return None


def assay_text(contract):
    if contract is None:
        return '待验证 · 连接所需输入后可查看形状'
    shape = '[' + ', '.join('N' if value is None else str(value) for value in contract['shape']) + ']'
    axes = contract.get('axes')
    return contract['dtype'] + ' ' + shape + (' · 轴 ' + ', '.join(axes) if axes else ' · 轴语义未声明')


def assess_link(document, call_id, port, source):
    reader = PortContracts(document)
    actual = expected = None
    try:
        call = reader.calls[call_id]
        instance = reader.instances[call['instance_id']]
        actual = reader.source(source, (call_id,))
        arguments = {}
        for name, other in call['inputs'].items():
            if name != port:
                try:
                    arguments[name] = reader.source(other, (call_id,))
                except (ValueError, KeyError, StopIteration):
                    arguments[name] = None
        arguments[port] = actual
        expected = reader.expected(call, port, actual, arguments)
        if actual and expected:
            for key, label in [('dtype', '数据类型'), ('axes', '轴语义')]:
                if expected.get(key) is not None and actual.get(key) != expected[key]:
                    raise ValueError(f'{label}不匹配，要求 {expected[key]}，收到 {actual.get(key)}')
            current, wanted = actual['shape'], expected['shape']
            if len(current) != len(wanted):
                raise ValueError(f'张量秩不匹配，要求 {len(wanted)} 维，收到 {len(current)} 维')
            for number, (available, required) in enumerate(zip(current, wanted), 1):
                if required is not None and available != required:
                    raise ValueError(f'第 {number} 维不匹配，要求 {required}，收到 {available}')
        if reader.typed:
            from .tensor_panel import check_connection
            check_connection(document, call, port, source)
        output = output_port_ids(instance, reader.definitions)[0]
        inferred = reader.infer(call, instance, arguments, output)
        status = 'compatible' if actual is not None and (expected is not None or inferred is not None) else 'pending'
        message = '端口匹配' if status == 'compatible' else '待验证 · 请补齐上游输入'
    except (ValueError, KeyError, StopIteration, TypeError) as error:
        status, message = 'incompatible', str(error)
    return {'status': status, 'call': call_id, 'port': port, 'source': source, 'actual': actual,
            'expected': expected, 'message': f'{call_id}.{port} · {message}'}
