"""Finite, expandable component graphs. No model objects or weights are created."""
import math
from copy import deepcopy

from .component_types import ComponentBundle
from .schema import prepare_graph, resource_limits


_CONFIG = {
    'head.mlp': (('in_features', 'hidden_features', 'num_classes'), ()),
    'fusion.concat_projection': (('a_features', 'b_features', 'out_features'), ()),
    'fusion.weighted_sum': (('features',), ('alpha',)),
    'fusion.gated': (('features',), ()),
    'block.residual_basic': (('input_shape', 'in_channels', 'out_channels'), ('stride',)),
}
_DEFINITION_IDS = {
    'head.mlp': 'MLPHead', 'fusion.concat_projection': 'ConcatProjection',
    'fusion.weighted_sum': 'WeightedSum', 'fusion.gated': 'GatedFusion',
    'block.residual_basic': 'ResidualBasicBlock',
}


def _config(template_id, config):
    if type(template_id) is not str or template_id not in _CONFIG:
        raise ValueError('Unknown module template ID')
    if type(config) is not dict:
        raise ValueError('Module configuration must be a finite JSON object')
    required, optional = _CONFIG[template_id]
    if not set(required) <= config.keys():
        raise ValueError('Missing module configuration: ' + ', '.join(key for key in required if key not in config))
    if not config.keys() <= set(required) | set(optional) | {'resource_batch_bound'}:
        raise ValueError('Unknown module configuration field')
    for key in required:
        if key != 'input_shape' and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(key + ' must be a positive integer')
    bound = config.get('resource_batch_bound', 2)
    resource_limits('vision-v4', batch_bound=bound)
    result = dict(config)
    result['resource_batch_bound'] = bound
    if template_id == 'fusion.weighted_sum':
        alpha = config.get('alpha', 0.5)
        try:
            finite = type(alpha) in (int, float) and math.isfinite(alpha)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('alpha must be a finite number, excluding bool')
        result['alpha'] = float(alpha)
    if template_id == 'block.residual_basic':
        shape = config['input_shape']
        if type(shape) is not list or len(shape) != 4 or any(
                not (type(axis) is int and axis > 0) and not (index == 0 and axis is None)
                for index, axis in enumerate(shape)):
            raise ValueError('input_shape must be [N,C,H,W] with positive integer axes and optional null N')
        if shape[1] != config['in_channels']:
            raise ValueError('input_shape C must match in_channels')
        if shape[0] is not None and shape[0] > bound:
            raise ValueError('input_shape batch exceeds resource_batch_bound')
        stride = config.get('stride', 1)
        if type(stride) is not int or stride not in (1, 2):
            raise ValueError('stride must be integer 1 or 2')
        result.update(input_shape=list(shape), stride=stride)
    return result


class _Graph:
    """Small local graph assembler; all data belongs to one factory invocation."""
    def __init__(self, inputs):
        self.value = {'inputs': [
            {'id': name, 'shape': list(shape), 'dtype': 'float32',
             'position': {'x': 0, 'y': 80 + 200 * index}}
            for index, (name, shape) in enumerate(inputs.items())],
            'instances': [], 'calls': [], 'outputs': []}

    def node(self, identifier, kind, inputs, params=None, position=None):
        self.value['instances'].append({'id': identifier, 'kind': kind, 'params': dict(params or {})})
        x, y = position if position is not None else (260 * (len(self.value['calls']) + 1), 80)
        self.value['calls'].append({'id': identifier, 'instance_id': identifier,
                                   'inputs': dict(inputs), 'position': {'x': x, 'y': y}})
        return 'call:' + identifier

    def outputs(self, **sources):
        self.value['outputs'] = [{'id': name, 'source': source,
            'position': {'x': 260 * (len(self.value['calls']) + 1), 'y': 80 + 200 * index}}
            for index, (name, source) in enumerate(sources.items())]


def _mlp(config):
    graph = _Graph({'x': [None, config['in_features']]})
    x = graph.node('linear1', 'Linear', {'x': 'input:x'},
                   {'in_features': config['in_features'], 'out_features': config['hidden_features']})
    x = graph.node('relu', 'ReLU', {'x': x})
    x = graph.node('linear2', 'Linear', {'x': x},
                   {'in_features': config['hidden_features'], 'out_features': config['num_classes']})
    graph.outputs(logits=x)
    return graph


def _concat_projection(config):
    graph = _Graph({'a': [None, config['a_features']], 'b': [None, config['b_features']]})
    x = graph.node('concat', 'Concat', {'a': 'input:a', 'b': 'input:b'}, {'dim': 1})
    x = graph.node('linear', 'Linear', {'x': x},
                   {'in_features': config['a_features'] + config['b_features'], 'out_features': config['out_features']})
    graph.outputs(features=graph.node('relu', 'ReLU', {'x': x}))
    return graph


def _weighted_sum(config):
    graph = _Graph({'a': [None, config['features']], 'b': [None, config['features']]})
    negative_a = graph.node('negative_a', 'Scale', {'x': 'input:a'}, {'factor': -1.0})
    delta = graph.node('delta', 'Add', {'a': 'input:b', 'b': negative_a})
    weighted = graph.node('weighted', 'Scale', {'x': delta}, {'factor': config['alpha']})
    graph.outputs(features=graph.node('sum_result', 'Add', {'a': 'input:a', 'b': weighted}))
    return graph


def _gated(config):
    features = config['features']
    graph = _Graph({'a': [None, features], 'b': [None, features]})
    merged = graph.node('concat', 'Concat', {'a': 'input:a', 'b': 'input:b'}, {'dim': 1})
    logits = graph.node('gate_linear', 'Linear', {'x': merged}, {'in_features': 2 * features, 'out_features': features})
    gate = graph.node('sigmoid', 'Sigmoid', {'x': logits})
    inverse = graph.node('inverse', 'OneMinus', {'x': gate})
    a = graph.node('weight_a', 'Multiply', {'a': gate, 'b': 'input:a'}, position=(1300, 80))
    b = graph.node('weight_b', 'Multiply', {'a': inverse, 'b': 'input:b'}, position=(1300, 320))
    graph.outputs(features=graph.node('sum_result', 'Add', {'a': a, 'b': b}, position=(1560, 80)), gate=gate)
    return graph


def _residual_basic(config):
    incoming, outgoing, stride = (config[name] for name in ('in_channels', 'out_channels', 'stride'))
    graph = _Graph({'x': config['input_shape']})
    x = graph.node('conv1', 'Conv2d', {'x': 'input:x'},
        {'in_channels': incoming, 'out_channels': outgoing, 'kernel_size': 3, 'stride': stride, 'padding': 1, 'bias': False})
    x = graph.node('bn1', 'BatchNorm2d', {'x': x}, {'num_features': outgoing})
    x = graph.node('relu1', 'ReLU', {'x': x})
    x = graph.node('conv2', 'Conv2d', {'x': x},
        {'in_channels': outgoing, 'out_channels': outgoing, 'kernel_size': 3, 'padding': 1, 'bias': False})
    x = graph.node('bn2', 'BatchNorm2d', {'x': x}, {'num_features': outgoing})
    if incoming != outgoing or stride != 1:
        shortcut = graph.node('shortcut_conv', 'Conv2d', {'x': 'input:x'},
            {'in_channels': incoming, 'out_channels': outgoing, 'kernel_size': 1, 'stride': stride, 'bias': False},
            position=(260, 340))
        shortcut = graph.node('shortcut_bn', 'BatchNorm2d', {'x': shortcut}, {'num_features': outgoing}, position=(520, 340))
    else:
        shortcut = graph.node('shortcut', 'Identity', {'x': 'input:x'}, position=(520, 340))
    x = graph.node('add', 'Add', {'a': x, 'b': shortcut}, position=(1560, 80))
    graph.outputs(features=graph.node('relu2', 'ReLU', {'x': x}, position=(1820, 80)))
    return graph


_FACTORIES = {'head.mlp': _mlp, 'fusion.concat_projection': _concat_projection,
              'fusion.weighted_sum': _weighted_sum, 'fusion.gated': _gated,
              'block.residual_basic': _residual_basic}


def make_module(template_id, config) -> ComponentBundle:
    """Create an independent JSON closure, validating shape and policy before use.

    Structural settings generate a new definition. WeightedSum exposes only
    alpha, bound directly to the final Scale in a + alpha * (b - a).
    """
    config = _config(template_id, config)
    graph = _FACTORIES[template_id](config).value
    definition = {'id': _DEFINITION_IDS[template_id], 'version': '1.0.0',
                  'output_mode': 'named', 'public_parameters': [], 'bindings': {}, 'graph': graph}
    if template_id == 'fusion.weighted_sum':
        definition['public_parameters'] = [{'id': 'alpha', 'type': 'float', 'default': config['alpha']}]
        definition['bindings'] = {'alpha': ['weighted.factor']}
    entry = {'id': 'entry', 'kind': 'chip', 'definition_id': definition['id'], 'params': {}}
    policy = {'resource_profile': 'vision-v4', 'resource_batch_bound': config['resource_batch_bound']}
    # The existing pure schema is the allocation/resource boundary; no Torch or
    # runtime import is needed to reject an oversized or inconsistent closure.
    prepare_graph({'schema_version': 1, 'id': 'module_validation', 'revision': 1, **policy,
        'inputs': deepcopy(graph['inputs']), 'instances': [entry], 'definitions': [definition],
        'calls': [{'id': 'entry', 'instance_id': 'entry',
                   'inputs': {port['id']: 'input:' + port['id'] for port in graph['inputs']}}],
        'outputs': [{'id': port['id'], 'source': 'call:entry:' + port['id']} for port in graph['outputs']]})
    return ComponentBundle(entry, [definition], {'operator_set': 'vision-ops-v4', **policy})
