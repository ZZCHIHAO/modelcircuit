"""Finite JSON factories for torchvision 0.21 model topologies.

Only architecture is generated. Operators use the platform's default random
initialization; no Torch/GUI imports, pretrained weights, or downloads occur.
VGG exposes standard convolutional features with an explicit GAP512 adapter,
not the original large fully connected VGG classifier.
"""
import math
from copy import deepcopy

from .component_types import ComponentBundle
from .module_templates import _Graph
from .schema import prepare_graph, resource_limits


VGG_STAGES = {
    'vgg11': ((64,), (128,), (256, 256), (512, 512), (512, 512)),
    'vgg13': ((64, 64), (128, 128), (256, 256), (512, 512), (512, 512)),
    'vgg16': ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)),
    'vgg19': ((64, 64), (128, 128), (256, 256, 256, 256), (512, 512, 512, 512), (512, 512, 512, 512)),
}
RESNET_LAYOUTS = {'resnet18': ((2, 2, 2, 2), 1), 'resnet34': ((3, 4, 6, 3), 1),
                  'resnet50': ((3, 4, 6, 3), 4)}
# Official feature-module order; P is MaxPool2d(3,2,ceil_mode=True).
SQUEEZE_FEATURES = {
    'squeezenet1_0': (96, 7, ((96, 16, 64), (128, 16, 64), (128, 32, 128),
        'P', (256, 32, 128), (256, 48, 192), (384, 48, 192), (384, 64, 256), 'P', (512, 64, 256))),
    'squeezenet1_1': (64, 3, ((64, 16, 64), (128, 16, 64), 'P', (128, 32, 128),
        (256, 32, 128), 'P', (256, 48, 192), (384, 48, 192), (384, 64, 256), (512, 64, 256))),
}


def _settings(input_shape, num_classes, feature_outputs, resource_batch_bound, allowed, resource_profile):
    if type(resource_profile) is not str or resource_profile not in ('hardware-v5', 'vision-v4'):
        raise ValueError('resource_profile must be hardware-v5 or legacy vision-v4')
    resource_limits(resource_profile, batch_bound=resource_batch_bound)
    if type(input_shape) is not list or len(input_shape) != 4 or any(
            not (type(axis) is int and axis > 0) and not (index == 0 and axis is None)
            for index, axis in enumerate(input_shape)):
        raise ValueError('input_shape must be [N,C,H,W] with positive integer axes and optional null N')
    if input_shape[1] not in (1, 3):
        raise ValueError('input_shape C must be 1 (explicit channel variant) or 3 (standard)')
    if input_shape[0] is not None and input_shape[0] > resource_batch_bound:
        raise ValueError('input_shape batch exceeds resource_batch_bound')
    if num_classes is not None and (type(num_classes) is not int or num_classes <= 0):
        raise ValueError('num_classes must be a positive integer')
    if type(feature_outputs) not in (list, tuple) or any(type(port) is not str or port not in allowed for port in feature_outputs):
        raise ValueError('Unsupported feature_outputs; allowed: ' + ', '.join(allowed))
    if len(set(feature_outputs)) != len(feature_outputs):
        raise ValueError('feature_outputs must not contain duplicates')
    if not feature_outputs and num_classes is None:
        raise ValueError('At least one output is required')
    return list(input_shape), tuple(feature_outputs)


class _Closure:
    def __init__(self):
        self.definitions = []

    def define(self, identifier, graph):
        self.definitions.append({'id': identifier, 'version': '1.0.0', 'output_mode': 'named',
            'public_parameters': [], 'bindings': {}, 'graph': graph.value})
        return identifier

    @staticmethod
    def chip(graph, identifier, definition_id, source, port='features'):
        graph.node(identifier, 'chip', {'x': source})
        graph.value['instances'][-1]['definition_id'] = definition_id
        return 'call:' + identifier + ':' + port

    def finish(self, definition_id, bound, resource_profile):
        entry = {'id': 'entry', 'kind': 'chip', 'definition_id': definition_id, 'params': {}}
        policy = {'resource_profile': resource_profile, 'resource_batch_bound': bound}
        root = next(d['graph'] for d in self.definitions if d['id'] == definition_id)
        prepare_graph({'schema_version': 1, 'id': 'model_validation', 'revision': 1, **policy,
            'inputs': deepcopy(root['inputs']), 'instances': [entry], 'definitions': self.definitions,
            'calls': [{'id': 'entry', 'instance_id': 'entry',
                       'inputs': {port['id']: 'input:' + port['id'] for port in root['inputs']}}],
            'outputs': [{'id': port['id'], 'source': 'call:entry:' + port['id']} for port in root['outputs']]})
        return ComponentBundle(entry, self.definitions, {'operator_set': 'vision-ops-v4', **policy})


def _conv(graph, identifier, source, incoming, outgoing, kernel=3, stride=1, padding=0, bias=True):
    return graph.node(identifier, 'Conv2d', {'x': source}, {'in_channels': incoming,
        'out_channels': outgoing, 'kernel_size': kernel, 'stride': stride, 'padding': padding, 'bias': bias})


def _relu(graph, identifier, source):
    return graph.node(identifier, 'ReLU', {'x': source})


def _pool(graph, identifier, source, kernel=3, stride=2, padding=0, ceil_mode=False):
    return graph.node(identifier, 'MaxPool2d', {'x': source}, {'kernel_size': kernel,
        'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode})


def _gap(graph, source):
    pooled = graph.node('avgpool', 'AdaptiveAvgPool2d', {'x': source}, {'output_size': [1, 1]})
    return graph.node('flatten', 'Flatten', {'x': pooled}, {'start_dim': 1, 'end_dim': -1})


def _spatial(shape, channels, transform):
    return [shape[0], channels, transform(shape[2]), transform(shape[3])]


def _resnet_block(closure, name, shape, planes, stride, expansion):
    graph = _Graph({'x': shape})
    outgoing = planes * expansion
    # torchvision ResNet v1.5 puts Bottleneck downsampling in 3x3 conv2.
    x = _conv(graph, 'conv1', 'input:x', shape[1], planes,
              kernel=1 if expansion == 4 else 3, stride=1 if expansion == 4 else stride,
              padding=0 if expansion == 4 else 1, bias=False)
    x = graph.node('bn1', 'BatchNorm2d', {'x': x}, {'num_features': planes})
    x = _relu(graph, 'relu1', x)
    x = _conv(graph, 'conv2', x, planes, planes, stride=stride if expansion == 4 else 1, padding=1, bias=False)
    x = graph.node('bn2', 'BatchNorm2d', {'x': x}, {'num_features': planes})
    if expansion == 4:
        x = _relu(graph, 'relu2', x)
        x = _conv(graph, 'conv3', x, planes, outgoing, kernel=1, bias=False)
        x = graph.node('bn3', 'BatchNorm2d', {'x': x}, {'num_features': outgoing})
    shortcut = 'input:x'
    if shape[1] != outgoing or stride != 1:
        projection = _Graph({'x': shape})
        y = _conv(projection, 'conv', 'input:x', shape[1], outgoing, 1, stride, bias=False)
        projection.outputs(features=projection.node('bn', 'BatchNorm2d', {'x': y}, {'num_features': outgoing}))
        projection_id = closure.define(name + 'Projection', projection)
        shortcut = closure.chip(graph, 'projection', projection_id, 'input:x')
    x = graph.node('add', 'Add', {'a': x, 'b': shortcut})
    graph.outputs(features=_relu(graph, 'relu3' if expansion == 4 else 'relu2', x))
    return closure.define(name, graph)


def _make_resnet(variant, input_shape, *, num_classes=None, feature_outputs=('features',),
                 resource_batch_bound=2, resource_profile='hardware-v5'):
    layout, expansion = RESNET_LAYOUTS[variant]
    shape, ports = _settings(input_shape, num_classes, feature_outputs, resource_batch_bound,
                            ('features', 'stage1', 'stage2', 'stage3', 'stage4'), resource_profile)
    prefix = 'ResNet' + variant.removeprefix('resnet')
    closure = _Closure()
    stem = _Graph({'x': shape})
    x = _conv(stem, 'conv', 'input:x', shape[1], 64, 7, 2, 3, False)
    x = stem.node('bn', 'BatchNorm2d', {'x': x}, {'num_features': 64})
    x = _relu(stem, 'relu', x)
    stem.outputs(features=_pool(stem, 'pool', x, padding=1))
    stem_id = closure.define(prefix + 'Stem', stem)
    root = _Graph({'x': shape})
    x = closure.chip(root, 'stem', stem_id, 'input:x')
    shape = _spatial(shape, 64, lambda side: ((side + 1) // 2 + 1) // 2)
    sources = {}
    for number, planes in enumerate((64, 128, 256, 512), 1):
        stage = _Graph({'x': shape})
        y = 'input:x'
        for block_number in range(layout[number - 1]):
            stride = 2 if number > 1 and block_number == 0 else 1
            block_id = _resnet_block(closure, prefix + 'Stage' + str(number) + 'Block' + str(block_number), shape, planes, stride, expansion)
            y = closure.chip(stage, 'block' + str(block_number), block_id, y)
            shape = _spatial(shape, planes * expansion, lambda side: (side + stride - 1) // stride)
        stage.outputs(features=y)
        stage_id = closure.define(prefix + 'Stage' + str(number), stage)
        x = closure.chip(root, 'stage' + str(number), stage_id, x)
        sources['stage' + str(number)] = x
    # The standard classifier consumes the full unshrunk final-stage features.
    if 'features' in ports or num_classes is not None:
        sources['features'] = _gap(root, x)
    outputs = {name: sources[name] for name in ports}
    if num_classes is not None:
        outputs['logits'] = root.node('fc', 'Linear', {'x': sources['features']},
            {'in_features': 512 * expansion, 'out_features': num_classes})
    root.outputs(**outputs)
    return closure.finish(closure.define(prefix + ('Classifier' if num_classes is not None else 'Backbone'), root), resource_batch_bound, resource_profile)


def make_resnet18(input_shape, *, num_classes=None, feature_outputs=('features',), resource_batch_bound=2,
                   resource_profile='hardware-v5'):
    """Standard [2,2,2,2] ResNet18; one-channel inputs are an explicit variant."""
    return _make_resnet('resnet18', input_shape, num_classes=num_classes, feature_outputs=feature_outputs,
                        resource_batch_bound=resource_batch_bound, resource_profile=resource_profile)


def make_resnet34(input_shape, *, num_classes=None, feature_outputs=('features',), resource_batch_bound=2,
                   resource_profile='hardware-v5'):
    """Standard [3,4,6,3] BasicBlock ResNet34 without channel reductions."""
    return _make_resnet('resnet34', input_shape, num_classes=num_classes, feature_outputs=feature_outputs,
                        resource_batch_bound=resource_batch_bound, resource_profile=resource_profile)


def make_resnet50(input_shape, *, num_classes=None, feature_outputs=('features',), resource_batch_bound=2,
                   resource_profile='hardware-v5'):
    """Standard v1.5 [3,4,6,3] Bottleneck ResNet50 with GAP2048 output."""
    return _make_resnet('resnet50', input_shape, num_classes=num_classes, feature_outputs=feature_outputs,
                        resource_batch_bound=resource_batch_bound, resource_profile=resource_profile)


def make_vgg(variant, input_shape, *, feature_outputs=('features',), resource_batch_bound=2,
              resource_profile='hardware-v5'):
    """Standard non-BN VGG convolutional features plus an explicit GAP512 adapter."""
    if type(variant) is not str or variant not in VGG_STAGES:
        raise ValueError('Unsupported VGG feature variant')
    shape, ports = _settings(input_shape, None, feature_outputs, resource_batch_bound, ('features', 'featuremap'), resource_profile)
    closure, root = _Closure(), _Graph({'x': shape})
    x = 'input:x'
    prefix = variant.upper()
    for number, channels in enumerate(VGG_STAGES[variant], 1):
        stage = _Graph({'x': shape})
        y, incoming = 'input:x', shape[1]
        for index, outgoing in enumerate(channels):
            y = _conv(stage, 'conv' + str(index), y, incoming, outgoing, padding=1)
            y = _relu(stage, 'relu' + str(index), y)
            incoming = outgoing
        stage.outputs(features=_pool(stage, 'pool', y, 2, 2))
        stage_id = closure.define(prefix + 'Stage' + str(number), stage)
        x = closure.chip(root, 'stage' + str(number), stage_id, x)
        shape = _spatial(shape, incoming, lambda side: side // 2)
    sources = {'featuremap': x}
    if 'features' in ports:
        sources['features'] = _gap(root, x)
    root.outputs(**{name: sources[name] for name in ports})
    return closure.finish(closure.define(prefix + 'FeaturesGAP512', root), resource_batch_bound, resource_profile)


def _fire(closure, identifier, shape, squeeze, expand):
    graph = _Graph({'x': shape})
    x = _conv(graph, 'squeeze', 'input:x', shape[1], squeeze, 1)
    x = _relu(graph, 'squeeze_activation', x)
    a = _conv(graph, 'expand1x1', x, squeeze, expand, 1)
    a = _relu(graph, 'expand1x1_activation', a)
    b = _conv(graph, 'expand3x3', x, squeeze, expand, 3, padding=1)
    b = _relu(graph, 'expand3x3_activation', b)
    graph.outputs(features=graph.node('concat', 'Concat', {'a': a, 'b': b}, {'dim': 1}))
    return closure.define(identifier, graph)


def make_squeezenet(variant, input_shape, *, num_classes=None, feature_outputs=('features',),
                     dropout=0.5, resource_batch_bound=2, resource_profile='hardware-v5'):
    """Standard Fire backbone and Dropout→Conv→ReLU→GAP→Flatten classifier."""
    if type(variant) is not str or variant not in SQUEEZE_FEATURES:
        raise ValueError('Unsupported SqueezeNet variant')
    shape, ports = _settings(input_shape, num_classes, feature_outputs, resource_batch_bound, ('features', 'featuremap'), resource_profile)
    if type(dropout) not in (int, float) or not 0 <= dropout <= 1 or not math.isfinite(dropout):
        raise ValueError('dropout must be a finite number between 0 and 1, excluding bool')
    if num_classes is None and dropout != 0.5:
        raise ValueError('dropout is configurable only for the SqueezeNet classifier')
    closure = _Closure()
    incoming_shape = list(shape)
    features = _Graph({'x': shape})
    outgoing, kernel, operations = SQUEEZE_FEATURES[variant]
    x = _conv(features, 'conv0', 'input:x', shape[1], outgoing, kernel, 2)
    x = _relu(features, 'relu0', x)
    shape = _spatial(shape, outgoing, lambda side: (side - kernel) // 2 + 1)
    x = _pool(features, 'pool2', x, ceil_mode=True)
    shape = _spatial(shape, outgoing, lambda side: (side - 3 + 1) // 2 + 1)
    prefix = 'SqueezeNet1_0' if variant.endswith('1_0') else 'SqueezeNet1_1'
    for index, operation in enumerate(operations, 3):
        if operation == 'P':
            x = _pool(features, 'pool' + str(index), x, ceil_mode=True)
            shape = _spatial(shape, shape[1], lambda side: (side - 3 + 1) // 2 + 1)
        else:
            incoming, squeeze, expand = operation
            if shape[1] != incoming:
                raise ValueError('Invalid native Fire topology')
            fire = _fire(closure, prefix + 'Fire' + str(index), shape, squeeze, expand)
            x = closure.chip(features, 'fire' + str(index), fire, x)
            shape[1] = 2 * expand
    features.outputs(featuremap=x)
    feature_id = closure.define(prefix + 'FeatureLayers', features)
    root = _Graph({'x': incoming_shape})
    x = closure.chip(root, 'feature_layers', feature_id, 'input:x', 'featuremap')
    sources = {'featuremap': x}
    if 'features' in ports:
        sources['features'] = _gap(root, x)
    outputs = {name: sources[name] for name in ports}
    if num_classes is not None:
        classifier = _Graph({'x': shape})
        y = classifier.node('dropout', 'Dropout', {'x': 'input:x'}, {'p': float(dropout)})
        y = _conv(classifier, 'conv', y, 512, num_classes, 1)
        y = _relu(classifier, 'relu', y)
        classifier.outputs(logits=_gap(classifier, y))
        classifier_id = closure.define(prefix + 'Classifier', classifier)
        outputs['logits'] = closure.chip(root, 'classifier', classifier_id, x, 'logits')
    root.outputs(**outputs)
    return closure.finish(closure.define(prefix + ('Model' if num_classes is not None else 'BackboneGAP512'), root), resource_batch_bound, resource_profile)
