"""Trusted native model registrations, independent of Torch and desktop UI.

Extend an immutable ModelRegistry with ``with_model(spec, factory)`` and pass
it to catalog list/build operations. The factory accepts merged finite JSON
configuration and returns a ComponentBundle of supported architecture only.
This is a source-level native extension API, not an arbitrary-Python loader.
Local user extensions remain finite JSON chip packages.
"""
import math
import re
from copy import deepcopy
from dataclasses import dataclass

from .backbone_templates import make_resnet18, make_resnet34, make_resnet50, make_squeezenet, make_vgg
from .component_types import ComponentBundle
from .schema import _preflight, prepare_graph, resource_limits


@dataclass
class ModelSpec:
    id: str
    category: str
    label: str
    aliases: tuple
    description: str
    required_config: tuple
    capabilities: dict
    config_defaults: dict
    input_ports: tuple
    output_ports: tuple
    family: str
    architecture: str
    config_fields: dict
    inputs: dict
    outputs: dict
    initialization: dict
    resources: dict


def _finite_json(value):
    _preflight(value)
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise ValueError('Configuration keys must be strings')
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Configuration numbers must be finite')
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError('Configuration must contain finite JSON values only')


def _port_metadata(metadata, *, symbolic):
    for identifier, port in metadata.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', identifier) or type(port) is not dict:
            raise ValueError('Port metadata must map valid names to objects')
        if port.get('dtype') != 'float32' or type(port.get('shape')) is not list or not 1 <= len(port['shape']) <= 8:
            raise ValueError('Port metadata requires a float32 dtype and bounded nonempty shape array')
        for index, axis in enumerate(port['shape']):
            if type(axis) is int and axis > 0 or index == 0 and axis is None:
                continue
            if symbolic and type(axis) is str and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', axis):
                continue
            raise ValueError('Port shape axes must be positive integers, optional null batch, or named output dimensions')


def _validate_bundle(bundle, spec, settings):
    if not isinstance(bundle, ComponentBundle):
        raise ValueError('Native model factory must return a ComponentBundle')
    _finite_json(bundle.__dict__)
    if type(bundle.entry) is not dict or type(bundle.definitions) is not list or type(bundle.requirements) is not dict or any(type(d) is not dict for d in bundle.definitions):
        raise ValueError('Native model bundle requires entry/requirements objects and a definition array')
    expected_profile = settings.get('resource_profile', spec.resources['resource_profile'])
    if bundle.requirements.get('operator_set') != 'vision-ops-v4' or bundle.requirements.get('resource_profile') != expected_profile:
        raise ValueError('Native model factory must declare the requested operator/resource requirements')
    if set(bundle.requirements) != {'operator_set', 'resource_profile', 'resource_batch_bound'}:
        raise ValueError('Unsupported native model requirements')
    root = next((d.get('graph') for d in bundle.definitions if d.get('id') == bundle.entry.get('definition_id')), None)
    if type(root) is not dict or any(type(root.get(key)) is not list for key in ('inputs', 'outputs')) or any(
            type(port) is not dict or type(port.get('id')) is not str for key in ('inputs', 'outputs') for port in root[key]):
        raise ValueError('Native model entry definition is missing')
    expected_outputs = list(settings['feature_outputs']) if 'feature_outputs' in settings else list(spec.output_ports)
    if 'feature_outputs' in settings and spec.category == 'model':
        expected_outputs.append('logits')
    if tuple(port['id'] for port in root['inputs']) != spec.input_ports or [port['id'] for port in root['outputs']] != expected_outputs:
        raise ValueError('Native model ports do not match its registration descriptor/configuration')
    for port in root['inputs']:
        expected_shape = settings.get('input_shape') if port['id'] == 'x' and 'input_shape' in settings else spec.inputs[port['id']]['shape']
        if port['shape'] != expected_shape:
            raise ValueError('Native model input shape does not match its descriptor/configuration')
    if 'resource_batch_bound' in settings and bundle.requirements['resource_batch_bound'] != settings['resource_batch_bound']:
        raise ValueError('Native model factory changed resource_batch_bound')
    policy = {key: value for key, value in bundle.requirements.items() if key != 'operator_set'}
    _, shapes, _ = prepare_graph({'schema_version': 1, 'id': 'registered_model', 'revision': 1, **policy,
        'inputs': deepcopy(root['inputs']), 'instances': [bundle.entry], 'definitions': bundle.definitions,
        'calls': [{'id': 'entry', 'instance_id': bundle.entry['id'],
                   'inputs': {port['id']: 'input:' + port['id'] for port in root['inputs']}}],
        'outputs': [{'id': port['id'], 'source': 'call:entry:' + port['id']} for port in root['outputs']]})
    for port in root['outputs']:
        descriptor = spec.outputs[port['id']]['shape']
        actual = shapes['call:entry:' + port['id']]
        if len(descriptor) != len(actual) or any(
                (axis in settings and settings[axis] != actual[index]) if type(axis) is str else
                (axis is not None and axis != actual[index]) for index, axis in enumerate(descriptor)):
            raise ValueError('Native model output shape does not match its descriptor/configuration')
    return bundle


class ModelRegistry:
    """An independent registry; extending one does not mutate global entries."""
    def __init__(self, registrations=()):
        records = []
        identifiers = set()
        for spec, factory in registrations:
            if not isinstance(spec, ModelSpec) or type(spec.id) is not str or not re.fullmatch(r'(backbone|model)\.[A-Za-z][A-Za-z0-9_]*', spec.id) or spec.id in identifiers:
                raise ValueError('Model registration requires a unique stable string ID and ModelSpec')
            if spec.category not in ('backbone', 'model') or not callable(factory):
                raise ValueError('Model registration requires a backbone/model category and native factory')
            if spec.id.split('.')[0] != spec.category:
                raise ValueError('Model ID prefix must match its category')
            if any(type(value) is not str or not value for value in (spec.label, spec.description, spec.family, spec.architecture)):
                raise ValueError('Model description fields must be nonempty strings')
            if type(spec.aliases) not in (tuple, list) or any(type(alias) is not str for alias in spec.aliases):
                raise ValueError('Model aliases must be strings')
            for metadata in (spec.config_fields, spec.inputs, spec.outputs, spec.initialization, spec.resources, spec.capabilities):
                if type(metadata) is not dict:
                    raise ValueError('Model metadata must be finite JSON objects')
                _finite_json(metadata)
            expected_capabilities = {'expandable': True, 'forward': True, 'train': True, 'python_roundtrip': True, 'weights': 'none'}
            if spec.capabilities != expected_capabilities or any(type(spec.capabilities[key]) is not bool for key in expected_capabilities if key != 'weights'):
                raise ValueError('Native model capabilities must describe expandable architecture without weights')
            if spec.initialization.get('weights') != 'none':
                raise ValueError('Pretrained weights are unsupported')
            if type(spec.config_defaults) is not dict:
                raise ValueError('Model defaults must be a finite JSON object')
            _finite_json(spec.config_defaults)
            if type(spec.required_config) not in (tuple, list) or any(type(key) is not str for key in spec.required_config) or len(set(spec.required_config)) != len(spec.required_config) or not set(spec.required_config) <= spec.config_defaults.keys():
                raise ValueError('Required configuration must have explicit descriptor defaults')
            if set(spec.config_fields) != set(spec.config_defaults):
                raise ValueError('Configuration field descriptors must match defaults exactly')
            if not spec.input_ports or not spec.output_ports:
                raise ValueError('Model descriptors must name inputs and outputs')
            if type(spec.input_ports) is not tuple or type(spec.output_ports) is not tuple or any(
                    type(port) is not str or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', port) for port in (*spec.input_ports, *spec.output_ports)) or len(set(spec.input_ports)) != len(spec.input_ports) or len(set(spec.output_ports)) != len(spec.output_ports):
                raise ValueError('Model descriptor ports must be unique tuples')
            if set(spec.inputs) != set(spec.input_ports) or not set(spec.output_ports) <= set(spec.outputs):
                raise ValueError('Model input/output metadata must cover all descriptor ports')
            _port_metadata(spec.inputs, symbolic=False)
            _port_metadata(spec.outputs, symbolic=True)
            bound = spec.config_defaults.get('resource_batch_bound')
            profile = spec.config_defaults.get('resource_profile', spec.resources.get('resource_profile'))
            if type(profile) is not str or profile not in ('hardware-v5', 'vision-v4'):
                raise ValueError('Model resource_profile must be hardware-v5 or legacy vision-v4')
            limits = resource_limits(profile, batch_bound=bound)
            expected_resources = {'operator_set': 'vision-ops-v4', 'resource_profile': profile, 'resource_batch_bound': bound,
                                  'max_parameters': limits.max_parameters, 'max_tensor_elements': limits.max_tensor_elements,
                                  'max_execution_elements': limits.max_inferred_elements}
            if spec.resources != expected_resources or type(spec.resources['resource_batch_bound']) is not int or any(
                    type(spec.resources[key]) is not int if expected_resources[key] is not None else spec.resources[key] is not None
                    for key in ('max_parameters', 'max_tensor_elements', 'max_execution_elements')):
                raise ValueError('Resource metadata must match the declared policy and default bound')
            identifiers.add(spec.id)
            records.append((deepcopy(spec), factory))
        self._registrations = tuple(records)

    def with_model(self, spec, factory):
        """Return an extended registry of trusted native graph constructors."""
        return ModelRegistry((*self._registrations, (spec, factory)))

    def list_models(self):
        return [deepcopy(spec) for spec, _ in self._registrations]

    def build(self, model_id, config):
        if type(model_id) is not str or type(config) is not dict:
            raise ValueError('Model ID and configuration must be a string and finite JSON object')
        _finite_json(config)
        record = next(((spec, factory) for spec, factory in self._registrations if spec.id == model_id), None)
        if record is None:
            raise ValueError('Unknown model ID: ' + model_id)
        spec, factory = record
        if not config.keys() <= spec.config_defaults.keys():
            raise ValueError('Unknown model configuration field: ' + ', '.join(sorted(config.keys() - spec.config_defaults.keys())))
        settings = {**deepcopy(spec.config_defaults), **deepcopy(config)}
        if 'resource_profile' in settings and (type(settings['resource_profile']) is not str or settings['resource_profile'] not in ('hardware-v5', 'vision-v4')):
            raise ValueError('resource_profile must be hardware-v5 or legacy vision-v4')
        if 'feature_outputs' in settings and type(settings['feature_outputs']) is not list:
            raise ValueError('feature_outputs must be a finite JSON array')
        if spec.category == 'model' and 'num_classes' in settings and (type(settings['num_classes']) is not int or settings['num_classes'] <= 0):
            raise ValueError('Classifier num_classes must be a positive integer')
        return _validate_bundle(factory(deepcopy(settings)), spec, settings)


def _spec(identifier, family, architecture, label, description, classifier=False):
    expansion = 4 if architecture == 'resnet50' else 1
    defaults = {'input_shape': [None, 3, 224, 224], 'feature_outputs': [] if classifier else ['features'],
                'resource_batch_bound': 2, 'resource_profile': 'hardware-v5'}
    fields = {'input_shape': {'type': 'shape', 'rank': 4, 'channels': [1, 3]},
              'feature_outputs': {'type': 'array', 'items': ['features', 'stage1', 'stage2', 'stage3', 'stage4']
                                  if family == 'resnet' else ['features', 'featuremap']},
              'resource_batch_bound': {'type': 'integer', 'minimum': 1, 'maximum': 64},
              'resource_profile': {'type': 'string', 'items': ['hardware-v5', 'vision-v4']}}
    if classifier:
        defaults['num_classes'] = 1000
        fields['num_classes'] = {'type': 'integer', 'minimum': 1}
    if family == 'squeezenet' and classifier:
        defaults['dropout'] = 0.5
        fields['dropout'] = {'type': 'number', 'minimum': 0, 'maximum': 1}
    output = 'logits' if classifier else 'features'
    output_metadata = {'features': {'shape': [None, 512 * expansion], 'dtype': 'float32'}}
    if family == 'resnet':
        output_metadata.update({'stage' + str(number): {'shape': [None, channels * expansion, 'H_stage' + str(number), 'W_stage' + str(number)], 'dtype': 'float32'}
                               for number, channels in enumerate((64, 128, 256, 512), 1)})
    else:
        output_metadata['featuremap'] = {'shape': [None, 512, 'H_out', 'W_out'], 'dtype': 'float32'}
    if classifier:
        output_metadata['logits'] = {'shape': [None, 'num_classes'], 'dtype': 'float32'}
    return ModelSpec(identifier, 'model' if classifier else 'backbone', label,
        (architecture, family, 'classifier' if classifier else 'backbone'),
        description + ' · 平台默认随机初始化，无预训练权重；1 通道为结构变体', ('input_shape',),
        {'expandable': True, 'forward': True, 'train': True, 'python_roundtrip': True, 'weights': 'none'},
        defaults, ('x',), (output,), family, architecture, fields,
        {'x': {'shape': [None, 3, 224, 224], 'dtype': 'float32', 'layout': 'NCHW'}},
        output_metadata,
        {'weights': 'none', 'policy': 'platform default random initialization; differs from torchvision custom initialization',
         'topology_source': 'torchvision v0.21.0'},
        {'operator_set': 'vision-ops-v4', 'resource_profile': 'hardware-v5', 'resource_batch_bound': 2,
         'max_parameters': None, 'max_tensor_elements': None, 'max_execution_elements': None})


def _native_registrations():
    records = []
    for architecture, factory, layout in (('resnet18', make_resnet18, '[2,2,2,2]'),
                                         ('resnet34', make_resnet34, '[3,4,6,3]'),
                                         ('resnet50', make_resnet50, '[3,4,6,3] Bottleneck v1.5')):
        label = architecture.replace('resnet', 'ResNet')
        for classifier in (False, True):
            identifier = 'model.' + architecture + '_classifier' if classifier else 'backbone.' + architecture
            spec = _spec(identifier, 'resnet', architecture, label + (' 分类器' if classifier else ' 主干'),
                         '标准 ' + label + ' ' + layout + '，stem/stage/block/projection 可展开', classifier)
            records.append((spec, lambda config, factory=factory: factory(**config)))
    for variant in ('vgg11', 'vgg13', 'vgg16', 'vgg19'):
        spec = _spec('backbone.' + variant, 'vgg', variant, variant.upper() + ' 卷积特征 + GAP512',
                     '标准 ' + variant.upper() + ' 非 BN 卷积特征与 GAP512 读出，无原版大 FC 分类头')
        records.append((spec, lambda config, variant=variant: make_vgg(variant, **config)))
    for variant in ('squeezenet1_0', 'squeezenet1_1'):
        for classifier in (False, True):
            identifier = 'model.' + variant + '_classifier' if classifier else 'backbone.' + variant
            label = variant.replace('squeezenet', 'SqueezeNet').replace('_', '.')
            spec = _spec(identifier, 'squeezenet', variant, label + (' 标准分类器' if classifier else ' 主干 + GAP512'),
                         '标准 Fire 卷积特征；' + ('Dropout/Conv1x1/ReLU/GAP 分类头' if classifier else 'GAP512 特征读出'), classifier)
            records.append((spec, lambda config, variant=variant: make_squeezenet(variant, **config)))
    return records


DEFAULT_MODEL_REGISTRY = ModelRegistry(_native_registrations())


def list_registered_models(*, registry=None):
    """Fresh, complete descriptors in stable registration order."""
    selected = DEFAULT_MODEL_REGISTRY if registry is None else registry
    if not isinstance(selected, ModelRegistry):
        raise ValueError('registry must be a ModelRegistry')
    return selected.list_models()


def build_registered_model(model_id, config, *, registry=None):
    selected = DEFAULT_MODEL_REGISTRY if registry is None else registry
    if not isinstance(selected, ModelRegistry):
        raise ValueError('registry must be a ModelRegistry')
    return selected.build(model_id, config)
