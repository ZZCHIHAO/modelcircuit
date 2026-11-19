"""Finite, JSON-only descriptors for components that can actually be placed."""
from copy import deepcopy
from dataclasses import dataclass
import math

from .component_types import ComponentBundle
from .operators import operator_defaults, operator_kinds, operator_spec
from .operator_rules import normalize_operator
from .module_templates import make_module
from .schema import _preflight
from .model_registry import list_registered_models, build_registered_model, list_tensor_models, build_tensor_model


@dataclass
class ComponentSpec:
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


# Display metadata only. Supported primitives, parameters and ports come from
# the operator registry; this table cannot register an unsupported constructor.
_DISPLAY = {
    'Conv3d': ('layers', '三维卷积'), 'MaxPool3d': ('shape_pool', '三维最大池化'),
    'AdaptiveAvgPool3d': ('shape_pool', '三维自适应平均池化'),
    'SVC': ('classical', '支持向量分类'), 'SVR': ('classical', '支持向量回归'),
    'Linear': ('layers', '全连接层'), 'Conv2d': ('layers', '二维卷积'),
    'ReLU': ('activation', 'ReLU 激活'), 'GELU': ('activation', 'GELU 激活'),
    'Sigmoid': ('activation', 'Sigmoid 激活'),
    'BatchNorm2d': ('normalization', '二维批归一化'), 'LayerNorm': ('normalization', '层归一化'),
    'Dropout': ('layers', '随机失活'),
    'Add': ('math_fusion', '逐元素相加'), 'Concat': ('math_fusion', '沿轴拼接'),
    'Multiply': ('math_fusion', '逐元素相乘'), 'Scale': ('math_fusion', '固定标量缩放'),
    'OneMinus': ('math_fusion', '一减输入'), 'Identity': ('shape_pool', '恒等映射'),
    'Flatten': ('shape_pool', '展平'), 'MaxPool2d': ('shape_pool', '二维最大池化'),
    'AdaptiveAvgPool2d': ('shape_pool', '二维自适应平均池化'),
}
_MODULES = (
    ('head.mlp', 'MLP 分类头', ('MLPHead',), ('in_features', 'hidden_features', 'num_classes'),
     {'in_features': 4, 'hidden_features': 8, 'num_classes': 2}, ('x',), ('logits',)),
    ('fusion.concat_projection', '拼接投影融合', ('ConcatProjection',), ('a_features', 'b_features', 'out_features'),
     {'a_features': 4, 'b_features': 4, 'out_features': 8}, ('a', 'b'), ('features',)),
    ('fusion.weighted_sum', '加权求和', ('WeightedSum',), ('features',),
     {'features': 4, 'alpha': 0.5}, ('a', 'b'), ('features',)),
    ('fusion.gated', '门控融合', ('GatedFusion',), ('features',),
     {'features': 4}, ('a', 'b'), ('features', 'gate')),
    ('block.residual_basic', '基础残差块', ('ResidualBasicBlock',), ('input_shape', 'in_channels', 'out_channels'),
     {'input_shape': [None, 8, 8, 8], 'in_channels': 8, 'out_channels': 8, 'stride': 1}, ('x',), ('features',)),
)


def _finite_json(value):
    """Reject Python-only containers/types before copying or normalization."""
    _preflight(value)
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if not all(type(key) is str for key in item): raise ValueError('Configuration keys must be strings')
            pending.extend(item.values())
        elif type(item) is list: pending.extend(item)
        elif type(item) is float:
            if not math.isfinite(item): raise ValueError('Configuration numbers must be finite')
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError('Configuration must contain finite JSON values only')


def _capabilities(expandable):
    return {'expandable': expandable, 'forward': True, 'train': True,
            'python_roundtrip': True, 'weights': 'none'}


def list_components(query='', category=None, *, registry=None, include_tensor=False) -> list[ComponentSpec]:
    """Return fresh descriptors in registry then template order."""
    if type(query) is not str or category is not None and type(category) is not str:
        raise ValueError('Component query/category must be strings')
    result = []
    for kind in operator_kinds():
        operator = operator_spec(kind)
        group, label = _DISPLAY.get(kind, ('layers', kind))
        classical = group == 'classical'
        aliases = (kind, 'SVM', '支持向量机', 'Support Vector Machine') if classical else (kind,)
        description = label + (' · ' + kind + ' · CPU 拟合 · 单模型 CSV 实验' if classical else ' · ' + kind)
        result.append(ComponentSpec('primitive.' + kind, group, label, aliases, description,
            tuple(operator.required), _capabilities(False), {**operator.defaults, **operator_defaults(kind)}, tuple(operator.input_ports), ('out',)))
    for identifier, label, aliases, required, defaults, inputs, outputs in _MODULES:
        result.append(ComponentSpec(identifier, 'functional', label, aliases,
            label + ' · 可展开功能芯片；结构参数通过新建配置修改', required,
            _capabilities(True), {**deepcopy(defaults), 'resource_batch_bound': 2}, inputs, outputs))
    for model in list_registered_models(registry=registry):
        result.append(ComponentSpec(model.id, model.category, model.label, model.aliases,
            model.description, model.required_config, model.capabilities, model.config_defaults,
            model.input_ports, model.output_ports))
    if include_tensor:
        for identifier, group, label, _, defaults, outputs in list_tensor_models():
            result.append(ComponentSpec(identifier, group, label, ('Transformer',),
                label + ' · tensor-v1 · 可展开张量运算', (), _capabilities(True), defaults, ('x',), outputs))
    words = query.casefold().split()
    return [item for item in result if (category is None or item.category == category)
            and all(word in ' '.join((item.id, item.label, item.description, *item.aliases)).casefold() for word in words)]


def build_component(component_id, config, *, registry=None) -> ComponentBundle:
    """Build canonical IDs; bare registered primitive kinds remain aliases."""
    if type(component_id) is not str or type(config) is not dict:
        raise ValueError('Component ID and configuration must be a string and object')
    _finite_json(config)
    if component_id in operator_kinds(): component_id = 'primitive.' + component_id
    if any(item[0] == component_id for item in list_tensor_models()):
        return build_tensor_model(component_id, config)
    spec = next((item for item in list_components(registry=registry) if item.id == component_id), None)
    if spec is None: raise ValueError('Unknown component ID: ' + component_id)
    settings = {**spec.config_defaults, **deepcopy(config)}
    if component_id.startswith('primitive.'):
        kind = component_id.removeprefix('primitive.')
        params = normalize_operator(kind, settings)
        return ComponentBundle({'kind': kind, 'params': params}, [], {'operator_set': 'vision-ops-v4'})
    if any(item.id == component_id for item in list_registered_models(registry=registry)):
        return build_registered_model(component_id, config, registry=registry)
    return make_module(component_id, settings)
