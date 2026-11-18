"""Finite tensor-v1 operators; static rules have no Torch dependency."""

import math
from copy import deepcopy
from functools import lru_cache
from types import MappingProxyType

from .operators import operator_spec
from .operator_rules import _finite, _parameter_count, _positive, _require, infer_operator, normalize_operator
from .tensor_specs import normalize_tensor_spec


TENSOR_KINDS = MappingProxyType({
    'Reshape': ('x',), 'Permute': ('x',), 'AxisRename': ('x',),
    'BroadcastTo': ('x', 'like'), 'MatMul': ('a', 'b'), 'Softmax': ('x',),
    'Select': ('x',), 'ParameterTensor': (), 'CausalMask': (), 'MaskApply': ('scores', 'keep'),
})
_FIELDS = {
    'Reshape': ('shape', 'axes'), 'BroadcastTo': ('shape', 'axes'),
    'Permute': ('dims',), 'AxisRename': ('axes',), 'Softmax': ('dim',),
    'Select': ('dim', 'index'), 'ParameterTensor': ('spec', 'init'),
    'CausalMask': ('length',), 'MatMul': (), 'MaskApply': (),
}
_EXTRAS = {'Linear': ('bias',), 'LayerNorm': ('eps', 'elementwise_affine')}


def tensor_input_ports(kind):
    return TENSOR_KINDS[kind] if kind in TENSOR_KINDS else operator_spec(kind).input_ports


def _without_extras(kind, params):
    return {name: value for name, value in params.items() if name not in _EXTRAS.get(kind, ())}


def _spec(shape, axes, dtype, batch_axis):
    return normalize_tensor_spec(dict(shape=shape, axes=axes, dtype=dtype, batch_axis=batch_axis))


def normalize_tensor_operator(kind, params):
    """Validate constructor data without inserting omitted legacy defaults."""
    _require(isinstance(params, dict), 'params must be an object')
    if kind not in TENSOR_KINDS:
        normalized = normalize_operator(kind, _without_extras(kind, params))
        for name in _EXTRAS.get(kind, ()):
            if name not in params:
                continue
            value = params[name]
            if name == 'eps':
                _require(_finite(value) and value > 0, 'LayerNorm eps must be positive and finite', name)
            else:
                _require(type(value) is bool, f'{kind} {name} must be a boolean', name)
            normalized[name] = value
        return normalized

    _require(params.keys() == set(_FIELDS[kind]), f'{kind} params must be exactly: ' + ', '.join(_FIELDS[kind]))
    if kind in ('Reshape', 'BroadcastTo'):
        shape = params['shape']
        _require(isinstance(shape, list) and 1 <= len(shape) <= 8 and
                 all(_positive(dim) or (index == 0 and dim == 'B') for index, dim in enumerate(shape)),
                 f'{kind} shape requires positive integers and may use B only as its first dimension', 'shape')
        batched = shape[0] == 'B'
        _require(kind != 'BroadcastTo' or batched, 'BroadcastTo requires B as its first dimension', 'shape')
        _spec([1 if dim == 'B' else dim for dim in shape], params['axes'], 'float32', 0 if batched else None)
    elif kind == 'AxisRename':
        axes = params['axes']
        _require(isinstance(axes, list) and 1 <= len(axes) <= 8, 'Axes must be a list of 1..8 labels', 'axes')
        _spec([1] * len(axes), axes, 'float32', 0 if axes[0] == 'batch' else None)
    elif kind == 'Permute':
        dims = params['dims']
        _require(isinstance(dims, list) and 1 <= len(dims) <= 8 and all(type(dim) is int for dim in dims)
                 and sorted(dims) == list(range(len(dims))), 'Permute dims must be a permutation of 0..rank-1', 'dims')
    elif kind in ('Softmax', 'Select'):
        for key in _FIELDS[kind]:
            _require(type(params[key]) is int, f'{kind} {key} must be an integer', key)
    elif kind == 'ParameterTensor':
        spec = normalize_tensor_spec(params['spec'])
        _require(spec['batch_axis'] is None and spec['dtype'] == 'float32',
                 'ParameterTensor requires float32 without batch', 'spec')
        init = params['init']
        _require(isinstance(init, dict) and ((init.keys() == {'kind'} and init['kind'] == 'zeros') or
                 (init.keys() == {'kind', 'mean', 'std'} and init['kind'] == 'normal' and
                  _finite(init['mean']) and _finite(init['std']) and init['std'] >= 0)),
                 'ParameterTensor init must be zeros or normal with finite mean and nonnegative std', 'init')
    elif kind == 'CausalMask':
        _require(_positive(params['length']), 'CausalMask length must be a positive integer', 'length')
    return deepcopy(params)


def tensor_parameter_count(kind, params):
    """Count state per instance, excluding persistent buffers."""
    if kind == 'ParameterTensor':
        return math.prod(params['spec']['shape'])
    if kind == 'Linear' and not params.get('bias', True):
        return params['in_features'] * params['out_features']
    if kind == 'LayerNorm' and not params.get('elementwise_affine', True):
        return 0
    return _parameter_count(kind, params)


def _nonbatch_dim(kind, dim, spec):
    rank = len(spec['shape'])
    _require(-rank <= dim < rank, f'{kind} dim is outside tensor rank', 'dim')
    dim %= rank
    _require(dim != spec['batch_axis'], f'{kind} cannot operate on the batch axis', 'dim')
    return dim


def _float_inputs(kind, values):
    _require(all(spec['dtype'] == 'float32' for spec in values), f'{kind} requires float32 input', 'inputs')


def _infer_legacy(kind, params, inputs):
    template = next(iter(inputs.values()))
    _float_inputs(kind, inputs.values())
    _require(all(spec['axes'] == template['axes'] and spec['batch_axis'] == template['batch_axis']
                 for spec in inputs.values()), f'{kind} inputs must have matching axes and batch placement', 'inputs')
    rank = len(template['shape'])
    if kind in ('Conv2d', 'BatchNorm2d', 'MaxPool2d', 'AdaptiveAvgPool2d'):
        _require(template['batch_axis'] == 0 and rank == 4,
                 f'{kind} requires an explicit batch axis in a rank-four tensor', 'inputs')
    if template['batch_axis'] == 0:
        if kind == 'Linear':
            _require(rank > 1, 'Linear cannot consume the batch axis as features', 'inputs')
        elif kind == 'LayerNorm':
            dimensions = params['normalized_shape']
            normalized_rank = len(dimensions) if isinstance(dimensions, list) else 1
            _require(normalized_rank < rank, 'LayerNorm cannot normalize across the batch axis', 'params')
        elif kind == 'Concat':
            _nonbatch_dim(kind, params.get('dim', -1), template)
        elif kind == 'Flatten':
            _nonbatch_dim(kind, params.get('start_dim', 1), template)
    output, _ = infer_operator(kind, _without_extras(kind, params), {key: spec['shape'] for key, spec in inputs.items()})
    _require(len(output) == rank, f'{kind} changes rank without an explicit output axis contract', 'params')
    return {**template, 'shape': output}


def infer_tensor_operator(kind, params, inputs):
    """Infer a single output TensorSpec from typed operands."""
    params = normalize_tensor_operator(kind, params)
    ports = tensor_input_ports(kind)
    _require(isinstance(inputs, dict) and inputs.keys() == set(ports),
             'Call arguments must be exactly: ' + ', '.join(ports), 'inputs')
    inputs = {name: normalize_tensor_spec(spec) for name, spec in inputs.items()}
    if kind not in TENSOR_KINDS:
        return _infer_legacy(kind, params, inputs)
    if kind == 'ParameterTensor':
        return params['spec']
    if kind == 'CausalMask':
        size = params['length']
        return _spec([1, 1, size, size], ['mask_batch', 'mask_head', 'query', 'key'], 'bool', None)
    if kind == 'MatMul':
        a, b = inputs['a'], inputs['b']
        left, right = a['shape'], b['shape']
        _float_inputs(kind, (a, b))
        _require(len(left) >= 2 and len(right) >= 2, 'MatMul operands must have rank at least two', 'inputs')
        _require(a['batch_axis'] == b['batch_axis'] and (a['batch_axis'] is None or len(left) >= 3),
                 'MatMul must preserve batch in its prefix', 'inputs')
        _require(left[:-2] == right[:-2], 'MatMul prefix dimensions must match; broadcasting is not implicit', 'inputs')
        _require(left[-1] == right[-2], 'MatMul contraction dimensions must match', 'inputs')
        return _spec(left[:-1] + [right[-1]], a['axes'][:-1] + b['axes'][-1:], 'float32', a['batch_axis'])
    if kind == 'MaskApply':
        scores, keep = inputs['scores'], inputs['keep']
        _float_inputs(kind, (scores,))
        _require(keep['dtype'] == 'bool', 'MaskApply keep must be bool', 'inputs')
        _require(all(scores[key] == keep[key] for key in ('shape', 'axes', 'batch_axis')),
                 'MaskApply scores and keep must have the same shape, axes and batch placement', 'inputs')
        _require(len(scores['shape']) - (scores['batch_axis'] is not None) >= 1,
                 'MaskApply requires a nonbatch key axis', 'inputs')
        return scores

    x = inputs['x']
    shape, axes, batch_axis = x['shape'], x['axes'], x['batch_axis']
    if kind == 'Reshape':
        target = params['shape']
        has_batch = batch_axis == 0
        _require((target[0] == 'B') == has_batch, 'Reshape must preserve the batch axis using B', 'shape')
        first_data_dim = int(has_batch)
        _require(math.prod(shape[first_data_dim:]) == math.prod(target[first_data_dim:]),
                 'Reshape must keep the element count of each sample', 'shape')
        output = shape[:1] + target[1:] if has_batch else target
        return _spec(output, params['axes'], x['dtype'], batch_axis)
    if kind == 'Permute':
        dims = params['dims']
        _require(len(dims) == len(shape), 'Permute dims must match input rank', 'dims')
        _require(batch_axis is None or dims[0] == 0, 'Permute cannot move the batch axis', 'dims')
        return _spec([shape[i] for i in dims], [axes[i] for i in dims], x['dtype'], batch_axis)
    if kind == 'AxisRename':
        return _spec(shape, params['axes'], x['dtype'], batch_axis)
    if kind == 'BroadcastTo':
        like = inputs['like']
        _require(like['batch_axis'] == 0, 'BroadcastTo like requires a batch axis', 'inputs')
        target = [like['shape'][0], *params['shape'][1:]]
        _require(len(shape) <= len(target), 'BroadcastTo cannot reduce tensor rank', 'shape')
        _require(batch_axis is None or len(shape) == len(target),
                 'BroadcastTo cannot move the source batch axis', 'shape')
        _require(all(source == 1 or source == dest for source, dest in zip(reversed(shape), reversed(target))),
                 'BroadcastTo source dimensions must be 1 or match the target', 'shape')
        return _spec(target, params['axes'], x['dtype'], 0)
    dim = _nonbatch_dim(kind, params['dim'], x)
    if kind == 'Softmax':
        _float_inputs(kind, (x,))
        return x
    _require(len(shape) > 1, 'Select must preserve positive tensor rank', 'dim')
    _require(-shape[dim] <= params['index'] < shape[dim], 'Select index is out of range', 'index')
    return _spec(shape[:dim] + shape[dim + 1:], axes[:dim] + axes[dim + 1:], x['dtype'], batch_axis)


@lru_cache(maxsize=1)
def _torch_components():
    import torch
    from torch import nn

    class ParameterTensor(nn.Module):
        def __init__(self, spec, init):
            super().__init__()
            value = torch.empty(spec['shape'], dtype=torch.float32)
            if init['kind'] == 'zeros':
                nn.init.zeros_(value)
            else:
                nn.init.normal_(value, mean=init['mean'], std=init['std'])
            self.value = nn.Parameter(value)

        def forward(self):
            return self.value

    class CausalMask(nn.Module):
        def __init__(self, length):
            super().__init__()
            self.register_buffer('keep', torch.ones((1, 1, length, length), dtype=torch.bool).tril_())

        def forward(self):
            return self.keep

    class TensorOperation(nn.Module):
        def __init__(self, kind, params):
            super().__init__()
            self.kind = kind
            self.params = params

        def forward(self, **args):
            kind, params = self.kind, self.params
            if args.keys() != set(TENSOR_KINDS[kind]):
                raise ValueError('Call arguments must be exactly: ' + ', '.join(TENSOR_KINDS[kind]))
            if kind == 'MatMul':
                a, b = args['a'], args['b']
                if a.dtype != torch.float32 or b.dtype != torch.float32:
                    raise ValueError('MatMul requires float32 input')
                if a.ndim < 2 or b.ndim < 2 or a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
                    raise ValueError('MatMul requires matching runtime prefix and contraction dimensions')
                return torch.matmul(a, b)
            if kind == 'MaskApply':
                scores, keep = args['scores'], args['keep']
                if scores.dtype != torch.float32 or keep.dtype != torch.bool or scores.shape != keep.shape:
                    raise ValueError('MaskApply requires same-shape float32 scores and bool keep')
                if not scores.ndim or not scores.numel():
                    raise ValueError('MaskApply requires nonempty scores')
                low, high = torch.aminmax(scores.detach())
                if not bool(torch.isfinite(low) & torch.isfinite(high)):
                    raise ValueError('MaskApply scores must be finite before masking')
                if not bool(keep.any(dim=-1).all()):
                    raise ValueError('MaskApply received an all-masked query row')
                return scores.masked_fill(~keep, -torch.inf)
            x = args['x']
            if kind == 'AxisRename':
                return x
            if kind == 'Reshape':
                return x.reshape(tuple(x.shape[0] if dim == 'B' else dim for dim in params['shape']))
            if kind == 'Permute':
                return x.permute(params['dims'])
            if kind == 'BroadcastTo':
                like = args['like']
                if like.ndim < 1 or x.device != like.device:
                    raise ValueError('BroadcastTo like must provide a batch on the source device')
                target = (like.shape[0], *params['shape'][1:])
                if x.ndim > len(target) or any(a != 1 and a != b for a, b in zip(reversed(x.shape), reversed(target))):
                    raise ValueError('BroadcastTo runtime source dimensions must be 1 or match the target')
                return x.expand(target)
            if kind == 'Softmax':
                if x.dtype != torch.float32:
                    raise ValueError('Softmax requires float32 input')
                return torch.softmax(x, params['dim'])
            return x.select(params['dim'], params['index'])

    return {'ParameterTensor': ParameterTensor, 'CausalMask': CausalMask}, TensorOperation


@lru_cache(maxsize=None)
def _legacy_tensor_constructor(kind):
    from .operator_runtime import _CONSTRUCTORS

    base = _CONSTRUCTORS[kind]
    if tensor_input_ports(kind) != ('x',):
        return base

    class UnaryTensorOperator(base):
        # Inherit the original module rather than nesting it, preserving state keys.
        def forward(self, x):
            return super().forward(x)

    return UnaryTensorOperator


def make_tensor_operator(kind, params):
    """Construct one registered module after graph and resource validation."""
    params = normalize_tensor_operator(kind, params)
    if kind not in TENSOR_KINDS:
        if kind in ('Linear', 'LayerNorm', 'Conv2d', 'BatchNorm2d'):
            from torch import float32
            params['dtype'] = float32
        return _legacy_tensor_constructor(kind)(**params)
    sources, stateless = _torch_components()
    return sources[kind](**params) if kind in sources else stateless(kind, params)
