"""Expandable tensor-v1 Transformer graphs; factories allocate no Torch state."""
import math
from copy import deepcopy

from .component_types import ComponentBundle, tensor_requirements
from .schema import MAX_ITEMS, prepare_graph, resource_limits


_TOKEN_AXES = ('batch', 'token', 'feature')
_IMAGE_AXES = ('batch', 'channel', 'height', 'width')
_ATTENTION_AXES = ('batch', 'head', 'query', 'key')


def _spec(shape, axes, dtype='float32', batch_axis=0):
    return dict(shape=list(shape), axes=list(axes), dtype=dtype, batch_axis=batch_axis)


def _positive(**settings):
    for name, value in settings.items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')


def _dropout(**settings):
    for name, value in settings.items():
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a number between 0 and 1')


def _check_heads(sequence_length, hidden_dim, num_heads):
    _positive(sequence_length=sequence_length, hidden_dim=hidden_dim, num_heads=num_heads)
    if hidden_dim % num_heads:
        raise ValueError('hidden_dim must be divisible by num_heads')


class _Scope:
    def __init__(self, **inputs):
        self.graph = {'inputs': [dict(id=name, **deepcopy(spec), position={'x': 0, 'y': 80 + 200 * index})
                                 for index, (name, spec) in enumerate(inputs.items())],
                      'instances': [], 'calls': [], 'outputs': []}

    def node(self, name, kind, inputs, params=None, *, position=None, definition=None):
        instance = dict(id=name, kind=kind, params=dict(params or {}))
        if definition is not None:
            instance['definition_id'] = definition['id']
        self.graph['instances'].append(instance)
        x, y = position or (260 * (len(self.graph['calls']) + 1), 80)
        self.graph['calls'].append(dict(id=name, instance_id=name, inputs=dict(inputs), position=dict(x=x, y=y)))
        return 'call:' + name

    def unary(self, name, kind, x, *, position=None, **params):
        return self.node(name, kind, {'x': x}, params, position=position)

    def chip(self, name, definition, x):
        return self.node(name, 'chip', {'x': x}, definition=definition) + ':features'

    def definition(self, name, **outputs):
        right = max(node['position']['x'] for node in self.graph['calls']) + 260
        self.graph['outputs'] = [dict(id=port, source=source, position={'x': right, 'y': 80 + 200 * index})
                                 for index, (port, source) in enumerate(outputs.items())]
        return dict(id=name, version='1.0.0', output_mode='named', public_parameters=[], bindings={}, graph=self.graph)


def _attention_definition(tokens, dim, heads, dropout, *, masked=False, causal=False, return_attention=False):
    width = dim // heads
    try:
        factor = 1 / math.sqrt(width)
    except OverflowError:
        # Avoid converting an arbitrarily large integer directly to float.
        factor = math.exp(-0.5 * math.log(width))
    inputs = {'x': _spec([None, tokens, dim], _TOKEN_AXES)}
    if masked:
        inputs['keep'] = _spec([None, tokens], ('batch', 'key'), 'bool')
    scope = _Scope(**inputs)
    if masked:
        scope.graph['inputs'][1]['position']['y'] = 680

    def project(name, token_axis, row):
        linear = scope.unary(name, 'Linear', 'input:x', position=(260, row), in_features=dim, out_features=dim)
        split = scope.unary(name + '_heads', 'Reshape', linear, position=(520, row),
                            shape=['B', tokens, heads, width], axes=['batch', token_axis, 'head', 'head_feature'])
        return scope.unary(name + '_transpose', 'Permute', split, position=(780, row),
                           dims=[0, 2, 3, 1] if name == 'k_proj' else [0, 2, 1, 3])

    q = project('q_proj', 'query', 80)
    k = project('k_proj', 'key', 280)
    v = project('v_proj', 'key', 480)
    scores = scope.node('scores', 'MatMul', {'a': q, 'b': k}, position=(1040, 80))
    scores = scope.unary('scaled_scores', 'Scale', scores, position=(1300, 80), factor=factor)
    column = 1560
    if masked or causal:
        if masked:
            keep = scope.unary('key_dimensions', 'Reshape', 'input:keep', position=(1040, 680),
                               shape=['B', 1, 1, tokens], axes=list(_ATTENTION_AXES))
        else:
            keep = scope.node('causal_mask', 'CausalMask', {}, {'length': tokens}, position=(1040, 680))
        keep = scope.node('keep_broadcast', 'BroadcastTo', {'x': keep, 'like': scores},
                          {'shape': ['B', heads, tokens, tokens], 'axes': list(_ATTENTION_AXES)}, position=(1300, 680))
        scores = scope.node('masked_scores', 'MaskApply', {'scores': scores, 'keep': keep}, position=(column, 80))
        column += 260
    weights = scope.unary('attention', 'Softmax', scores, position=(column, 80), dim=-1)
    dropped = scope.unary('attention_dropped', 'Dropout', weights, position=(column + 260, 80), p=dropout)
    context = scope.node('context', 'MatMul', {'a': dropped, 'b': v}, position=(column + 520, 80))
    context = scope.unary('token_heads', 'Permute', context, position=(column + 780, 80), dims=[0, 2, 1, 3])
    context = scope.unary('merge_heads', 'Reshape', context, position=(column + 1040, 80),
                          shape=['B', tokens, dim], axes=['batch', 'query', 'feature'])
    context = scope.unary('token_features', 'AxisRename', context, position=(column + 1300, 80), axes=list(_TOKEN_AXES))
    features = scope.unary('out_proj', 'Linear', context, position=(column + 1560, 80), in_features=dim, out_features=dim)
    outputs = {'features': features}
    if return_attention:
        outputs['attention'] = weights
    return scope.definition('TensorAttention', **outputs)


def _encoder_definitions(tokens, dim, heads, mlp_dim, dropout, attention_dropout, depth):
    attention = _attention_definition(tokens, dim, heads, attention_dropout)
    spec = _spec([None, tokens, dim], _TOKEN_AXES)
    norm = dict(normalized_shape=dim, eps=1e-6)
    scope = _Scope(x=spec)
    x = 'input:x'
    features = scope.unary('attention_norm', 'LayerNorm', x, **norm)
    features = scope.chip('self_attention', attention, features)
    features = scope.unary('attention_residual_dropout', 'Dropout', features, p=dropout)
    residual = scope.node('attention_residual', 'Add', {'a': x, 'b': features})
    hidden = scope.unary('mlp_norm', 'LayerNorm', residual, **norm)
    hidden = scope.unary('mlp_up', 'Linear', hidden, in_features=dim, out_features=mlp_dim)
    hidden = scope.unary('mlp_gelu', 'GELU', hidden)
    hidden = scope.unary('mlp_hidden_dropout', 'Dropout', hidden, p=dropout)
    hidden = scope.unary('mlp_down', 'Linear', hidden, in_features=mlp_dim, out_features=dim)
    hidden = scope.unary('mlp_residual_dropout', 'Dropout', hidden, p=dropout)
    out = scope.node('mlp_residual', 'Add', {'a': residual, 'b': hidden})
    block = scope.definition('TensorEncoderBlock', features=out)

    scope = _Scope(x=spec)
    for index in range(depth):
        x = scope.chip(f'block{index}', block, x)
    x = scope.unary('final_norm', 'LayerNorm', x, **norm)
    return [scope.definition('TensorEncoder', features=x), block, attention]


def _bundle(definitions, bound):
    graph = definitions[0]['graph']
    entry = dict(id='entry', kind='chip', definition_id=definitions[0]['id'], params={})
    requirements = tensor_requirements(bound)
    policy = {key: value for key, value in requirements.items() if key != 'operator_set'}
    prepare_graph(dict(id='transformer_validation', revision=1, **policy, inputs=graph['inputs'],
                       instances=[entry], definitions=definitions,
                       calls=[dict(id='entry', instance_id='entry',
                                   inputs={p['id']: 'input:' + p['id'] for p in graph['inputs']})],
                       outputs=[dict(id=p['id'], source='call:entry:' + p['id']) for p in graph['outputs']]))
    return ComponentBundle(entry, definitions, requirements)


def make_attention(*, sequence_length, hidden_dim, num_heads, dropout=0.0, masked=False,
                   resource_batch_bound=2, causal=False, return_attention=False):
    """Create self-attention, optionally with a bool key mask or fixed causal mask."""
    _check_heads(sequence_length, hidden_dim, num_heads)
    _dropout(dropout=dropout)
    if any(type(flag) is not bool for flag in (masked, causal, return_attention)):
        raise ValueError('masked, causal and return_attention must be booleans')
    if masked and causal:
        raise ValueError('External keep and causal masks cannot be combined')
    return _bundle([_attention_definition(sequence_length, hidden_dim, num_heads, dropout,
                                          masked=masked, causal=causal, return_attention=return_attention)],
                   resource_batch_bound)


def _check_encoder(sequence_length, hidden_dim, num_heads, mlp_dim, depth, dropout, attention_dropout, bound):
    _check_heads(sequence_length, hidden_dim, num_heads)
    _positive(mlp_dim=mlp_dim, depth=depth)
    if depth >= MAX_ITEMS:
        raise ValueError('depth exceeds the graph item limit')
    _dropout(dropout=dropout, attention_dropout=attention_dropout)
    resource_limits('hardware-v5', batch_bound=bound)


def make_encoder(*, sequence_length, hidden_dim, num_heads, mlp_dim, dropout=0.0,
                 attention_dropout=0.0, depth=1, resource_batch_bound=2):
    """Create independent pre-norm Encoder blocks followed by final LayerNorm."""
    _check_encoder(sequence_length, hidden_dim, num_heads, mlp_dim, depth, dropout,
                   attention_dropout, resource_batch_bound)
    return _bundle(_encoder_definitions(sequence_length, hidden_dim, num_heads, mlp_dim,
                                        dropout, attention_dropout, depth), resource_batch_bound)


def make_small_vit(*, image_size=32, patch_size=4, channels=3, hidden_dim=64, num_heads=4, mlp_dim=128,
                   depth=2, num_classes=2, resource_batch_bound=2, dropout=0.0, attention_dropout=0.0):
    """Create a random-init image classifier with a visible trainable class token and position."""
    _positive(image_size=image_size, patch_size=patch_size, channels=channels, num_classes=num_classes)
    if image_size % patch_size:
        raise ValueError('image_size must be divisible by patch_size')
    if num_classes < 2:
        raise ValueError('num_classes must be at least 2 for classification')
    patches = (image_size // patch_size) ** 2
    tokens = patches + 1
    _check_encoder(tokens, hidden_dim, num_heads, mlp_dim, depth, dropout, attention_dropout, resource_batch_bound)
    definitions = _encoder_definitions(tokens, hidden_dim, num_heads, mlp_dim, dropout, attention_dropout, depth)
    scope = _Scope(x=_spec([None, channels, image_size, image_size], _IMAGE_AXES))
    image = 'input:x'
    x = scope.unary('patch_embed', 'Conv2d', image, in_channels=channels, out_channels=hidden_dim,
                    kernel_size=patch_size, stride=patch_size, padding=0)
    x = scope.unary('flatten_patches', 'Reshape', x, shape=['B', hidden_dim, patches], axes=['batch', 'feature', 'token'])
    x = scope.unary('patch_tokens', 'Permute', x, dims=[0, 2, 1])

    def parameter(name, length, init, like):
        spec = _spec([1, length, hidden_dim], ('singleton', 'token', 'feature'), batch_axis=None)
        value = scope.node(name, 'ParameterTensor', {}, {'spec': spec, 'init': init})
        return scope.node(name + '_broadcast', 'BroadcastTo', {'x': value, 'like': like},
                          {'shape': ['B', length, hidden_dim], 'axes': list(_TOKEN_AXES)})

    cls = parameter('cls_token', 1, {'kind': 'zeros'}, image)
    x = scope.node('tokens', 'Concat', {'a': cls, 'b': x}, {'dim': 1})
    pos = parameter('pos_token', tokens, {'kind': 'normal', 'mean': 0.0, 'std': 0.02}, x)
    x = scope.node('positioned_tokens', 'Add', {'a': x, 'b': pos})
    x = scope.chip('encoder', definitions[0], x)
    x = scope.unary('class_token', 'Select', x, dim=1, index=0)
    x = scope.unary('classifier', 'Linear', x, in_features=hidden_dim, out_features=num_classes)
    logits = scope.unary('class_logits', 'AxisRename', x, axes=['batch', 'class'])
    return _bundle([scope.definition('SmallVisionTransformer', logits=logits), *definitions], resource_batch_bound)
