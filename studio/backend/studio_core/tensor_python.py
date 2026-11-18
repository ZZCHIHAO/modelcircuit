"""Canonical standalone helpers for the statically parsed tensor profile."""
import ast
import re

from .tensor_specs import normalize_tensor_spec

PARAMS = {
    'Reshape': ('shape', 'axes'), 'Permute': ('dims',), 'AxisRename': ('axes',),
    'BroadcastTo': ('shape', 'axes'), 'MatMul': (), 'Softmax': ('dim',),
    'Select': ('dim', 'index'), 'ParameterTensor': ('spec', 'init'),
    'CausalMask': ('length',), 'MaskApply': (),
}

HELPERS = '''class Reshape(nn.Module):
    def __init__(self, shape, axes):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(tuple(x.shape[0] if dim == 'B' else dim for dim in self.shape))


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class AxisRename(nn.Module):
    def __init__(self, axes):
        super().__init__()

    def forward(self, x):
        return x


class BroadcastTo(nn.Module):
    def __init__(self, shape, axes):
        super().__init__()
        self.shape = shape

    def forward(self, x, like):
        return x.expand(like.shape[0], *self.shape[1:])


class MatMul(nn.Module):
    def forward(self, a, b):
        return a @ b


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, self.dim)


class Select(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim, self.index = dim, index

    def forward(self, x):
        return x.select(self.dim, self.index)


class ParameterTensor(nn.Module):
    def __init__(self, spec, init):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(spec['shape'], dtype=torch.float32))
        if init['kind'] == 'normal':
            nn.init.normal_(self.value, mean=init['mean'], std=init['std'])

    def forward(self):
        return self.value


class CausalMask(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.register_buffer('keep', torch.ones((1, 1, length, length), dtype=torch.bool).tril())

    def forward(self):
        return self.keep


class MaskApply(nn.Module):
    def forward(self, scores, keep):
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('MaskApply scores must be finite before masking')
        if not bool(keep.any(dim=-1).all()):
            raise ValueError('MaskApply received an all-masked query row')
        return scores.masked_fill(~keep, -torch.inf)
'''

GUARD = '''def _check_execution_policy(*entries):
    batch = None
    for value, dtype, shape, batch_axis in entries:
        if value.dtype != (torch.bool if dtype == 'bool' else torch.float32):
            raise ValueError('Input dtype differs from its tensor contract')
        if value.ndim != len(shape) or any(d is not None and value.shape[i] != d for i, d in enumerate(shape)):
            raise ValueError('Input shape differs from its tensor contract')
        if batch_axis == 0:
            current = value.shape[0]
            if not 1 <= current <= MLFORGE_EXECUTION_POLICY['resource_batch_bound'] or (batch is not None and batch != current):
                raise ValueError('Input batch differs from its tensor contract')
            batch = current
'''


def annotation(port):
    shape = ','.join('?' if dim is None else str(dim) for dim in port['shape'])
    batch = '0' if port['batch_axis'] == 0 else 'none'
    return f"{port['dtype']}[{shape}];axes={','.join(port['axes'])};batch={batch}"


def parse_annotation(node):
    text = node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else ''
    match = re.fullmatch(r'(float32|bool)\[([?0-9,]+)\];axes=([A-Za-z0-9_,]+);batch=(0|none)', text)
    if not match:
        raise ValueError('tensor-v1 requires a literal dtype, shape, axes and batch annotation')
    return normalize_tensor_spec({'dtype': match[1], 'shape': [None if d == '?' else int(d) for d in match[2].split(',')],
                                  'axes': match[3].split(','), 'batch_axis': 0 if match[4] == '0' else None})


def guard_call(inputs):
    args = (f"({p['id']}, {p['dtype']!r}, {p['shape']!r}, {p['batch_axis']!r})" for p in inputs)
    return '_check_execution_policy(' + ', '.join(args) + ')'


def strip_guard(cls):
    method = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == 'forward'), None)
    if not method or not method.body:
        return False
    first = method.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and isinstance(first.value.func, ast.Name) and first.value.func.id == '_check_execution_policy'):
        return False
    params = method.args.args[1:]
    if any(arg.arg == '_check_execution_policy' for arg in params) or any(
            isinstance(node, ast.Name) and node.id == '_check_execution_policy' and isinstance(node.ctx, (ast.Store, ast.Del))
            for node in ast.walk(method)):
        raise ValueError('Local binding shadows the tensor execution guard')
    expected = ast.parse(guard_call([dict(id=arg.arg, **parse_annotation(arg.annotation)) for arg in params])).body[0]
    if ast.dump(first) != ast.dump(expected):
        raise ValueError('Tensor execution guard must match the input annotations')
    del method.body[0]
    return True
