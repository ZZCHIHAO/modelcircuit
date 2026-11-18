"""Finite, Torch-independent tensor contracts for schema v2."""
import math
import re

_AXIS = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}')
_ITEM_BYTES = {'float32': 4, 'bool': 1}
SPEC_FIELDS = ('shape', 'dtype', 'axes', 'batch_axis')


class TensorSpecError(ValueError):
    def __init__(self, message, field):
        super().__init__(message)
        self.field = field


def normalize_tensor_spec(spec):
    """Return a detached JSON descriptor; only the leading batch can vary."""
    if not isinstance(spec, dict) or set(spec) != set(SPEC_FIELDS):
        raise TensorSpecError('Tensor spec requires shape, dtype, axes and batch_axis', 'tensor_spec')
    shape, dtype, axes, batch_axis = (spec[key] for key in SPEC_FIELDS)
    if not isinstance(dtype, str) or dtype not in _ITEM_BYTES:
        raise TensorSpecError('Tensor dtype must be float32 or bool', 'dtype')
    if batch_axis is not None and (type(batch_axis) is not int or batch_axis != 0):
        raise TensorSpecError('batch_axis must be 0 or null', 'batch_axis')
    if not isinstance(shape, list) or not 1 <= len(shape) <= 8:
        raise TensorSpecError('Tensor shape must have rank 1..8', 'shape')
    if any(not ((type(dim) is int and dim > 0) or (dim is None and index == batch_axis))
           for index, dim in enumerate(shape)):
        raise TensorSpecError('Shape dimensions must be positive integers; only the batch axis may be null', 'shape')
    if (not isinstance(axes, list) or len(axes) != len(shape)
            or any(not isinstance(axis, str) or not _AXIS.fullmatch(axis) for axis in axes)
            or len(set(axes)) != len(axes)):
        raise TensorSpecError('Axes must be distinct identifiers matching tensor rank', 'axes')
    if (batch_axis == 0 and axes[0] != 'batch') or (batch_axis is None and 'batch' in axes):
        raise TensorSpecError('The batch axis must be named batch and must be explicitly declared', 'axes')
    return dict(shape=list(shape), dtype=dtype, axes=list(axes), batch_axis=batch_axis)


def tensor_nbytes(spec, *, batch_size=None):
    """Count logical storage, not allocator workspace or memory admission."""
    spec = normalize_tensor_spec(spec)
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')
    shape = spec['shape']
    if spec['batch_axis'] == 0:
        if shape[0] is None:
            if batch_size is None:
                raise ValueError('Dynamic batch requires batch_size')
            shape[0] = batch_size
        elif batch_size is not None and shape[0] != batch_size:
            raise ValueError('batch_size differs from the fixed batch dimension')
    return math.prod(shape) * _ITEM_BYTES[spec['dtype']]


def require_executable_profile(graph):
    """Do not let static v2 validation imply support in a v1 consumer."""
    if isinstance(graph, dict) and graph.get('schema_version') == 2:
        raise ValueError('tensor-v1 is not supported by the legacy synthetic runner; select a real classification task')
