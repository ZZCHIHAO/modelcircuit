"""Pure parameter and mathematical shape rules for registered operators."""
from copy import deepcopy
import math

from .operators import operator_spec
from .shape_rules import CNN_PARAMETERS, cnn_output_shape, normalize_cnn_params

_MAX_RANK = 8


class OperatorError(ValueError):
    """An optional graph-field location without depending on schema."""
    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field


def _require(condition, message, field=None):
    if not condition:
        raise OperatorError(message, field)


def _positive(value):
    return type(value) is int and value > 0


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def normalize_operator(kind, params):
    """Validate graph parameters without adding omitted defaults to old graphs."""
    spec = operator_spec(kind)
    _require(isinstance(params, dict), "params must be an object")
    required = set(spec.required)
    _require(required <= params.keys(), kind + " is missing required parameters: " + ", ".join(sorted(required - params.keys())))
    _require(params.keys() <= required | spec.defaults.keys(), kind + " has unsupported parameters")
    if kind in CNN_PARAMETERS:
        return deepcopy(normalize_cnn_params(kind, params))
    result = deepcopy(params)
    if kind == "Linear":
        for name in spec.required:
            _require(_positive(result[name]), name + " must be a positive integer", name)
    elif kind == "Upsample":
        size = result["size"]
        _require(isinstance(size, list) and len(size) in (2, 3) and all(_positive(axis) for axis in size),
                 "Upsample size must be [H, W] or [D, H, W] with positive integers", "size")
    elif kind == "Concat":
        _require(type(result.get("dim", -1)) is int, "Concat dim must be an integer", "dim")
    elif kind == "LayerNorm":
        shape = result["normalized_shape"]
        _require(_positive(shape) or (isinstance(shape, list) and 0 < len(shape) <= _MAX_RANK and all(_positive(d) for d in shape)),
                 "normalized_shape must be a positive integer or list of positive integers up to rank " + str(_MAX_RANK), "normalized_shape")
    elif kind == "Dropout":
        _require(_finite(result["p"]) and 0 <= result["p"] <= 1, "Dropout p must be between 0 and 1", "p")
    elif kind == "BatchNorm2d":
        result = {**spec.defaults, **result}
        _require(_positive(result["num_features"]), "num_features must be a positive integer")
        _require(_finite(result["eps"]) and result["eps"] > 0, "BatchNorm2d eps must be positive and finite")
        momentum = result["momentum"]
        _require(momentum is None or (_finite(momentum) and 0 <= momentum <= 1), "BatchNorm2d momentum must be None or finite between 0 and 1")
        for name in ("affine", "track_running_stats"):
            _require(type(result[name]) is bool, "BatchNorm2d " + name + " must be a boolean")
    elif kind in ("SVC", "SVR"):
        result = {**spec.defaults, **result}
        _require(_finite(result["C"]) and result["C"] > 0, "SVM C must be positive and finite", "C")
        _require(result["kernel"] in ("rbf", "linear"), "SVM kernel must be rbf or linear", "kernel")
    elif kind == "Scale":
        result = {**spec.defaults, **result}
        _require(_finite(result["factor"]), "Scale factor must be a finite scalar")
    return result


def _parameter_count(kind, params):
    """Count an instance even when it has no calls; buffers are excluded."""
    if kind == "Linear":
        return params["in_features"] * params["out_features"] + params["out_features"]
    if kind in ("Conv2d", "Conv3d"):
        kernel = params["kernel_size"]
        kernel = kernel if isinstance(kernel, list) else [kernel] * (3 if kind == "Conv3d" else 2)
        count = params["out_channels"] * (params["in_channels"] // params["groups"]) * math.prod(kernel)
        return count + (params["out_channels"] if params["bias"] else 0)
    if kind == "LayerNorm":
        dims = params["normalized_shape"]
        return 2 * math.prod(dims if isinstance(dims, list) else [dims])
    if kind == "BatchNorm2d":
        return 2 * params["num_features"] if params["affine"] else 0
    return 0


def _buffer_shapes(kind, params):
    """Persistent state allocations, separate from parameter/gradient counts."""
    if kind == "BatchNorm2d" and params["track_running_stats"]:
        return ((params["num_features"],), (params["num_features"],), ())
    return ()


def infer_operator(kind, params, input_shapes):
    """Return an independent output shape and the instance's parameter count."""
    params = normalize_operator(kind, params)
    spec = operator_spec(kind)
    _require(isinstance(input_shapes, dict) and input_shapes.keys() == set(spec.input_ports),
             "Call arguments must be exactly: " + ", ".join(sorted(spec.input_ports)))
    for shape in input_shapes.values():
        _require(isinstance(shape, (list, tuple)) and 0 < len(shape) <= _MAX_RANK and
                 all(_positive(axis) or (index == 0 and axis is None) for index, axis in enumerate(shape)),
                 "Input shape dimensions must be positive integers; only batch dimension may be null")
    if kind in ("Add", "Multiply", "Concat"):
        a, b = list(input_shapes["a"]), list(input_shapes["b"])
        if kind != "Concat":
            _require(a == b, kind + " requires equal shapes; broadcasting is not implicit")
            output = a
        else:
            dim = params.get("dim", -1)
            _require(len(a) == len(b) and -len(a) <= dim < len(a), "Concat rank or dim mismatch", "dim")
            dim %= len(a)
            _require(all(x == y for index, (x, y) in enumerate(zip(a, b)) if index != dim), "Concat non-concatenated dimensions must match")
            output = a
            output[dim] = None if a[dim] is None or b[dim] is None else a[dim] + b[dim]
    else:
        output = list(input_shapes["x"])
        if kind in CNN_PARAMETERS:
            output = cnn_output_shape(kind, params, output)
        elif kind == "Upsample":
            _require(len(output) == 2 + len(params["size"]), "Upsample size must match the spatial input rank", "size")
            output = output[:2] + params["size"]
        elif kind in ("SVC", "SVR"):
            _require(len(output) == 2, "SVM requires tabular input [batch, feature]")
            output[-1] = 1
        elif kind == "Linear":
            _require(output[-1] == params["in_features"], "Linear input dimension mismatch", "in_features")
            output[-1] = params["out_features"]
        elif kind == "LayerNorm":
            dims = params["normalized_shape"]
            dims = dims if isinstance(dims, list) else [dims]
            _require(output[-len(dims):] == dims, "LayerNorm normalized_shape does not match input", "normalized_shape")
        elif kind == "BatchNorm2d":
            _require(len(output) == 4, "BatchNorm2d requires rank-four [batch, channels, height, width] input")
            _require(output[1] == params["num_features"], "BatchNorm2d input channels do not match num_features")
    return list(output), _parameter_count(kind, params)
