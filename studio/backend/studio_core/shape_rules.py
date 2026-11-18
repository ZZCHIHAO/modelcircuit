"""Pure shape/constructor rules shared by static validation and execution."""
import math

CNN_PARAMETERS = {
    "Conv2d": (("in_channels", "out_channels", "kernel_size"), ("stride", "padding", "dilation", "groups", "bias", "padding_mode")),
    "MaxPool2d": (("kernel_size",), ("stride", "padding", "dilation", "return_indices", "ceil_mode")),
    "AdaptiveAvgPool2d": (("output_size",), ()),
    "Flatten": ((), ("start_dim", "end_dim")),
}
CNN_DEFAULTS = {
    "Conv2d": {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"},
    "MaxPool2d": {"stride": None, "padding": 0, "dilation": 1, "return_indices": False, "ceil_mode": False},
    "AdaptiveAvgPool2d": {}, "Flatten": {"start_dim": 1, "end_dim": -1},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pair(value, name, minimum=1, allow_none=False):
    values = list(value) if isinstance(value, (list, tuple)) else [value, value]
    _require(len(values) == 2 and all((v is None and allow_none) or (type(v) is int and v >= minimum) for v in values),
             name + " must be an integer or pair of integers >= " + str(minimum) + (" (None preserves that axis)" if allow_none else ""))
    return values


def normalize_cnn_params(kind, params):
    """Validate a supported constructor and return an independent JSON mapping."""
    required, optional = CNN_PARAMETERS[kind]
    _require(isinstance(params, dict), "params must be an object")
    _require(set(required) <= params.keys(), kind + " is missing required parameters: " + ", ".join(sorted(set(required) - params.keys())))
    _require(params.keys() <= set(required) | set(optional), kind + " has unsupported parameters")
    result = dict(CNN_DEFAULTS[kind])
    result.update(params)
    if kind == "Conv2d":
        for name in ("in_channels", "out_channels", "groups"):
            _require(type(result[name]) is int and result[name] > 0, name + " must be a positive integer")
        _require(result["in_channels"] % result["groups"] == 0 and result["out_channels"] % result["groups"] == 0,
                 "Conv2d groups must divide in_channels and out_channels")
        _require(type(result["bias"]) is bool, "Conv2d bias must be a boolean")
        _require(result["padding_mode"] == "zeros", "Conv2d supports padding_mode='zeros' only")
    if kind in ("Conv2d", "MaxPool2d"):
        for name in ("kernel_size", "stride", "padding", "dilation"):
            value = result[name]
            if name == "stride" and kind == "MaxPool2d" and value is None:
                continue
            pair = _pair(value, name, 0 if name == "padding" else 1)
            result[name] = pair if isinstance(value, (list, tuple)) else value
        if kind == "MaxPool2d":
            _require(type(result["return_indices"]) is bool and not result["return_indices"], "MaxPool2d return_indices must be False; tuple outputs are unsupported")
            _require(type(result["ceil_mode"]) is bool, "MaxPool2d ceil_mode must be a boolean")
            _require(all(p <= k // 2 for p, k in zip(_pair(result["padding"], "padding", 0), _pair(result["kernel_size"], "kernel_size"))),
                     "MaxPool2d padding must not exceed half the kernel size")
    elif kind == "AdaptiveAvgPool2d":
        value = result["output_size"]
        _require(value is not None, "output_size must be a positive integer or pair")
        pair = _pair(value, "output_size", allow_none=isinstance(value, (list, tuple)))
        result["output_size"] = pair if isinstance(value, (list, tuple)) else value
    elif kind == "Flatten":
        _require(all(type(result[name]) is int for name in ("start_dim", "end_dim")), "Flatten dimensions must be integers")
    return result


def cnn_output_shape(kind, params, shape):
    """Use PyTorch's integer output formulas without allocating tensors."""
    if kind == "Flatten":
        rank = len(shape)
        start, end = params["start_dim"], params["end_dim"]
        _require(-rank <= start < rank and -rank <= end < rank, "Flatten dimensions are outside input rank")
        start, end = start % rank, end % rank
        _require(start <= end, "Flatten start_dim must not exceed end_dim")
        flattened = shape[start:end + 1]
        product = None if None in flattened else math.prod(flattened)
        return list(shape[:start]) + [product] + list(shape[end + 1:])
    _require(len(shape) == 4, kind + " requires rank-four [batch, channels, height, width] input")
    output = list(shape)
    if kind == "AdaptiveAvgPool2d":
        dimensions = _pair(params["output_size"], "output_size", allow_none=True)
        output[-2:] = [original if size is None else size for original, size in zip(shape[-2:], dimensions)]
        return output
    if kind == "Conv2d":
        _require(shape[1] == params["in_channels"], "Conv2d input channels do not match in_channels")
        output[1] = params["out_channels"]
    kernel = _pair(params["kernel_size"], "kernel_size")
    stride = _pair(params["stride"] if params["stride"] is not None else params["kernel_size"], "stride")
    padding = _pair(params["padding"], "padding", 0)
    dilation = _pair(params["dilation"], "dilation")
    for index, (size, k, s, p, d) in enumerate(zip(shape[-2:], kernel, stride, padding, dilation), 2):
        numerator = size + 2 * p - d * (k - 1) - 1
        ceil_mode = kind == "MaxPool2d" and params["ceil_mode"]
        dimension = (numerator + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (dimension - 1) * s >= size + p:
            dimension -= 1
        _require(dimension > 0, kind + " produces an empty spatial output")
        output[index] = dimension
    return output
