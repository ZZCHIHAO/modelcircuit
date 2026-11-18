"""Finite operator declarations; safe to import without PyTorch or a GUI.

Constructor defaults describe supported constructor arguments. Required graph
arguments remain required even when a constructor has its own default (Dropout).
Editor presets are independent, intentionally useful starting configurations.
"""
from copy import deepcopy
from dataclasses import dataclass, field

from .shape_rules import CNN_DEFAULTS, CNN_PARAMETERS


@dataclass(frozen=True)
class OperatorSpec:
    kind: str
    required: tuple
    defaults: dict
    input_ports: tuple
    state_kind: str
    python_constructor: str | None
    editor_defaults: dict = field(default_factory=dict)


_EDITOR_DEFAULTS = {
    "Linear": {"in_features": 4, "out_features": 8}, "ReLU": {},
    "GELU": {}, "Add": {}, "Concat": {"dim": -1},
    "LayerNorm": {"normalized_shape": 8}, "Dropout": {"p": 0.1},
    "Conv2d": {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
    "MaxPool2d": {"kernel_size": 2}, "AdaptiveAvgPool2d": {"output_size": [1, 1]},
    "Flatten": {"start_dim": 1, "end_dim": -1},
    "BatchNorm2d": {"num_features": 8}, "Identity": {}, "Sigmoid": {},
    "Multiply": {}, "Scale": {"factor": 1.0}, "OneMinus": {}, "SVC": {"C": 1.0, "kernel": "rbf"}, "SVR": {"C": 1.0, "kernel": "rbf"},
}


def _spec(kind, required=(), defaults=None, *, ports=("x",), state="none", constructor=None):
    return OperatorSpec(kind, required, defaults or {}, ports, state,
                        constructor, _EDITOR_DEFAULTS[kind])


_SPECS = {
    "Linear": _spec("Linear", ("in_features", "out_features"), state="parameters", constructor="nn.Linear"),
    "ReLU": _spec("ReLU", constructor="nn.ReLU"),
    "GELU": _spec("GELU", constructor="nn.GELU"),
    "Add": _spec("Add", ports=("a", "b"), constructor="Add"),
    "Concat": _spec("Concat", defaults={"dim": -1}, ports=("a", "b"), constructor="Concat"),
    "LayerNorm": _spec("LayerNorm", ("normalized_shape",), state="parameters", constructor="nn.LayerNorm"),
    "Dropout": _spec("Dropout", ("p",), {"p": 0.5}, constructor="nn.Dropout"),
}
for _kind, (_required, _optional) in CNN_PARAMETERS.items():
    _SPECS[_kind] = _spec(_kind, _required, deepcopy(CNN_DEFAULTS[_kind]),
                         state="parameters" if _kind == "Conv2d" else "none", constructor="nn." + _kind)
_SPECS.update({
    "BatchNorm2d": _spec("BatchNorm2d", ("num_features",),
                         {"eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True},
                         state="parameters_and_buffers", constructor="nn.BatchNorm2d"),
    "Identity": _spec("Identity", constructor="nn.Identity"),
    "Sigmoid": _spec("Sigmoid", constructor="nn.Sigmoid"),
    "Multiply": _spec("Multiply", ports=("a", "b")),
    "Scale": _spec("Scale", defaults={"factor": 1.0}),
    "OneMinus": _spec("OneMinus"),
    "SVC": _spec("SVC", defaults={"C": 1.0, "kernel": "rbf"}),
    "SVR": _spec("SVR", defaults={"C": 1.0, "kernel": "rbf"}),
})


def operator_spec(kind):
    """Return an independent declaration, rejecting arbitrary constructor names."""
    if not isinstance(kind, str) or kind not in _SPECS:
        raise ValueError("Unsupported operator kind: " + str(kind))
    return deepcopy(_SPECS[kind])


def operator_kinds():
    return tuple(_SPECS)


def operator_defaults(kind):
    """Return editable presets, never the constructor's omitted arguments."""
    return operator_spec(kind).editor_defaults
