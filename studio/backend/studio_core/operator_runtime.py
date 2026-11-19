"""Explicit PyTorch constructors for the finite declared operator set."""
import torch
from torch import nn

from .operator_rules import normalize_operator


class _Add(nn.Module):
    def forward(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Add requires equal runtime shapes; broadcasting is not implicit")
        return a + b


class _Concat(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat((a, b), dim=self.dim)


class _Multiply(nn.Module):
    def forward(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Multiply requires equal runtime shapes; broadcasting is not implicit")
        return a * b


class _Scale(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return self.factor * x


class _OneMinus(nn.Module):
    def forward(self, x):
        return 1 - x


class _BatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        if x.shape[1] != self.num_features:
            raise ValueError("BatchNorm2d input channels do not match num_features")
        uses_batch_statistics = self.training or (self.running_mean is None and self.running_var is None)
        if uses_batch_statistics and x.shape[0] * x.shape[2] * x.shape[3] <= 1:
            raise ValueError("BatchNorm2d batch statistics require more than 1 value per channel (N*H*W > 1)")
        return super().forward(x)


_CONSTRUCTORS = {
    "Linear": nn.Linear, "ReLU": nn.ReLU, "GELU": nn.GELU,
    "Add": _Add, "Concat": _Concat, "LayerNorm": nn.LayerNorm,
    "Dropout": nn.Dropout, "Upsample": nn.Upsample, "Conv2d": nn.Conv2d, "MaxPool2d": nn.MaxPool2d,
    "Conv3d": nn.Conv3d, "MaxPool3d": nn.MaxPool3d, "AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d, "Flatten": nn.Flatten,
    "BatchNorm2d": _BatchNorm2d, "Identity": nn.Identity, "Sigmoid": nn.Sigmoid,
    "Multiply": _Multiply, "Scale": _Scale, "OneMinus": _OneMinus,
}


def make_operator(kind, params):
    """Validate and construct an operator; build_model owns graph allocation budgets.

    Direct calls are low-level constructors, like native nn constructors, and
    have no graph/profile context for cumulative parameter or buffer budgets.
    """
    normalized = normalize_operator(kind, params)
    return _CONSTRUCTORS[kind](**normalized)
