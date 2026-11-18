"""Independent Torch VGG/SqueezeNet topology oracles for graph-factory tests.

Topology source: torchvision v0.21.0 models/vgg.py and models/squeezenet.py.
These helpers have no platform imports and never derive layers from a graph.
VGG intentionally has a GAP512 feature adapter, not the original VGG FC head.
"""
import torch
from torch import nn

from resnet_reference import ResNetReference


_VGG_STAGES = {
    'vgg11': ((64,), (128,), (256, 256), (512, 512), (512, 512)),
    'vgg13': ((64, 64), (128, 128), (256, 256), (512, 512), (512, 512)),
    'vgg16': ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512),
              (512, 512, 512)),
    'vgg19': ((64, 64), (128, 128), (256, 256, 256, 256), (512, 512, 512, 512),
              (512, 512, 512, 512)),
}


class VGGFeaturesReference(nn.Module):
    def __init__(self, variant='vgg11', input_channels=3):
        super().__init__()
        if variant not in _VGG_STAGES:
            raise ValueError('Unsupported VGG feature variant: ' + str(variant))
        layers = []
        incoming = input_channels
        for stage in _VGG_STAGES[variant]:
            for outgoing in stage:
                layers.append(nn.Conv2d(incoming, outgoing, 3, padding=1))
                layers.append(nn.ReLU())
                incoming = outgoing
            layers.append(nn.MaxPool2d(2, stride=2))
        # The numeric indices match torchvision's non-BN `features` module.
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        featuremap = self.features(x)
        return {'featuremap': featuremap,
                'features': torch.flatten(self.avgpool(featuremap), 1)}


class FireReference(nn.Module):
    def __init__(self, incoming, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Conv2d(incoming, squeeze, 1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze, expand, 1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze, expand, 3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x):
        squeezed = self.squeeze_activation(self.squeeze(x))
        a = self.expand1x1_activation(self.expand1x1(squeezed))
        b = self.expand3x3_activation(self.expand3x3(squeezed))
        return torch.cat((a, b), dim=1)


class SqueezeNetReference(nn.Module):
    def __init__(self, variant='squeezenet1_0', input_channels=3,
                 num_classes=None, dropout=0.5):
        super().__init__()
        if variant == 'squeezenet1_0':
            modules = [
                nn.Conv2d(input_channels, 96, 7, stride=2), nn.ReLU(),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(96, 16, 64), FireReference(128, 16, 64),
                FireReference(128, 32, 128),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(256, 32, 128), FireReference(256, 48, 192),
                FireReference(384, 48, 192), FireReference(384, 64, 256),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(512, 64, 256),
            ]
        elif variant == 'squeezenet1_1':
            modules = [
                nn.Conv2d(input_channels, 64, 3, stride=2), nn.ReLU(),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(64, 16, 64), FireReference(128, 16, 64),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(128, 32, 128), FireReference(256, 32, 128),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                FireReference(256, 48, 192), FireReference(384, 48, 192),
                FireReference(384, 64, 256), FireReference(512, 64, 256),
            ]
        else:
            raise ValueError('Unsupported SqueezeNet variant: ' + str(variant))
        self.features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.classifier is not None and module is self.classifier[1]:
                    nn.init.normal_(module.weight, mean=0., std=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        featuremap = self.features(x)
        result = {'featuremap': featuremap,
                  'features': torch.flatten(self.avgpool(featuremap), 1)}
        if self.classifier is not None:
            result['logits'] = torch.flatten(self.classifier(featuremap), 1)
        return result


def make_reference(model_id, *, input_channels=3, num_classes=None, dropout=0.5):
    """Build an independent oracle using a registered ID or variant name."""
    variant = model_id.split('.')[-1].removesuffix('_classifier')
    if variant in ('resnet18', 'resnet34', 'resnet50'):
        return ResNetReference(variant, input_channels=input_channels, num_classes=num_classes)
    if variant in _VGG_STAGES:
        if num_classes is not None:
            raise ValueError('VGG reference exposes GAP512 features without a classifier')
        return VGGFeaturesReference(variant, input_channels=input_channels)
    if variant in ('squeezenet1_0', 'squeezenet1_1'):
        return SqueezeNetReference(variant, input_channels=input_channels,
                                  num_classes=num_classes, dropout=dropout)
    raise ValueError('Unsupported model reference: ' + str(model_id))
