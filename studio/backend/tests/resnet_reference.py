"""Independent Torch ResNet18/34/50 oracles with torchvision state names.

The topology is pinned to torchvision v0.21.0 models/resnet.py.  This test
helper never reads a platform graph, schema, factory, or exported source.
It uses no torchvision package, pretrained weights, or network operations.
"""
import torch
from torch import nn


class BasicBlockReference(nn.Module):
    expansion = 1

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = None
        if stride != 1 or incoming != outgoing:
            self.downsample = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outgoing),
            )

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + shortcut)


class BottleneckReference(nn.Module):
    """Torchvision ResNet v1.5: the 3x3 conv2 carries the stage stride."""
    expansion = 4

    def __init__(self, incoming, planes, stride=1):
        super().__init__()
        outgoing = planes * self.expansion
        self.conv1 = nn.Conv2d(incoming, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, outgoing, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or incoming != outgoing:
            self.downsample = nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(outgoing))

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return self.relu(self.bn3(self.conv3(y)) + shortcut)


class ResNetReference(nn.Module):
    """Standard ResNet, optionally exposing only its pooled backbone.

    ``num_classes=None`` omits fc.  The result dictionary includes every
    independently calculated stage, so the caller may select named ports.
    Convolution/BN initialization follows torchvision's policy; comparisons
    must explicitly copy named states because platform defaults can differ.
    """

    def __init__(self, variant='resnet18', input_channels=3, num_classes=None):
        super().__init__()
        layouts = {'resnet18': (BasicBlockReference, (2, 2, 2, 2)),
                   'resnet34': (BasicBlockReference, (3, 4, 6, 3)),
                   'resnet50': (BottleneckReference, (3, 4, 6, 3))}
        if variant not in layouts:
            raise ValueError('Unsupported independent ResNet reference')
        block, counts = layouts[variant]
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        incoming = 64
        for number, planes in enumerate((64, 128, 256, 512), 1):
            blocks = []
            for index in range(counts[number - 1]):
                blocks.append(block(incoming, planes, stride=2 if number > 1 and index == 0 else 1))
                incoming = planes * block.expansion
            setattr(self, 'layer' + str(number), nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes) if num_classes is not None else None
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        stem = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        stage1 = self.layer1(stem)
        stage2 = self.layer2(stage1)
        stage3 = self.layer3(stage2)
        stage4 = self.layer4(stage3)
        features = torch.flatten(self.avgpool(stage4), 1)
        result = {'stem': stem, 'stage1': stage1, 'stage2': stage2,
                  'stage3': stage3, 'stage4': stage4, 'features': features}
        if self.fc is not None:
            result['logits'] = self.fc(features)
        return result


def resnet18_reference(input_channels=3, num_classes=None):
    return ResNet18Reference(input_channels=input_channels, num_classes=num_classes)


class ResNet18Reference(ResNetReference):
    def __init__(self, input_channels=3, num_classes=None):
        super().__init__('resnet18', input_channels, num_classes)


class ResNet34Reference(ResNetReference):
    def __init__(self, input_channels=3, num_classes=None):
        super().__init__('resnet34', input_channels, num_classes)


class ResNet50Reference(ResNetReference):
    def __init__(self, input_channels=3, num_classes=None):
        super().__init__('resnet50', input_channels, num_classes)
