from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNormFactory:
    def __init__(self, max_groups: int = 32):
        self.max_groups = max_groups

    def __call__(self, channels: int) -> nn.Module:
        groups = min(self.max_groups, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)


def make_norm_factory(norm: str) -> Callable[[int], nn.Module]:
    if norm == "group":
        return GroupNormFactory()
    if norm == "batch":
        return lambda channels: nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported norm: {norm}")


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int, norm_factory):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_factory(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.norm2 = norm_factory(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                norm_factory(planes),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18Small(nn.Module):
    def __init__(self, num_classes: int, norm: str = "group"):
        super().__init__()
        factory = make_norm_factory(norm)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.norm1 = factory(64)
        self.layer1 = self._make_layer(64, 2, 1, factory)
        self.layer2 = self._make_layer(128, 2, 2, factory)
        self.layer3 = self._make_layer(256, 2, 2, factory)
        self.layer4 = self._make_layer(512, 2, 2, factory)
        self.head = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, planes, blocks, stride, factory):
        layers = []
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, current_stride, factory))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None: nn.init.ones_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel, stride, groups, norm_factory, activation=True):
        padding = (kernel - 1) // 2
        modules = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False), norm_factory(out_ch)]
        if activation:
            modules.append(nn.ReLU6(inplace=True))
        super().__init__(*modules)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, norm_factory):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_ch, hidden, 1, 1, 1, norm_factory))
        layers.extend([
            ConvNormAct(hidden, hidden, 3, stride, hidden, norm_factory),
            ConvNormAct(hidden, out_ch, 1, 1, 1, norm_factory, activation=False),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV2Small(nn.Module):
    def __init__(self, num_classes: int, norm: str = "group", image_size: int = 32, width_mult: float = 1.0):
        super().__init__()
        factory = make_norm_factory(norm)
        first_stride = 1 if image_size <= 32 else 2
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 1 if image_size <= 32 else 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        features = [ConvNormAct(3, input_channel, 3, first_stride, 1, factory)]
        for t, c, n, s in settings:
            out_ch = int(c * width_mult)
            for index in range(n):
                stride = s if index == 0 else 1
                features.append(InvertedResidual(input_channel, out_ch, stride, t, factory))
                input_channel = out_ch
        features.append(ConvNormAct(input_channel, last_channel, 1, 1, 1, factory))
        self.features = nn.Sequential(*features)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(last_channel, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None: nn.init.ones_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.dropout(x))


def build_model(model_name: str, num_classes: int, norm: str, image_size: int = 32) -> nn.Module:
    if model_name == "resnet18":
        return ResNet18Small(num_classes=num_classes, norm=norm)
    if model_name == "mobilenet_v2":
        return MobileNetV2Small(num_classes=num_classes, norm=norm, image_size=image_size)
    raise ValueError(f"Unsupported model_name: {model_name}")
