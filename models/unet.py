"""VGG16-UNet backbone used by CFC-Net.

The encoder channel widths are [64, 128, 256, 512], matching the paper.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url


VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def _make_vgg_layers(in_channels: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    channels = in_channels
    for width in VGG16_CFG:
        if width == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend(
                [nn.Conv2d(channels, int(width), kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            )
            channels = int(width)
    return nn.Sequential(*layers)


class VGG16Encoder(nn.Module):
    """VGG16 encoder that exposes five spatial feature levels."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        self.features = _make_vgg_layers(in_channels)
        self._initialize_weights()
        if pretrained:
            state = load_state_dict_from_url(VGG16_URL, map_location="cpu", progress=True)
            if in_channels != 3:
                state.pop("features.0.weight", None)
                state.pop("features.0.bias", None)
            self.load_state_dict(
                {key: value for key, value in state.items() if key.startswith("features.")},
                strict=False,
            )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: Tensor) -> List[Tensor]:
        feature1 = self.features[:4](image)
        feature2 = self.features[4:9](feature1)
        feature3 = self.features[9:16](feature2)
        feature4 = self.features[16:23](feature3)
        feature5 = self.features[23:-1](feature4)
        return [feature1, feature2, feature3, feature4, feature5]


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip: Tensor, deeper: Tensor) -> Tensor:
        return self.block(torch.cat([skip, self.up(deeper)], dim=1))


class VGG16UNet(nn.Module):
    """UNet decoder on top of a VGG16 encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = VGG16Encoder(in_channels=in_channels, pretrained=pretrained)
        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(768, 256)
        self.up2 = UpBlock(384, 128)
        self.up1 = UpBlock(192, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        features = self.encoder(image)
        decoded4 = self.up4(features[3], features[4])
        decoded3 = self.up3(features[2], decoded4)
        decoded2 = self.up2(features[1], decoded3)
        decoded1 = self.up1(features[0], decoded2)
        return self.classifier(decoded1)
