"""Joint image-mask transforms used by the released data pipeline."""

from __future__ import annotations

import random
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class JointCompose:
    def __init__(self, transforms: Iterable) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, target: Image.Image):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class JointResize:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = tuple(size)

    def __call__(self, image: Image.Image, target: Image.Image):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        target = F.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        return image, target


class JointRandomHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image, target: Image.Image):
        if random.random() < self.probability:
            return F.hflip(image), F.hflip(target)
        return image, target


class JointRandomVerticalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image, target: Image.Image):
        if random.random() < self.probability:
            return F.vflip(image), F.vflip(target)
        return image, target


class JointRandomRotation:
    def __init__(self, degrees: float = 10.0, probability: float = 0.5) -> None:
        self.degrees = float(degrees)
        self.probability = probability

    def __call__(self, image: Image.Image, target: Image.Image):
        if random.random() >= self.probability:
            return image, target
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        target = F.rotate(target, angle, interpolation=InterpolationMode.NEAREST)
        return image, target


class JointToTensor:
    def __call__(self, image: Image.Image, target: Image.Image):
        image_tensor = F.to_tensor(image)
        target_tensor = torch.as_tensor(np.asarray(target).copy(), dtype=torch.long)
        return image_tensor, target_tensor


class JointNormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        return F.normalize(image, mean=self.mean, std=self.std), target
