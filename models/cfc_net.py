"""Three-branch Cross-Frequency Collaborative Network container."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .unet import VGG16UNet


class CFCNet(nn.Module):
    """Low-frequency student, high-frequency student, and comprehensive teacher.

    The public container exposes the model topology and the weighted teacher
    update used by CFC-MT. Training entry points are intentionally not included.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        kwargs = dict(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
        self.low_frequency_student = VGG16UNet(**kwargs)
        self.high_frequency_student = VGG16UNet(**kwargs)
        self.comprehensive_teacher = VGG16UNet(**kwargs)

    def forward(self, raw: Tensor, low: Tensor, high: Tensor) -> Dict[str, Tensor]:
        return {
            "low_student": self.low_frequency_student(low),
            "high_student": self.high_frequency_student(high),
            "teacher": self.comprehensive_teacher(raw),
        }

    @torch.no_grad()
    def update_teacher(
        self,
        student_weight: float = 0.99,
        ema_decay: float = 0.99,
        step: Optional[int] = None,
    ) -> None:
        """Update the comprehensive teacher from the two specialized students."""
        if not 0.0 <= student_weight <= 1.0:
            raise ValueError("student_weight must lie in [0, 1].")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("ema_decay must lie in [0, 1].")
        if step is not None:
            ema_decay = min(1.0 - 1.0 / (step + 1.0), ema_decay)

        for teacher, low, high in zip(
            self.comprehensive_teacher.parameters(),
            self.low_frequency_student.parameters(),
            self.high_frequency_student.parameters(),
        ):
            teacher.mul_(ema_decay)
            teacher.add_(low, alpha=student_weight * (1.0 - ema_decay))
            teacher.add_(high, alpha=(1.0 - student_weight) * (1.0 - ema_decay))
