"""Wavelet-based low- and high-frequency input construction."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import pywt
import torch
from torch import Tensor, nn


def _minmax(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    return ((array - minimum) / max(maximum - minimum, eps)).astype(np.float32)


@torch.no_grad()
def wavelet_frequency_split(image: Tensor, wavelet: str = "db2") -> Tuple[Tensor, Tensor]:
    """Return full-resolution low- and high-frequency representations.

    Args:
        image: Tensor with shape ``[B, C, H, W]``.
        wavelet: PyWavelets wavelet identifier.

    Each channel is decomposed independently. LL, LH, HL, and HH are min-max
    normalized; the three normalized detail bands are summed and normalized
    again. Both outputs are bilinearly resized to the input resolution.
    """
    if image.ndim != 4:
        raise ValueError("image must have shape [B, C, H, W].")
    device, dtype = image.device, image.dtype
    batch, channels, height, width = image.shape
    source = image.detach().cpu().numpy()
    low_batch, high_batch = [], []

    for batch_index in range(batch):
        low_channels, high_channels = [], []
        for channel_index in range(channels):
            ll, (lh, hl, hh) = pywt.dwt2(source[batch_index, channel_index], wavelet=wavelet)
            low = _minmax(ll)
            high = _minmax(_minmax(lh) + _minmax(hl) + _minmax(hh))
            low_channels.append(cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR))
            high_channels.append(cv2.resize(high, (width, height), interpolation=cv2.INTER_LINEAR))
        low_batch.append(np.stack(low_channels, axis=0))
        high_batch.append(np.stack(high_channels, axis=0))

    low_tensor = torch.from_numpy(np.stack(low_batch, axis=0)).to(device=device, dtype=dtype)
    high_tensor = torch.from_numpy(np.stack(high_batch, axis=0)).to(device=device, dtype=dtype)
    return low_tensor, high_tensor


class WaveletFrequencyDecomposer(nn.Module):
    def __init__(self, wavelet: str = "db2") -> None:
        super().__init__()
        self.wavelet = wavelet

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        return wavelet_frequency_split(image, wavelet=self.wavelet)
