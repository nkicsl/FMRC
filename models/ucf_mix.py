"""Uncertainty-guided Cross-Frequency Mix (UCF-Mix)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class UCFMixResult:
    raw_a: Tensor
    low_a: Tensor
    high_a: Tensor
    raw_b: Tensor
    low_b: Tensor
    high_b: Tensor
    selected_indices: List[List[int]]


def _patchify(tensor: Tensor, patch_size: int) -> Tensor:
    batch, channels, _, _ = tensor.shape
    patches = F.unfold(tensor, kernel_size=patch_size, stride=patch_size)
    return patches.view(batch, channels, patch_size, patch_size, -1)


def _fold(patches: Tensor, output_size: Tuple[int, int], patch_size: int) -> Tensor:
    batch, channels, _, _, count = patches.shape
    flattened = patches.reshape(batch, channels * patch_size * patch_size, count)
    return F.fold(flattened, output_size=output_size, kernel_size=patch_size, stride=patch_size)


@torch.no_grad()
def uncertainty_guided_cross_frequency_mix(
    raw: Tensor,
    low: Tensor,
    high: Tensor,
    logits: Sequence[Tensor],
    foreground_mask: Tensor,
    patch_size: int = 16,
    top_k: int = 4,
) -> UCFMixResult:
    """Generate the two same-location cyclic cross-frequency views.

    Low-entropy foreground patches are selected independently for every sample.
    The reverse cycle uses the same sample and the same spatial patch index,
    preventing accidental cross-batch or cross-location replacement.
    """
    if raw.shape != low.shape or raw.shape != high.shape:
        raise ValueError("raw, low, and high tensors must have identical shapes.")
    if len(logits) != 3 or any(item.shape != logits[0].shape for item in logits):
        raise ValueError("logits must contain three tensors with identical shapes.")
    if foreground_mask.ndim == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    if foreground_mask.ndim != 4 or foreground_mask.shape[0] != raw.shape[0]:
        raise ValueError("foreground_mask must have shape [B, H, W] or [B, 1, H, W].")

    batch, _, height, width = raw.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Input height and width must be divisible by patch_size.")

    mean_probability = torch.stack([torch.softmax(item, dim=1) for item in logits]).mean(dim=0)
    entropy = -(mean_probability * torch.log(mean_probability.clamp_min(1e-6))).sum(dim=1, keepdim=True)
    entropy_patches = _patchify(entropy, patch_size).mean(dim=(1, 2, 3))
    mask_patches = _patchify((foreground_mask > 0).float(), patch_size)

    order = torch.argsort(entropy_patches, dim=1, descending=False)
    selected: List[List[int]] = []
    for sample in range(batch):
        indices: List[int] = []
        for index in order[sample].tolist():
            if torch.any(mask_patches[sample, :, :, :, index] > 0):
                indices.append(index)
            if len(indices) == top_k:
                break
        selected.append(indices)

    raw_original = _patchify(raw, patch_size)
    low_original = _patchify(low, patch_size)
    high_original = _patchify(high, patch_size)
    raw_a, low_a, high_a = raw_original.clone(), low_original.clone(), high_original.clone()
    raw_b, low_b, high_b = raw_original.clone(), low_original.clone(), high_original.clone()

    for sample, indices in enumerate(selected):
        for index in indices:
            raw_a[sample, :, :, :, index] = low_original[sample, :, :, :, index]
            low_a[sample, :, :, :, index] = high_original[sample, :, :, :, index]
            high_a[sample, :, :, :, index] = raw_original[sample, :, :, :, index]

            raw_b[sample, :, :, :, index] = high_original[sample, :, :, :, index]
            low_b[sample, :, :, :, index] = raw_original[sample, :, :, :, index]
            high_b[sample, :, :, :, index] = low_original[sample, :, :, :, index]

    output_size = (height, width)
    return UCFMixResult(
        raw_a=_fold(raw_a, output_size, patch_size),
        low_a=_fold(low_a, output_size, patch_size),
        high_a=_fold(high_a, output_size, patch_size),
        raw_b=_fold(raw_b, output_size, patch_size),
        low_b=_fold(low_b, output_size, patch_size),
        high_b=_fold(high_b, output_size, patch_size),
        selected_indices=selected,
    )
