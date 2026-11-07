"""2D segmentation datasets and semi-supervised sampling utilities."""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


def build_sample_list(
    image_dir: Union[str, Path],
    output_file: Union[str, Path],
    image_suffix: str = ".jpg",
) -> List[str]:
    """Write a deterministic, extension-free sample list."""
    image_dir = Path(image_dir)
    names = sorted(path.stem for path in image_dir.glob(f"*{image_suffix}") if path.is_file())
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    return names


def split_labeled_unlabeled(
    samples: Sequence[str],
    labeled_fraction: float,
    seed: int = 3522,
) -> Tuple[List[str], List[str]]:
    if not 0.0 < labeled_fraction < 1.0:
        raise ValueError("labeled_fraction must lie strictly between 0 and 1.")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    boundary = int(round(len(shuffled) * labeled_fraction))
    return shuffled[:boundary], shuffled[boundary:]


class SegmentationDataset2D(Dataset):
    """Paired 2D images and masks stored under a split directory.

    Expected layout::

        root/{train,test}/image/<sample>.jpg
        root/{train,test}/label/<sample>.png
        root/{train,test}/{train,test}.txt

    Palette-mode masks are read by their underlying indices, which preserves
    multi-class labels such as ``0, 1, 2, 3``.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        list_file: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
        image_suffix: str = ".jpg",
        label_suffix: str = ".png",
        rgb: bool = True,
        label_map: Optional[Dict[int, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.image_size = tuple(image_size)
        self.transform = transform
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.rgb = rgb
        self.label_map = dict(label_map or {})

        list_path = Path(list_file) if list_file is not None else self.split_dir / f"{split}.txt"
        self.samples = sorted(line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not self.samples:
            raise ValueError(f"No samples found in {list_path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name = self.samples[index]
        image_path = self.split_dir / "image" / f"{name}{self.image_suffix}"
        label_path = self.split_dir / "label" / f"{name}{self.label_suffix}"
        image = Image.open(image_path).convert("RGB" if self.rgb else "L")
        label = Image.open(label_path)
        image = image.resize(self.image_size, resample=Image.Resampling.BICUBIC)
        label = label.resize(self.image_size, resample=Image.Resampling.NEAREST)

        if self.transform is not None:
            image, label = self.transform(image, label)
        else:
            image_array = np.asarray(image, dtype=np.float32)
            if image_array.ndim == 2:
                image_array = image_array[None, ...]
            else:
                image_array = image_array.transpose(2, 0, 1)
            image = torch.from_numpy(image_array / 255.0)
            label = torch.from_numpy(np.asarray(label, dtype=np.int64).copy())

        if self.label_map:
            mapped = label.clone()
            for source, target in self.label_map.items():
                mapped[label == source] = target
            label = mapped
        return image, label, name


def _iterate_once(indices: Sequence[int]) -> np.ndarray:
    return np.random.permutation(indices)


def _iterate_eternally(indices: Sequence[int]) -> Iterator[int]:
    def infinite_shuffles() -> Iterator[np.ndarray]:
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def _grouper(iterable: Iterable[int], size: int):
    arguments = [iter(iterable)] * size
    return zip(*arguments)


class SemiSupervisedBatchSampler(Sampler[List[int]]):
    """Yield each batch with fixed labeled and unlabeled sub-batches."""

    def __init__(
        self,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        batch_size: int,
        unlabeled_batch_size: int,
    ) -> None:
        if not 0 < unlabeled_batch_size < batch_size:
            raise ValueError("unlabeled_batch_size must lie in (0, batch_size).")
        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.batch_size = batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled_batch_size = batch_size - unlabeled_batch_size

    def __iter__(self):
        labeled = _iterate_once(self.labeled_indices)
        unlabeled = _iterate_eternally(self.unlabeled_indices)
        return (
            list(labeled_batch + unlabeled_batch)
            for labeled_batch, unlabeled_batch in zip(
                _grouper(labeled, self.labeled_batch_size),
                _grouper(unlabeled, self.unlabeled_batch_size),
            )
        )

    def __len__(self) -> int:
        return len(self.labeled_indices) // self.labeled_batch_size
