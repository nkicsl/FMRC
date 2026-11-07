# CFC-Net & FMRC

[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.8-3776AB.svg)](https://www.python.org/)
[![License: CC BY-NC-ND](https://img.shields.io/badge/License-CC%20BY--NC--ND-blue.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Official model-component and data-processing release for:

> **Cross-Frequency Collaborative Training for Robust Semi-supervised Segmentation in Endodontics and Beyond**

CFC-Net learns complementary representations through a low-frequency student, a high-frequency student, and a comprehensive teacher. Its UCF-Mix module selects low-uncertainty foreground patches and performs same-location cyclic exchange among raw, low-frequency, and high-frequency views.

## Release scope

This repository intentionally contains only:

- the VGG16-UNet backbone and three-branch CFC-Net container;
- wavelet frequency decomposition and the corrected UCF-Mix operation;
- dataset loading, paired augmentation, split-list generation, and semi-supervised batch sampling.

Training entry points, training loops, checkpoints, pretrained weights, experiment logs, and datasets are **not** included.

## Repository structure

```text
FMRC/
├── models/
│   ├── cfc_net.py       # three collaborative branches and teacher update
│   ├── frequency.py     # db2 wavelet low/high-frequency construction
│   ├── ucf_mix.py       # uncertainty-guided cyclic patch exchange
│   └── unet.py          # VGG16-UNet backbone
├── datasets/
│   ├── segmentation.py  # paired 2D dataset and semi-supervised sampler
│   └── transforms.py    # synchronized image-mask transforms
└── README.md
```

## Environment

The released components require Python 3.8+ and the following packages:

```bash
pip install torch torchvision numpy pillow opencv-python pywavelets
```

No pretrained weights are downloaded unless `pretrained=True` is explicitly passed when constructing the model.

## Data preparation

We use 2D slices for training, including slices extracted from 3D datasets. Images and labels are resized to `256 x 256` by the dataset pipeline.

Arrange a dataset as follows:

```text
DATASET_ROOT/
├── train/
│   ├── image/
│   │   ├── sample_0001.jpg
│   │   └── ...
│   ├── label/
│   │   ├── sample_0001.png
│   │   └── ...
│   └── train.txt
└── test/
    ├── image/
    ├── label/
    └── test.txt
```

Each line of `train.txt` or `test.txt` contains a filename without its extension:

```text
sample_0001
sample_0002
```

Generate a deterministic list file with:

```python
from datasets import build_sample_list

build_sample_list(
    image_dir="/path/to/DATASET_ROOT/train/image",
    output_file="/path/to/DATASET_ROOT/train/train.txt",
    image_suffix=".jpg",
)
```

Palette-mode masks are read using their underlying class indices. For binary masks encoded as `0/255`, pass `label_map={255: 1}`. Multi-class masks such as `0/1/2/3` can be loaded without remapping.

## Quick start

### 1. Build the data pipeline

```python
from torch.utils.data import DataLoader
from datasets import (
    JointCompose,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    JointRandomVerticalFlip,
    JointToTensor,
    SegmentationDataset2D,
)

transform = JointCompose([
    JointRandomHorizontalFlip(0.5),
    JointRandomVerticalFlip(0.5),
    JointRandomRotation(degrees=10, probability=0.5),
    JointToTensor(),
])

dataset = SegmentationDataset2D(
    root="/path/to/DATASET_ROOT",
    split="train",
    image_size=(256, 256),
    transform=transform,
    rgb=True,
    label_map={255: 1},
)

loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
```

For semi-supervised batches, use `SemiSupervisedBatchSampler` to place labeled samples first and unlabeled samples second in every batch.

### 2. Construct CFC-Net

```python
import torch
from models import CFCNet, WaveletFrequencyDecomposer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CFCNet(in_channels=3, num_classes=2, pretrained=False).to(device)
decomposer = WaveletFrequencyDecomposer(wavelet="db2")

image = torch.randn(2, 3, 256, 256, device=device)
low_frequency, high_frequency = decomposer(image)
outputs = model(image, low_frequency, high_frequency)

print(outputs["low_student"].shape)   # [2, 2, 256, 256]
print(outputs["high_student"].shape)  # [2, 2, 256, 256]
print(outputs["teacher"].shape)       # [2, 2, 256, 256]
```

### 3. Apply corrected UCF-Mix

```python
from models import uncertainty_guided_cross_frequency_mix

pseudo_mask = outputs["teacher"].argmax(dim=1)
mixed = uncertainty_guided_cross_frequency_mix(
    raw=image,
    low=low_frequency,
    high=high_frequency,
    logits=[outputs["low_student"], outputs["high_student"], outputs["teacher"]],
    foreground_mask=pseudo_mask,
    patch_size=16,
    top_k=4,
)

print(mixed.selected_indices)
```

UCF-Mix sorts patches by ascending entropy, selects the lowest-uncertainty foreground patches, and applies both frequency cycles within the same sample and at the same spatial location.

## FMRC dataset

An initial set of 60 manually annotated samples is available at:

- [Baidu Netdisk](https://pan.baidu.com/s/1y5wayt1N3UPbtbc5tklTcA?pwd=jsqf) (access code: `jsqf`)

The data are provided for non-commercial research use only. Please follow the terms distributed with the dataset.

## Citation

The final BibTeX entry will be added after publication. Until then, please cite the manuscript by title:

```bibtex
@misc{zhou2026cfcnet,
  title  = {Cross-Frequency Collaborative Training for Robust Semi-supervised Segmentation in Endodontics and Beyond},
  author = {Zhou, ZhenHuan and Zhang, Yuchen and He, Along and Wang, Peng and Li, Tao},
  year   = {2026}
}
```

## License

This project and the linked FMRC samples are intended for academic, non-commercial research. Commercial use is not permitted.
