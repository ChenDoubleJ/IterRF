<div align="center">

# IterRF

### Iterative and Interpretable Mutual Enhancement for Joint Multi-Modal Image Registration and Fusion

[![Paper](https://img.shields.io/badge/Paper-IJCV(CCF--A)-blue)](#citation)
[![Code](https://img.shields.io/badge/Code-released-brightgreen)](#testing)
[![Models](https://img.shields.io/badge/Models-released-brightgreen)](#pretrained-models)

Official implementation of **IterRF**.

</div>

---

## Overview

IterRF addresses unaligned multi-modal image fusion by optimizing spatial registration and image fusion in one interpretable, iterative framework. Registration benefits from progressively enhanced fused representations, while fusion benefits from increasingly reliable spatial correspondence.

<p align="center">
  <img width="600" alt="IterRF overview" src="https://github.com/user-attachments/assets/f59bf430-32dd-4fcd-a6fb-72b52c111e80" />
</p>

## Highlights

- Joint registration and fusion in a unified optimization framework.
- Iterative bidirectional refinement between registration and fusion.
- Optimization-inspired, interpretable network unfolding.
- Support for infrared-visible and medical multi-modal images.

## Framework

<p align="center">
  <img width="1000" alt="IterRF framework" src="https://github.com/user-attachments/assets/61d8565f-fce4-44af-a676-abc135ccec12" />
</p>

## Installation

The original experiments used Python 3.8, PyTorch 1.11.0, torchvision 0.12.0, and CUDA 11.5. Create an isolated environment and install the PyTorch build matching your machine first.

```bash
conda create -n iterrf python=3.8 -y
conda activate iterrf

pip install -r requirements.txt
```

## Data Preparation

Each image pair must use the same filename stem in every subdirectory. Images are loaded as RGB tensors; `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff` are supported. 

```text
data/
└── RoadScene/                 # or M3FD, MRI-PET, SPECT-MRI
    ├── train/
    │   ├── ir/
    │   ├── vi/
    │   ├── warp_ir/
    │   ├── warp_vi/
    │   └── disp/
    └── test/
        ├── ir/
        ├── vi/
        └── warp_vi/
```

Custom locations can be passed through the command-line options below.

## Pretrained Models

Download the four task checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/19YCLZSKtE3Q0HuKdI3HUDw) with extraction code `b5kn`, then place them in `checkpoint/` using these filenames:

| Task | Checkpoint |
| --- | --- |
| RoadScene | `checkpoint/weight_roadscene.pth` |
| M3FD | `checkpoint/weight_m3fd.pth` |
| MRI-PET | `checkpoint/weight_mri_pet.pth` |
| SPECT-MRI | `checkpoint/weight_spect_mri.pth` |


## Testing

The task selects the default dataset and checkpoint paths. Results are written to `results/<task>/registered/` and `results/<task>/fused/`.

```bash
python test.py --task RoadScene
python test.py --task M3FD
python test.py --task MRI-PET
python test.py --task SPECT-MRI
```

Override paths when data or weights are stored elsewhere:

```bash
python test.py \
  --task RoadScene \
  --data-path /path/to/RoadScene/test \
  --weights /path/to/weight_roadscene.pth \
  --output-dir /path/to/results \
  --device cuda --gpu-id 0
```

Use `python test.py --help` for all options. CPU inference is supported with `--device cpu`, although it is substantially slower.

## Training

By default, training reads `data/<task>/train` and validates on `data/<task>/test`.

```bash
python train.py --task RoadScene --device cuda --gpu-id 0
```

Paths, initialization weights, and output location are configurable:

```bash
python train.py \
  --task RoadScene \
  --data-path-train /path/to/RoadScene/train \
  --data-path-test /path/to/RoadScene/test \
  --weights /path/to/initial_checkpoint.pth \
  --output-dir /path/to/experiments
```

Resume a complete training state with:

```bash
python train.py --task RoadScene --resume /path/to/checkpoint_100.pth
```

Run `python train.py --help` to inspect epoch, batch-size, learning-rate, validation, and checkpoint options.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{xu2026iterrf,
  title={IterRF: Iterative and Interpretable Mutual Enhancement for Joint Multi-Modal Image Registration and Fusion},
  author={Xu, Han and Chen, Jiajun and Yi, Xunpeng and Li, Peili and Ma, Jiayi},
  journal={International Journal of Computer Vision},
  volume={134},
  number={10},
  pages={442},
  year={2026},
  publisher={Springer}
}
```

## Contact

For questions about IterRF, please open an issue in this repository.

