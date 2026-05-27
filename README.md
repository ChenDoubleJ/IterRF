<div align="center">

# IterRF

### Iterative and Interpretable Mutual Enhancement for Joint Multi-Modal Image Registration and Fusion

[![Paper](https://img.shields.io/badge/Paper-IJCV%20revision-blue)](#citation)
[![Code](https://img.shields.io/badge/Code-coming%20soon-orange)](#release-plan)
[![Models](https://img.shields.io/badge/Models-coming%20soon-orange)](#release-plan)
[![License](https://img.shields.io/badge/License-coming%20soon-lightgrey)](#license)

Official repository for **IterRF**, an iterative and interpretable mutual enhancement for jointly improving multi-modal image registration and fusion.

</div>

---

## 📌 Overview

**IterRF** targets the challenging problem of **unaligned multi-modal image fusion**, where accurate spatial registration and high-quality image fusion must be optimized together rather than handled as isolated steps.

<p align="center">
  <img width="600" alt="IterRF overview" src="https://github.com/user-attachments/assets/f59bf430-32dd-4fcd-a6fb-72b52c111e80" />
</p>

Existing approaches commonly rely on a cascaded **register-then-fuse** pipeline, or introduce only limited one-step feedback between registration and fusion. These designs can restrict the mutual enhancement between spatial alignment and information integration.

IterRF formulates registration and fusion within a unified optimization framework and unfolds the optimization process into alternating registration-related and fusion-related subproblems. With iterative bidirectional refinement, the model progressively improves alignment accuracy and fusion quality.

## ✨ Highlights

- **Joint registration and fusion**: optimizes spatial alignment and fused image generation in a unified framework.
- **Iterative mutual enhancement**: alternates between registration-oriented and fusion-oriented refinement stages.
- **Optimization-inspired design**: unfolds interpretable subproblems into a learnable end-to-end framework.
- **Broad multi-modal applicability**: designed for infrared-visible and medical multi-modal image scenarios.
- **Research-friendly release plan**: code, pretrained models, evaluation scripts, and visualization results will be released after paper acceptance.

## 🧩 Framework

<p align="center">
  <img width="100%" alt="IterRF framework" src="https://github.com/user-attachments/assets/dd3ac80c-1c0b-4e11-a241-ff690d9f6ddf" />
</p>

The framework alternates between registration and fusion modules so that the two tasks can refine each other over multiple iterations. Registration benefits from progressively enhanced fused representations, while fusion benefits from increasingly reliable spatial correspondence.

## 🗓️ Release Plan

The full project will be released upon paper acceptance. Planned materials include:

- [ ] Source code for the official IterRF implementation
- [ ] Pretrained model checkpoints
- [ ] Testing and evaluation scripts
- [ ] Training scripts and configuration files
- [ ] Visualization tools and qualitative comparison results
- [ ] Detailed usage instructions and reproducibility notes

## 🛠️ Installation

The installation guide will be provided together with the code release.

## 🧪 Testing

Testing and evaluation instructions will be added after the official implementation is released.

## 🚀 Training

Training scripts, configuration files, and dataset preparation instructions will be released after paper acceptance.

## 📦 Pretrained Models

Pretrained checkpoints will be made available after the paper is accepted.

## 📈 Results

Quantitative results, qualitative comparisons, and visualization examples will be added in the official release.

## 📄 License

The license will be announced before the public code release.

## 📚 Citation

If you find this work useful for your research, please consider citing our paper. The BibTeX entry will be updated after publication.

<!--
```bibtex
@article{iterrf,
  title   = {IterRF: Iterative and Interpretable Mutual Enhancement for Joint Multi-Modal Image Registration and Fusion},
  author  = {Han Xu and Jiajun Chen and Xunpeng Yi and Peili Li and Jiayi Ma},
  journal = {International Journal of Computer Vision},
  year    = {2026}
}
```
-->

## 📬 Contact

For questions about IterRF, please open an issue in this repository.
