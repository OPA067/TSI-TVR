<div align="center">

# TSI-TVR: Temporal Semantic Interaction for Bridging Information Asymmetry in Text-Video Retrieval

[![Python ≥3.8](https://img.shields.io/badge/python-≥3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch ≥2.0](https://img.shields.io/badge/pytorch-≥2.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**TSI-TVR**

</div>

This is a **CLIP-based cross-modal text-video retrieval model** that achieves fine-grained text-video alignment through **temporal semantic interaction** mechanisms.

---

## Updates

- **[2026.04]** Initial release — complete training & evaluation pipeline for text-video retrieval.
- **[2026.05]** Added ActionFlow module with PCM (Progressive Clustering) for patch token compression.

---

## Table of Contents

- [Overview](#overview)
- [Updates](#updates)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Download Pre-trained Weights](#download-pre-trained-weights)
  - [Training](#training)
  - [Testing / Evaluation](#testing--evaluation)
  - [Using Scripts](#using-scripts)
- [Architecture](#architecture)
- [Key Modules](#key-modules)
- [Evaluation Metrics](#evaluation-metrics)
- [Training Outputs](#training-outputs)
- [Key CLI Arguments](#key-cli-arguments)
- [Citation](#citation)

---

## Overview

Text-Video Retrieval (TVR) aims to retrieve the most relevant video given a text query (Text → Video) or vice versa (Video → Text). This project provides a unified training and evaluation pipeline built on CLIP, enhanced with multi-granularity feature representations and temporal semantic interaction.

### Key Features

| Feature | Description |
|---------|-------------|
| **Backbone** | OpenAI CLIP (`ViT-B/32` or `ViT-B/16`) with pretrained weight loading |
| **Text Features** | Triple-granularity: `subject_feat` (main entity) + `whole_feat` (full sentence) + `caption_feat` (multi-sentence aggregation) |
| **Video Features** | Dual-granularity: `frame_feat` (frame-level global) + `pooled_feat` (patch-level local) |
| **ActionFlow** | Progressive Clustering Module (PCM) + 3× Attention blocks for patch token compression and refinement |
| **Interaction** | Temporal Semantic Interaction with KL divergence alignment |
| **Datasets** | 7 standard benchmarks: MSRVTT, MSVD, LSMDC, Charades, ActivityNet, DiDeMo, VATEX |
| **Training** | Single-GPU or Multi-GPU (DDP via `torch.distributed.launch`) with warmup + cosine scheduling |
| **Loss** | Cross-modal contrastive loss (InfoNCE) + KL divergence loss for temporal semantic consistency |

---

## Project Structure

```
TSI-TVR/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main_retrieval.py                   # Main training / evaluation entry point
│                                       #   - Parses args, sets up DDP
│                                       #   - Builds dataloaders, model, optimizer
│                                       #   - Runs do_train() / do_eval() loop
│
# ─── Core Model Definitions ──────────────────────────────────────────
├── models/
│   ├── __init__.py                     # Package init
│   ├── modeling.py                     # Core Model: CLIP + video aggregation
│   │                                   #   + ActionFlow + multi-matching heads
│   ├── module_clip.py                  # CLIP pretrained model loading (ViT-B/32, etc.)
│   ├── module_cross.py                 # Cross-modal Transformer (video frame position encoding + temporal Transformer)
│   ├── module_transformer.py           # CLIP internal Transformer blocks
│   ├── cluster.py                      # PCM (Progressive Clustering Module)
│   │                                   #   + Att_Block_Patch (ActionFlow building blocks)
│   ├── module_CAttention.py            # Cross-modal Attention Module (CAM)
│   ├── until_module.py                 # CrossEn (InfoNCE loss) + KL divergence loss
│   ├── until_config.py                 # Model configuration utilities
│   ├── optimization.py                 # BertAdam optimizer (warmup + cosine schedule)
│   ├── tokenization_clip.py            # CLIP BPE tokenizer
│   ├── file_utils.py                   # File I/O helpers
│   └── cross-base/                     # Cross-modal base model configs
│       └── config.json
│
# ─── Data Loading & Preprocessing ────────────────────────────────────
├── dataloaders/
│   ├── __init__.py                     # Package init
│   ├── data_dataloaders.py             # Dataset dispatcher (supports all 7 datasets)
│   ├── dataloader_retrieval.py         # Base retrieval dataloader
│   ├── dataloader_msrvtt_retrieval.py  # MSRVTT dataloader
│   ├── dataloader_msvd_retrieval.py    # MSVD dataloader
│   ├── dataloader_lsmdc_retrieval.py   # LSMDC dataloader
│   ├── dataloader_charades_retrieval.py # Charades dataloader
│   ├── dataloader_activitynet_retrieval.py # ActivityNet dataloader
│   ├── dataloader_didemo_retrieval.py  # DiDeMo dataloader
│   ├── dataloader_vatex_retrieval.py   # VATEX dataloader
│   ├── rawvideo_util.py                # Video reading / frame extraction utilities
│   ├── video_transforms.py             # Video preprocessing transforms
│   ├── functional.py                   # Functional video ops
│   ├── rand_augment.py                 # Random augmentation policies
│   └── random_erasing.py               # Random erasing augmentation
│
# ─── Utilities ───────────────────────────────────────────────────────
├── utils/
│   ├── __init__.py                     # Package init
│   ├── metrics.py                      # Standard retrieval metrics: R@1/5/10, MdR, MnR
│   ├── metrics_qa.py                   # QA-specific metrics
│   ├── metric_logger.py                # Training metric logging
│   ├── logger.py                       # Logger setup
│   ├── util.py                         # General utilities
│   └── comm.py                         # DDP communication helpers
│
# ─── Training Scripts ────────────────────────────────────────────────
├── script/
│   ├── run_MSRVTT.sh                   # MSRVTT training script
│   ├── run_Charades.sh                 # Charades training script
│   ├── run_ActivityNet.sh              # ActivityNet training script
│   ├── run_DiDeMo.sh                   # DiDeMo training script
│   ├── run_LSMDC.sh                    # LSMDC training script
│   └── run_test.sh                     # Evaluation script template
│
# ─── Preprocessing ───────────────────────────────────────────────────
├── preprocess/
│   └── compress_video.py               # Video preprocessing / compression script
│
# ─── Visualization & Analysis ────────────────────────────────────────
├── visualize/
│   ├── get_frame.py                    # Frame extraction for visualization
│   ├── get_retrieval.py                # Retrieval result visualization
│   ├── sim_matrix.txt                  # Similarity matrix sample
│   ├── retrieval.json                  # Retrieval results sample
│   ├── sim_matrix_kl_loss.png          # KL loss visualization
│   ├── infer_info.png                  # Inference info visualization
│   ├── params.png                      # Model parameters visualization
│   └── video0_frames.jpg               # Sample frame visualization
│
# ─── Reviews ─────────────────────────────────────────────────────────
└── reviews/
    ├── batch32.png                     # Batch analysis figure
    └── batch10000.png                  # Batch analysis figure
```

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/OPA067/TSI-TVR.git
cd TSI-TVR

# 2. Create a conda environment (recommended)
conda create -n tsi-tvr python=3.10
conda activate tsi-tvr

# 3. Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` pins core packages:
- `torch>=2.0`, `torchvision`
- `transformers>=4.39` (Hugging Face transformers)
- `decord` (video decoding)
- `opencv-python>=4.9` (video I/O)
- `numpy`, `pandas`, `tqdm` (data & logging)
- `timm` (PyTorch image models)
- `boto3` (optional, for weight downloading)

### Dataset Preparation

The project supports **7 standard text-video retrieval datasets**. Please organize your data as follows:

```
TSI-TVR/
├── MSRVTT/
│   ├── videos/                         # Video files (.mp4)
│   ├── MSRVTT_data.json               # Metadata / annotations
│   ├── MSRVTT_train.7000.csv          # Training split
│   └── MSRVTT_test.1000.csv           # Testing split
│
├── MSVD/
│   ├── videos/
│   └── ...
├── LSMDC/
│   ├── videos/
│   └── ...
├── Charades/
│   ├── videos/
│   └── ...
├── ActivityNet/
│   ├── videos/
│   └── ...
├── DiDeMo/
│   ├── videos/
│   └── ...
└── VATEX/
    ├── videos/
    └── ...
```

Each dataset dataloader (`dataloaders/dataloader_<name>_retrieval.py`) reads the corresponding annotation format. Refer to individual dataloader files for expected JSON/CSV schemas.

### Download Pre-trained Weights

CLIP pretrained weights are downloaded **automatically** on first use (cached in `~/.cache/clip/`). Supported variants:

| Model | Architecture | Notes |
|:-----:|:-------------|:------|
| **ViT-B/32** | ViT-Base, patch 32 | **Recommended baseline** |
| **ViT-B/16** | ViT-Base, patch 16 | Finer spatial patches |

Specify via `--base_encoder ViT-B/32` (default).

### Training

#### Single-GPU Quick Start

```bash
bash script/run_MSRVTT.sh
```

This runs with:
- CLIP: `ViT-B/32`
- Dataset: `MSRVTT`
- Epochs: `5`
- Batch size: `8`
- Max words: `32` / Max frames: `12`

#### Manual Training (Full Control)

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
  --master_port 2501 \
  --nproc_per_node=1 \
  main_retrieval.py \
  --do_train 1 \
  --datatype msrvtt \
  --anno_path MSRVTT \
  --video_path MSRVTT/videos \
  --max_words 32 \
  --max_frames 12 \
  --epochs 5 \
  --batch_size 32 \
  --batch_size_val 32 \
  --lr 1e-4 \
  --coef_lr 1e-3 \
  --output_dir experiments/MSRVTT
```

#### Multi-GPU DDP

```bash
GPU="0,1,2,3"
NPROC=$(echo "$GPU" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$GPU \
python -m torch.distributed.launch \
  --master_port 2501 \
  --nproc_per_node=$NPROC \
  main_retrieval.py \
  --do_train 1 \
  --datatype msrvtt \
  --batch_size 8 \
  --batch_size_val 8 \
  --split_batch 8 \
  ...
```

> DDP is enabled via `torch.distributed.launch`. The script `run_MSRVTT.sh` automatically derives `NPROC` from the GPU list.

### Testing / Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
  --master_port 2502 \
  --nproc_per_node=1 \
  main_retrieval.py \
  --do_eval 1 \
  --datatype msrvtt \
  --anno_path MSRVTT \
  --video_path MSRVTT/videos \
  --max_words 32 \
  --max_frames 12 \
  --batch_size_val 32 \
  --output_dir experiments/MSRVTT \
  --init_model experiments/MSRVTT/pytorch_model.bin.best
```

### Using Scripts

Pre-configured scripts are provided under `script/`:

```bash
# MSRVTT
bash script/run_MSRVTT.sh

# Charades
bash script/run_Charades.sh

# ActivityNet
bash script/run_ActivityNet.sh

# DiDeMo
bash script/run_DiDeMo.sh

# LSMDC
bash script/run_LSMDC.sh
```

---

## Architecture

### Overall Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Training / Evaluation Pipeline                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Text Input                                                                  │
│       │                                                                      │
│       ▼                                                                      │
│  CLIP Text Encoder ──┬──► subject_feat (s_feat)    [Global / Entity]       │
│                      └──► whole_feat (w_feat)       [Sequence / Full]        │
│                      └──► caption_feat (c_feat)     [Multi-sentence]         │
│                                                                              │
│  Video Input                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  CLIP Vision Encoder ──┬──► frame_feat (f_feat)     [Frame-level global]   │
│                        └──► pooled_feat (p_feat)    [Patch-level local]      │
│                              │                                               │
│                              ▼                                               │
│                        ActionFlow (PCM × 3 Att_Block_Patch)                  │
│                              │  Patch token compression & refinement          │
│                              ▼                                               │
│                                                                              │
│  Temporal Semantic Interaction Matching Layer                                │
│  ════════════════════════════════════════════════════════════════════════   │
│  Temporal:  c_feat ↔ f_feat   (caption global → video frames global)        │
│  Spatial:   w_feat ↔ p_feat   (whole text sequence → video patches local)   │
│  Temporal:  s_feat ↔ f_feat   (subject entity → video frames global)        │
│  Spatial:   c_feat ↔ p_feat   (caption global → video patches local)        │
│  ════════════════════════════════════════════════════════════════════════   │
│                              │                                               │
│                              ▼                                               │
│  KL Alignment Loss ←───────  Temporal & Spatial distribution consistency    │
│                              │                                               │
│                              ▼                                               │
│  Similarity Matrix ──► Softmax ──► Recall@K                                  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### Matching Heads

In `models/modeling.py`, the `forward()` method computes **4 cross-modal similarity matrices** simultaneously:

```python
# Temporal: caption_feat (global) ↔ frame_feat
sims_cf = self.c_and_f(c_feat, f_feat)

# Spatial: whole_feat ↔ pooled_feat (patch)
sims_wp = self.w_and_p(w_feat, p_feat)

# Temporal: subject_feat ↔ frame_feat
sims_sf = self.s_and_f(s_feat, f_feat)

# Spatial: caption_feat ↔ pooled_feat (patch)
-sims_cp = self.c_and_p(c_feat, p_feat)
```

Each similarity matrix is constrained by **CrossEn** (symmetric InfoNCE):

```python
loss = (loss_fct(sims * logit_scale) + loss_fct(sims.T * logit_scale)) / 2.0
```

And **KL divergence** aligns Temporal and Spatial distributions:

```python
loss_kl_sf = (loss_kl(sims_sf, sims_cf) + loss_kl(sims_sf.T, sims_cf.T)) / 2.0
```

---

## Key Modules

| Module | File | Function |
|--------|------|----------|
| `CLIP` | `models/module_clip.py` | Pretrained image/text dual-stream encoder |
| `TransformerClip` | `models/module_cross.py` | Video frame position encoding + temporal Transformer (for `seqTransf` aggregation) |
| `PCM` + `Att_Block_Patch` | `models/cluster.py` | **ActionFlow**: 3-layer progressive clustering + attention for patch token compression |
| `CAM` | `models/module_CAttention.py` | Cross-modal attention module |
| `CrossEn` / `KL` | `models/until_module.py` | Contrastive loss (InfoNCE) + KL divergence alignment loss |

---

## Evaluation Metrics

Standard retrieval metrics computed by `utils/metrics.py`:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **R@1** | — | % queries where ground truth is ranked 1st |
| **R@5** | — | % queries where ground truth is in top-5 |
| **R@10** | — | % queries where ground truth is in top-10 |
| **RSum** | — | `R@1 + R@5 + R@10` (comprehensive metric) |
| **MdR** | — | Median rank of the correct match (lower is better) |
| **MnR** | — | Mean rank of the correct match (lower is better) |

**Two retrieval directions:**
- **Text → Video**: Given a text query, retrieve the matching video
- **Video → Text**: Given a video, retrieve the matching text description

Example evaluation output:

```
Text-to-Video:  R@1: 32.5  R@5: 58.3  R@10: 68.9  RSum: 159.7  MdR: 5.0  MnR: 18.2
Video-to-Text:  R@1: 28.3  R@5: 52.1  R@10: 63.7  RSum: 144.1  MdR: 6.0  MnR: 21.5
```

---

## Training Outputs

Each training run creates a timestamped directory under `--output_dir`:

```
experiments/
└── MSRVTT/
    ├── 2025-10-19_12:07:07/
    │   ├── pytorch_model.bin.0       # Epoch checkpoint
    │   └── log.txt                   # Training log
    └── ...
```

During training, logs are printed every `--n_display` iterations:

```
eta: 0:10:00, epoch: 1/5, iteration: 100/219,
 time: 0.234, data: 0.012, loss: 2.345
 lr: 0.000100000/0.000000100, logit: 4.605, memory: 8.23GB
```

Best checkpoints are typically saved as `pytorch_model.bin.best`.

---

## Key CLI Arguments

All arguments are defined in `main_retrieval.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--do_train` | `0` | Enable training |
| `--do_eval` | `0` | Enable evaluation |
| `--datatype` | `msrvtt` | Dataset name: `msrvtt`, `msvd`, `lsmdc`, `charades`, `activitynet`, `didemo`, `vatex` |
| `--base_encoder` | `ViT-B/32` | CLIP backbone variant |
| `--agg_module` | `seqTransf` | Video temporal aggregation: `None` / `seqLSTM` / `seqTransf` |
| `--interaction` | `wti` | Interaction type (Weighted Token Interaction) |
| `--num_hidden_layers` | `4` | Video temporal Transformer layers |
| `--max_words` | `24` | Max text token length |
| `--max_frames` | `12` | Max video sampling frames |
| `--split_batch` | `32` | GPU memory split size during evaluation |
| `--alpha` | `0.5` | Attention hyperparameter |
| `--beta` | `0.1` | Attention hyperparameter |
| `--gamma` | `0.01` | Attention hyperparameter |
| `--lr` | `1e-4` | Learning rate for non-CLIP modules |
| `--coef_lr` | `1e-3` | CLIP module LR coefficient (`lr_CLIP = lr × coef_lr`) |
| `--init_model` | `None` | Pretrained weight path for evaluation or resume |
| `--epochs` | `5` | Total training epochs |
| `--batch_size` | `32` | Training batch size per GPU |
| `--batch_size_val` | `32` | Validation batch size |
| `--video_framerate` | `1` | Frame sampling rate |

---

## Citation

If you use this code, please cite the original CLIP work:

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## License

This codebase is modified from the open-source CLIP implementation and distributed under the [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE).

---

<div align="center">
  <sub>Built for the Text-Video Retrieval community.</sub>
</div>
