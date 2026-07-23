# WRDNet — Weather-Resilient Detection Unified Network

Joint Defogging and Detection with Multi-Level Frequency-Aware Domain Adaptation for Adverse Weather.

## Overview

WRDNet addresses two critical research gaps in adverse-weather object detection:
1. **Gap 3.3**: Joint optimization of image restoration and object detection (not sequential)
2. **Gap 2.1**: Synthetic-to-real domain generalization for foggy scenes

**Core Innovation**: Feature Selection Gate (FSG) — a learned per-pixel gate that dynamically fuses restored and original features based on local fog density.

**Extended Contribution**: Depth-Guided FSG (DG-FSG) — uses monocular depth estimation to make fusion decisions physically interpretable (distant objects → trust defogger more).

## Architecture

- **DehazeFormer-T** (0.69M params): ViT-based defogging at 320×320
- **YOLOv11s** (9.4M params): Detection at 640×640
- **FSG / DG-FSG** (~0.20M params): Dynamic feature fusion at P3/P4/P5 scales
- **Domain Adaptation**: FDA + DCT Alignment + FSG Consistency + TTA
- **Depth Decoder** (0.30M params): Monocular depth from defogging bottleneck

**Total**: ~10.5M params, ~24.5 GMACs

## Project Structure

```
object_detection/
├── configs/              # Training configurations
├── src/
│   ├── models/           # Model definitions (WRDNet, FSG, DG-FSG, etc.)
│   ├── domain_adaptation/ # FDA, DCT Alignment, FSG Consistency
│   ├── data/             # Dataset loaders (Foggy Cityscapes, ACDC, DAWN)
│   ├── training/         # Training loop, losses, optimizers
│   ├── evaluation/       # mAP, PSNR, SSIM, alpha visualization
│   └── utils/            # Config, logging, metrics, FLOPs
├── scripts/              # train.py, evaluate.py, visualize_alpha.py
├── experiments/          # Checkpoints, logs, results
└── notebooks/            # Analysis notebooks
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

- **Foggy Cityscapes** (synthetic fog, labeled): https://www.cityscapes-dataset.com/
- **ACDC** (real fog, unlabeled): https://acdc.vision.ee.ethz.ch/
- **DAWN** (real adverse, unlabeled): https://data.mendeley.com/datasets/766ygrbt8y/3
- **RESIDE-6K** (DehazeFormer pretraining): https://sites.google.com/view/reside-dehaze-datasets

Organize under `data/` as described in `IMPLEMENTATION_PLAN.md`.

### 3. Train

```bash
# Phase 0: Warmup (joint training without DA)
python scripts/train.py --config configs/default.yaml --phase warmup

# Phase 1: Domain Adaptation
python scripts/train.py --config configs/wrnet_s.yaml --phase da

# With depth (DG-FSG)
python scripts/train.py --config configs/wrnet_s.yaml --use_depth true
```

### 4. Evaluate

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pth --dataset acdc

# With TTA
python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pth --dataset acdc --tta

# Alpha visualization
python scripts/visualize_alpha.py --checkpoint experiments/checkpoints/best.pth --output_dir experiments/results/alpha_maps/
```

### 5. Generate "Money Shot" Plot

```bash
python scripts/plot_alpha_vs_depth.py --checkpoint experiments/checkpoints/best.pth --dataset foggy_cityscapes --output experiments/results/alpha_vs_depth.png
```

## Ablation Experiments

| ID | Experiment | Config |
|----|-----------|--------|
| E0 | YOLOv11s on foggy (no defog) | `configs/ablations/e0_baseline.yaml` |
| E1 | Sequential (DehazeFormer→YOLO) | `configs/ablations/e1_sequential.yaml` |
| E2 | Joint (concat, no FSG) | `configs/ablations/e2_joint_no_fsg.yaml` |
| E3 | Joint + FSG | `configs/ablations/e3_joint_fsg.yaml` |
| E4 | E3 + FDA + Entropy | `configs/ablations/e4_fda.yaml` |
| E5 | E3 + DCT Alignment | `configs/ablations/e5_dct_align.yaml` |
| E6 | E3 + FSG Consistency | `configs/ablations/e6_fsg_consistency.yaml` |
| E7 | E3 + All DA | `configs/ablations/e7_full_da.yaml` |
| E8 | E7 + TTA | `configs/ablations/e8_tta.yaml` |
| E9 | E3 - MAA | `configs/ablations/e9_no_maa.yaml` |
| E10 | E3 - CDMSA | `configs/ablations/e10_no_cdmsa.yaml` |
| E11 | E3 + Depth Decoder (no DG-FSG) | `configs/ablations/e11_depth_aux.yaml` |
| E12 | E3 + DG-FSG | `configs/ablations/e12_dg_fsg.yaml` |
| E13 | E7 + DG-FSG | `configs/ablations/e13_full_da_depth.yaml` |

## Citation

If you use WRDNet in your research, please cite:

```bibtex
@article{wrdnet2026,
  title={WRDNet: Weather-Resilient Detection Unified Network},
  author={[Your Name]},
  journal={[Venue]},
  year={2026}
}
```

## License

MIT License
