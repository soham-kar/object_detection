"""Visualize WRDNet detection results on foggy images.

Generates a qualitative figure showing the model's predicted bounding boxes
drawn on real foggy images (ACDC val). This is the "show, don't just tell"
evidence that complements the quantitative mAP tables.

Usage (on Modal):
    modal run modal_train.py::visualize_detections --phase phase1 --num_samples 4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# DehazeFormer module (cloned to /tmp/DehazeFormer on Modal)
sys.path.insert(0, '/tmp/DehazeFormer')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.ops import nms as tv_nms

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.data.dataset import build_dataloaders

# 8 detection classes (same as training / evaluator)
CLASS_NAMES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
# Distinct colors per class for the boxes
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 8))

# ImageNet denormalization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def decode_predictions(raw_preds, b, conf_thres=0.25, input_h=512, input_w=1024):
    """Decode raw YOLO predictions into normalized [x1,y1,x2,y2] boxes.

    Mirrors WRDNetEvaluator._decode_predictions but with a higher default
    confidence threshold (0.25) so the figure shows confident detections
    rather than the low-threshold (0.01) used for mAP.
    """
    pred = raw_preds[b]  # [84, N]
    box_preds = pred[:4, :].T  # [N, 4] cx, cy, w, h (pixels)
    cls_preds = pred[4:, :].T  # [N, 80]
    cls_preds_8 = cls_preds[:, :8]  # our 8 classes

    max_conf, max_cls = cls_preds_8.max(dim=1)
    mask = max_conf > conf_thres
    if mask.sum() == 0:
        return None, None, None

    boxes = box_preds[mask]
    confs = max_conf[mask]
    cls_ids = max_cls[mask]

    # Normalize to [0,1]
    boxes_norm = boxes.clone()
    boxes_norm[:, 0] = boxes[:, 0] / input_w
    boxes_norm[:, 1] = boxes[:, 1] / input_h
    boxes_norm[:, 2] = boxes[:, 2] / input_w
    boxes_norm[:, 3] = boxes[:, 3] / input_h

    # cx,cy,w,h -> x1,y1,x2,y2
    x1 = boxes_norm[:, 0] - boxes_norm[:, 2] / 2
    y1 = boxes_norm[:, 1] - boxes_norm[:, 3] / 2
    x2 = boxes_norm[:, 0] + boxes_norm[:, 2] / 2
    y2 = boxes_norm[:, 1] + boxes_norm[:, 3] / 2
    x1, y1, x2, y2 = x1.clamp(0, 1), y1.clamp(0, 1), x2.clamp(0, 1), y2.clamp(0, 1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes, confs, cls_ids


def draw_detections(ax, img, boxes, confs, cls_ids):
    """Draw bounding boxes + labels on an image axis."""
    ax.imshow(img)
    ax.axis('off')
    if boxes is None:
        return
    H, W = img.shape[:2]
    for i in range(len(confs)):
        x1, y1, x2, y2 = boxes[i].tolist()
        c = int(cls_ids[i].item())
        conf = confs[i].item()
        color = CLASS_COLORS[c]
        # Convert normalized coords to pixel coords
        px1, py1 = x1 * W, y1 * H
        pw, ph = (x2 - x1) * W, (y2 - y1) * H
        rect = mpatches.Rectangle((px1, py1), pw, ph, linewidth=2,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = f"{CLASS_NAMES[c]} {conf:.2f}"
        ax.text(px1, max(py1 - 4, 0), label, fontsize=8, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=1))


def main():
    parser = argparse.ArgumentParser(description='Visualize WRDNet detections')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualizations/detections.png')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    return parser.parse_args()


def run(phase: str = "phase1", num_samples: int = 4, conf_thres: float = 0.25,
        output: str = None):
    """Main entrypoint (also called from Modal)."""
    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WRDNet(config).to(device)
    ckpt_path = f"/checkpoints/{phase}/best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ACDC val loader (real fog with labels)
    config.batch_size = 1
    _, val_loader = build_dataloaders(config)

    # Collect samples
    samples = []
    with torch.no_grad():
        for batch in val_loader:
            if len(samples) >= num_samples:
                break
            images = batch['image'].to(device)
            outputs = model(images)
            det_output = outputs['detections']
            raw_preds = det_output[0] if isinstance(det_output, (tuple, list)) else det_output

            for b in range(images.shape[0]):
                if len(samples) >= num_samples:
                    break
                boxes, confs, cls_ids = decode_predictions(
                    raw_preds, b, conf_thres=conf_thres
                )
                # Denormalize image
                img = images[b].cpu().permute(1, 2, 0).numpy()
                img = img * STD + MEAN
                img = np.clip(img, 0, 1)
                samples.append((img, boxes, confs, cls_ids))

    # Build figure grid
    n = len(samples)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (img, boxes, confs, cls_ids) in enumerate(samples):
        draw_detections(axes[i], img, boxes, confs, cls_ids)
        axes[i].set_title(f"WRDNet detections (sample {i+1})", fontsize=11)
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if output is None:
        output = f"/checkpoints/{phase}/detections.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved detection visualization to {output}")


if __name__ == '__main__':
    args = parse_args()
    run(phase="phase1", num_samples=args.num_samples, conf_thres=args.conf_thres,
        output=args.output)
