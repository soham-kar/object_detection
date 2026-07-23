"""Visualization utilities for WRDNet evaluation."""

import os
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_detection_results(
    image: torch.Tensor,
    detections: torch.Tensor,
    save_path: str,
    class_names: List[str] = None,
    score_threshold: float = 0.5,
):
    """
    Visualize detection bounding boxes on image.

    Args:
        image: [3, H, W] tensor
        detections: [N, 6] tensor (x1, y1, x2, y2, score, class)
        save_path: path to save figure
        class_names: list of class names
        score_threshold: minimum confidence score
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Convert image to numpy
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)

    # Draw boxes
    for det in detections:
        x1, y1, x2, y2, score, cls = det.cpu().numpy()
        if score < score_threshold:
            continue

        width = x2 - x1
        height = y2 - y1

        rect = Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        label = f"{class_names[int(cls)] if class_names else int(cls)}: {score:.2f}"
        ax.text(x1, y1 - 5, label, color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_restoration_comparison(
    foggy: torch.Tensor,
    restored: torch.Tensor,
    clear_gt: torch.Tensor,
    save_path: str,
):
    """
    Visualize foggy, restored, and clear images side by side.

    Args:
        foggy: [3, H, W] foggy image
        restored: [3, H, W] restored image
        clear_gt: [3, H, W] ground truth clear image
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [foggy, restored, clear_gt]
    titles = ['Foggy Input', 'Restored', 'Ground Truth']

    for ax, img, title in zip(axes, images, titles):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_depth_map(
    image: torch.Tensor,
    depth: torch.Tensor,
    save_path: str,
    colormap: str = 'plasma',
):
    """
    Visualize depth map overlaid on image.

    Args:
        image: [3, H, W] image
        depth: [1, H, W] or [H, W] depth map
        save_path: path to save figure
        colormap: matplotlib colormap
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Depth
    depth_np = depth.cpu().squeeze().numpy()
    im = axes[1].imshow(depth_np, cmap=colormap)
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    log_file: str,
    save_dir: str,
    metrics: List[str] = None,
):
    """
    Plot training curves from log file.

    Args:
        log_file: path to JSON log file
        save_dir: directory to save plots
        metrics: list of metrics to plot
    """
    import json

    os.makedirs(save_dir, exist_ok=True)

    with open(log_file, 'r') as f:
        entries = json.load(f)

    if not entries:
        return

    # Extract data
    steps = [e['step'] for e in entries]

    if metrics is None:
        # Auto-detect metrics (exclude metadata keys)
        metrics = [k for k in entries[0].keys()
                   if k not in ['step', 'epoch', 'timestamp']]

    for metric in metrics:
        values = [e.get(metric, 0) for e in entries]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, linewidth=1.5)
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.title(f'Training Curve: {metric}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'), dpi=150)
        plt.close()
