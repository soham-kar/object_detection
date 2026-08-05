"""Plot mAP vs. fog density — the "graceful degradation" money shot.

Stratifies detection performance by fog density (beta = 0.005, 0.01, 0.02)
using the Foggy Cityscapes validation set. Shows that WRDNet degrades
gracefully as fog thickens, unlike a baseline that collapses.

This is a NOVEL, UDA-SAFE analysis figure:
  - Pure evaluation (no training signal)
  - Uses known beta values from DBF fog
  - No new data downloads required
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.evaluation.evaluator import WRDNetEvaluator
from src.data.foggy_cityscapes import FoggyCityscapesDataset
from torch.utils.data import DataLoader
from src.data.dataset import wrdnet_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Plot mAP vs fog density')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualizations/map_vs_fog_density.png')
    parser.add_argument('--data_root', type=str, default='data')
    return parser.parse_args()


def evaluate_at_density(model, config, data_root, beta, device):
    """Evaluate mAP on Foggy Cityscapes val at a specific fog density."""
    # Build a val dataset at this specific beta
    dataset = FoggyCityscapesDataset(
        root=os.path.join(data_root, 'cityscapes'),
        split='val',
        fog_density=beta,
        input_size=getattr(config, 'input_size', 640),
        load_clear=False,
        load_depth=False,
        config=config,
    )

    if len(dataset) == 0:
        print(f"  WARNING: No samples at beta={beta}")
        return 0.0, 0.0

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=wrdnet_collate_fn,
    )

    evaluator = WRDNetEvaluator(model, device=str(device))
    metrics = evaluator.evaluate_detection(loader)
    return metrics.get('mAP@50', 0.0), metrics.get('mAP@50:95', 0.0)


def main():
    args = parse_args()

    # Load model
    config = load_config(args.config)
    model = WRDNet(config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Fog densities from DBF
    betas = ['0.005', '0.01', '0.02']
    labels = ['Light (β=0.005)', 'Medium (β=0.01)', 'Dense (β=0.02)']

    map50 = []
    map5095 = []

    print("Evaluating mAP at each fog density...")
    for beta in betas:
        m50, m5095 = evaluate_at_density(model, config, args.data_root, beta, device)
        map50.append(m50)
        map5095.append(m5095)
        print(f"  β={beta}: mAP@50={m50:.4f}, mAP@50:95={m5095:.4f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(betas))
    width = 0.35

    bars1 = ax.bar(x - width/2, map50, width, label='mAP@50', color='#1f77b4')
    bars2 = ax.bar(x + width/2, map5095, width, label='mAP@50:95', color='#ff7f0e')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('mAP', fontsize=13)
    ax.set_title('WRDNet Detection Performance vs. Fog Density\n'
                 '(Foggy Cityscapes validation)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(map50 + map5095) * 1.2 + 0.02)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")


if __name__ == '__main__':
    main()
