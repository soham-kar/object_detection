"""Plot alpha map values vs. depth — the "money shot" figure."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.data.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Plot alpha vs depth')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualizations/alpha_vs_depth.png')
    parser.add_argument('--num_samples', type=int, default=500)
    return parser.parse_args()


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

    # Build data loader
    _, val_loader = build_dataloaders(config)

    # Collect alpha and depth values
    alphas = []
    depths = []

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= args.num_samples:
                break

            images = batch['image'].to(device)
            outputs = model(images, return_depth=True, return_alpha=True)

            alpha = outputs['alpha_maps']['stage2'].cpu().numpy().flatten()
            depth = outputs['depth_640'].cpu().numpy().flatten()

            alphas.extend(alpha.tolist())
            depths.extend(depth.tolist())

            count += images.shape[0]

    alphas = np.array(alphas)
    depths = np.array(depths)

    # Create scatter plot with density coloring
    fig, ax = plt.subplots(figsize=(8, 6))

    # Hexbin for density
    hb = ax.hexbin(depths, alphas, gridsize=50, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')

    # Add trend line
    z = np.polyfit(depths, alphas, 1)
    p = np.poly1d(z)
    x_line = np.linspace(depths.min(), depths.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    ax.set_xlabel('Estimated Depth (m)', fontsize=12)
    ax.set_ylabel('FSG Alpha Value', fontsize=12)
    ax.set_title('Alpha Map vs. Depth: DG-FSG learns to trust depth cues', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.output}")


if __name__ == '__main__':
    main()
