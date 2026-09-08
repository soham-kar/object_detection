"""Plot alpha map values vs. depth — the "money shot" figure."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# DehazeFormer module (cloned to /tmp/DehazeFormer on Modal)
sys.path.insert(0, '/tmp/DehazeFormer')

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
    parser.add_argument('--data_output', type=str, default=None,
                        help='Optional .npz path to save the raw (depths, alphas) arrays')
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

            # Alpha maps use keys 'P3', 'P4', 'P5' (not 'stage2'). Use P3
            # (highest resolution) for the alpha-vs-depth correlation.
            alpha_maps = outputs['alpha_maps']
            if isinstance(alpha_maps, dict):
                alpha_key = 'P3' if 'P3' in alpha_maps else list(alpha_maps.keys())[0]
                alpha = alpha_maps[alpha_key]  # [1, 1, H_a, W_a]
            else:
                alpha = alpha_maps
            # The model's forward returns depth under key 'depth' (not 'depth_640').
            # Depth is full-resolution; resize it to match alpha's spatial size so
            # the two arrays have the same number of elements.
            depth = outputs['depth']  # [1, 1, H_d, W_d]
            if depth.shape[2:] != alpha.shape[2:]:
                depth = torch.nn.functional.interpolate(
                    depth, size=alpha.shape[2:], mode='bilinear', align_corners=False
                )
            alpha = alpha.cpu().numpy().flatten()
            depth = depth.cpu().numpy().flatten()

            alphas.extend(alpha.tolist())
            depths.extend(depth.tolist())

            count += images.shape[0]

    alphas = np.array(alphas)
    depths = np.array(depths)

    # Save raw arrays for exact downstream statistics (Pearson/Spearman, CI).
    if args.data_output:
        os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
        np.savez(args.data_output, depths=depths, alphas=alphas)
        print(f"Saved data arrays to {args.data_output}")

    # Exact statistics on the raw data (not read off the plot).
    from scipy import stats
    pearson_r, pearson_p = stats.pearsonr(depths, alphas)
    spearman_r, spearman_p = stats.spearmanr(depths, alphas)
    z = np.polyfit(depths, alphas, 1)
    slope = z[0]
    # 95% CI on the slope via bootstrap (deterministic seed for reproducibility).
    rng = np.random.default_rng(0)
    n = len(depths)
    boot_slopes = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boot_slopes.append(np.polyfit(depths[idx], alphas[idx], 1)[0])
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    print(f"Pearson  r = {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"Slope = {slope:.4f}  [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")

    # Create scatter plot with density coloring
    fig, ax = plt.subplots(figsize=(8, 6))

    # Hexbin for density
    hb = ax.hexbin(depths, alphas, gridsize=50, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')

    # Add trend line
    p = np.poly1d(z)
    x_line = np.linspace(depths.min(), depths.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=2,
            label=f'Trend (slope={slope:.4f}, r={pearson_r:.3f})')

    # Depth is produced by a Sigmoid head, so it is normalized to [0, 1],
    # NOT metric meters. Label the axis honestly.
    ax.set_xlabel('Estimated Relative Depth (normalized)', fontsize=12)
    ax.set_ylabel('FSG Alpha Value', fontsize=12)
    ax.set_title('Alpha Map vs. Depth: DG-FSG learns to trust depth cues', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.output}")


if __name__ == '__main__':
    main()
