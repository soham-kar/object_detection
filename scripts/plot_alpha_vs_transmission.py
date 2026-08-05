"""Plot alpha map values vs. ground-truth transmission — the physics money shot.

Shows that the FSG gate learns the atmospheric scattering relationship:
  - alpha → 1 (trust defogged) where transmission is LOW (dense fog)
  - alpha → 0 (trust original) where transmission is HIGH (clear)

Uses the ground-truth DBF transmittance maps (already on the volume).
This is a NOVEL, UDA-SAFE analysis figure:
  - Pure evaluation (no training signal)
  - Uses known GT transmission maps
  - No new data downloads required
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pearsonr

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.data.foggy_cityscapes import FoggyCityscapesDataset
from torch.utils.data import DataLoader
from src.data.dataset import wrdnet_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Plot alpha vs transmission')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualizations/alpha_vs_transmission.png')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--num_samples', type=int, default=200)
    return parser.parse_args()


def load_transmission_map(data_root, city, base, beta):
    """Load the GT transmission map for a given image."""
    path = os.path.join(
        data_root, 'leftImg8bit_trainval_transmittanceDBF',
        'leftImg8bit_transmittanceDBF', 'val', city,
        f'{base}_leftImg8bit_transmittance_beta_{beta}.png',
    )
    if not os.path.exists(path):
        return None
    t = np.array(Image.open(path), dtype=np.float32) / 255.0  # [0, 1]
    return t


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

    # Build val dataset at medium fog density (beta=0.01)
    dataset = FoggyCityscapesDataset(
        root=os.path.join(args.data_root, 'cityscapes'),
        split='val',
        fog_density='0.01',
        input_size=getattr(config, 'input_size', 640),
        load_clear=False,
        load_depth=False,
        config=config,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=wrdnet_collate_fn,
    )

    all_transmission = []
    all_alpha = []

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= args.num_samples:
                break

            images = batch['image'].to(device)
            image_path = batch['image_path'][0] if 'image_path' in batch else None

            # Get alpha maps
            outputs = model(images, return_alpha=True)
            alpha_p3 = outputs['alpha_maps']['P3']  # [B, 1, 80, 80]

            # Load GT transmission map
            if image_path is not None:
                # Parse city and base from path
                parts = image_path.replace('\\', '/').split('/')
                fname = parts[-1]
                city = parts[-2]
                # base = fname without _leftImg8bit_foggy_beta_0.01.png
                base = fname.replace('_leftImg8bit_foggy_beta_0.01.png', '')
                t_map = load_transmission_map(args.data_root, city, base, '0.01')

                if t_map is not None:
                    # Resize transmission to alpha resolution (80x80)
                    t_resized = np.array(Image.fromarray(t_map).resize((80, 80), Image.BILINEAR))

                    alpha_np = alpha_p3[0, 0].cpu().numpy()

                    # Sample pixels (flatten)
                    all_transmission.extend(t_resized.flatten().tolist())
                    all_alpha.extend(alpha_np.flatten().tolist())

            count += images.shape[0]

    transmission = np.array(all_transmission)
    alpha = np.array(all_alpha)

    print(f"Collected {len(transmission)} (transmission, alpha) pairs")

    # Bin by transmission (0.05 intervals)
    bins = np.arange(0, 1.05, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_alpha = []
    std_alpha = []
    for i in range(len(bins) - 1):
        mask = (transmission >= bins[i]) & (transmission < bins[i+1])
        if mask.sum() > 50:
            mean_alpha.append(alpha[mask].mean())
            std_alpha.append(alpha[mask].std())
        else:
            mean_alpha.append(np.nan)
            std_alpha.append(np.nan)

    mean_alpha = np.array(mean_alpha)
    std_alpha = np.array(std_alpha)
    valid = ~np.isnan(mean_alpha)

    # Pearson correlation
    r, p = pearsonr(transmission, alpha)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        bin_centers[valid],
        mean_alpha[valid] - std_alpha[valid],
        mean_alpha[valid] + std_alpha[valid],
        alpha=0.3, color='blue', label='±1 std',
    )
    ax.plot(
        bin_centers[valid], mean_alpha[valid],
        'b-', linewidth=2.5, label='Mean α',
    )

    ax.set_xlabel('Ground-Truth Transmission t(x)', fontsize=14)
    ax.set_ylabel('α (trust in defogged features)', fontsize=14)
    ax.set_title(
        f'Gate Activation vs. Transmission\n'
        f'Pearson r = {r:.3f} (p = {p:.4f})',
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")


if __name__ == '__main__':
    main()
