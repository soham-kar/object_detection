"""Visualize alpha maps and depth maps."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.data.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize WRDNet alpha maps')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/alpha_maps')
    parser.add_argument('--num_samples', type=int, default=10)
    return parser.parse_args()


def visualize_sample(image, alpha_map, depth_map, save_path):
    """Create visualization figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Input (Foggy)')
    axes[0].axis('off')

    # Alpha map
    alpha_np = alpha_map.cpu().squeeze().numpy()
    im = axes[1].imshow(alpha_np, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('FSG Alpha Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    # Depth map
    depth_np = depth_map.cpu().squeeze().numpy()
    im = axes[2].imshow(depth_np, cmap='plasma')
    axes[2].set_title('Depth Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and model
    config = load_config(args.config)
    model = WRDNet(config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Build data loader
    _, val_loader = build_dataloaders(config)

    # Visualize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= args.num_samples:
                break

            images = batch['image'].to(device)
            outputs = model(images, return_depth=True, return_alpha=True)

            for i in range(images.shape[0]):
                if count >= args.num_samples:
                    break

                save_path = os.path.join(args.output_dir, f'sample_{count:03d}.png')
                visualize_sample(
                    images[i],
                    outputs['alpha_maps']['stage2'],
                    outputs['depth_640'][i],
                    save_path,
                )
                count += 1

    print(f"Saved {count} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
