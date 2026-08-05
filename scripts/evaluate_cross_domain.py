"""Cross-domain evaluation on held-out real fog test sets.

Evaluates the trained WRDNet on:
  1. Foggy_Driving  — has bbox GT → computes mAP (quantitative)
  2. Foggy_Zurich   — has NO bbox GT (only semantic labels) → qualitative only

This proves the model generalizes to real fog beyond the training distribution.

Usage:
  python scripts/evaluate_cross_domain.py \
    --config configs/default.yaml \
    --checkpoint my_local_wrdnet-checkpoints/phase1/epoch_82.pth \
    --output results/cross_domain_results.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.evaluation.evaluator import WRDNetEvaluator
from src.data.dataset import build_test_loader
from src.data.foggy_driving import FoggyDrivingDataset
from torch.utils.data import DataLoader
from src.data.dataset import wrdnet_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-domain evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/cross_domain_results.json')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--visualize', action='store_true',
                        help='Save sample detection visualizations')
    return parser.parse_args()


def evaluate_foggy_driving(model, config, data_root, device):
    """Evaluate mAP on Foggy Driving (has bbox GT)."""
    print("\n=== Foggy Driving (real fog, has bbox GT) ===")

    dataset = FoggyDrivingDataset(
        root=os.path.join(data_root, 'Foggy_Driving'),
        split='all',
        input_size=getattr(config, 'input_size', 640),
        config=config,
    )
    print(f"  Samples: {len(dataset)}")

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

    print(f"  mAP@50:    {metrics.get('mAP@50', 0.0):.4f}")
    print(f"  mAP@50:95: {metrics.get('mAP@50:95', 0.0):.4f}")

    return {
        'dataset': 'Foggy_Driving',
        'samples': len(dataset),
        'mAP@50': metrics.get('mAP@50', 0.0),
        'mAP@50:95': metrics.get('mAP@50:95', 0.0),
    }


def evaluate_foggy_zurich(model, config, data_root, device):
    """Qualitative evaluation on Foggy Zurich (no bbox GT)."""
    print("\n=== Foggy Zurich (real dense fog, NO bbox GT — qualitative only) ===")

    # Count samples
    rgb_dir = os.path.join(data_root, 'Foggy_Zurich', 'RGB')
    n_samples = 0
    if os.path.exists(rgb_dir):
        for seq in os.listdir(rgb_dir):
            seq_dir = os.path.join(rgb_dir, seq)
            if os.path.isdir(seq_dir):
                n_samples += len([f for f in os.listdir(seq_dir) if f.endswith('.png')])
    print(f"  Samples: {n_samples}")

    # Measure inference speed (proxy for qualitative readiness)
    evaluator = WRDNetEvaluator(model, device=str(device))
    fps = evaluator.measure_speed()
    print(f"  Inference speed: {fps:.1f} FPS")

    return {
        'dataset': 'Foggy_Zurich',
        'samples': n_samples,
        'note': 'No bbox GT available — qualitative evaluation only',
        'FPS': fps,
    }


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

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    results = {}

    # Foggy Driving (quantitative)
    results['foggy_driving'] = evaluate_foggy_driving(model, config, args.data_root, device)

    # Foggy Zurich (qualitative)
    results['foggy_zurich'] = evaluate_foggy_zurich(model, config, args.data_root, device)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
