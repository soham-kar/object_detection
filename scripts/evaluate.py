"""Evaluation script for WRDNet."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.evaluation.evaluator import WRDNetEvaluator
from src.data.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WRDNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Build model
    model = WRDNet(config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Build data loaders
    _, val_loader = build_dataloaders(config)

    # Evaluate
    evaluator = WRDNetEvaluator(model)

    # Detection metrics
    det_metrics = evaluator.evaluate_detection(val_loader)
    print("Detection Metrics:")
    for k, v in det_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Restoration metrics
    rest_metrics = evaluator.evaluate_restoration(val_loader)
    print("\nRestoration Metrics:")
    for k, v in rest_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Speed
    fps = evaluator.measure_speed()
    print(f"\nInference Speed: {fps:.2f} FPS")


if __name__ == '__main__':
    main()
