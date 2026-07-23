"""Training script for WRDNet."""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.training.trainer import WRDNetTrainer
from src.data.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train WRDNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Build data loaders
    train_loader, val_loader = build_dataloaders(config)

    # Create trainer
    trainer = WRDNetTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
