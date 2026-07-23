"""Training script for WRDNet.

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --resume experiments/checkpoints/best.pth
  python scripts/train.py --config configs/default.yaml --smoke-test  # Quick 5-batch test
"""

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
    parser.add_argument('--smoke-test', action='store_true', help='Run 5 batches only for quick verification')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.max_epochs is not None:
        config.training.epochs = args.max_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    # Smoke test: small batch, 1 epoch, few batches
    if args.smoke_test:
        config.training.epochs = 1
        config.training.batch_size = 2
        config.training.num_workers = 0
        config.use_fda = False
        config.use_dct_align = False
        config.use_fsg_consistency = False
        print("=" * 60)
        print("SMOKE TEST MODE: 1 epoch, batch_size=2, no DA")
        print("=" * 60)

    # Build data loaders
    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_dataloaders(config)

    # Create trainer
    print("\nCreating trainer...")
    trainer = WRDNetTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    # Smoke test: only run a few batches
    if args.smoke_test:
        print("\nRunning 5 training batches...")
        trainer.model.train()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:
                break
            if 'synth' in batch and 'real' in batch:
                synth_batch = trainer._move_to_device(batch['synth'])
                real_batch = trainer._move_to_device(batch['real'])
                loss_batch = synth_batch
            else:
                synth_batch = trainer._move_to_device(batch)
                real_batch = None
                loss_batch = synth_batch

            outputs = trainer.model.forward_train(synth_batch, real_batch)
            losses = trainer.criterion(outputs, loss_batch)
            loss = losses['total']

            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()

            print(f"  Batch {batch_idx}: total={loss.item():.4f}, "
                  f"det={losses['det'].item():.4f}, "
                  f"rest={losses['rest'].item():.4f}")

        print("\n✅ Smoke test PASSED! Model can train on real data.")
        return

    # Full training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
