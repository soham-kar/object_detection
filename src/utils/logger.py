"""Logging and checkpointing utilities."""

import os
import json
from datetime import datetime

import torch


class Logger:
    """Simple logger for training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.entries = []

    def log(self, metrics: dict, step: int, epoch: int):
        """Log metrics."""
        entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics,
        }
        self.entries.append(entry)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        filename: str,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str,
    ) -> int:
        """Load model checkpoint. Returns epoch."""
        path = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
