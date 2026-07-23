"""Main training loop for WRDNet."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.wrnet import WRDNet
from ..utils.config import Config
from .losses import WRDNetLoss
from .optimizer import build_optimizer, build_scheduler


class WRDNetTrainer:
    """Trainer for WRDNet."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = WRDNet(config).to(self.device)

        # Loss
        self.criterion = WRDNetLoss(config)

        # Optimizer and scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Logging
        self.log_dir = getattr(config, 'log_dir', 'experiments/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Checkpointing
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', 'experiments/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_interval = getattr(config, 'save_interval', 5)

        # Early stopping
        self.early_stopping = getattr(config, 'early_stopping', False)
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
        self.early_stopping_metric = getattr(config, 'early_stopping_metric', 'mAP@50')
        self.best_metric = 0.0
        self.patience_counter = 0

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop.

        Args:
            train_loader: training data loader
            val_loader: validation data loader (optional)
        """
        epochs = getattr(self.config, 'epochs', 100)
        log_interval = getattr(self.config, 'log_interval', 100)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch(train_loader, log_interval)

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(val_loader)

                # Early stopping
                if self.early_stopping:
                    current_metric = val_metrics.get(self.early_stopping_metric, 0.0)
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.patience_counter = 0
                        self._save_checkpoint('best.pth')
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.early_stopping_metric, 0.0))
                else:
                    self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pth')

        self.writer.close()

    def _train_epoch(self, train_loader: DataLoader, log_interval: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Handle paired (synth, real) batches from PairedDADataset
            if 'synth' in batch and 'real' in batch:
                synth_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch['synth'].items()}
                real_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch['real'].items()}
            else:
                # Single dataset mode — move to device
                synth_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
                real_batch = None

            # Forward pass
            outputs = self.model.forward_train(synth_batch, real_batch)

            # Compute loss
            losses = self.criterion(outputs, batch)
            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            self.global_step += 1

            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                })
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> dict:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle paired format if present
                if 'synth' in batch:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch['synth'].items()}
                else:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                outputs = self.model.forward_train(batch)
                losses = self.criterion(outputs, batch)
                total_loss += losses['total'].item()

        avg_loss = total_loss / len(val_loader)
        metrics = {'val_loss': avg_loss, 'mAP@50': 0.0}  # TODO: Compute actual mAP

        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)

        return metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        print(f"Loaded checkpoint from {path}")
