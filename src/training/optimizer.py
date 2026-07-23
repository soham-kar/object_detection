"""Optimizer and scheduler utilities for WRDNet."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


def build_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """
    Build optimizer based on config.

    Args:
        model: WRDNet model
        config: Config with optimizer settings
    Returns:
        optimizer: PyTorch optimizer
    """
    lr = getattr(config, 'lr', 1e-3)
    weight_decay = getattr(config, 'weight_decay', 1e-4)
    optimizer_name = getattr(config, 'optimizer', 'AdamW')

    # Separate parameters with/without weight decay
    # Typically: biases and BN parameters don't use weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name or 'batch_norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(param_groups, lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Config with scheduler settings
    Returns:
        scheduler: PyTorch LR scheduler
    """
    scheduler_name = getattr(config, 'scheduler', 'cosine')
    epochs = getattr(config, 'epochs', 100)
    warmup_epochs = getattr(config, 'warmup_epochs', 5)

    if scheduler_name == 'cosine':
        # Cosine annealing with warmup
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            # Use SequentialLR for warmup + cosine
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    else:
        scheduler = None

    return scheduler
