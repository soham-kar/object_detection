"""Metrics computation for WRDNet."""

import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: predicted image [B, 3, H, W]
        target: target image [B, 3, H, W]
        max_val: maximum pixel value
    Returns:
        psnr: PSNR in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (simplified).

    Args:
        pred: predicted image [B, 3, H, W]
        target: target image [B, 3, H, W]
        window_size: SSIM window size
    Returns:
        ssim: SSIM score
    """
    # Simplified SSIM — for accurate results, use skimage.metrics.structural_similarity
    # or pytorch-msssim
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(pred, window_size, 1, padding=window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred ** 2, window_size, 1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target ** 2, window_size, 1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> dict:
    """
    Compute depth estimation metrics.

    Args:
        pred_depth: predicted depth [B, 1, H, W]
        gt_depth: ground-truth depth [B, 1, H, W]
    Returns:
        metrics: dict with RMSE, AbsRel, delta thresholds
    """
    # Mask valid pixels
    mask = gt_depth > 0
    pred = pred_depth[mask]
    gt = gt_depth[mask]

    if pred.numel() == 0:
        return {'RMSE': 0.0, 'AbsRel': 0.0, 'delta_1.25': 0.0}

    # RMSE
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()

    # AbsRel
    abs_rel = torch.mean(torch.abs(pred - gt) / gt).item()

    # Delta < 1.25
    ratio = torch.max(pred / gt, gt / pred)
    delta_125 = (ratio < 1.25).float().mean().item()

    return {
        'RMSE': rmse,
        'AbsRel': abs_rel,
        'delta_1.25': delta_125,
    }
