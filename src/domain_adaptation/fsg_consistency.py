"""FSG Consistency Loss.

Novel domain adaptation signal: the model's own gating decisions must be
consistent across domains for images with similar fog density.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_fog_density(dehazeformer, image: torch.Tensor) -> torch.Tensor:
    """
    Estimate fog density from DehazeFormer bottleneck features.

    The bottleneck encodes fog density implicitly through the restoration task.
    Feature activation magnitude correlates with fog density.

    Bottleneck shape: [B, 96, 80, 80] for DehazeFormer-T at 320×320 input.
    """
    with torch.no_grad():
        bottleneck = dehazeformer.get_bottleneck(image)  # [B, 96, 80, 80]
        density = bottleneck.norm(dim=[1, 2, 3])
        density = density / (density.max() + 1e-8)
    return density


def fsg_consistency_loss(
    alpha_synth: dict,
    alpha_real: dict,
    density_synth: torch.Tensor,
    density_real: torch.Tensor,
    threshold: float = 0.15,
) -> torch.Tensor:
    """
    FSG Consistency Loss with soft weighting based on density similarity.

    Args:
        alpha_synth: dict with keys 'P3', 'P4', 'P5', each [B, 1, H, W]
        alpha_real: dict with keys 'P3', 'P4', 'P5', each [B, 1, H, W]
        density_synth: [B_s] fog density estimates
        density_real: [B_r] fog density estimates
        threshold: max density difference for matching
    Returns:
        consistency_loss: scalar
    """
    B_s = density_synth.shape[0]
    B_r = density_real.shape[0]

    # Compute density difference matrix [B_s, B_r]
    density_diff = torch.abs(
        density_synth.unsqueeze(1) - density_real.unsqueeze(0)
    )

    # Soft weighting: Gaussian kernel
    weights = torch.exp(-density_diff ** 2 / (2 * threshold ** 2))

    # Compute alpha difference for all scales
    total_loss = 0.0
    for scale_name in ['P3', 'P4', 'P5']:
        a_s = alpha_synth[scale_name]  # [B_s, 1, H, W]
        a_r = alpha_real[scale_name]   # [B_r, 1, H, W]

        # Flatten spatial dimensions
        a_s_flat = a_s.view(B_s, -1)  # [B_s, H*W]
        a_r_flat = a_r.view(B_r, -1)  # [B_r, H*W]

        # Pairwise MSE [B_s, B_r]
        alpha_diff = torch.cdist(a_s_flat, a_r_flat, p=2) ** 2
        alpha_diff = alpha_diff / a_s_flat.shape[1]  # Normalize by spatial size

        # Weighted average
        scale_loss = (alpha_diff * weights).sum() / (weights.sum() + 1e-8)
        total_loss += scale_loss

    return total_loss / 3.0  # Average over scales
