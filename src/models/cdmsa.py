"""Cross-Dimensional Multi-Scale Attention (CDMSA).

Inspired by YOLOv8s-WAMNet (Jaiswal et al., 2026).
Integrated into the FSG module.

Three attention dimensions:
  1. Channel attention: which channels are important?
  2. Spatial attention: which locations are important?
  3. Cross-scale attention: how do features interact across scales?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CrossDimensionalMSA(nn.Module):
    """
    Cross-Dimensional Multi-Scale Attention.

    Args:
        channels: number of input channels for current scale
        prev_channels: number of channels from previous scale (for cross-scale fusion)
                      If None, defaults to channels (same-scale mode)
    """

    def __init__(self, channels: int, prev_channels: int = None):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

        # Cross-scale fusion conv (projects previous scale to current scale)
        prev_ch = prev_channels if prev_channels is not None else channels
        self.cross_scale_conv = nn.Conv2d(prev_ch, channels, kernel_size=1, bias=False)

    def forward(
        self,
        restored_feat: torch.Tensor,
        original_feat: torch.Tensor,
        prev_scale_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            restored_feat: [B, C, H, W]
            original_feat: [B, C, H, W]
            prev_scale_feat: [B, C_prev, H_prev, W_prev] or None
        Returns:
            enhanced: [B, C, H, W] — enhanced combined feature for gating
        """
        # Combine restored and original
        combined = restored_feat + original_feat

        # Channel attention
        combined = self.channel_attn(combined)

        # Spatial attention
        combined = self.spatial_attn(combined)

        # Cross-scale attention (if previous scale available)
        if prev_scale_feat is not None:
            if prev_scale_feat.shape[2:] != combined.shape[2:]:
                prev_scale_feat = F.interpolate(
                    prev_scale_feat,
                    size=combined.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            cross = self.cross_scale_conv(prev_scale_feat)
            combined = combined + cross

        return combined
