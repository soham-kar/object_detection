"""Depth-Guided Feature Selection Gate (DG-FSG) — PRIMARY DEPTH INNOVATION.

Extends the standard FSG with depth awareness. The estimated depth map is
encoded and fed as a THIRD input to the gate, allowing fusion decisions
that are aware of object distance.

CORE INSIGHT: Fog severity increases exponentially with depth (per the
atmospheric scattering model). Close objects need almost no defogging;
distant objects are invisible without it. The DG-FSG learns this physical
relationship from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DepthEncoder(nn.Module):
    """
    Minimal encoder for depth maps.
    Converts [B, 1, 160, 160] depth into [B, C_d, H, W] feature representation.

    Parameters: ~0.01M
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_map: [B, 1, 160, 160] from DepthDecoder
        Returns:
            d_encoded: [B, 16, 160, 160] encoded depth features
        """
        x = self.conv1(depth_map)
        x = self.conv2(x)
        return x


class DepthGuidedFSG(nn.Module):
    """
    Depth-Guided Feature Selection Gate.

    Applied at 3 detection scales (P3, P4, P5).
    At each scale, the depth encoding is resized to match.
    """

    def __init__(self, channels_list: list, depth_channels: int = 16, use_cdmsa: bool = True):
        super().__init__()
        self.channels_list = channels_list
        self.depth_channels = depth_channels
        self.use_cdmsa = use_cdmsa

        self.depth_encoder = DepthEncoder(out_channels=depth_channels)

        # One DG-FSG gate per scale
        self.gates = nn.ModuleList()
        for i, ch in enumerate(channels_list):
            prev_ch = channels_list[i - 1] if i > 0 else None
            gate = self._make_gate(ch, depth_channels, prev_ch)
            self.gates.append(gate)

    def _make_gate(self, channels: int, depth_channels: int, prev_channels: int = None) -> nn.Module:
        """Create a single DG-FSG gate for one scale."""
        from .cdmsa import CrossDimensionalMSA

        cdmsa = CrossDimensionalMSA(channels, prev_channels=prev_channels) if self.use_cdmsa else nn.Identity()

        # Gating network: 2C + C_d input channels (includes depth)
        gating_net = nn.Sequential(
            nn.Conv2d(2 * channels + depth_channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # alpha in [0, 1]
        )

        return nn.ModuleDict({
            'cdmsa': cdmsa,
            'gating': gating_net,
        })

    def forward(
        self,
        restored_features: Dict[str, torch.Tensor],
        original_features: Dict[str, torch.Tensor],
        depth_map: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            restored_features: dict with keys 'P3', 'P4', 'P5'
            original_features: dict with keys 'P3', 'P4', 'P5'
            depth_map: [B, 1, 160, 160] from DepthDecoder
        Returns:
            fused_features: dict with keys 'P3', 'P4', 'P5'
            alpha_maps: dict with keys 'P3', 'P4', 'P5'
        """
        # Encode depth once
        d_encoded = self.depth_encoder(depth_map)  # [B, 16, 160, 160]

        fused_features = {}
        alpha_maps = {}

        scale_names = ['P3', 'P4', 'P5']
        for i, name in enumerate(scale_names):
            f_rest = restored_features[name]
            f_orig = original_features[name]

            # Ensure spatial sizes match
            if f_rest.shape[2:] != f_orig.shape[2:]:
                f_rest = F.interpolate(f_rest, size=f_orig.shape[2:],
                                       mode='bilinear', align_corners=False)

            # Resize depth encoding to match this scale
            d_resized = F.interpolate(
                d_encoded,
                size=f_rest.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            # Apply CDMSA for cross-scale context (enhances gating input)
            gate_module = self.gates[i]
            if self.use_cdmsa and i > 0:
                prev_fused = fused_features[scale_names[i - 1]]
                _ = gate_module['cdmsa'](f_rest, f_orig, prev_fused)
            else:
                _ = gate_module['cdmsa'](f_rest, f_orig)

            # Concatenate with depth for depth-aware gating
            concat = torch.cat([f_rest, f_orig, d_resized], dim=1)

            # Gating
            alpha = gate_module['gating'](concat)

            # Fusion
            fused = alpha * f_rest + (1.0 - alpha) * f_orig

            fused_features[name] = fused
            alpha_maps[name] = alpha

        return fused_features, alpha_maps
