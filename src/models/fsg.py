"""Feature Selection Gate (FSG) — Core contribution of WRDNet.

Learns per-pixel weights alpha in [0,1] to fuse restored and original features:
    F_fused = alpha * F_restored + (1-alpha) * F_original

Applied at 3 detection scales (P3, P4, P5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FeatureSelectionGate(nn.Module):
    """
    Feature Selection Gate with optional CDMSA.

    Args:
        channels_list: list of channel counts for P3, P4, P5
        use_cdmsa: whether to use Cross-Dimensional Multi-Scale Attention
    """

    def __init__(self, channels_list: list, use_cdmsa: bool = True):
        super().__init__()
        self.channels_list = channels_list
        self.use_cdmsa = use_cdmsa
        self.num_scales = len(channels_list)

        if self.use_cdmsa:
            from .cdmsa import CrossDimensionalMSA
            self.cdma_modules = nn.ModuleList()
            for i, ch in enumerate(channels_list):
                prev_ch = channels_list[i - 1] if i > 0 else None
                self.cdma_modules.append(CrossDimensionalMSA(ch, prev_channels=prev_ch))
        else:
            self.cdma_modules = nn.ModuleList([nn.Identity() for _ in channels_list])

        # Gating network per scale
        self.gates = nn.ModuleList()
        for ch in channels_list:
            gate = nn.Sequential(
                # Input: concatenated restored + original = 2C channels
                nn.Conv2d(2 * ch, ch // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, ch // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),  # alpha in [0, 1]
            )
            # Initialize the final alpha conv bias to -2.0 so Sigmoid(-2.0) ≈ 0.12.
            # This forces the gate to TRUST ORIGINAL features early in training,
            # preventing the detection loss from spiking on blurred restored features.
            nn.init.constant_(gate[-2].bias, -2.0)
            self.gates.append(gate)

        # Output BatchNorm per scale to normalize fused features before they
        # enter the YOLO neck. This stabilizes the feature magnitudes so the
        # DFL in the detection head doesn't saturate at the box-edge bins.
        # NOTE: The original YOLO backbone features also have magnitude ~7-10,
        # so this is a defensive normalization, not a guaranteed fix.
        self.output_bns = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in channels_list
        ])

        # Magnitude clamp on fused features before they enter the YOLO neck.
        # The FSG fusion (especially with CDMSA) can produce features with
        # magnitude 37-62, which shatters the DFL in the detection head
        # (x-coordinates collapse to the left edge). Clamping to a safe range
        # (~YOLO's expected magnitude of 8) prevents this while still letting
        # FSG learn adaptive fusion.
        self.fsg_clamp = 10.0

    def forward(
        self,
        restored_features: Dict[str, torch.Tensor],
        original_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            restored_features: dict with keys 'P3', 'P4', 'P5'
            original_features: dict with keys 'P3', 'P4', 'P5'
        Returns:
            fused_features: dict with keys 'P3', 'P4', 'P5'
            alpha_maps: dict with keys 'P3', 'P4', 'P5'
        """
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

            # Apply CDMSA to get enhanced combined feature for gating
            if self.use_cdmsa and i > 0:
                prev_fused = fused_features[scale_names[i - 1]]
                enhanced = self.cdma_modules[i](f_rest, f_orig, prev_fused)
            else:
                enhanced = self.cdma_modules[i](f_rest, f_orig)

            # Concatenate original features with enhanced context for gating
            concat = torch.cat([f_rest, f_orig], dim=1)

            # Compute alpha
            alpha = self.gates[i](concat)

            # Fuse
            fused = alpha * f_rest + (1.0 - alpha) * f_orig

            # Normalize fused features before they enter the YOLO neck.
            # Stabilizes magnitudes so the DFL doesn't saturate at box edges.
            fused = self.output_bns[i](fused)

            # Clamp to a safe magnitude range to prevent DFL collapse.
            # The CDMSA can produce spikes of 37-62; clamping to ±fsg_clamp
            # keeps features in YOLO's expected range (~8).
            fused = torch.clamp(fused, -self.fsg_clamp, self.fsg_clamp)

            fused_features[name] = fused
            alpha_maps[name] = alpha

        return fused_features, alpha_maps
