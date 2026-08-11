"""Lightweight Depth Decoder for WRDNet.

Attached to DehazeFormer-T's bottleneck (Stage 3, stride 4).
Bottleneck: [B, 96, 80, 80] for 320×320 input.
Produces metric depth maps at 160×160 (upsampled to 640×640).

WHY THIS WORKS: DehazeFormer already learns depth implicitly through the
atmospheric scattering model. The bottleneck contains transmission map
information — the depth decoder just makes it explicit.

Reference: DPT (Ranftl et al., ICCV 2021), MiDaS (Ranftl et al., TPAMI 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthDecoder(nn.Module):
    """
    Progressive upsampling depth decoder.

    Bottleneck: [B, 96, H/4, W/4] → H/2×W/2 → H×W (dynamic)

    Args:
        bottleneck_channels: channels from DehazeFormer bottleneck (default 96 for T variant)
        output_size: final upsampling target (default 640 for YOLO input)
    """

    def __init__(self, bottleneck_channels: int = 96, output_size: int = 640):
        super().__init__()

        # Normalize to (H, W) tuple to support 2:1 aspect ratio
        if isinstance(output_size, (list, tuple)):
            self.output_size = tuple(output_size)
        else:
            self.output_size = (output_size, output_size)

        # Stage 1: bottleneck → 2× upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 160×160 → 160×160 (refinement)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Output: 160×160 → 1 channel depth
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalized depth [0, 1]
        )

    def forward(self, bottleneck: torch.Tensor) -> tuple:
        """
        Args:
            bottleneck: [B, 96, 80, 80] from DehazeFormer-T Stage 3
        Returns:
            depth_160: [B, 1, 160, 160] normalized depth map
            depth_640: [B, 1, 640, 640] upsampled to full resolution
        """
        x = self.up1(bottleneck)     # [B, 64, 160, 160]
        x = self.refine(x)           # [B, 16, 160, 160]
        depth_160 = self.output(x)   # [B, 1, 160, 160]
        depth_640 = F.interpolate(depth_160, size=self.output_size,
                                  mode='bilinear', align_corners=False)
        return depth_160, depth_640
