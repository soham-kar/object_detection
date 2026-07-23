"""Multi-Angle Attention Module (MAA).

Inspired by TCL-Net (Tang et al., ACCV 2024).
Applied to DehazeFormer encoder stages 1-2 only.

Fixed differential kernels in 5 directions:
  - Horizontal (Sobel-X): detects vertical edges
  - Vertical (Sobel-Y): detects horizontal edges
  - Diagonal /: detects diagonal edges
  - Diagonal \\: detects anti-diagonal edges
  - Center-surround (Laplacian): detects blob-like patterns

Learned components:
  - Per-direction importance weights
  - Spatial attention gate for fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAngleAttention(nn.Module):
    """
    Multi-Angle Attention with fixed differential kernels.

    Args:
        channels: number of input channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.num_directions = 5

        # Fixed differential kernels (non-learnable)
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))  # [1, 1, 3, 3]

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('diag1', torch.tensor([
            [[0, 0, 1],
             [0, -1, 0],
             [1, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('diag2', torch.tensor([
            [[1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('laplacian', torch.tensor([
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        ], dtype=torch.float32).unsqueeze(0))

        # Learnable direction importance weights
        self.direction_weights = nn.Parameter(torch.ones(self.num_directions))

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * self.num_directions, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Spatial attention gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _apply_kernel(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply a kernel to each channel independently."""
        B, C, H, W = x.shape
        # Expand kernel for all channels
        kernel_expanded = kernel.repeat(C, 1, 1, 1)  # [C, 1, 3, 3]
        return F.conv2d(x, kernel_expanded, padding=1, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced: [B, C, H, W]
        """
        # Apply fixed kernels
        responses = [
            self._apply_kernel(x, self.sobel_x),
            self._apply_kernel(x, self.sobel_y),
            self._apply_kernel(x, self.diag1),
            self._apply_kernel(x, self.diag2),
            self._apply_kernel(x, self.laplacian),
        ]

        # Weight by learnable direction weights
        weights = F.softmax(self.direction_weights, dim=0)
        weighted = [w * r for w, r in zip(weights, responses)]

        # Concatenate all direction responses
        concat = torch.cat(weighted, dim=1)  # [B, 5*C, H, W]

        # Fuse
        fused = self.fusion_conv(concat)

        # Spatial attention
        gate = self.spatial_gate(fused)
        out = fused * gate

        # Residual connection
        return x + out
