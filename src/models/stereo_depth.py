
"""Stereo Depth Module for WRDNet — UDA-Compatible Depth Estimation.

DESIGN (per mentor consensus):
  - Source supervision: Cityscapes disparity GT (SGM)
  - Target supervision: ACDC photometric consistency (SSIM + L1) on stereo pairs
  - Integration: feed predicted depth into DG-FSG

WHY STEREO (vs. current monocular bottleneck decoder):
  - Monocular depth from a single image is ambiguous and noisy in fog
  - Stereo matching uses geometric cues (disparity) that are more reliable
  - ACDC provides stereo pairs → enables self-supervised photometric loss on target
  - Depth transfers across domains (a car is 5m in both Cityscapes and ACDC)

ARCHITECTURE:
  left_img ──┐
             ├── FeatureNet (shared) ──→ L_feat, R_feat
  right_img ─┘
                     │
                     ▼
              StereoCostVolume(L_feat, R_feat, max_disp)
                     │
                     ▼
              StereoDepthDecoder(cost_volume) → disparity → depth
                     │
                     ▼
              DepthEncoder → DG-FSG (existing)

LOSSES:
  L_source = SILog(depth_pred, depth_gt)          # Cityscapes disparity GT
  L_target = SSIM + L1 photometric consistency    # ACDC left/right pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoFeatureNet(nn.Module):
    """
    Shared feature extractor for left/right images.
    Produces multi-scale features for cost volume construction.

    Args:
        in_channels: input image channels (3)
        base_channels: base feature channels (default 32)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] image
        Returns:
            features: [B, 4*base, H/4, W/4] downsampled features
        """
        x = self.conv1(x)   # [B, 32, H, W]
        x = F.max_pool2d(x, 2)  # [B, 32, H/2, W/2]
        x = self.conv2(x)   # [B, 64, H/2, W/2]
        x = F.max_pool2d(x, 2)  # [B, 64, H/4, W/4]
        x = self.conv3(x)   # [B, 128, H/4, W/4]
        return x


class StereoCostVolume(nn.Module):
    """
    Build a correlation cost volume from left/right features.

    For each disparity d in [0, max_disp), shift right features by d
    and compute correlation with left features.

    Args:
        max_disp: maximum disparity (default 96 at 1/4 resolution)
    """

    def __init__(self, max_disp: int = 96):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left_feat: [B, C, H, W] left image features
            right_feat: [B, C, H, W] right image features
        Returns:
            cost_volume: [B, max_disp, H, W] correlation volume
        """
        B, C, H, W = left_feat.shape
        cost_volume = torch.zeros(B, self.max_disp, H, W, device=left_feat.device)

        for d in range(self.max_disp):
            if d == 0:
                cost_volume[:, d] = (left_feat * right_feat).mean(dim=1)
            else:
                # Shift right features left by d, then slice to original width
                shifted = F.pad(right_feat[:, :, :, d:], (0, d, 0, 0))[:, :, :, :W]
                cost_volume[:, d] = (left_feat * shifted).mean(dim=1)

        return cost_volume


class StereoDepthDecoder(nn.Module):
    """
    Decode disparity from the cost volume, then convert to depth.

    Args:
        cost_channels: number of disparity levels (max_disp)
        output_size: final depth resolution (default 640)
    """

    def __init__(self, cost_channels: int = 96, output_size: int = 640):
        super().__init__()
        self.output_size = output_size

        # 3D conv to aggregate cost volume
        self.cost_agg = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Squeeze disparity dimension → disparity map
        self.disparity_head = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # normalized disparity [0, 1]
        )

    def forward(self, cost_volume: torch.Tensor) -> tuple:
        """
        Args:
            cost_volume: [B, max_disp, H, W]
        Returns:
            disparity: [B, 1, H, W] normalized
            depth: [B, 1, output_size, output_size]
        """
        # Add channel dim for 3D conv: [B, 1, D, H, W]
        cv = cost_volume.unsqueeze(1)
        cv = self.cost_agg(cv)  # [B, 32, D, H, W]

        # Squeeze disparity: soft-argmax over disparity dimension
        # cv: [B, 32, D, H, W] → mean over channels → [B, D, H, W]
        scores = cv.mean(dim=1)  # [B, D, H, W]
        weights = F.softmax(scores, dim=1)  # [B, D, H, W]
        disp_idx = torch.arange(cost_volume.shape[1], device=cost_volume.device) \
            .float().view(1, -1, 1, 1)  # [1, D, 1, 1]
        disparity = (weights * disp_idx).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        disparity = disparity / cost_volume.shape[1]  # normalize to [0, 1]

        # Refine with 2D conv (input is 1 channel disparity)
        disparity = self.disparity_head(disparity)  # [B, 1, H, W]

        # Convert disparity to depth (inverse, normalized)
        depth = 1.0 / (disparity + 1e-6)
        depth = depth / depth.max()  # normalize to [0, 1]

        # Upsample to full resolution
        depth_full = F.interpolate(depth, size=(self.output_size, self.output_size),
                                   mode='bilinear', align_corners=False)
        return disparity, depth_full


class StereoDepthModule(nn.Module):
    """
    Complete stereo depth module: features → cost volume → depth.

    Args:
        max_disp: maximum disparity
        output_size: final depth resolution
    """

    def __init__(self, max_disp: int = 96, output_size: int = 640):
        super().__init__()
        self.feature_net = StereoFeatureNet()
        self.cost_volume = StereoCostVolume(max_disp=max_disp)
        self.decoder = StereoDepthDecoder(cost_channels=max_disp, output_size=output_size)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        """
        Args:
            left_img: [B, 3, H, W] left image
            right_img: [B, 3, H, W] right image
        Returns:
            disparity: [B, 1, H/4, W/4] normalized
            depth: [B, 1, output_size, output_size]
        """
        left_feat = self.feature_net(left_img)   # [B, 128, H/4, W/4]
        right_feat = self.feature_net(right_img)  # [B, 128, H/4, W/4]

        cost_vol = self.cost_volume(left_feat, right_feat)  # [B, D, H/4, W/4]
        disparity, depth = self.decoder(cost_vol)

        return disparity, depth


def photometric_loss(left_img: torch.Tensor, right_img: torch.Tensor,
                     disparity: torch.Tensor) -> torch.Tensor:
    """
    Self-supervised photometric consistency loss for target domain.

    Warps the right image to the left using predicted disparity, then
    computes SSIM + L1 between the warped right and the original left.

    Args:
        left_img: [B, 3, H, W] left image
        right_img: [B, 3, H, W] right image
        disparity: [B, 1, H, W] predicted disparity (normalized [0,1])
    Returns:
        loss: scalar photometric loss
    """
    # Convert normalized disparity to pixel shift
    H, W = left_img.shape[-2:]
    # Upsample disparity to image resolution
    disp_up = F.interpolate(disparity, size=(H, W), mode='bilinear', align_corners=False)
    disp_px = disp_up * W  # [B, 1, H, W] in pixels

    # Build sampling grid for warping right → left
    # x_new = x - disp (shift left by disparity)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=left_img.device),
        torch.linspace(-1, 1, W, device=left_img.device),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(left_img.shape[0], -1, -1)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(left_img.shape[0], -1, -1)

    # Shift x by disparity (normalized to [-1, 1])
    shift = (disp_px.squeeze(1) / W) * 2  # [B, H, W] in [-2, 2]
    grid_x_warped = grid_x - shift

    grid = torch.stack([grid_x_warped, grid_y], dim=-1)  # [B, H, W, 2]

    # Warp right image to left
    warped_right = F.grid_sample(right_img, grid, mode='bilinear',
                                 padding_mode='border', align_corners=False)

    # SSIM + L1 loss
    l1 = (left_img - warped_right).abs().mean()

    # Simple SSIM (luminance + contrast)
    mu_x = F.avg_pool2d(left_img, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(warped_right, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(left_img ** 2, 3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(warped_right ** 2, 3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(left_img * warped_right, 3, stride=1, padding=1) - mu_x * mu_y

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    ssim_loss = (1 - ssim).mean()

    return 0.85 * ssim_loss + 0.15 * l1
