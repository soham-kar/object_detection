"""DehazeFormer-T wrapper for WRDNet.

Integrates the actual DehazeFormer from external/DehazeFormer.
Extracts intermediate encoder features for FSG fusion.

Architecture (DehazeFormer-T at 320×320 input):
  Stage 1 (layer1):  [B, 24,  320, 320]  — 1× spatial
  Stage 2 (layer2):  [B, 48,  160, 160]  — 1/2× spatial
  Stage 3 (layer3):  [B, 96,   80,  80]  — 1/4× spatial (bottleneck)
  Stage 4 (layer4):  [B, 48,  160, 160]  — 1/2× spatial (decoder)
  Stage 5 (layer5):  [B, 24,  320, 320]  — 1× spatial (decoder)

For FSG, we use encoder stages 1-3 and project them to YOLO feature dims.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Add DehazeFormer to path
_DEHAZEFORMER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'DehazeFormer')
if os.path.isdir(_DEHAZEFORMER_PATH):
    if _DEHAZEFORMER_PATH not in sys.path:
        sys.path.insert(0, _DEHAZEFORMER_PATH)


class DehazeFormerWrapper(nn.Module):
    """
    Wrapper around DehazeFormer-T with WRDNet-specific modifications.

    - Runs at 320×320 for efficiency
    - Upsamples restored image to 640×640 for YOLO
    - Extracts encoder features (stage1-3) for FSG fusion
    - Projects DehazeFormer channels to YOLO-compatible dimensions
    - Applies MAA to stages 1-2 (optional)
    """

    def __init__(
        self,
        variant: str = 'T',
        pretrained: bool = True,
        input_size: int = 320,
        output_size: int = 640,
        use_maa: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant
        self.input_size = input_size
        self.output_size = output_size
        self.use_maa = use_maa

        # ── Build actual DehazeFormer ──
        self.dehazeformer = self._build_dehazeformer(variant)

        # Load pretrained weights if available
        if pretrained and pretrained_path is not None and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)

        # DehazeFormer-T channel dimensions
        # encoder: 24 → 48 → 96, decoder: 48 → 24
        self.enc_channels = [24, 48, 96]   # stage1, stage2, stage3 (bottleneck)
        self.bottleneck_channels = 96

        # ── Projection layers: DehazeFormer channels → YOLO feature dims ──
        # YOLOv11s backbone: P3=256ch, P4=512ch, P5=1024ch
        self.yolo_channels = [256, 256, 512]  # Verified from YOLOv11s backbone
        self.proj_stage1 = nn.Conv2d(24, 256, kernel_size=1)
        self.proj_stage2 = nn.Conv2d(48, 256, kernel_size=1)
        self.proj_stage3 = nn.Conv2d(96, 512, kernel_size=1)

        # ── MAA modules (optional, stages 1-2 only) ──
        if self.use_maa:
            from .maa import MultiAngleAttention
            self.maa_stage1 = MultiAngleAttention(24)
            self.maa_stage2 = MultiAngleAttention(48)
        else:
            self.maa_stage1 = nn.Identity()
            self.maa_stage2 = nn.Identity()

    def _build_dehazeformer(self, variant: str) -> nn.Module:
        """Build the actual DehazeFormer model."""
        try:
            from models import dehazeformer_t, dehazeformer_s, dehazeformer_b
        except ImportError:
            raise ImportError(
                "DehazeFormer not found. Clone it to external/DehazeFormer:\n"
                "  git clone https://github.com/IDKiro/DehazeFormer.git external/DehazeFormer"
            )

        variant_map = {
            'T': dehazeformer_t,
            'S': dehazeformer_s,
            'B': dehazeformer_b,
        }
        if variant not in variant_map:
            raise ValueError(f"Unknown DehazeFormer variant: {variant}. Choose from {list(variant_map.keys())}")

        return variant_map[variant]()

    def _load_pretrained(self, path: str):
        """Load pretrained DehazeFormer weights."""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'module.' prefix if trained with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.dehazeformer.load_state_dict(state_dict, strict=False)
        print(f"[DehazeFormer] Loaded pretrained weights from {path}")

    def _extract_encoder_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run DehazeFormer encoder and extract intermediate features.

        Returns dict with keys: 'stage1', 'stage2', 'stage3' (bottleneck)
        """
        m = self.dehazeformer

        # Stage 0: patch embed
        x = m.patch_embed(x)

        # Stage 1: layer1 (24ch, 1×)
        x = m.layer1(x)
        stage1 = x

        # Stage 2: patch_merge1 + layer2 (48ch, 1/2×)
        x = m.patch_merge1(x)
        x = m.layer2(x)
        stage2 = x

        # Stage 3: patch_merge2 + layer3 (96ch, 1/4×) — bottleneck
        x = m.patch_merge2(x)
        x = m.layer3(x)
        stage3 = x

        return {
            'stage1': stage1,   # [B, 24,  320, 320]
            'stage2': stage2,   # [B, 48,  160, 160]
            'stage3': stage3,   # [B, 96,   80,  80]
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dehaze at 320×320, upsample restored image to 640×640.

        Args:
            x: [B, 3, H, W] foggy image (any size, will be resized to 320×320)
        Returns:
            restored_image: [B, 3, 640, 640]
        """
        # Resize input to dehaze resolution
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=False)

        # Run actual DehazeFormer
        restored = self.dehazeformer(x)  # [B, 3, 320, 320]

        # Upsample restored IMAGE to detection resolution
        if restored.shape[-2:] != (self.output_size, self.output_size):
            restored = F.interpolate(
                restored,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            )

        return restored

    def get_encoder_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract encoder features projected to YOLO channel dimensions.

        Returns:
            dict with keys 'P3', 'P4', 'P5' (matching YOLO backbone scales)
        """
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=False)

        features = self._extract_encoder_features(x)

        # Apply MAA to stages 1-2
        features['stage1'] = self.maa_stage1(features['stage1'])
        features['stage2'] = self.maa_stage2(features['stage2'])

        # Project to YOLO channel dimensions
        return {
            'P3': self.proj_stage1(features['stage1']),  # [B, 256, 320, 320]
            'P4': self.proj_stage2(features['stage2']),  # [B, 512, 160, 160]
            'P5': self.proj_stage3(features['stage3']),  # [B, 1024, 80, 80]
        }

    def get_stage2_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw Stage 2 features for DCT alignment.
        Shape: [B, 48, 160, 160] for 320×320 input
        """
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=False)
        features = self._extract_encoder_features(x)
        return features['stage2']

    def get_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns bottleneck features for fog density estimation and depth decoder.
        Shape: [B, 96, 80, 80] for 320×320 input
        """
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=False)
        features = self._extract_encoder_features(x)
        return features['stage3']
