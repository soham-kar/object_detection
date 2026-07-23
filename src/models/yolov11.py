"""YOLOv11s wrapper for WRDNet.

Integrates ultralytics YOLOv11s.
Exposes backbone features (P3, P4, P5) for FSG fusion.
Injects fused features back into the neck+head for detection.

YOLOv11s Architecture (verified from actual model):
  BACKBONE (layers 0-10):
    L0:  Conv    [B, 32,  320, 320]   stride 2
    L1:  Conv    [B, 64,  160, 160]   stride 4
    L2:  C3k2    [B, 128, 160, 160]
    L3:  Conv    [B, 128, 80,  80]    stride 8
    L4:  C3k2    [B, 256, 80,  80]    ← P3 output (256ch, 80×80)
    L5:  Conv    [B, 256, 40,  40]    stride 16
    L6:  C3k2    [B, 256, 40,  40]    ← P4 output (256ch, 40×40)
    L7:  Conv    [B, 512, 20,  20]    stride 32
    L8:  C3k2    [B, 512, 20,  20]
    L9:  SPPF    [B, 512, 20,  20]
    L10: C2PSA  [B, 512, 20,  20]    ← P5 output (512ch, 20×20)

  NECK (layers 11-22):
    L11: Upsample [B, 512, 40, 40]
    L12: Concat   [-1, 6] → [B, 768, 40, 40]   (upsamples P5, concats with L6/P4)
    L13: C3k2     [B, 256, 40, 40]
    L14: Upsample [B, 256, 80, 80]
    L15: Concat   [-1, 4] → [B, 512, 80, 80]   (upsamples L13, concats with L4/P3)
    L16: C3k2     [B, 128, 80, 80]             ← Detect input 1
    L17: Conv     [B, 128, 40, 40]
    L18: Concat   [-1, 13] → [B, 384, 40, 40]
    L19: C3k2     [B, 256, 40, 40]             ← Detect input 2
    L20: Conv     [B, 256, 20, 20]
    L21: Concat   [-1, 10] → [B, 768, 20, 20]
    L22: C3k2     [B, 512, 20, 20]             ← Detect input 3

  HEAD (layer 23):
    L23: Detect   f=[16, 19, 22]  → [B, 84, 8400]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class YOLOv11sWrapper(nn.Module):
    """
    Wrapper around YOLOv11s with WRDNet-specific modifications.

    - Exposes backbone features at P3 (80×80, 256ch), P4 (40×40, 256ch), P5 (20×20, 512ch)
    - Accepts fused features from FSG and injects them into the neck
    - Runs neck+head on fused features for detection
    - Supports gradient flow for end-to-end training
    """

    # Layer indices in YOLOv11s
    BACKBONE_P3 = 4    # C3k2 output: [B, 256, 80, 80]
    BACKBONE_P4 = 6    # C3k2 output: [B, 256, 40, 40]
    BACKBONE_P5 = 10   # C2PSA output: [B, 512, 20, 20]
    NECK_START = 11    # First neck layer (Upsample)
    DETECT = 23        # Detection head

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.pretrained = pretrained

        # ── Build actual YOLOv11s from ultralytics ──
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        # Load YOLOv11s model
        model_name = 'yolo11s.pt'
        # Use YOLO to load weights, but extract the nn.Module immediately
        # Store the YOLO loader as a non-module attribute to prevent .train() override
        yolo_loader = YOLO(model_name) if pretrained else None

        # Extract the underlying nn.Module (the actual PyTorch DetectionModel)
        if yolo_loader is not None:
            self.model = yolo_loader.model  # DetectionModel (nn.Module)
        else:
            raise RuntimeError("Failed to load YOLOv11s model")

        # Don't store yolo_loader as an attribute — it's not an nn.Module
        # and its .train() method would interfere with PyTorch's train mode

        # Verified backbone channel dimensions (from actual model inspection)
        self.backbone_channels = {
            'P3': 256,   # 80×80  (stride 8)
            'P4': 256,   # 40×40  (stride 16)
            'P5': 512,   # 20×20  (stride 32)
        }

        # Number of detection classes (COCO = 80, but we use 8 for foggy driving)
        # The Detect head outputs [B, 4 + nc, num_anchors] per scale
        # For COCO: 4 + 80 = 84 channels
        self.nc = 80  # Will be updated if we swap the detection head

    def get_backbone_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract YOLOv11s backbone features at P3, P4, P5.

        Runs a manual forward through backbone layers (0-10) with gradient flow.
        NO torch.no_grad() — gradients flow through backbone for training.

        Args:
            x: [B, 3, 640, 640] image
        Returns:
            dict with keys 'P3', 'P4', 'P5'
        """
        layers = self.model.model

        # Manual forward through backbone
        out = x
        backbone_outputs = {}

        for i in range(self.BACKBONE_P5 + 1):  # layers 0-10
            out = layers[i](out)
            if i == self.BACKBONE_P3:
                backbone_outputs['P3'] = out  # [B, 256, 80, 80]
            elif i == self.BACKBONE_P4:
                backbone_outputs['P4'] = out  # [B, 256, 40, 40]
            elif i == self.BACKBONE_P5:
                backbone_outputs['P5'] = out  # [B, 512, 20, 20]

        return backbone_outputs

    def forward_neck_head(self, fused_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run YOLO neck and head on fused features from FSG.

        Delegates to _forward_neck_head_proper which correctly handles
        all skip connections in the YOLOv11s neck.

        Args:
            fused_features: dict with keys 'P3', 'P4', 'P5' from FSG
        Returns:
            detections: YOLO output tuple (raw_preds, decoded_dict)
        """
        return self._forward_neck_head_proper(fused_features)

    def _forward_neck_head_proper(self, fused_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Proper neck+head forward with saved intermediate outputs.

        YOLOv11s neck uses skip connections:
          L12: Concat[-1, 6]  → needs P4 (fused replaces L6)
          L15: Concat[-1, 4]  → needs P3 (fused replaces L4)
          L18: Concat[-1, 13] → needs L13 output
          L21: Concat[-1, 10] → needs P5 (fused replaces L10)
        """
        layers = self.model.model
        p3 = fused_features['P3']  # [B, 256, 80, 80]
        p4 = fused_features['P4']  # [B, 256, 40, 40]
        p5 = fused_features['P5']  # [B, 512, 20, 20]

        # Save intermediates for skip connections
        y = {}  # layer index → output

        # L11: Upsample P5
        y[11] = layers[11](p5)  # [B, 512, 40, 40]

        # L12: Concat [L11, P4_fused]  (replaces L6 with fused P4)
        y[12] = torch.cat([y[11], p4], dim=1)  # [B, 768, 40, 40]

        # L13: C3k2
        y[13] = layers[13](y[12])  # [B, 256, 40, 40]

        # L14: Upsample
        y[14] = layers[14](y[13])  # [B, 256, 80, 80]

        # L15: Concat [L14, P3_fused]  (replaces L4 with fused P3)
        y[15] = torch.cat([y[14], p3], dim=1)  # [B, 512, 80, 80]

        # L16: C3k2 → Detect input 1
        y[16] = layers[16](y[15])  # [B, 128, 80, 80]

        # L17: Conv
        y[17] = layers[17](y[16])  # [B, 128, 40, 40]

        # L18: Concat [L17, L13]
        y[18] = torch.cat([y[17], y[13]], dim=1)  # [B, 384, 40, 40]

        # L19: C3k2 → Detect input 2
        y[19] = layers[19](y[18])  # [B, 256, 40, 40]

        # L20: Conv
        y[20] = layers[20](y[19])  # [B, 256, 20, 20]

        # L21: Concat [L20, P5_fused]  (replaces L10 with fused P5)
        y[21] = torch.cat([y[20], p5], dim=1)  # [B, 768, 20, 20]

        # L22: C3k2 → Detect input 3
        y[22] = layers[22](y[21])  # [B, 512, 20, 20]

        # L23: Detect head
        detections = layers[23]([y[16], y[19], y[22]])

        return detections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward for standalone YOLOv11s inference.
        Runs full backbone → neck → head.
        """
        features = self.get_backbone_features(x)
        return self._forward_neck_head_proper(features)

    def get_loss_function(self):
        """
        Get the YOLOv11s detection loss function for training.

        Returns the model's built-in loss function (v8DetectionLoss).
        """
        # ultralytics stores the loss function init in model attribute
        # We need to create it from the model's detection head
        from ultralytics.utils.loss import v8DetectionLoss

        # The Detect head needs to be initialized for loss computation
        # v8DetectionLoss expects a model with stride and nc attributes
        return v8DetectionLoss(self.model)

    def prepare_targets(self, bboxes: List[torch.Tensor], batch_size: int,
                         img_size: int = 640, device: str = 'cpu') -> torch.Tensor:
        """
        Convert YOLO-format bboxes to ultralytics target format.

        Ultralytics expects targets as [B, N, 5] where each row is
        [class_id, cx, cy, w, h] (normalized), padded with zeros.

        Args:
            bboxes: list of [N, 5] tensors (class_id, cx, cy, w, h)
            batch_size: batch size
            img_size: image size (for scaling if needed)
            device: target device
        Returns:
            targets: [B, max_N, 5] padded tensor
        """
        if not bboxes:
            return torch.zeros(batch_size, 0, 5, device=device)

        max_n = max(b.shape[0] for b in bboxes) if bboxes else 0
        if max_n == 0:
            return torch.zeros(batch_size, 0, 5, device=device)

        targets = torch.zeros(batch_size, max_n, 5, device=device)
        for i, b in enumerate(bboxes):
            n = b.shape[0]
            if n > 0:
                targets[i, :n] = b

        return targets
