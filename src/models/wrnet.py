"""WRDNet: Weather-Resilient Detection Unified Network.

Complete model combining all components:
  - DehazeFormer-T (restoration)
  - YOLOv11s (detection)
  - FSG / DG-FSG (feature fusion)
  - DCT Alignment (domain adaptation)
  - Depth Decoder (monocular depth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .dehazeformer import DehazeFormerWrapper
from .yolov11 import YOLOv11sWrapper
from .fsg import FeatureSelectionGate
from .dg_fsg import DepthGuidedFSG
from .depth_decoder import DepthDecoder
from .dct_alignment import DCTFeatureAlignment


class WRDNet(nn.Module):
    """
    Weather-Resilient Detection Unified Network.

    Args:
        config: Config object with model settings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Restoration encoder
        self.dehazeformer = DehazeFormerWrapper(
            variant=getattr(config, 'dehazeformer_variant', 'T'),
            pretrained=getattr(config, 'pretrained', True),
            input_size=getattr(config, 'input_size_dehaze', 320),
            output_size=getattr(config, 'input_size_detect', 640),
            use_maa=getattr(config, 'use_maa', True),
        )

        # Detection encoder
        self.yolo = YOLOv11sWrapper(pretrained=getattr(config, 'pretrained', True))

        # Feature fusion
        fsg_channels = getattr(config, 'fsg_channels', [256, 512, 1024])
        use_depth = getattr(config, 'use_depth', False)
        use_dg_fsg = getattr(config, 'use_dg_fsg', False)
        use_cdmsa = getattr(config, 'use_cdmsa', True)

        if use_depth and use_dg_fsg:
            self.fsg = DepthGuidedFSG(
                channels_list=fsg_channels,
                depth_channels=16,
                use_cdmsa=use_cdmsa,
            )
            self.depth_decoder = DepthDecoder(bottleneck_channels=96)
        else:
            self.fsg = FeatureSelectionGate(
                channels_list=fsg_channels,
                use_cdmsa=use_cdmsa,
            )
            self.depth_decoder = None
            if use_depth and not use_dg_fsg:
                # Depth as auxiliary task only (E11)
                self.depth_decoder = DepthDecoder(bottleneck_channels=96)

        # DCT Alignment (training only)
        # Stage 2 raw features: [B, 48, 160, 160] from DehazeFormer-T
        self.dct_alignment = DCTFeatureAlignment(
            channels=48,  # Stage 2 channels (DehazeFormer-T layer2 output)
            mode='mmd',
        )

        # Domain adaptation flags
        self.use_fda = getattr(config, 'use_fda', False)
        self.use_dct_align = getattr(config, 'use_dct_align', False)
        self.use_fsg_consistency = getattr(config, 'use_fsg_consistency', False)

    def forward(
        self,
        x: torch.Tensor,
        return_depth: bool = False,
        return_alpha: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for inference.

        Args:
            x: [B, 3, 640, 640] foggy image
            return_depth: whether to return depth map
            return_alpha: whether to return alpha maps
        Returns:
            dict with keys:
                - 'detections': YOLO output
                - 'restored': restored image
                - 'depth': depth map (optional)
                - 'alpha_maps': alpha maps (optional)
        """
        outputs = {}

        # DehazeFormer: 320×320 input → 640×640 restored IMAGE
        restored = self.dehazeformer(x)
        outputs['restored'] = restored

        # Depth decoder (if enabled)
        depth = None
        if self.depth_decoder is not None:
            bottleneck = self.dehazeformer.get_bottleneck(x)  # [B, 96, 80, 80]
            depth_160, depth_640 = self.depth_decoder(bottleneck)
            depth = depth_640
            if return_depth:
                outputs['depth'] = depth

        # YOLO backbone features from RESTORED image
        orig_features = self.yolo.get_backbone_features(restored)

        # DehazeFormer encoder features (already projected to YOLO dims)
        # Returns {'P3': [B,256,320,320], 'P4': [B,512,160,160], 'P5': [B,1024,80,80]}
        rest_features = self.dehazeformer.get_encoder_features(x)

        # Upsample restored features to match YOLO spatial scales
        # DehazeFormer P3: 320×320 → YOLO P3: 80×80
        # DehazeFormer P4: 160×160 → YOLO P4: 40×40
        # DehazeFormer P5: 80×80   → YOLO P5: 20×20
        rest_features_up = {}
        for name in ['P3', 'P4', 'P5']:
            target_size = orig_features[name].shape[2:]
            rest_features_up[name] = F.interpolate(
                rest_features[name], size=target_size,
                mode='bilinear', align_corners=False,
            )

        # Feature Selection Gate
        if isinstance(self.fsg, DepthGuidedFSG) and depth is not None:
            fused_features, alpha_maps = self.fsg(
                rest_features_up, orig_features, depth
            )
        else:
            fused_features, alpha_maps = self.fsg(
                rest_features_up, orig_features
            )

        if return_alpha:
            outputs['alpha_maps'] = alpha_maps

        # Detection
        detections = self.yolo.forward_neck_head(fused_features)
        outputs['detections'] = detections

        return outputs

    def forward_train(
        self,
        synth_input: Dict[str, torch.Tensor],
        real_input: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with domain adaptation.

        Args:
            synth_input: dict with 'image', 'clear_gt', 'labels', 'depth_gt'
            real_input: dict with 'image' (optional, for DA)
        Returns:
            dict with all outputs and intermediate features for loss computation
        """
        outputs = {}

        # ── Synthetic path ──
        synth_img = synth_input['image']

        # DehazeFormer
        restored_s = self.dehazeformer(synth_img)
        outputs['restored_s'] = restored_s

        # Depth
        depth_s = None
        if self.depth_decoder is not None:
            bottleneck_s = self.dehazeformer.get_bottleneck(synth_img)  # [B, 96, 80, 80]
            _, depth_s = self.depth_decoder(bottleneck_s)
            outputs['depth_s'] = depth_s

        # YOLO features from restored image
        orig_features_s = self.yolo.get_backbone_features(restored_s)

        # DehazeFormer encoder features (already projected: P3/P4/P5)
        rest_features_s = self.dehazeformer.get_encoder_features(synth_img)

        # Upsample to match YOLO spatial scales
        rest_features_s_up = {}
        for name in ['P3', 'P4', 'P5']:
            target_size = orig_features_s[name].shape[2:]
            rest_features_s_up[name] = F.interpolate(
                rest_features_s[name], size=target_size,
                mode='bilinear', align_corners=False,
            )

        # FSG
        if isinstance(self.fsg, DepthGuidedFSG) and depth_s is not None:
            fused_s, alpha_s = self.fsg(rest_features_s_up, orig_features_s, depth_s)
        else:
            fused_s, alpha_s = self.fsg(rest_features_s_up, orig_features_s)

        outputs['fused_s'] = fused_s
        outputs['alpha_s'] = alpha_s

        # Detection
        detections_s = self.yolo.forward_neck_head(fused_s)
        outputs['detections_s'] = detections_s

        # ── Real path (for domain adaptation) ──
        if real_input is not None:
            real_img = real_input['image']

            with torch.set_grad_enabled(self.use_fsg_consistency):
                restored_r = self.dehazeformer(real_img)
                orig_features_r = self.yolo.get_backbone_features(restored_r)
                rest_features_r = self.dehazeformer.get_encoder_features(real_img)

                rest_features_r_up = {}
                for name in ['P3', 'P4', 'P5']:
                    target_size = orig_features_r[name].shape[2:]
                    rest_features_r_up[name] = F.interpolate(
                        rest_features_r[name], size=target_size,
                        mode='bilinear', align_corners=False,
                    )

                if isinstance(self.fsg, DepthGuidedFSG):
                    # Depth decoder works on any image
                    bottleneck_r = self.dehazeformer.get_bottleneck(real_img)
                    _, depth_r = self.depth_decoder(bottleneck_r)
                    fused_r, alpha_r = self.fsg(rest_features_r_up, orig_features_r, depth_r)
                else:
                    fused_r, alpha_r = self.fsg(rest_features_r_up, orig_features_r)

                detections_r = self.yolo.forward_neck_head(fused_r)
                outputs['detections_r'] = detections_r
                outputs['alpha_r'] = alpha_r

            # DCT Alignment
            if self.use_dct_align:
                features_s = self.dehazeformer.get_stage2_features(synth_img)
                features_r = self.dehazeformer.get_stage2_features(real_img)
                features_all = torch.cat([features_s, features_r], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(len(synth_img), device=synth_img.device),
                    torch.ones(len(real_img), device=real_img.device),
                ])
                _, domain_loss = self.dct_alignment(features_all, domain_labels)
                outputs['domain_loss'] = domain_loss

        return outputs
