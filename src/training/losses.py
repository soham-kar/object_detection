"""Combined loss for WRDNet training.

L_total = L_det + lambda_rest * L_rest + lambda_depth * L_depth
        + lambda_entropy * L_entropy + lambda_domain * L_domain
        + lambda_fsg * L_fsg_cons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class WRDNetLoss(nn.Module):
    """
    Combined loss for WRDNet.

    Args:
        config: Config object with loss weights
        yolo_model: YOLOv11s model (for v8DetectionLoss initialization)
    """

    def __init__(self, config, yolo_model=None):
        super().__init__()
        self.lambda_rest = getattr(config, 'lambda_rest', 0.5)
        self.lambda_depth = getattr(config, 'lambda_depth', 0.1)
        self.lambda_entropy = getattr(config, 'lambda_entropy', 0.01)
        self.lambda_domain = getattr(config, 'lambda_domain', 0.1)
        self.lambda_fsg = getattr(config, 'lambda_fsg', 0.01)

        self.restoration_loss = nn.MSELoss()

        # YOLO detection loss (from ultralytics)
        self.yolo_loss = None
        if yolo_model is not None:
            try:
                from ultralytics.utils.loss import v8DetectionLoss
                from ultralytics.utils import IterableSimpleNamespace

                # v8DetectionLoss reads model.args for hyperparameters (hyp)
                # It expects attribute-style access (hyp.box, hyp.cls, hyp.dfl)
                # Ensure model.args is an IterableSimpleNamespace with loss weights
                if not hasattr(yolo_model, 'args') or isinstance(yolo_model.args, dict):
                    yolo_model.args = IterableSimpleNamespace(
                        box=7.5,       # box loss gain
                        cls=0.5,       # cls loss gain
                        dfl=1.5,       # dfl loss gain
                        box_pos_weight=-1.0,
                        cls_pw=1.0,
                        dfl_pw=1.0,
                    )

                self.yolo_loss = v8DetectionLoss(yolo_model)
                print("  YOLO detection loss initialized (v8DetectionLoss)")
            except Exception as e:
                print(f"  WARNING: Could not init YOLO loss: {e}")
                import traceback
                traceback.print_exc()
                self.yolo_loss = None

    def silog_loss(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        variance_focus: float = 0.5,
    ) -> torch.Tensor:
        """
        Scale-Invariant Log loss for monocular depth.

        Reference: Eigen et al., NeurIPS 2014
        """
        mask = (gt_depth > 0).float()
        n_valid = mask.sum() + 1e-8

        g = torch.log(pred_depth * mask + 1e-8) - torch.log(gt_depth * mask + 1e-8)
        g = g * mask

        Dg = variance_focus * (g.pow(2).sum() / n_valid)
        term2 = (1 - variance_focus) * (g.sum() / n_valid).pow(2)
        return torch.sqrt(torch.clamp(Dg - term2, min=0.0))

    def _prepare_yolo_batch(self, bboxes: List[torch.Tensor], batch_size: int,
                              device: str) -> dict:
        """
        Convert list of [N, 5] bbox tensors to ultralytics batch format.

        Ultralytics v8DetectionLoss expects a batch dict with:
          - 'batch_idx': [total_N] image index for each annotation
          - 'cls': [total_N] class IDs
          - 'bboxes': [total_N, 4] bounding boxes (cx, cy, w, h normalized)
        """
        batch_idx_list = []
        cls_list = []
        bbox_list = []

        for img_idx, b in enumerate(bboxes):
            if b.shape[0] == 0:
                continue
            batch_idx_list.append(torch.full((b.shape[0],), img_idx, dtype=torch.float32, device=device))
            cls_list.append(b[:, 0].float())  # class ID
            bbox_list.append(b[:, 1:5].float())  # cx, cy, w, h

        if not batch_idx_list:
            return {
                'batch_idx': torch.zeros(0, device=device),
                'cls': torch.zeros(0, 1, device=device),
                'bboxes': torch.zeros(0, 4, device=device),
            }

        return {
            'batch_idx': torch.cat(batch_idx_list, dim=0),
            'cls': torch.cat(cls_list, dim=0).view(-1, 1),
            'bboxes': torch.cat(bbox_list, dim=0),
        }

    def forward(self, outputs: dict, batch: dict) -> dict:
        """
        Compute all losses.

        Args:
            outputs: dict from model forward_train
            batch: dict with ground truth data
        Returns:
            losses: dict with individual and total loss
        """
        losses = {}

        # ── Detection loss (synthetic only) ──
        if 'detections_s' in outputs and 'bboxes' in batch and batch['bboxes'] is not None:
            if self.yolo_loss is not None:
                try:
                    # Determine device from detections (may be tuple)
                    det_s = outputs['detections_s']
                    if isinstance(det_s, (tuple, list)) and len(det_s) > 0:
                        if isinstance(det_s[0], torch.Tensor):
                            device = det_s[0].device
                        elif isinstance(det_s[0], dict) and 'boxes' in det_s[0]:
                            device = det_s[0]['boxes'].device
                        else:
                            device = 'cpu'
                    elif isinstance(det_s, torch.Tensor):
                        device = det_s.device
                    else:
                        device = 'cpu'

                    # Ensure YOLO loss internal tensors are on the right device
                    self.yolo_loss.device = device
                    if hasattr(self.yolo_loss, 'proj') and self.yolo_loss.proj.device != device:
                        self.yolo_loss.proj = self.yolo_loss.proj.to(device)

                    yolo_batch = self._prepare_yolo_batch(
                        batch['bboxes'],
                        batch_size=len(batch['bboxes']),
                        device=device,
                    )
                    # v8DetectionLoss returns (loss, loss_items)
                    det_loss_result = self.yolo_loss(det_s, yolo_batch)
                    if isinstance(det_loss_result, (tuple, list)):
                        loss_tensor = det_loss_result[0]
                        if loss_tensor.dim() == 0:
                            losses['det'] = loss_tensor
                        else:
                            losses['det'] = loss_tensor.sum()
                    else:
                        losses['det'] = det_loss_result
                except Exception as e:
                    print(f"  WARNING: YOLO loss computation failed: {e}")
                    # Create zero loss on the correct device with gradient
                    dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
                    losses['det'] = torch.tensor(0.0, device=dev, requires_grad=True)
            else:
                dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
                losses['det'] = torch.tensor(0.0, device=dev, requires_grad=True)
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['det'] = torch.tensor(0.0, device=dev, requires_grad=True)

        # ── Restoration loss (synthetic only) ──
        if 'restored_s' in outputs and 'clear_gt' in batch and batch['clear_gt'] is not None:
            losses['rest'] = self.restoration_loss(outputs['restored_s'], batch['clear_gt'])
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['rest'] = torch.tensor(0.0, device=dev)

        # ── Depth loss (synthetic only — SILog) ──
        if 'depth_s' in outputs and 'depth_gt' in batch and batch['depth_gt'] is not None:
            losses['depth'] = self.silog_loss(outputs['depth_s'], batch['depth_gt'])
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['depth'] = torch.tensor(0.0, device=dev)

        # ── Entropy loss on real fog detections (FDA paper) ──
        if 'detections_r' in outputs and outputs['detections_r'] is not None:
            try:
                det_r = outputs['detections_r']
                if isinstance(det_r, (tuple, list)):
                    det_r = det_r[0]

                if isinstance(det_r, torch.Tensor) and det_r.dim() >= 2:
                    cls_scores = det_r[:, 4:, :] if det_r.dim() == 3 else det_r[..., 4:]
                    probs = torch.softmax(cls_scores, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    losses['entropy'] = (entropy ** 2 + 0.001 ** 2) ** 0.75
                else:
                    dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
                    losses['entropy'] = torch.tensor(0.0, device=dev)
            except Exception:
                dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
                losses['entropy'] = torch.tensor(0.0, device=dev)
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['entropy'] = torch.tensor(0.0, device=dev)

        # ── Domain alignment loss ──
        if 'domain_loss' in outputs and outputs['domain_loss'] is not None:
            losses['domain'] = outputs['domain_loss']
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['domain'] = torch.tensor(0.0, device=dev)

        # ── FSG consistency loss ──
        if 'fsg_cons_loss' in outputs and outputs['fsg_cons_loss'] is not None:
            losses['fsg_cons'] = outputs['fsg_cons_loss']
        else:
            dev = outputs['restored_s'].device if 'restored_s' in outputs else 'cpu'
            losses['fsg_cons'] = torch.tensor(0.0, device=dev)

        # ── Total ──
        total = losses['det']
        total = total + self.lambda_rest * losses['rest']
        total = total + self.lambda_depth * losses['depth']
        total = total + self.lambda_entropy * losses['entropy']
        total = total + self.lambda_domain * losses['domain']
        total = total + self.lambda_fsg * losses['fsg_cons']
        losses['total'] = total

        return losses
