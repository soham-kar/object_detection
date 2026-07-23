"""Combined loss for WRDNet training.

L_total = L_det + lambda_rest * L_rest + lambda_depth * L_depth
        + lambda_entropy * L_entropy + lambda_domain * L_domain
        + lambda_fsg * L_fsg_cons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WRDNetLoss(nn.Module):
    """
    Combined loss for WRDNet.

    Args:
        config: Config object with loss weights
    """

    def __init__(self, config):
        super().__init__()
        self.lambda_rest = getattr(config, 'lambda_rest', 0.5)
        self.lambda_depth = getattr(config, 'lambda_depth', 0.1)
        self.lambda_entropy = getattr(config, 'lambda_entropy', 0.01)
        self.lambda_domain = getattr(config, 'lambda_domain', 0.1)
        self.lambda_fsg = getattr(config, 'lambda_fsg', 0.01)

        self.restoration_loss = nn.MSELoss()

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
        # Mask valid pixels (gt > 0)
        mask = (gt_depth > 0).float()
        n_valid = mask.sum() + 1e-8

        # Log difference
        g = torch.log(pred_depth * mask + 1e-8) - torch.log(gt_depth * mask + 1e-8)
        g = g * mask

        # SILog
        Dg = variance_focus * (g.pow(2).sum() / n_valid)
        term2 = (1 - variance_focus) * (g.sum() / n_valid).pow(2)
        return torch.sqrt(torch.clamp(Dg - term2, min=0.0))

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

        # Detection loss (synthetic only)
        if 'detections_s' in outputs and 'labels' in batch:
            # TODO: Implement actual YOLO detection loss
            # For now, placeholder
            losses['det'] = torch.tensor(0.0, device=outputs['detections_s'].device)
        else:
            losses['det'] = torch.tensor(0.0)

        # Restoration loss (synthetic only)
        if 'restored_s' in outputs and 'clear_gt' in batch:
            losses['rest'] = self.restoration_loss(outputs['restored_s'], batch['clear_gt'])
        else:
            losses['rest'] = torch.tensor(0.0)

        # Depth loss (synthetic only — SILog)
        if 'depth_s' in outputs and 'depth_gt' in batch:
            losses['depth'] = self.silog_loss(outputs['depth_s'], batch['depth_gt'])
        else:
            losses['depth'] = torch.tensor(0.0)

        # Entropy loss on real fog detections (FDA paper)
        if 'detections_r' in outputs:
            # Simplified entropy computation
            # Actual implementation depends on YOLO output format
            losses['entropy'] = torch.tensor(0.0, device=outputs['detections_r'].device)
        else:
            losses['entropy'] = torch.tensor(0.0)

        # Domain alignment loss
        if 'domain_loss' in outputs:
            losses['domain'] = outputs['domain_loss']
        else:
            losses['domain'] = torch.tensor(0.0)

        # FSG consistency loss
        if 'fsg_cons_loss' in outputs:
            losses['fsg_cons'] = outputs['fsg_cons_loss']
        else:
            losses['fsg_cons'] = torch.tensor(0.0)

        # Total
        total = losses['det']
        total += self.lambda_rest * losses['rest']
        total += self.lambda_depth * losses['depth']
        total += self.lambda_entropy * losses['entropy']
        total += self.lambda_domain * losses['domain']
        total += self.lambda_fsg * losses['fsg_cons']
        losses['total'] = total

        return losses
