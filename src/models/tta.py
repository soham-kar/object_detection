"""Test-Time Adaptation for real fog inference.

Inspired by TENT (Wang et al., ICLR 2021).
Reference: https://github.com/DequanWang/tent

Adapts BatchNorm statistics to real fog distribution via entropy minimization.
Applied only during evaluation on real fog images.
"""

import torch
import torch.nn as nn


def test_time_adapt(
    model: nn.Module,
    test_batch: torch.Tensor,
    num_iterations: int = 10,
    lr: float = 1e-4,
) -> nn.Module:
    """
    Adapt model BN statistics to test batch via entropy minimization.

    Args:
        model: WRDNet model
        test_batch: [B, 3, 640, 640] real fog images
        num_iterations: number of adaptation steps
        lr: learning rate for BN parameter updates
    Returns:
        model: adapted model (in eval mode)
    """
    # Switch to train mode, freeze all except BN
    model.train()
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name or 'batch_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Freeze detection head BN (preserve calibration)
    # Try to find head parameters and freeze them
    for module in model.modules():
        # Heuristic: modules named 'head' or in the last few layers
        if hasattr(module, 'weight') and module.weight is not None:
            # Check if this is in the detection head
            module_name = module.__class__.__name__
            if 'Detect' in module_name or 'Head' in module_name:
                for p in module.parameters():
                    p.requires_grad = False

    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # No BN layers found, skip TTA
        model.eval()
        return model

    optimizer = torch.optim.SGD(trainable_params, lr=lr)

    for _ in range(num_iterations):
        # Simple augmentation: horizontal flip
        augmented = torch.flip(test_batch, dims=[-1])

        # Forward pass
        detections = model(augmented)

        # Entropy minimization
        # detections shape depends on model output format
        # For YOLO-style: [B, num_classes + 5, num_anchors]
        if isinstance(detections, torch.Tensor):
            # Assume class logits are in the output
            # This is a simplified entropy computation
            # Actual implementation depends on YOLO output format
            probs = torch.softmax(detections[:, 4:, :], dim=1)  # class probs
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        else:
            # If detections is a dict or list, skip entropy
            entropy = torch.tensor(0.0, device=test_batch.device)

        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()

    # Final inference with adapted BN
    model.eval()
    return model
