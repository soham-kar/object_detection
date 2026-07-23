"""FLOPs and parameter counting utilities."""

import torch
from thop import profile


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: torch.nn.Module, input_size: tuple = (1, 3, 640, 640)) -> float:
    """
    Count FLOPs using thop.

    Args:
        model: PyTorch model
        input_size: input tensor shape
    Returns:
        gflops: GFLOPs
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)

    # Set to eval mode for profiling
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    gflops = flops / 1e9
    return gflops


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 640, 640)):
    """Print model summary with params and FLOPs."""
    params = count_parameters(model)
    gflops = count_flops(model, input_size)

    print("=" * 60)
    print(f"Model Summary")
    print("=" * 60)
    print(f"Total Parameters: {params / 1e6:.2f}M")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Input Size: {input_size}")
    print("=" * 60)
