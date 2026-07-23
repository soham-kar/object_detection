"""Measure FLOPs and parameters for WRDNet."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.utils.config import load_config
from src.models.wrnet import WRDNet
from src.utils.flops import count_parameters, count_flops, print_model_summary


def main():
    # Use default config
    config = load_config('configs/default.yaml')

    # Build model
    model = WRDNet(config)

    # Print summary
    print_model_summary(model, input_size=(1, 3, 640, 640))

    # Detailed breakdown
    print("\nDetailed Parameter Count:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params / 1e6:.2f}M")


if __name__ == '__main__':
    main()
