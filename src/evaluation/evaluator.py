"""Evaluation utilities for WRDNet."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


class WRDNetEvaluator:
    """Evaluator for WRDNet."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate_detection(self, dataloader) -> Dict[str, float]:
        """
        Compute mAP@50 and mAP@50:95 using COCO metrics.

        Args:
            dataloader: validation/test data loader
        Returns:
            metrics: dict with mAP scores
        """
        # TODO: Implement actual COCO mAP computation
        # For now, return placeholder
        return {'mAP@50': 0.0, 'mAP@50:95': 0.0}

    def evaluate_restoration(self, dataloader, has_gt: bool = True) -> Dict[str, float]:
        """
        Compute restoration quality metrics.

        Args:
            dataloader: data loader with foggy and clear images
            has_gt: whether ground-truth clear images are available
        Returns:
            metrics: dict with PSNR, SSIM, or BRISQUE/NIQE
        """
        from ..utils.metrics import compute_psnr, compute_ssim

        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating restoration"):
                foggy = batch['image'].to(self.device)
                restored = self.model(foggy)['restored']

                if has_gt and 'clear_gt' in batch:
                    clear = batch['clear_gt'].to(self.device)
                    psnr = compute_psnr(restored, clear)
                    ssim = compute_ssim(restored, clear)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        metrics = {}
        if psnr_list:
            metrics['PSNR'] = sum(psnr_list) / len(psnr_list)
            metrics['SSIM'] = sum(ssim_list) / len(ssim_list)

        return metrics

    def measure_speed(self, input_size: tuple = (1, 3, 640, 640), num_runs: int = 100) -> float:
        """
        Measure inference FPS.

        Args:
            input_size: input tensor shape
            num_runs: number of inference runs for averaging
        Returns:
            fps: frames per second
        """
        dummy_input = torch.randn(*input_size).to(self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # Measure
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        import time
        start = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        fps = num_runs / elapsed

        return fps

    def visualize_alpha_maps(
        self,
        dataloader,
        save_dir: str,
        num_samples: int = 10,
    ):
        """
        Generate alpha map visualizations.

        Args:
            dataloader: data loader
            save_dir: directory to save visualizations
            num_samples: number of samples to visualize
        """
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= num_samples:
                    break

                images = batch['image'].to(self.device)
                outputs = self.model(images, return_alpha=True)

                alpha_maps = outputs.get('alpha_maps', {})
                # TODO: Save alpha map visualizations

                count += images.shape[0]
