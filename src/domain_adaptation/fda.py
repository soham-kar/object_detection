"""Fourier Domain Adaptation (FDA) transform.

Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
Segmentation", CVPR 2020.

During training, randomly swaps low-frequency amplitudes between synthetic
and real fog images.
"""

import torch
import torch.nn as nn


class FDATransform:
    """
    Fourier Domain Adaptation transform.

    Args:
        beta: fraction of low-frequency spectrum to swap (default 0.01)
    """

    def __init__(self, beta: float = 0.01):
        self.beta = beta

    def __call__(
        self,
        synth_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            synth_img: [B, 3, H, W] synthetic fog image
            real_img: [B, 3, H, W] real fog image
        Returns:
            adapted_img: [B, 3, H, W] style-transferred image
        """
        # Ensure same spatial size
        if synth_img.shape != real_img.shape:
            real_img = torch.nn.functional.interpolate(
                real_img,
                size=synth_img.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        # FFT
        fft_synth = torch.fft.rfft2(synth_img)
        fft_real = torch.fft.rfft2(real_img)

        # Amplitude and phase
        amp_s = torch.abs(fft_synth)
        phase_s = torch.angle(fft_synth)
        amp_r = torch.abs(fft_real)

        # Low-frequency mask
        h, w = synth_img.shape[-2:]
        ch = max(1, int(self.beta * h))
        cw = max(1, int(self.beta * w))

        # Create mask on the same device as input
        # Shape: [1, 1, H, W//2+1] for broadcasting over [B, C, H, W//2+1]
        mask = torch.zeros(1, 1, h, w // 2 + 1, device=synth_img.device)
        mask[:, :, :ch, :cw] = 1.0

        # Swap amplitudes
        amp_new = amp_s * (1 - mask) + amp_r * mask

        # Reconstruct
        adapted = amp_new * torch.exp(1j * phase_s)
        return torch.fft.irfft2(adapted, s=(h, w))
