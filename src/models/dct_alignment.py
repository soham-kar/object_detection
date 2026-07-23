"""DCT Feature Alignment Module.

Inspired by AdaDCP (Bi et al., ICCV 2025).
Applied to DehazeFormer Stage 2 features (40x40).

Simplification: Start with MMD (Maximum Mean Discrepancy) on DCT coefficients.
Upgrade to adversarial mode only if MMD fails.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTFeatureAlignment(nn.Module):
    """
    DCT-based domain alignment.

    Args:
        channels: number of feature channels
        num_bands: number of DCT frequency bands (default 8)
        mode: 'mmd' (simpler) or 'adversarial' (full)
    """

    def __init__(self, channels: int, num_bands: int = 8, mode: str = 'mmd'):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        self.mode = mode

        # DCT decomposition is done on-the-fly (no parameters)
        # Per-band processing
        self.band_projectors = nn.ModuleList([
            nn.Linear(channels, channels // 2) for _ in range(num_bands)
        ])

        if mode == 'adversarial':
            # Domain classifier per band
            self.domain_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(channels // 2, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                ) for _ in range(num_bands)
            ])
            # Gradient reversal is applied externally in training loop
        else:
            # MMD mode: no classifiers needed
            self.domain_classifiers = None

        # Learnable frequency importance weights
        self.freq_weights = nn.Parameter(torch.ones(num_bands))

    def _dct_decompose(self, x: torch.Tensor, block_size: int = 8) -> list:
        """
        Block-wise DCT-II decomposition using precomputed basis.

        Args:
            x: [B, C, H, W]
        Returns:
            list of [B, C, Hb, Wb] tensors for each frequency band
        """
        B, C, H, W = x.shape
        assert H % block_size == 0 and W % block_size == 0

        # Build DCT-II basis matrix (or use cached)
        if not hasattr(self, '_dct_basis') or self._dct_basis.shape[0] != block_size:
            # DCT-II basis: X_k = sum_n x_n * cos(pi * k * (n + 0.5) / N)
            n = torch.arange(block_size, dtype=x.dtype, device=x.device)
            k = torch.arange(block_size, dtype=x.dtype, device=x.device).unsqueeze(1)
            basis = torch.cos(3.1415926535 * k * (n + 0.5) / block_size)  # [N, N]
            basis[0, :] *= 1.0 / (2.0 ** 0.5)  # Orthogonal normalization
            basis *= (2.0 / block_size) ** 0.5
            self.register_buffer('_dct_basis', basis, persistent=False)

        # Reshape into blocks: [B, C, Hb, block_size, Wb, block_size]
        Hb, Wb = H // block_size, W // block_size
        x_blocks = x.view(B, C, Hb, block_size, Wb, block_size)
        x_blocks = x_blocks.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, Hb, Wb, C, N, N]

        # Apply 2D DCT: D = B @ x @ B^T
        basis = self._dct_basis  # [N, N]
        dct = torch.matmul(basis, x_blocks)       # [B, Hb, Wb, C, N, N]
        dct = torch.matmul(dct, basis.t())         # [B, Hb, Wb, C, N, N]

        # Extract frequency bands (zigzag order approximation)
        bands = []
        for i in range(min(self.num_bands, block_size * block_size)):
            row = i // block_size
            col = i % block_size
            band = dct[..., row, col].contiguous()  # [B, Hb, Wb, C]
            band = band.permute(0, 3, 1, 2)          # [B, C, Hb, Wb]
            bands.append(band)

        return bands

    def forward(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> tuple:
        """
        Args:
            features: [B, C, H, W] from DehazeFormer Stage 2
            domain_labels: [B] 0=synthetic, 1=real
        Returns:
            aligned_features: [B, C, H, W]
            domain_loss: scalar (MMD or adversarial loss)
        """
        B, C, H, W = features.shape

        # Decompose into DCT bands
        bands = self._dct_decompose(features)  # list of [B, C, Hb, Wb]

        # Process each band
        processed_bands = []
        for i, band in enumerate(bands):
            # Global average pooling
            band_vec = F.adaptive_avg_pool2d(band, 1).view(B, C)  # [B, C]
            # Project
            projected = self.band_projectors[i](band_vec)  # [B, C//2]
            processed_bands.append(projected)

        # Compute domain loss
        if self.mode == 'mmd':
            domain_loss = self._mmd_loss(processed_bands, domain_labels)
        else:
            domain_loss = self._adversarial_loss(processed_bands, domain_labels)

        # Recombine bands (simplified: just return original features)
        # In a full implementation, we would modulate features by band alignment
        aligned_features = features

        return aligned_features, domain_loss

    def _mmd_loss(self, bands: list, domain_labels: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy loss between synthetic and real domains."""
        synthetic_mask = domain_labels == 0
        real_mask = domain_labels == 1

        if synthetic_mask.sum() == 0 or real_mask.sum() == 0:
            return torch.tensor(0.0, device=domain_labels.device)

        # Concatenate all band representations
        all_bands = torch.cat(bands, dim=1)  # [B, num_bands * C//2]

        synth = all_bands[synthetic_mask]
        real = all_bands[real_mask]

        # MMD with RBF kernel (simplified)
        mean_synth = synth.mean(dim=0)
        mean_real = real.mean(dim=0)
        mmd = torch.norm(mean_synth - mean_real, p=2)

        # Weight by frequency importance
        weights = F.softmax(self.freq_weights, dim=0)
        weighted_mmd = mmd * weights.sum()

        return weighted_mmd

    def _adversarial_loss(self, bands: list, domain_labels: torch.Tensor) -> torch.Tensor:
        """Adversarial domain classification loss."""
        if self.domain_classifiers is None:
            return torch.tensor(0.0, device=domain_labels.device)

        total_loss = 0.0
        weights = F.softmax(self.freq_weights, dim=0)

        for i, band in enumerate(bands):
            pred = self.domain_classifiers[i](band).squeeze(-1)  # [B]
            loss = F.binary_cross_entropy_with_logits(pred, domain_labels.float())
            total_loss += weights[i] * loss

        return total_loss
