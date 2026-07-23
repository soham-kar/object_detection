"""Foggy Zurich dataset loader.

Loads real dense fog images for unsupervised domain adaptation.
No labels (unlabeled dataset).

Expected structure:
  data/Foggy_Zurich/RGB/{sequence}/{frame}.png
"""

import os
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .transforms import get_train_transforms, get_val_transforms


class FoggyZurichDataset(Dataset):
    """
    Foggy Zurich dataset for real dense fog domain adaptation.

    3,808 unlabeled real fog images across 4 sequences.
    Used only for domain adaptation (FDA, entropy loss, FSG consistency).

    Args:
        root: Foggy Zurich root directory (containing RGB/)
        input_size: target image size (default 640)
        config: optional config object
    """

    def __init__(
        self,
        root: str,
        input_size: int = 640,
        config=None,
    ):
        self.root = root
        self.input_size = input_size
        self.config = config

        self.img_dir = os.path.join(root, 'RGB')
        self.samples = self._collect_samples()

        # Use train transforms (augmentation helps DA)
        self.transform = get_train_transforms(input_size)

    def _collect_samples(self) -> List[Dict]:
        """Collect all image paths."""
        samples = []

        if not os.path.exists(self.img_dir):
            print(f"WARNING: Foggy Zurich RGB directory not found: {self.img_dir}")
            return samples

        for sequence in sorted(os.listdir(self.img_dir)):
            seq_dir = os.path.join(self.img_dir, sequence)
            if not os.path.isdir(seq_dir):
                continue

            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith('.png'):
                    samples.append({
                        'image': os.path.join(seq_dir, fname),
                        'sequence': sequence,
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = Image.open(sample['image']).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # No labels, no clear_gt, no depth_gt
        target = {
            'clear_gt': None,
            'depth_gt': None,
            'bboxes': torch.zeros((0, 5), dtype=torch.float32),
        }

        image, target = self.transform(image, target)

        return {
            'image': image,
            'clear_gt': None,
            'depth_gt': None,
            'bboxes': target['bboxes'],
            'image_path': sample['image'],
        }