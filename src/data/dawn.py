"""DAWN (Detection in Adverse Weather Nature) dataset loader."""

import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .transforms import Compose, Normalize


class DAWNDataset(Dataset):
    """
    DAWN dataset for foggy/rainy weather detection.

    Expected structure:
        root/
            images/
                train/
                val/
            annotations/
                train/
                val/
    """

    def __init__(self, root: str, split: str = 'train', config=None):
        self.root = root
        self.split = split
        self.config = config

        self.img_dir = os.path.join(root, 'images', split)
        self.ann_dir = os.path.join(root, 'annotations', split)

        self.samples = self._collect_samples()
        self.transform = Compose([Normalize()])

    def _collect_samples(self):
        """Collect sample paths."""
        samples = []

        if not os.path.exists(self.img_dir):
            return samples

        for fname in sorted(os.listdir(self.img_dir)):
            if not fname.endswith(('.png', '.jpg')):
                continue

            img_path = os.path.join(self.img_dir, fname)
            ann_path = os.path.join(self.ann_dir, fname.replace('.jpg', '.xml').replace('.png', '.xml'))

            samples.append({'image': img_path, 'annotation': ann_path})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image']).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        image, _ = self.transform(image)

        return {
            'image': image,
            'annotation_path': sample['annotation'],
        }
