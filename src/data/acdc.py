"""ACDC (Adverse Conditions Dataset with Correspondences) loader.

Loads real foggy images with YOLO-format bounding box labels.
No clear GT or depth GT (real fog — no supervision for restoration/depth).

Expected structure:
  data/rgb_anon_trainvaltest/rgb_anon/fog/{train,val,test}/{sequence}/{frame}_rgb_anon.png
  data/acdc_labels/{train,val}/{sequence}/{frame}.txt  (YOLO format)
"""

import os
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .transforms import get_train_transforms, get_val_transforms


class ACDCDataset(Dataset):
    """
    ACDC fog dataset for real fog training and validation.

    Args:
        root: ACDC root directory (containing rgb_anon/fog/)
        split: 'train', 'val', or 'test'
        labels_dir: directory containing YOLO .txt label files
        input_size: target image size (default 640)
        config: optional config object
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        labels_dir: str = None,
        input_size: int = 640,
        config=None,
    ):
        self.root = root
        self.split = split
        self.input_size = input_size
        self.config = config

        # Image directory: root/rgb_anon/fog/{split}/{sequence}/
        self.img_dir = os.path.join(root, 'rgb_anon', 'fog', split)

        # Labels directory: data/acdc_labels/{split}/{sequence}/
        if labels_dir is None:
            labels_dir = os.path.join(os.path.dirname(root), 'acdc_labels', split)
        self.labels_dir = labels_dir

        self.samples = self._collect_samples()

        # Transforms (no clear_gt or depth_gt for ACDC)
        if split == 'train':
            self.transform = get_train_transforms(input_size)
        else:
            self.transform = get_val_transforms(input_size)

    def _collect_samples(self) -> List[Dict]:
        """Collect all valid sample paths."""
        samples = []

        if not os.path.exists(self.img_dir):
            print(f"WARNING: ACDC image directory not found: {self.img_dir}")
            return samples

        # Walk through sequence directories
        for sequence in sorted(os.listdir(self.img_dir)):
            seq_dir = os.path.join(self.img_dir, sequence)
            if not os.path.isdir(seq_dir):
                continue

            for fname in sorted(os.listdir(seq_dir)):
                if not fname.endswith('_rgb_anon.png'):
                    continue

                img_path = os.path.join(seq_dir, fname)

                # Derive label path
                # Image: GOPR0475_frame_000041_rgb_anon.png
                # Label: GOPR0475_frame_000041.txt
                base = fname.replace('_rgb_anon.png', '')
                label_path = os.path.join(self.labels_dir, sequence, f"{base}.txt")

                samples.append({
                    'image': img_path,
                    'label': label_path if os.path.exists(label_path) else None,
                    'sequence': sequence,
                    'frame': base,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_yolo_labels(self, label_path: str) -> torch.Tensor:
        """Load YOLO format labels. Returns [N, 5] tensor."""
        if label_path is None or not os.path.exists(label_path):
            return torch.zeros((0, 5), dtype=torch.float32)

        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = [float(p) for p in parts]
                    bboxes.append([cls, cx, cy, w, h])

        if len(bboxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(bboxes, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load foggy image
        image = Image.open(sample['image']).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load labels
        bboxes = self._load_yolo_labels(sample['label'])

        # Build target dict (no clear_gt, no depth_gt for real fog)
        target = {
            'clear_gt': None,
            'depth_gt': None,
            'bboxes': bboxes,
        }

        # Apply transforms
        image, target = self.transform(image, target)

        return {
            'image': image,
            'clear_gt': None,
            'depth_gt': None,
            'bboxes': target['bboxes'],
            'image_path': sample['image'],
        }
