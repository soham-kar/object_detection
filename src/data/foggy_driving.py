"""Foggy Driving dataset loader.

Loads real fog images with bounding box annotations for evaluation.
101 images total (test + test_extra).

Expected structure:
  data/Foggy_Driving/leftImg8bit/{test,test_extra}/{category}/{base}_leftImg8bit.png
  data/Foggy_Driving/bboxGt/{test,test_extra}/{category}/{base}.txt

Bbox format: class_id x1 y1 x2 y2 (pixel coordinates, NOT normalized)
  class_id: 0=person, 1=rider, 2=car, 3=truck, 4=bus, 5=train, 6=motorcycle, 7=bicycle
"""

import os
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .transforms import get_val_transforms


class FoggyDrivingDataset(Dataset):
    """
    Foggy Driving dataset for real fog evaluation.

    101 real fog images with bounding box annotations.
    Used as a test set (no training).

    Args:
        root: Foggy Driving root directory
        split: 'test' or 'test_extra' (default: load both)
        input_size: target image size (default 640)
        config: optional config object
    """

    def __init__(
        self,
        root: str,
        split: str = 'all',
        input_size: int = 640,
        config=None,
    ):
        self.root = root
        self.input_size = input_size
        self.config = config

        self.img_base = os.path.join(root, 'leftImg8bit')
        self.bbox_base = os.path.join(root, 'bboxGt')

        self.samples = self._collect_samples(split)
        self.transform = get_val_transforms(input_size)

    def _collect_samples(self, split: str) -> List[Dict]:
        """Collect all image + annotation paths."""
        samples = []

        # Determine which splits to load
        if split == 'all':
            splits = ['test', 'test_extra']
        else:
            splits = [split]

        for s in splits:
            img_split_dir = os.path.join(self.img_base, s)
            bbox_split_dir = os.path.join(self.bbox_base, s)

            if not os.path.exists(img_split_dir):
                continue

            # Walk through category subdirectories
            for category in sorted(os.listdir(img_split_dir)):
                cat_img_dir = os.path.join(img_split_dir, category)
                cat_bbox_dir = os.path.join(bbox_split_dir, category)

                if not os.path.isdir(cat_img_dir):
                    continue

                for fname in sorted(os.listdir(cat_img_dir)):
                    if not fname.endswith('_leftImg8bit.png'):
                        continue

                    img_path = os.path.join(cat_img_dir, fname)
                    base = fname.replace('_leftImg8bit.png', '')
                    bbox_path = os.path.join(cat_bbox_dir, f"{base}.txt")

                    samples.append({
                        'image': img_path,
                        'bbox': bbox_path if os.path.exists(bbox_path) else None,
                        'split': s,
                        'category': category,
                        'base': base,
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_driving_bboxes(self, bbox_path: str, img_w: int, img_h: int) -> torch.Tensor:
        """
        Load Foggy Driving bbox annotations.

        Format: class_id x1 y1 x2 y2 (pixel coordinates)
        Convert to YOLO format: class_id cx cy w h (normalized)
        """
        if bbox_path is None or not os.path.exists(bbox_path):
            return torch.zeros((0, 5), dtype=torch.float32)

        bboxes = []
        with open(bbox_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(float(parts[0]))
                    x1, y1, x2, y2 = [float(p) for p in parts[1:5]]

                    # Convert to YOLO format
                    cx = (x1 + x2) / 2.0 / img_w
                    cy = (y1 + y2) / 2.0 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h

                    # Clamp
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))

                    bboxes.append([cls, cx, cy, w, h])

        if len(bboxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(bboxes, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image']).convert('RGB')
        img_w, img_h = image.size  # PIL: (width, height)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load bboxes (convert from pixel coords to normalized YOLO)
        bboxes = self._load_driving_bboxes(sample['bbox'], img_w, img_h)

        # Build target
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