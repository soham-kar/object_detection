"""Synchronized image transforms for WRDNet.

All transforms operate on (image, target_dict) pairs where target_dict
can contain 'clear_gt', 'depth_gt', and 'bboxes' that must be transformed
in sync with the image.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


class Resize:
    """Resize image and all targets to target size."""

    def __init__(self, size=(640, 640)):
        self.size = size  # (H, W)

    def __call__(self, image, target=None):
        h, w = self.size
        image = TF.resize(image, [h, w])

        if target is not None:
            if 'clear_gt' in target and target['clear_gt'] is not None:
                target['clear_gt'] = TF.resize(target['clear_gt'], [h, w])
            if 'depth_gt' in target and target['depth_gt'] is not None:
                target['depth_gt'] = TF.resize(
                    target['depth_gt'], [h, w],
                    interpolation=TF.InterpolationMode.NEAREST,
                )
            # Bboxes are normalized [0,1] so they don't change on resize

        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip — synchronizes image, clear_gt, depth_gt, bboxes."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = TF.hflip(image)

            if target is not None:
                if 'clear_gt' in target and target['clear_gt'] is not None:
                    target['clear_gt'] = TF.hflip(target['clear_gt'])
                if 'depth_gt' in target and target['depth_gt'] is not None:
                    target['depth_gt'] = TF.hflip(target['depth_gt'])
                if 'bboxes' in target and target['bboxes'] is not None:
                    bboxes = target['bboxes'].clone()
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]  # cx -> 1 - cx
                    target['bboxes'] = bboxes

        return image, target


class Normalize:
    """Normalize image to ImageNet statistics. Depth is normalized separately."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image, target=None):
        image = (image - self.mean) / self.std

        if target is not None:
            if 'clear_gt' in target and target['clear_gt'] is not None:
                target['clear_gt'] = (target['clear_gt'] - self.mean) / self.std

        return image, target


class DepthNormalize:
    """Normalize depth from [0, max_depth] to [0, 1]."""

    def __init__(self, max_depth=80.0):
        self.max_depth = max_depth

    def __call__(self, image, target=None):
        if target is not None:
            if 'depth_gt' in target and target['depth_gt'] is not None:
                target['depth_gt'] = torch.clamp(
                    target['depth_gt'] / self.max_depth, 0.0, 1.0
                )
        return image, target


class Compose:
    """Compose multiple transforms. Each receives (image, target)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_train_transforms(input_size=640, max_depth=80.0):
    """Get training transforms."""
    return Compose([
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=0.5),
        Normalize(),
        DepthNormalize(max_depth=max_depth),
    ])


def get_val_transforms(input_size=640, max_depth=80.0):
    """Get validation/test transforms (no augmentation)."""
    return Compose([
        Resize((input_size, input_size)),
        Normalize(),
        DepthNormalize(max_depth=max_depth),
    ])
