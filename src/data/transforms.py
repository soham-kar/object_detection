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


class ColorJitter:
    """Random color jitter (brightness, contrast, saturation, hue).

    Applied to BOTH the foggy image and the clear GT so the dehazing
    supervision stays consistent. Depth is unaffected.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Random jitter factors
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            s = 1.0 + random.uniform(-self.saturation, self.saturation)
            h = random.uniform(-self.hue, self.hue)

            # Apply to foggy image
            image = TF.adjust_brightness(image, b)
            image = TF.adjust_contrast(image, c)
            image = TF.adjust_saturation(image, s)
            image = TF.adjust_hue(image, h)

            # Apply same factors to clear GT (keeps dehazing supervision consistent)
            if target is not None and 'clear_gt' in target and target['clear_gt'] is not None:
                clear = target['clear_gt']
                clear = TF.adjust_brightness(clear, b)
                clear = TF.adjust_contrast(clear, c)
                clear = TF.adjust_saturation(clear, s)
                clear = TF.adjust_hue(clear, h)
                target['clear_gt'] = clear

        return image, target


class RandomScale:
    """Random scale + pad to target size.

    Scales the image by a random factor, then pads/crops back to the target
    size. Bboxes are scaled accordingly. This simulates objects at different
    distances/sizes, which is critical for detection generalization.
    """

    def __init__(self, scale_range=(0.5, 1.5), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            h, w = image.shape[-2:]
            scale = random.uniform(*self.scale_range)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize image and clear_gt
            image = TF.resize(image, [new_h, new_w])
            if target is not None and 'clear_gt' in target and target['clear_gt'] is not None:
                target['clear_gt'] = TF.resize(target['clear_gt'], [new_h, new_w])
            if target is not None and 'depth_gt' in target and target['depth_gt'] is not None:
                target['depth_gt'] = TF.resize(
                    target['depth_gt'], [new_h, new_w],
                    interpolation=TF.InterpolationMode.NEAREST,
                )

            # Pad or crop back to target size
            if new_h >= h and new_w >= w:
                # Pad (center)
                pad_h = (new_h - h) // 2
                pad_w = (new_w - w) // 2
                image = TF.crop(image, pad_h, pad_w, h, w)
                if target is not None and 'clear_gt' in target and target['clear_gt'] is not None:
                    target['clear_gt'] = TF.crop(target['clear_gt'], pad_h, pad_w, h, w)
                if target is not None and 'depth_gt' in target and target['depth_gt'] is not None:
                    target['depth_gt'] = TF.crop(target['depth_gt'], pad_h, pad_w, h, w)
                # Bboxes: scale then shift by -pad
                if target is not None and 'bboxes' in target and target['bboxes'] is not None:
                    bboxes = target['bboxes'].clone()
                    bboxes[:, 1] = bboxes[:, 1] * scale - pad_w / w  # cx
                    bboxes[:, 2] = bboxes[:, 2] * scale - pad_h / h  # cy
                    bboxes[:, 3] = bboxes[:, 3] * scale  # w
                    bboxes[:, 4] = bboxes[:, 4] * scale  # h
                    target['bboxes'] = bboxes
            else:
                # Crop (random offset)
                max_y = h - new_h
                max_x = w - new_w
                y0 = random.randint(0, max_y) if max_y > 0 else 0
                x0 = random.randint(0, max_x) if max_x > 0 else 0
                image = TF.pad(image, [x0, y0, w - new_w - x0, h - new_h - y0], fill=0)
                if target is not None and 'clear_gt' in target and target['clear_gt'] is not None:
                    target['clear_gt'] = TF.pad(target['clear_gt'], [x0, y0, w - new_w - x0, h - new_h - y0], fill=0)
                if target is not None and 'depth_gt' in target and target['depth_gt'] is not None:
                    target['depth_gt'] = TF.pad(target['depth_gt'], [x0, y0, w - new_w - x0, h - new_h - y0], fill=0)
                # Bboxes: scale then shift by +x0/y0
                if target is not None and 'bboxes' in target and target['bboxes'] is not None:
                    bboxes = target['bboxes'].clone()
                    bboxes[:, 1] = bboxes[:, 1] * scale + x0 / w  # cx
                    bboxes[:, 2] = bboxes[:, 2] * scale + y0 / h  # cy
                    bboxes[:, 3] = bboxes[:, 3] * scale  # w
                    bboxes[:, 4] = bboxes[:, 4] * scale  # h
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


def get_train_transforms(input_size=(512, 1024), max_depth=80.0):
    """Get training transforms.

    Includes RandomScale, RandomHorizontalFlip, and ColorJitter to reduce
    overfitting on the small (~2975 image) Foggy Cityscapes training set.
    The model was memorizing the training set by epoch 4 (mAP peaked then
    declined). These augmentations force it to learn general shapes instead.

    input_size: (H, W) tuple for 2:1 aspect ratio (e.g., (512, 1024)).
    """
    # Normalize to (H, W) tuple
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return Compose([
        Resize(input_size),
        RandomScale(scale_range=(0.5, 1.5), p=0.5),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        Normalize(),
        DepthNormalize(max_depth=max_depth),
    ])


def get_val_transforms(input_size=(512, 1024), max_depth=80.0):
    """Get validation/test transforms (no augmentation)."""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    return Compose([
        Resize(input_size),
        Normalize(),
        DepthNormalize(max_depth=max_depth),
    ])
