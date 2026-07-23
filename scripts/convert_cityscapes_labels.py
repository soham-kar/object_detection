#!/usr/bin/env python3
"""
Convert Cityscapes gtFine instance IDs to YOLO bounding box format.

Cityscapes instanceIds.png encoding:
  - Pixel value = class_id * 1000 + instance_id
  - class_id 24=person, 25=rider, 26=car, 27=truck, 28=bus, 31=train, 32=motorcycle, 33=bicycle
  - Values < 1000 are "stuff" classes (road, building, etc.) — ignored for detection

YOLO format (one .txt per image):
  class_id cx cy w h  (all normalized 0-1, one line per object)

Usage:
  python scripts/convert_cityscapes_labels.py --root data/cityscapes --split train
  python scripts/convert_cityscapes_labels.py --root data/cityscapes --split val
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Cityscapes instance class ID → YOLO class ID mapping
# Only 8 "thing" classes (humans + vehicles) are used for detection
CITYSCAPES_TO_YOLO = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: 5,  # train
    32: 6,  # motorcycle
    33: 7,  # bicycle
}

# Class names for verification
CLASS_NAMES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def instance_ids_to_yolo(instance_ids_path: str, img_width: int, img_height: int) -> list:
    """
    Convert a single Cityscapes instanceIds.png to YOLO bbox list.

    Args:
        instance_ids_path: path to *_gtFine_instanceIds.png
        img_width: original image width (2048 for Cityscapes)
        img_height: original image height (1024 for Cityscapes)
    Returns:
        list of (class_id, cx, cy, w, h) tuples, all normalized [0, 1]
    """
    inst_map = np.array(Image.open(instance_ids_path), dtype=np.int32)
    bboxes = []

    # Get unique instance IDs (exclude 0 = void, and values < 1000 = stuff)
    unique_ids = np.unique(inst_map)

    for inst_id in unique_ids:
        if inst_id < 1000:
            continue  # Skip stuff classes (road, building, sky, etc.)

        class_id_raw = inst_id // 1000

        if class_id_raw not in CITYSCAPES_TO_YOLO:
            continue  # Skip classes not in our 8 detection classes

        yolo_class = CITYSCAPES_TO_YOLO[class_id_raw]

        # Create binary mask for this instance
        mask = (inst_map == inst_id)
        if mask.sum() < 10:  # Skip tiny instances (< 10 pixels)
            continue

        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Convert to YOLO format: cx, cy, w, h (normalized)
        cx = (x_min + x_max) / 2.0 / img_width
        cy = (y_min + y_max) / 2.0 / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))

        bboxes.append((yolo_class, cx, cy, w, h))

    return bboxes


def write_yolo_labels(bboxes: list, output_path: str):
    """Write YOLO format labels to .txt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for class_id, cx, cy, w, h in bboxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def convert_split(root: str, split: str):
    """
    Convert all images in a split (train/val/test).

    Args:
        root: Cityscapes root directory
        split: 'train', 'val', or 'test'
    """
    gt_dir = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', split)
    output_dir = os.path.join(root, 'labels', split)

    if not os.path.exists(gt_dir):
        print(f"ERROR: gtFine directory not found: {gt_dir}")
        sys.exit(1)

    # Find all instance ID files
    instance_files = sorted(Path(gt_dir).rglob("*_gtFine_instanceIds.png"))
    print(f"Found {len(instance_files)} instance ID files in {split}")

    total_bboxes = 0
    empty_count = 0
    class_counts = {i: 0 for i in range(8)}

    for inst_path in tqdm(instance_files, desc=f"Converting {split}"):
        # Derive base name: aachen/aachen_000000_000019
        rel_path = inst_path.relative_to(gt_dir)
        city = rel_path.parent.name
        base = inst_path.name.replace('_gtFine_instanceIds.png', '')

        # Cityscapes images are 2048×1024
        img_width, img_height = 2048, 1024

        # Convert
        bboxes = instance_ids_to_yolo(str(inst_path), img_width, img_height)

        # Write
        output_path = os.path.join(output_dir, city, f"{base}.txt")
        write_yolo_labels(bboxes, output_path)

        # Stats
        total_bboxes += len(bboxes)
        if len(bboxes) == 0:
            empty_count += 1
        for class_id, _, _, _, _ in bboxes:
            class_counts[class_id] += 1

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Conversion complete: {split}")
    print(f"  Total images: {len(instance_files)}")
    print(f"  Total bboxes: {total_bboxes}")
    print(f"  Avg bboxes/image: {total_bboxes / len(instance_files):.1f}")
    print(f"  Empty images: {empty_count}")
    print(f"  Class distribution:")
    for class_id, count in class_counts.items():
        print(f"    {class_id} ({CLASS_NAMES[class_id]:12s}): {count:6d}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes gtFine to YOLO bbox format'
    )
    parser.add_argument('--root', type=str, required=True,
                        help='Cityscapes root directory (containing gtFine_trainvaltest/)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to convert')
    args = parser.parse_args()

    convert_split(args.root, args.split)


if __name__ == '__main__':
    main()