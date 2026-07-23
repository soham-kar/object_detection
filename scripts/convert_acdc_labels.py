#!/usr/bin/env python3
"""
Convert ACDC COCO-format detection annotations to YOLO bbox format.

ACDC provides annotations in COCO format JSON files:
  gt_detection_trainval/gt_detection/fog/instancesonly_fog_train_gt_detection.json
  gt_detection_trainval/gt_detection/fog/instancesonly_fog_val_gt_detection.json

COCO bbox format: [x, y, width, height] in absolute pixels
YOLO format:      class_id cx cy w h (normalized 0-1)

Category IDs in ACDC match Cityscapes raw IDs (24-33).
We map them to the same 0-7 YOLO IDs used for Cityscapes.

Usage:
  python scripts/convert_acdc_labels.py \
    --json data/gt_detection_trainval/gt_detection/fog/instancesonly_fog_train_gt_detection.json \
    --output data/acdc_labels/train

  python scripts/convert_acdc_labels.py \
    --json data/gt_detection_trainval/gt_detection/fog/instancesonly_fog_val_gt_detection.json \
    --output data/acdc_labels/val
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

# ACDC category IDs (same as Cityscapes raw IDs) → YOLO class IDs
ACDC_TO_YOLO = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: 5,  # train
    32: 6,  # motorcycle
    33: 7,  # bicycle
}

CLASS_NAMES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def convert_coco_to_yolo(coco_json_path: str, output_dir: str):
    """
    Convert ACDC COCO detection JSON to YOLO .txt files.

    Args:
        coco_json_path: path to instancesonly_fog_{split}_gt_detection.json
        output_dir: output directory for YOLO .txt files
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build image ID → image info mapping
    images = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Convert each image's annotations
    total_bboxes = 0
    empty_count = 0
    class_counts = {i: 0 for i in range(8)}

    os.makedirs(output_dir, exist_ok=True)

    for img_id, img_info in tqdm(images.items(), desc="Converting"):
        # ACDC file_name format: "fog/train/GP010475/GP010475_frame_001043_rgb_anon.png"
        # We want: output_dir/GP010475/GP010475_frame_001043.txt
        file_name = img_info['file_name']  # e.g., "fog/train/GP010475/GP010475_frame_001043_rgb_anon.png"
        parts = file_name.split('/')
        sequence = parts[2] if len(parts) > 2 else ''
        frame_name = parts[3].replace('_rgb_anon.png', '.txt') if len(parts) > 3 else ''

        output_path = os.path.join(output_dir, sequence, frame_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img_w = img_info['width']
        img_h = img_info['height']

        anns = annotations_by_image.get(img_id, [])
        bboxes = []

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in ACDC_TO_YOLO:
                continue

            yolo_class = ACDC_TO_YOLO[cat_id]

            # COCO bbox: [x, y, width, height] in pixels
            x, y, w, h = ann['bbox']

            # Convert to YOLO format: cx, cy, w, h (normalized)
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w_norm = max(0.001, min(1.0, w_norm))
            h_norm = max(0.001, min(1.0, h_norm))

            bboxes.append((yolo_class, cx, cy, w_norm, h_norm))
            class_counts[yolo_class] += 1

        # Write YOLO file
        with open(output_path, 'w') as f:
            for class_id, cx, cy, w, h in bboxes:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        total_bboxes += len(bboxes)
        if len(bboxes) == 0:
            empty_count += 1

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Conversion complete: {coco_json_path}")
    print(f"  Total images: {len(images)}")
    print(f"  Total bboxes: {total_bboxes}")
    print(f"  Avg bboxes/image: {total_bboxes / len(images):.1f}")
    print(f"  Empty images: {empty_count}")
    print(f"  Class distribution:")
    for class_id, count in class_counts.items():
        print(f"    {class_id} ({CLASS_NAMES[class_id]:12s}): {count:6d}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ACDC COCO detection annotations to YOLO format'
    )
    parser.add_argument('--json', type=str, required=True,
                        help='Path to instancesonly_fog_{split}_gt_detection.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for YOLO .txt files')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)

    convert_coco_to_yolo(args.json, args.output)


if __name__ == '__main__':
    main()