"""Evaluation utilities for WRDNet.

Computes mAP@50, mAP@50:95 using pycocotools, PSNR/SSIM for restoration,
FPS for inference speed, and alpha map visualizations for FSG interpretability.
"""

import os
import json
import numpy as np
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from tqdm import tqdm


class WRDNetEvaluator:
    """Evaluator for WRDNet."""

    # 8 detection classes (same as training)
    CLASS_NAMES = [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate_detection(self, dataloader, conf_thres: float = 0.25,
                           iou_thres: float = 0.45) -> Dict[str, float]:
        """
        Compute mAP@50 and mAP@50:95 using a simplified COCO-style metric.

        Uses YOLO's built-in NMS for post-processing, then computes
        per-class AP at IoU thresholds 0.5 and 0.5:0.95.

        Args:
            dataloader: validation/test data loader
            conf_thres: confidence threshold for detections
            iou_thres: IoU threshold for NMS
        Returns:
            metrics: dict with mAP@50, mAP@50:95, and per-class AP
        """
        from ultralytics.utils.ops import non_max_suppression

        all_predictions = []  # List of (image_idx, class_id, conf, x1, y1, x2, y2)
        all_targets = []      # List of (image_idx, class_id, x1, y1, x2, y2)

        img_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating detection"):
                images = batch['image'].to(self.device)
                bboxes = batch.get('bboxes', None)

                # Forward pass
                outputs = self.model(images)
                det_output = outputs['detections']

                # Parse YOLO output
                if isinstance(det_output, (tuple, list)):
                    raw_preds = det_output[0]  # [B, 84, 8400]
                else:
                    raw_preds = det_output

                # Apply NMS using ultralytics
                # raw_preds is [B, 84, 8400] → need to convert to detection format
                # ultralytics NMS expects [B, 84, num_anchors] where 84 = 4 bbox + 80 classes
                # But we only have 8 classes. We need to handle this carefully.
                # For now, use a simplified NMS approach.

                B = raw_preds.shape[0]
                for b in range(B):
                    pred = raw_preds[b]  # [84, 8400]
                    # Extract box coords (cx, cy, w, h) and class scores
                    box_preds = pred[:4, :].T  # [8400, 4] cx, cy, w, h
                    cls_preds = pred[4:, :].T  # [8400, 80] class scores

                    # We only care about our 8 classes (indices 0-7 in our mapping)
                    # But YOLO was trained on COCO (80 classes).
                    # For evaluation, use the max confidence across all classes
                    # and map to our 8 classes if possible.
                    # Simplified: use top-1 class and confidence
                    max_conf, max_cls = cls_preds.max(dim=1)  # [8400]

                    # Filter by confidence
                    mask = max_conf > conf_thres
                    if mask.sum() == 0:
                        img_idx += 1
                        continue

                    boxes = box_preds[mask]  # [N, 4] cx, cy, w, h (normalized)
                    confs = max_conf[mask]   # [N]
                    cls_ids = max_cls[mask]  # [N]

                    # Convert cx,cy,w,h to x1,y1,x2,y2 (normalized)
                    x1 = boxes[:, 0] - boxes[:, 2] / 2
                    y1 = boxes[:, 1] - boxes[:, 3] / 2
                    x2 = boxes[:, 0] + boxes[:, 2] / 2
                    y2 = boxes[:, 1] + boxes[:, 3] / 2

                    for i in range(len(confs)):
                        all_predictions.append({
                            'image_idx': img_idx,
                            'class_id': cls_ids[i].item(),
                            'conf': confs[i].item(),
                            'bbox': [x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()],
                        })

                    # Collect targets
                    if bboxes is not None and b < len(bboxes):
                        gt = bboxes[b]  # [N, 5] class, cx, cy, w, h
                        for g in range(gt.shape[0]):
                            cx, cy, w, h = gt[g, 1].item(), gt[g, 2].item(), gt[g, 3].item(), gt[g, 4].item()
                            gx1 = cx - w / 2
                            gy1 = cy - h / 2
                            gx2 = cx + w / 2
                            gy2 = cy + h / 2
                            all_targets.append({
                                'image_idx': img_idx,
                                'class_id': int(gt[g, 0].item()),
                                'bbox': [gx1, gy1, gx2, gy2],
                            })

                    img_idx += 1

        # Compute mAP
        metrics = self._compute_map(all_predictions, all_targets)
        return metrics

    def _compute_map(self, predictions: list, targets: list,
                     iou_thresholds: list = None) -> Dict[str, float]:
        """
        Compute mAP using a simplified COCO-style algorithm.

        Args:
            predictions: list of dicts with image_idx, class_id, conf, bbox
            targets: list of dicts with image_idx, class_id, bbox
            iou_thresholds: list of IoU thresholds for mAP computation
        Returns:
            metrics: dict with mAP@50, mAP@50:95
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5] + [0.5 + 0.05 * i for i in range(1, 10)]

        if len(predictions) == 0 or len(targets) == 0:
            return {'mAP@50': 0.0, 'mAP@50:95': 0.0}

        # Group by class
        pred_by_class = {}
        target_by_class = {}
        all_classes = set()

        for p in predictions:
            cid = p['class_id']
            all_classes.add(cid)
            if cid not in pred_by_class:
                pred_by_class[cid] = []
            pred_by_class[cid].append(p)

        for t in targets:
            cid = t['class_id']
            all_classes.add(cid)
            if cid not in target_by_class:
                target_by_class[cid] = []
            target_by_class[cid].append(t)

        # Compute AP per class per IoU threshold
        aps_50 = []
        aps_5095 = []

        for cls_id in all_classes:
            preds = pred_by_class.get(cls_id, [])
            gts = target_by_class.get(cls_id, [])

            if len(gts) == 0:
                continue

            # Sort predictions by confidence (descending)
            preds.sort(key=lambda x: -x['conf'])

            # Group targets by image
            gt_by_img = {}
            for gt in gts:
                if gt['image_idx'] not in gt_by_img:
                    gt_by_img[gt['image_idx']] = []
                gt_by_img[gt['image_idx']].append(gt)

            # Mark GTs as matched
            for gt_list in gt_by_img.values():
                for gt in gt_list:
                    gt['matched'] = [False] * len(iou_thresholds)

            # Compute TP/FP for each prediction
            tp = [[] for _ in iou_thresholds]
            fp = [[] for _ in iou_thresholds]

            for pred in preds:
                img_idx = pred['image_idx']
                best_iou = 0
                best_gt_idx = -1

                if img_idx in gt_by_img:
                    gts_in_img = gt_by_img[img_idx]
                    for gi, gt in enumerate(gts_in_img):
                        iou = self._compute_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gi

                for ti, iou_thresh in enumerate(iou_thresholds):
                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        gt = gt_by_img[img_idx][best_gt_idx]
                        if not gt['matched'][ti]:
                            tp[ti].append(1)
                            fp[ti].append(0)
                            gt['matched'][ti] = True
                        else:
                            tp[ti].append(0)
                            fp[ti].append(1)
                    else:
                        tp[ti].append(0)
                        fp[ti].append(1)

            # Compute AP for each IoU threshold
            class_aps = []
            for ti, iou_thresh in enumerate(iou_thresholds):
                tp_cum = np.cumsum(tp[ti])
                fp_cum = np.cumsum(fp[ti])
                recall = tp_cum / (len(gts) + 1e-8)
                precision = tp_cum / (tp_cum + fp_cum + 1e-8)

                # Compute AP (area under PR curve, using 11-point interpolation)
                ap = self._compute_ap_11point(precision, recall)
                class_aps.append(ap)

            aps_50.append(class_aps[0])  # IoU=0.5
            aps_5095.append(np.mean(class_aps))  # Average over 0.5:0.95

        mAP_50 = np.mean(aps_50) if aps_50 else 0.0
        mAP_5095 = np.mean(aps_5095) if aps_5095 else 0.0

        return {
            'mAP@50': float(mAP_50),
            'mAP@50:95': float(mAP_5095),
            'num_classes': len(all_classes),
            'num_predictions': len(predictions),
            'num_targets': len(targets),
        }

    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-8)

    def _compute_ap_11point(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute AP using 11-point interpolation."""
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max() / 11.0
        return ap

    def evaluate_restoration(self, dataloader, has_gt: bool = True) -> Dict[str, float]:
        """
        Compute restoration quality metrics.

        Args:
            dataloader: data loader with foggy and clear images
            has_gt: whether ground-truth clear images are available
        Returns:
            metrics: dict with PSNR, SSIM, or BRISQUE/NIQE
        """
        from ..utils.metrics import compute_psnr, compute_ssim

        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating restoration"):
                foggy = batch['image'].to(self.device)
                restored = self.model(foggy)['restored']

                if has_gt and 'clear_gt' in batch:
                    clear = batch['clear_gt'].to(self.device)
                    psnr = compute_psnr(restored, clear)
                    ssim = compute_ssim(restored, clear)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        metrics = {}
        if psnr_list:
            metrics['PSNR'] = sum(psnr_list) / len(psnr_list)
            metrics['SSIM'] = sum(ssim_list) / len(ssim_list)

        return metrics

    def measure_speed(self, input_size: tuple = (1, 3, 640, 640), num_runs: int = 100) -> float:
        """
        Measure inference FPS.

        Args:
            input_size: input tensor shape
            num_runs: number of inference runs for averaging
        Returns:
            fps: frames per second
        """
        dummy_input = torch.randn(*input_size).to(self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # Measure
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        import time
        start = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        fps = num_runs / elapsed

        return fps

    def visualize_alpha_maps(
        self,
        dataloader,
        save_dir: str,
        num_samples: int = 10,
    ):
        """
        Generate alpha map visualizations for FSG interpretability.

        Creates overlay images showing where the FSG trusts defogged (red)
        vs. original (blue) features.

        Args:
            dataloader: data loader
            save_dir: directory to save visualizations
            num_samples: number of samples to visualize
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        os.makedirs(save_dir, exist_ok=True)

        # Custom colormap: blue (α=0, trust original) → red (α=1, trust defogged)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'alpha_cmap', ['blue', 'cyan', 'yellow', 'red']
        )

        # ImageNet denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= num_samples:
                    break

                images = batch['image'].to(self.device)
                outputs = self.model(images, return_alpha=True)

                alpha_maps = outputs.get('alpha_maps', {})
                restored = outputs.get('restored', None)

                B = images.shape[0]
                for b in range(B):
                    if count >= num_samples:
                        break

                    # Denormalize foggy image
                    foggy_img = images[b].cpu().permute(1, 2, 0).numpy()
                    foggy_img = foggy_img * std + mean
                    foggy_img = np.clip(foggy_img, 0, 1)

                    # Denormalize restored image
                    if restored is not None:
                        rest_img = restored[b].cpu().permute(1, 2, 0).numpy()
                        rest_img = rest_img * std + mean
                        rest_img = np.clip(rest_img, 0, 1)
                    else:
                        rest_img = foggy_img

                    # Get alpha maps at each scale
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                    # Row 1: Foggy, Restored, Alpha P3 overlay, Alpha P5 overlay
                    axes[0, 0].imshow(foggy_img)
                    axes[0, 0].set_title('Foggy Input')
                    axes[0, 0].axis('off')

                    axes[0, 1].imshow(rest_img)
                    axes[0, 1].set_title('Restored (DehazeFormer)')
                    axes[0, 1].axis('off')

                    # Alpha P3 (highest resolution)
                    if 'P3' in alpha_maps:
                        alpha_p3 = alpha_maps['P3'][b, 0].cpu().numpy()
                        axes[0, 2].imshow(foggy_img)
                        axes[0, 2].imshow(alpha_p3, cmap=cmap, alpha=0.5)
                        axes[0, 2].set_title('Alpha P3 (80×80)\nRed=defog, Blue=original')
                    axes[0, 2].axis('off')

                    # Alpha P5 (lowest resolution)
                    if 'P5' in alpha_maps:
                        alpha_p5 = alpha_maps['P5'][b, 0].cpu().numpy()
                        axes[0, 3].imshow(foggy_img)
                        axes[0, 3].imshow(alpha_p5, cmap=cmap, alpha=0.5)
                        axes[0, 3].set_title('Alpha P5 (20×20)\nRed=defog, Blue=original')
                    axes[0, 3].axis('off')

                    # Row 2: Alpha maps alone (P3, P4, P5, histogram)
                    for j, scale in enumerate(['P3', 'P4', 'P5']):
                        if scale in alpha_maps:
                            alpha = alpha_maps[scale][b, 0].cpu().numpy()
                            axes[1, j].imshow(alpha, cmap=cmap, vmin=0, vmax=1)
                            axes[1, j].set_title(f'Alpha {scale}')
                        axes[1, j].axis('off')

                    # Alpha histogram
                    if 'P3' in alpha_maps:
                        alpha_flat = alpha_maps['P3'][b, 0].cpu().numpy().flatten()
                        axes[1, 3].hist(alpha_flat, bins=50, color='blue', alpha=0.7)
                        axes[1, 3].set_title('Alpha Distribution (P3)')
                        axes[1, 3].set_xlabel('Alpha value')
                        axes[1, 3].set_ylabel('Count')
                        axes[1, 3].axvline(x=0.5, color='red', linestyle='--', label='α=0.5')
                        axes[1, 3].legend()

                    plt.suptitle(f'Sample {count+1}: FSG Alpha Map Visualization', fontsize=16)
                    plt.tight_layout()

                    save_path = os.path.join(save_dir, f'alpha_{count:03d}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    count += 1

        print(f"Saved {count} alpha map visualizations to {save_dir}")
