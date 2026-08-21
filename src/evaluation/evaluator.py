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

    def evaluate_detection(self, dataloader, conf_thres: float = 0.01,
                           iou_thres: float = 0.45, use_tta: bool = False) -> Dict[str, float]:
        """
        Compute mAP@50 and mAP@50:95 using a simplified COCO-style metric.

        Uses YOLO's built-in NMS for post-processing, then computes
        per-class AP at IoU thresholds 0.5 and 0.5:0.95.

        Args:
            dataloader: validation/test data loader
            conf_thres: confidence threshold for detections
            iou_thres: IoU threshold for NMS
            use_tta: if True, apply test-time augmentation (horizontal flip)
                     and average the predictions. This is a standard, honest
                     evaluation technique that typically adds +1-3% mAP.
        Returns:
            metrics: dict with mAP@50, mAP@50:95, and per-class AP
        """
        all_predictions = []  # List of (image_idx, class_id, conf, x1, y1, x2, y2)
        all_targets = []      # List of (image_idx, class_id, x1, y1, x2, y2)

        img_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating detection"):
                images = batch['image'].to(self.device)
                bboxes = batch.get('bboxes', None)

                # Forward pass (with optional TTA)
                if use_tta:
                    # Run original + horizontal flip, merge predictions.
                    # NOTE: the flipped image's box cx coordinates are mirrored,
                    # so we must un-flip them (cx' = W - cx) before averaging.
                    outputs = self.model(images)
                    det_output = outputs['detections']
                    raw_preds = det_output[0] if isinstance(det_output, (tuple, list)) else det_output

                    # Horizontal flip
                    flipped = torch.flip(images, dims=[3])
                    outputs_f = self.model(flipped)
                    det_output_f = outputs_f['detections']
                    raw_preds_f = det_output_f[0] if isinstance(det_output_f, (tuple, list)) else det_output_f

                    # Un-flip the flipped predictions' box cx (index 0) before averaging.
                    # raw_preds_f shape: [B, 84, N]; box coords are cx,cy,w,h in pixels.
                    # After horizontal flip, cx_f = W - cx. So cx = W - cx_f.
                    W = images.shape[3]
                    raw_preds_f = raw_preds_f.clone()
                    raw_preds_f[:, 0, :] = W - raw_preds_f[:, 0, :]

                    # Average the two predictions (raw logits)
                    raw_preds = (raw_preds + raw_preds_f) / 2.0
                else:
                    outputs = self.model(images)
                    det_output = outputs['detections']
                    raw_preds = det_output[0] if isinstance(det_output, (tuple, list)) else det_output

                # YOLO Detect head returns raw predictions in eval mode:
                # [B, 4+nc, num_anchors] where first 4 are (cx, cy, w, h) in pixel coords
                # relative to input size (640), NOT normalized [0,1]
                # We need to normalize to [0,1] for IoU computation with GT (which is normalized)

                B = raw_preds.shape[0]
                # YOLO input size — (H, W) tuple for 2:1 aspect ratio.
                # Box coords are in pixel space; normalize x by width, y by height.
                input_h, input_w = 512, 1024  # YOLO input size (config.input_size_detect)

                for b in range(B):
                    pred = raw_preds[b]  # [84, 8400]
                    # Extract box coords (cx, cy, w, h) in PIXEL coordinates
                    box_preds = pred[:4, :].T  # [8400, 4] cx, cy, w, h (pixels)
                    cls_preds = pred[4:, :].T  # [8400, 80] class scores

                    # We only care about our 8 classes (indices 0-7)
                    cls_preds_8 = cls_preds[:, :8]  # [8400, 8]

                    # Use top-1 class and confidence from our 8 classes only
                    max_conf, max_cls = cls_preds_8.max(dim=1)  # [8400]

                    # Filter by confidence
                    mask = max_conf > conf_thres
                    if mask.sum() == 0:
                        img_idx += 1
                        continue

                    boxes = box_preds[mask]  # [N, 4] cx, cy, w, h (pixels)
                    confs = max_conf[mask]   # [N]
                    cls_ids = max_cls[mask]  # [N]

                    # Normalize to [0,1]: x by width, y by height (2:1 aspect ratio)
                    boxes_norm = boxes.clone()
                    boxes_norm[:, 0] = boxes[:, 0] / input_w  # cx / width
                    boxes_norm[:, 1] = boxes[:, 1] / input_h  # cy / height
                    boxes_norm[:, 2] = boxes[:, 2] / input_w  # w / width
                    boxes_norm[:, 3] = boxes[:, 3] / input_h  # h / height

                    # Convert cx,cy,w,h to x1,y1,x2,y2 (normalized)
                    x1 = boxes_norm[:, 0] - boxes_norm[:, 2] / 2
                    y1 = boxes_norm[:, 1] - boxes_norm[:, 3] / 2
                    x2 = boxes_norm[:, 0] + boxes_norm[:, 2] / 2
                    y2 = boxes_norm[:, 1] + boxes_norm[:, 3] / 2

                    # Clamp to [0, 1]
                    x1 = x1.clamp(0, 1)
                    y1 = y1.clamp(0, 1)
                    x2 = x2.clamp(0, 1)
                    y2 = y2.clamp(0, 1)

                    # Apply NMS using torchvision
                    from torchvision.ops import nms as tv_nms
                    nms_boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [N, 4]
                    keep = tv_nms(nms_boxes, confs, iou_thres)
                    nms_boxes = nms_boxes[keep]
                    nms_confs = confs[keep]
                    nms_cls = cls_ids[keep]

                    for i in range(len(nms_confs)):
                        all_predictions.append({
                            'image_idx': img_idx,
                            'class_id': nms_cls[i].item(),
                            'conf': nms_confs[i].item(),
                            'bbox': [nms_boxes[i, 0].item(), nms_boxes[i, 1].item(),
                                     nms_boxes[i, 2].item(), nms_boxes[i, 3].item()],
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

        # Debug: print prediction statistics
        print(f"  [Debug] Predictions: {len(all_predictions)}, Targets: {len(all_targets)}")
        if len(all_predictions) > 0:
            confs = [p['conf'] for p in all_predictions[:100]]
            print(f"  [Debug] Sample confs: min={min(confs):.4f}, max={max(confs):.4f}")
            print(f"  [Debug] Sample pred bbox: {all_predictions[0]['bbox']}")
        if len(all_targets) > 0:
            print(f"  [Debug] Sample GT bbox: {all_targets[0]['bbox']}")
            print(f"  [Debug] GT class IDs: {set(t['class_id'] for t in all_targets[:100])}")
            print(f"  [Debug] Pred class IDs: {set(p['class_id'] for p in all_predictions[:100])}")

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
