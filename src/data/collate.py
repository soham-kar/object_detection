"""Collate functions for WRDNet dataloaders.

Handles variable-length bounding box lists that standard
torch.utils.data.default_collate cannot stack.
"""

import torch
from typing import List, Dict, Any


def wrdnet_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for WRDNet batches with variable-length bboxes.

    Args:
        batch: list of dicts, each from a dataset __getitem__
    Returns:
        Collated dict with stacked tensors and list of bbox tensors.
    """
    result = {}

    # Stack image tensors (all same shape)
    result['image'] = torch.stack([b['image'] for b in batch])

    # Stack clear_gt if present (all samples should have it or none)
    if 'clear_gt' in batch[0] and batch[0]['clear_gt'] is not None:
        result['clear_gt'] = torch.stack([b['clear_gt'] for b in batch])
    else:
        result['clear_gt'] = None

    # Stack depth_gt if present
    if 'depth_gt' in batch[0] and batch[0]['depth_gt'] is not None:
        result['depth_gt'] = torch.stack([b['depth_gt'] for b in batch])
    else:
        result['depth_gt'] = None

    # Bboxes: variable length, keep as list of [N, 5] tensors
    if 'bboxes' in batch[0]:
        result['bboxes'] = [b['bboxes'] for b in batch]
    else:
        result['bboxes'] = None

    # Pass through any string fields (e.g., label_path)
    for key in batch[0]:
        if key not in ['image', 'clear_gt', 'depth_gt', 'bboxes']:
            result[key] = [b[key] for b in batch]

    return result


def paired_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for paired (synthetic, real) domain adaptation batches.

    Each item in batch is {'synth': [sample, sample, sample], 'real': sample}
    where 'synth' is a LIST of synth_per_real samples (1:3 ratio) and 'real'
    is a single sample. The collate flattens all synth samples into one batch
    and all real samples into another, so the resulting 'synth' batch is
    synth_per_real × larger than the 'real' batch.
    """
    # Flatten all synth samples (each item contributes synth_per_real samples)
    synth_batch = []
    for b in batch:
        synth_batch.extend(b['synth'])

    # Collect all real samples (one per item)
    real_batch = [b['real'] for b in batch]

    return {
        'synth': wrdnet_collate_fn(synth_batch),
        'real': wrdnet_collate_fn(real_batch),
    }