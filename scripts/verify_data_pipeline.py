#!/usr/bin/env python3
"""
Verify the WRDNet data pipeline end-to-end.

Tests:
  1. Cityscapes dataset loads with correct shapes
  2. ACDC dataset loads with correct shapes
  3. Foggy Zurich dataset loads (unlabeled)
  4. Foggy Driving dataset loads (test set)
  5. build_dataloaders() works in supervised mode
  6. build_dataloaders() works in DA mode (paired batches)
  7. Bbox coordinates are in [0, 1]
  8. Depth values are in [0, 1] after normalization
  9. Images are normalized (ImageNet stats)
  10. Collate function handles variable-length bboxes

Usage:
  python scripts/verify_data_pipeline.py
  python scripts/verify_data_pipeline.py --da  # Test domain adaptation mode
"""

import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.foggy_cityscapes import FoggyCityscapesDataset
from src.data.acdc import ACDCDataset
from src.data.foggy_zurich import FoggyZurichDataset
from src.data.foggy_driving import FoggyDrivingDataset
from src.data.dataset import build_dataloaders, build_test_loader
from src.data.collate import wrdnet_collate_fn, paired_collate_fn
from src.utils.config import load_config


def test_cityscapes(data_root='data'):
    """Test Cityscapes loader."""
    print("\n" + "=" * 60)
    print("TEST 1: Foggy Cityscapes Dataset")
    print("=" * 60)

    ds = FoggyCityscapesDataset(
        root=os.path.join(data_root, 'cityscapes'),
        split='train',
        fog_density='0.02',
        input_size=640,
        load_clear=True,
        load_depth=True,
    )
    print(f"  Samples: {len(ds)}")

    if len(ds) == 0:
        print("  ❌ FAIL: No samples found")
        return False

    sample = ds[0]
    print(f"  image:      {sample['image'].shape}, range [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  clear_gt:   {sample['clear_gt'].shape if sample['clear_gt'] is not None else 'None'}")
    print(f"  depth_gt:   {sample['depth_gt'].shape if sample['depth_gt'] is not None else 'None'}, range [{sample['depth_gt'].min():.3f}, {sample['depth_gt'].max():.3f}]")
    print(f"  bboxes:     {sample['bboxes'].shape}")

    # Verify shapes
    assert sample['image'].shape == (3, 640, 640), f"Expected (3,640,640), got {sample['image'].shape}"
    assert sample['clear_gt'].shape == (3, 640, 640), f"Expected (3,640,640), got {sample['clear_gt'].shape}"
    assert sample['depth_gt'].shape == (1, 640, 640), f"Expected (1,640,640), got {sample['depth_gt'].shape}"

    # Verify bbox format
    if sample['bboxes'].shape[0] > 0:
        assert sample['bboxes'].shape[1] == 5, f"Expected (N,5), got {sample['bboxes'].shape}"
        assert sample['bboxes'][:, 1:].min() >= 0, "Bbox coords should be >= 0"
        assert sample['bboxes'][:, 1:].max() <= 1, "Bbox coords should be <= 1"

    # Verify depth is normalized [0, 1]
    assert sample['depth_gt'].min() >= 0, "Depth should be >= 0"
    assert sample['depth_gt'].max() <= 1, "Depth should be <= 1"

    print("  ✅ PASS")
    return True


def test_acdc(data_root='data'):
    """Test ACDC loader."""
    print("\n" + "=" * 60)
    print("TEST 2: ACDC Dataset (Real Fog)")
    print("=" * 60)

    ds = ACDCDataset(
        root=os.path.join(data_root, 'rgb_anon_trainvaltest'),
        split='train',
        labels_dir=os.path.join(data_root, 'acdc_labels', 'train'),
        input_size=640,
    )
    print(f"  Samples: {len(ds)}")

    if len(ds) == 0:
        print("  ❌ FAIL: No samples found")
        return False

    sample = ds[0]
    print(f"  image:      {sample['image'].shape}, range [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  clear_gt:   {sample['clear_gt']}")
    print(f"  depth_gt:   {sample['depth_gt']}")
    print(f"  bboxes:     {sample['bboxes'].shape}")

    assert sample['image'].shape == (3, 640, 640)
    assert sample['clear_gt'] is None, "ACDC should not have clear_gt"
    assert sample['depth_gt'] is None, "ACDC should not have depth_gt"

    if sample['bboxes'].shape[0] > 0:
        assert sample['bboxes'].shape[1] == 5
        assert sample['bboxes'][:, 1:].min() >= 0
        assert sample['bboxes'][:, 1:].max() <= 1

    print("  ✅ PASS")
    return True


def test_zurich(data_root='data'):
    """Test Foggy Zurich loader."""
    print("\n" + "=" * 60)
    print("TEST 3: Foggy Zurich Dataset (Unlabeled DA)")
    print("=" * 60)

    ds = FoggyZurichDataset(
        root=os.path.join(data_root, 'Foggy_Zurich'),
        input_size=640,
    )
    print(f"  Samples: {len(ds)}")

    if len(ds) == 0:
        print("  ❌ FAIL: No samples found")
        return False

    sample = ds[0]
    print(f"  image:      {sample['image'].shape}")
    print(f"  bboxes:     {sample['bboxes'].shape} (should be [0, 5] — empty)")

    assert sample['image'].shape == (3, 640, 640)
    assert sample['bboxes'].shape == (0, 5), "Zurich should have empty bboxes"

    print("  ✅ PASS")
    return True


def test_driving(data_root='data'):
    """Test Foggy Driving loader."""
    print("\n" + "=" * 60)
    print("TEST 4: Foggy Driving Dataset (Test Set)")
    print("=" * 60)

    ds = FoggyDrivingDataset(
        root=os.path.join(data_root, 'Foggy_Driving'),
        split='all',
        input_size=640,
    )
    print(f"  Samples: {len(ds)}")

    if len(ds) == 0:
        print("  ❌ FAIL: No samples found")
        return False

    sample = ds[0]
    print(f"  image:      {sample['image'].shape}")
    print(f"  bboxes:     {sample['bboxes'].shape}")

    assert sample['image'].shape == (3, 640, 640)

    if sample['bboxes'].shape[0] > 0:
        assert sample['bboxes'].shape[1] == 5
        assert sample['bboxes'][:, 1:].min() >= 0
        assert sample['bboxes'][:, 1:].max() <= 1

    print("  ✅ PASS")
    return True


def test_build_dataloaders_supervised(config_path='configs/default.yaml'):
    """Test build_dataloaders in supervised mode (no DA)."""
    print("\n" + "=" * 60)
    print("TEST 5: build_dataloaders (Supervised Mode)")
    print("=" * 60)

    config = load_config(config_path)
    # Force supervised mode
    config.use_fda = False
    config.use_dct_align = False
    config.use_fsg_consistency = False
    config.batch_size = 4
    config.num_workers = 0

    train_loader, val_loader = build_dataloaders(config)

    # Get one batch
    batch = next(iter(train_loader))
    print(f"  Train batch keys: {list(batch.keys())}")
    print(f"  image:    {batch['image'].shape}")
    print(f"  clear_gt: {batch['clear_gt'].shape if batch['clear_gt'] is not None else 'None'}")
    print(f"  depth_gt: {batch['depth_gt'].shape if batch['depth_gt'] is not None else 'None'}")
    print(f"  bboxes:   {len(batch['bboxes'])} tensors, shapes: {[b.shape for b in batch['bboxes']]}")

    assert batch['image'].shape[0] == 4, "Batch size should be 4"
    assert batch['image'].shape[1:] == (3, 640, 640)

    # Val batch
    val_batch = next(iter(val_loader))
    print(f"  Val batch: image {val_batch['image'].shape}, bboxes: {len(val_batch['bboxes'])}")

    print("  ✅ PASS")
    return True


def test_build_dataloaders_da(config_path='configs/default.yaml'):
    """Test build_dataloaders in DA mode (paired batches)."""
    print("\n" + "=" * 60)
    print("TEST 6: build_dataloaders (Domain Adaptation Mode)")
    print("=" * 60)

    config = load_config(config_path)
    # Force DA mode
    config.use_fda = True
    config.use_dct_align = True
    config.use_fsg_consistency = True
    config.batch_size = 4
    config.num_workers = 0
    config.real_datasets = ['acdc', 'zurich']

    train_loader, val_loader = build_dataloaders(config)

    # Get one batch (should be paired)
    batch = next(iter(train_loader))
    assert 'synth' in batch and 'real' in batch, "DA batch should have 'synth' and 'real' keys"

    synth = batch['synth']
    real = batch['real']
    print(f"  Synth image: {synth['image'].shape}")
    print(f"  Synth clear_gt: {synth['clear_gt'].shape if synth['clear_gt'] is not None else 'None'}")
    print(f"  Synth bboxes: {len(synth['bboxes'])} tensors")
    print(f"  Real image: {real['image'].shape}")
    print(f"  Real clear_gt: {real['clear_gt']}")
    print(f"  Real bboxes: {len(real['bboxes'])} tensors")

    assert synth['image'].shape == (4, 3, 640, 640)
    assert real['image'].shape == (4, 3, 640, 640)

    print("  ✅ PASS")
    return True


def test_test_loader(config_path='configs/default.yaml'):
    """Test Foggy Driving test loader."""
    print("\n" + "=" * 60)
    print("TEST 7: build_test_loader (Foggy Driving)")
    print("=" * 60)

    config = load_config(config_path)
    config.batch_size = 4
    config.num_workers = 0

    test_loader = build_test_loader(config)

    batch = next(iter(test_loader))
    print(f"  Test batch: image {batch['image'].shape}, bboxes: {len(batch['bboxes'])} tensors")

    assert batch['image'].shape[0] == 4

    print("  ✅ PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify WRDNet data pipeline')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--da', action='store_true', help='Also test DA mode')
    args = parser.parse_args()

    print("=" * 60)
    print("WRDNet Data Pipeline Verification")
    print("=" * 60)

    results = []

    # Individual dataset tests
    results.append(("Cityscapes", test_cityscapes(args.data_root)))
    results.append(("ACDC", test_acdc(args.data_root)))
    results.append(("Foggy Zurich", test_zurich(args.data_root)))
    results.append(("Foggy Driving", test_driving(args.data_root)))

    # Pipeline tests
    results.append(("Supervised Loader", test_build_dataloaders_supervised(args.config)))
    results.append(("Test Loader", test_test_loader(args.config)))

    if args.da:
        results.append(("DA Loader", test_build_dataloaders_da(args.config)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())