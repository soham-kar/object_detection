"""Dataset builders for WRDNet.

Supports:
  - Single-dataset loading (Foggy Cityscapes, ACDC, DAWN)
  - Paired synthetic+real loading for domain adaptation training
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from .foggy_cityscapes import FoggyCityscapesDataset
from .acdc import ACDCDataset
from .dawn import DAWNDataset


class PairedDADataset(Dataset):
    """
    Wraps a synthetic dataset and a real dataset for domain adaptation.

    Each __getitem__ returns a dict with both 'synth' and 'real' sub-dicts.
    When datasets have different lengths, the shorter one is cycled.
    """

    def __init__(self, synth_dataset: Dataset, real_dataset: Dataset):
        self.synth_dataset = synth_dataset
        self.real_dataset = real_dataset
        self._len = max(len(synth_dataset), len(real_dataset))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        synth_idx = idx % len(self.synth_dataset)
        real_idx = idx % len(self.real_dataset)

        synth_sample = self.synth_dataset[synth_idx]
        real_sample = self.real_dataset[real_idx]

        return {
            'synth': synth_sample,
            'real': real_sample,
        }


def build_dataloaders(config) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build training and validation data loaders.

    When domain adaptation is enabled, the training loader returns
    paired (synthetic, real) batches.

    Args:
        config: Config with dataset settings
    Returns:
        train_loader, val_loader
    """
    dataset_name = getattr(config, 'dataset', 'foggy_cityscapes')
    batch_size = getattr(config, 'batch_size', 8)
    num_workers = getattr(config, 'num_workers', 4)
    data_root = getattr(config, 'data_root', 'data')
    use_da = getattr(config, 'use_dct_align', False) or getattr(config, 'use_fsg_consistency', False)

    # Build synthetic (labeled) dataset
    if dataset_name == 'foggy_cityscapes':
        synth_train = FoggyCityscapesDataset(
            root=os.path.join(data_root, 'foggy_cityscapes'),
            split='train',
            config=config,
        )
        synth_val = FoggyCityscapesDataset(
            root=os.path.join(data_root, 'foggy_cityscapes'),
            split='val',
            config=config,
        )
    elif dataset_name == 'acdc':
        synth_train = ACDCDataset(
            root=os.path.join(data_root, 'acdc'),
            split='train',
            config=config,
        )
        synth_val = ACDCDataset(
            root=os.path.join(data_root, 'acdc'),
            split='val',
            config=config,
        )
    elif dataset_name == 'dawn':
        synth_train = DAWNDataset(
            root=os.path.join(data_root, 'dawn'),
            split='train',
            config=config,
        )
        synth_val = DAWNDataset(
            root=os.path.join(data_root, 'dawn'),
            split='val',
            config=config,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Build real (unlabeled) dataset for DA
    if use_da:
        real_data_paths = getattr(config, 'real_data', ['data/acdc/rgb_anon/fog'])
        real_datasets = []
        for rp in real_data_paths:
            full_path = os.path.join(data_root, rp) if not os.path.isabs(rp) else rp
            if os.path.exists(full_path):
                real_datasets.append(ACDCDataset(root=full_path, split='train', config=config))

        if real_datasets:
            real_train = ConcatDataset(real_datasets) if len(real_datasets) > 1 else real_datasets[0]
            train_dataset = PairedDADataset(synth_train, real_train)
        else:
            train_dataset = synth_train
    else:
        train_dataset = synth_train

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        synth_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
