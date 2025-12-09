"""
Modality Classification Dataset for MPRAGE
Classes: 0=PD, 1=T1w, 2=T2star
"""

import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
from datasets.base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class ModalityMPRAGEDataset(Dataset):
    def __init__(self, root, split, splits_path, fold=0, transform=None):
        super().__init__()
        
        self.data_dir = Path(root) 
        print("Loading Modality MPRAGE Dataset from:", self.data_dir)
        #splits_file = self.data_dir / "SSL3D_FM_Classification" / "cli_configs" / "data" / "modality_mprage" / "splits.json"
        splits_file = Path(splits_path)
        # Load splits
        with open(splits_file) as f:
            splits_data = json.load(f)
        
        self.samples = splits_data[split]
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        #img_filename = f"{sample['case_id']}_{sample['modality_index']:04d}.nii.gz"
        #img_path = self.data_dir / Path(sample['data_source']) / sample['image_folder'] / img_filename
        img_filename = f"{sample['case_id']}.b2nd"
        img_path = self.data_dir / Path(sample['data_source']) / sample['image_folder'] / "Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z" / img_filename
        
        #nii = nib.load(str(img_path))
        #img_data = nii.get_fdata().astype(np.float32)
        img, _ = Blosc2IO.load(img_path, mode="r")
        img = img[int(sample['label'])]
        
        # Add channel dimension: (H, W, D) -> (1, H, W, D)
        #img_data = img_data[np.newaxis, ...]
        
        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img)})["image"]
        else:
            img = torch.from_numpy(img)
        
        return img, sample['label']
    
    def __len__(self):
        return len(self.samples)


class ModalityMPRAGEDataModule(BaseDataModule):
    def __init__(self, splits_path=None,**params):
        super(ModalityMPRAGEDataModule, self).__init__(**params)
        self.splits_path = splits_path
    
    def setup(self, stage: str):
        self.train_dataset = ModalityMPRAGEDataset(
            self.data_path,
            split="train",
            splits_path=self.splits_path,
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = ModalityMPRAGEDataset(
            self.data_path,
            split="val",
            splits_path=self.splits_path,
            transform=self.test_transforms,
            fold=self.fold,
        )