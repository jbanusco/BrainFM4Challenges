# SSL3D Model - Classification Fine-Tuning

Complete guide for fine-tuning the SSL3D ResEnc foundation model for brain MRI classification tasks. This example demonstrates modality classification using preprocessed nnUNet data.

## Overview

Unlike segmentation, classification is flexible in data format, but **our example uses preprocessed nnUNet data** (blosc2 format) for consistency with the segmentation workflow.

**Key Features:**
- Uses preprocessed blosc2 files from nnUNet preprocessing
- Flexible framework adaptable to various classification tasks
- Example: MPRAGE modality classification (4 classes)
- Based on original SSL3D_classification repository structure

## Docker Image

**Image:** `petermcgor/nnssl-classification`  
**Size:** ~11.1GB  
**Includes:** Pre-trained SSL3D ResEnc-L model, classification framework, all dependencies

The Dockerfile used to build this image is available in this repository at `SSL3D_model/classification/Dockerfile`.

Pull the image:
```bash
docker pull petermcgor/nnssl-classification
```

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with GPU support (nvidia-docker)
- **Preprocessed data:** If following our example, you must run the segmentation preprocessing first (see [segmentation guide](../segmentation/))

**Important:** All commands below should be run from the **repository root directory**.

## Example: Modality Classification

Our example classifies MPRAGE modalities into 4 classes:
- Class 0: PD (Proton Density)
- Class 1: T1w (T1-weighted)
- Class 2: T2star (T2*)
- Class 3: T1wDivPD (T1w divided by PD)

### Data Requirements

The example uses **preprocessed blosc2 files** created during segmentation preprocessing. If you haven't already, complete the preprocessing steps from the [segmentation guide](../segmentation/):

1. Plan and preprocess (creates fingerprint)
2. Preprocess like nnssl (creates blosc2 files)

This creates files in:
```
nnUNet_preprocessed/Dataset002_MPRAGE/Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z/
├── sub-02_000.b2nd
├── sub-03_001.b2nd
├── sub-05_002.b2nd
├── sub-06_003.b2nd
└── sub-07_000.b2nd
```

**Note:** While preprocessing is required to follow our example exactly, the classification framework is flexible and can work with other data formats if you adapt the dataset class accordingly.

### Training

Run classification training with Docker:
```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/SSL3D_model/classification/datasets/modality_mprage.py:/opt/SSL3D_classification/datasets/modality_mprage.py \
  -v $(pwd)/SSL3D_model/classification/cli_configs/data/modality_mprage.yaml:/opt/SSL3D_classification/cli_configs/data/modality_mprage.yaml \
  -v $(pwd)/SSL3D_model/classification/cli_configs/env/cluster.yaml:/opt/SSL3D_classification/cli_configs/env/cluster.yaml \
  -v $(pwd)/SSL3D_model/classification/cli_configs/data/modality_mprage/splits_prepro.json:/opt/SSL3D_classification/cli_configs/data/modality_mprage/splits.json \
  -w /workspace \
  --shm-size=40gb \
  petermcgor/nnssl-classification \
  python3 /opt/SSL3D_classification/main.py \
    data=modality_mprage \
    data_dir=/workspace \
    model=resenc \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    data.module.batch_size=2 \
    env=cluster \
    +splits_path=/opt/SSL3D_classification/cli_configs/data/modality_mprage/splits.json
```

**Parameters Explained:**
- `data=modality_mprage`: Uses the modality classification config
- `data_dir=/workspace`: Root directory for data (mounted repo)
- `model=resenc`: ResEnc architecture (matches pretrained model)
- `trainer.devices=1`: Use 1 GPU
- `trainer.max_epochs=100`: Training epochs (adjust as needed)
- `data.module.batch_size=2`: Batch size (adjust based on GPU memory)
- `env=cluster`: Uses cluster environment config (disables progress bars)
- `+splits_path=...`: Path to splits file with train/val/test splits

**Docker-specific flags:**
- `--shm-size=40gb`: Shared memory size (required for data loading)

**Results:** 
Training outputs are saved in `experiments/modality_mprage_classification/`

### Understanding the Example Structure

The example consists of these files in the repository:
```
SSL3D_model/classification/
├── datasets/
│   └── modality_mprage.py              # Dataset class
├── cli_configs/
│   ├── data/
│   │   ├── modality_mprage.yaml        # Data configuration
│   │   └── modality_mprage/
│   │       ├── dataset_info.json       # Dataset metadata
│   │       ├── splits.json             # Raw nifti splits (commented)
│   │       └── splits_prepro.json      # Blosc2 splits (active)
│   └── env/
│       └── cluster.yaml                # Environment config
└── Dockerfile
```

**Key Files:**

**`datasets/modality_mprage.py`:**
- Implements the dataset class
- Loads blosc2 files from preprocessed nnUNet data
- Extracts the correct modality channel based on label
- Applies transforms

**`cli_configs/data/modality_mprage.yaml`:**
- Hydra configuration for the dataset
- Specifies model parameters (input channels, patch size, etc.)
- Points to pretrained model: `/workspace/models/.../checkpoint_best.pth`
- Defines training hyperparameters

**`cli_configs/data/modality_mprage/splits_prepro.json`:**
- Defines train/val/test splits
- Points to preprocessed data location
- Includes metadata (case_id, modality_index, label)

**`cli_configs/env/cluster.yaml`:**
- Environment-specific settings
- Disables progress bars for cluster/Docker use
- Sets output directory

### Splits Files: splits.json vs splits_prepro.json

The repository includes both split files:

- **`splits.json`**: Points to raw nifti files in `nnUNet_raw/` (commented out in code)
- **`splits_prepro.json`**: Points to preprocessed blosc2 files in `nnUNet_preprocessed/` (actively used)

Our example uses `splits_prepro.json` because:
1. Blosc2 format loads faster
2. Data is already normalized and properly formatted
3. Consistent with the preprocessing done for segmentation

## Creating Your Own Classification Task

To adapt this for your own classification problem:

### 1. Create Your Dataset Class

Create `datasets/your_task.py`:
```python
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets.base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO  # If using preprocessed data

class YourTaskDataset(Dataset):
    def __init__(self, root, split, splits_path, fold=0, transform=None):
        self.data_dir = Path(root)
        
        # Load your splits
        with open(Path(splits_path)) as f:
            splits_data = json.load(f)
        self.samples = splits_data[split]
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load your data (adapt as needed)
        # Example for blosc2:
        img, _ = Blosc2IO.load(img_path, mode="r")
        
        # Example for nifti:
        # import nibabel as nib
        # img = nib.load(img_path).get_fdata()
        
        # Apply transforms
        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img)})["image"]
        
        return img, sample['label']
    
    def __len__(self):
        return len(self.samples)

class YourTaskDataModule(BaseDataModule):
    def __init__(self, splits_path=None, patch_size=(160, 160, 160), **params):
        super().__init__(**params)
        self.splits_path = splits_path
        self.patch_size = patch_size
    
    def setup(self, stage: str):
        # Setup train/val datasets
        # (similar to modality_mprage.py)
        pass
```

### 2. Create Configuration File

Create `cli_configs/data/your_task.yaml`:
```yaml
# @package _global_

data:
  module:
    _target_: datasets.your_task.YourTaskDataModule
    name: your_task
    data_root_dir: ${data_dir}
    splits_path: ${splits_path}
    batch_size: 2
    train_transforms:
      _target_: augmentation.policies.batchgenerators.get_training_transforms
      patch_size: ${data.patch_size}
      rotation_for_DA: 0.523599
      mirror_axes: [0,1,2]
      do_dummy_2d_data_aug: False
  
  num_classes: YOUR_NUM_CLASSES
  patch_size: [160, 160, 160]

model:
  task: 'Classification'
  input_channels: 1
  input_dim: 3
  pretrained: True
  chpt_path: /workspace/models/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso/fold_all/checkpoint_best.pth

trainer:
  max_epochs: 100
```

### 3. Create Splits File

Create `cli_configs/data/your_task/splits.json`:
```json
{
  "train": [
    {
      "case_id": "case_001",
      "label": 0,
      "data_source": "path/to/data",
      "image_folder": "imagesTr"
    }
  ],
  "val": [...],
  "test": [...]
}
```

### 4. Run Training
```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=40gb \
  petermcgor/nnssl-classification \
  python3 /opt/SSL3D_classification/main.py \
    data=your_task \
    data_dir=/workspace \
    model=resenc \
    +splits_path=/workspace/SSL3D_model/classification/cli_configs/data/your_task/splits.json
```

## Tips and Best Practices

1. **Data Format:** While our example uses blosc2, you can use any format by adapting the dataset class
2. **Batch Size:** Adjust `data.module.batch_size` based on GPU memory
3. **Epochs:** Start with fewer epochs for testing, increase for final training
4. **Patch Size:** Ensure patch size in config matches your data dimensions
5. **Augmentation:** Modify augmentation settings in the yaml config as needed
6. **Preprocessing:** If using preprocessed data, ensure augmentation is minimal (already normalized)

## Code Structure Note

The code structure is kept similar to the [original SSL3D_classification repository](https://github.com/MIC-DKFZ/SSL3D_classification) to maintain transparency about our challenge-winning approach. While this could be optimized from a software engineering perspective, the current structure prioritizes:
- Fidelity to the methods used in the SSL3D Challenge
- Clear separation of concerns (datasets, configs, environment)
- Easy adaptation for new tasks

## Troubleshooting

**Issue:** Out of memory errors  
**Solution:** Reduce `batch_size` or `patch_size`

**Issue:** Cannot find splits file  
**Solution:** Ensure `+splits_path` points to correct location (use absolute path within Docker: `/workspace/...`)

**Issue:** Blosc2 files not found  
**Solution:** Run segmentation preprocessing first to create blosc2 files

**Issue:** Progress bar clutter in logs  
**Solution:** Ensure `env=cluster` is set (disables progress bars)

**Issue:** Slow data loading  
**Solution:** Increase `--shm-size` (shared memory) parameter

## References

This classification workflow is based on:

1. **SSL3D Classification Repository** - [https://github.com/MIC-DKFZ/SSL3D_classification](https://github.com/MIC-DKFZ/SSL3D_classification)  
   Original classification framework used in SSL3D Challenge

2. **SSL3D Challenge** - [https://ssl3d-challenge.dkfz.de/](https://ssl3d-challenge.dkfz.de/)  
   Challenge website and leaderboard

3. **nnUNet Framework** - [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)  
   For data preprocessing and format specifications

For more details on the preprocessing format and nnUNet specifications, see the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).