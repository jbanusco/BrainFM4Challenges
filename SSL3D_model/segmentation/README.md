# SSL3D Model - Segmentation Fine-Tuning

Guide for fine-tuning the SSL3D ResEncL foundation model for brain MRI segmentation tasks using the nnUNet framework.

## Overview

This workflow uses our pre-trained SSL3D model embedded in a Docker image to fine-tune on your segmentation task. The process follows the nnssl<sup>w</supx> (nnUNet with self-supervised learning) framework and includes:

1. Data preparation (nnUNet format)
2. Preprocessing adapted for SSL3D model
3. Fine-tuning with pretrained weights
4. Inference on new data

## Docker Image

**Image:** `petermcgor/nnunetv2:1.1.0-nnssl`  
**Size:** ~11.02GB  
**Includes:** Pre-trained SSL3D ResEnc-L model, nnUNet framework, nnssl adaptations

Pull the image:
```bash
docker pull petermcgor/nnunetv2:1.1.0-nnssl
```

## Prerequisites

- Data in nnUNet format (see [data_preparation](../../data_preparation/))
- NVIDIA GPU with CUDA support
- Docker with GPU support (nvidia-docker)

## Complete Workflow

### Step 1: Prepare Your Data

Convert your data to nnUNet format. See [data_preparation](../../data_preparation/) for scripts and instructions.

Your data should be in:
```
nnUNet_raw/
└── DatasetXXX_YourDataset/
    ├── dataset.json
    ├── imagesTr/
    ├── imagesTs/
    └── labelsTr/
```

### Step 2: Preprocessing

Preprocessing consists of two steps:

#### 2a. Create Fingerprint and Plans
```bash 
docker run --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_plan_and_preprocess \
    -d DATASET_ID \
    --no_pp \
    -np 8
```

**Parameters:**
- `-d DATASET_ID`: Your dataset ID (e.g., 001 for Dataset001_MP2RAGE)
- `--no_pp`: Skip standard preprocessing (we'll use nnssl preprocessing)
- `-np 8`: Number of processes (adjust based on your CPU)

#### 2b. Preprocess Like nnssl
```bash
docker run --gpus all --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_preprocess_like_nnssl \
    -np 8 \
    -d DATASET_ID \
    -n YOUR_TRAINING_NAME \
    -pc /opt/nnunet_resources/models/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso/fold_all/checkpoint_best.pth \
    -am "like_pretrained"
```

**Parameters:**
- `-d DATASET_ID`: Your dataset ID
- `-n YOUR_TRAINING_NAME`: Unique name for this training run (e.g., "MySegmentation_SSL3D")
- `-pc`: Path to pretrained checkpoint (fixed path inside Docker)
- `-am "like_pretrained"`: Adaptation method
- `-np 8`: Number of processes

This creates a preprocessing plan file: `ptPlans__YOUR_TRAINING_NAME__...json`

### Step 3: Fine-Tuning
```bash
docker run --gpus all --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_results:/opt/nnunet_resources/nnUNet_results \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_train_pretrained \
    DATASET_ID \
    3d_fullres \
    FOLD \
    -p PLANS_IDENTIFIER \
    -tr PretrainedTrainer_15ep \
    --npz
```

**Parameters:**
- `DATASET_ID`: Your dataset ID
- `3d_fullres`: Configuration (standard for 3D full resolution)
- `FOLD`: Fold number (0-4 for 5-fold CV, or `all` for all training data)
- `-p PLANS_IDENTIFIER`: Plans file from preprocessing (e.g., `ptPlans__MySegmentation_SSL3D__...`)
- `-tr PretrainedTrainer_15ep`: Trainer using pretrained weights (15 epochs fine-tuning)
- `--npz`: Save predictions in npz format

**Example:**
```bash
docker run --gpus all --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_results:/opt/nnunet_resources/nnUNet_results \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_train_pretrained \
    001 \
    3d_fullres \
    all \
    -p ptPlans__MySegmentation_SSL3D__Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z_Z \
    -tr PretrainedTrainer_15ep \
    --npz
```

Training outputs will be saved in `nnUNet_results/DatasetXXX_YourDataset/`.

### Step 4: Inference

Run predictions on new data:
```bash
docker run --gpus all --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_results:/opt/nnunet_resources/nnUNet_results \
  -v $(pwd)/predictions:/predictions \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_predict \
    -i /opt/nnunet_resources/nnUNet_raw/DatasetXXX_YourDataset/imagesTs \
    -o /predictions \
    -d DATASET_ID \
    -c 3d_fullres \
    -f FOLD \
    -p PLANS_IDENTIFIER \
    -tr PretrainedTrainer_15ep \
    -npp 12 \
    -nps 12
```

**Parameters:**
- `-i`: Input folder with test images
- `-o`: Output folder for predictions
- `-d`: Dataset ID
- `-c`: Configuration
- `-f`: Fold used in training
- `-p`: Plans identifier
- `-tr`: Trainer name
- `-npp`: Number of processes for preprocessing
- `-nps`: Number of processes for segmentation

**Example:**
```bash
docker run --gpus all --rm \
  -v $(pwd)/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw \
  -v $(pwd)/nnUNet_results:/opt/nnunet_resources/nnUNet_results \
  -v $(pwd)/predictions:/predictions \
  petermcgor/nnunetv2:1.1.0-nnssl \
  nnUNetv2_predict \
    -i /opt/nnunet_resources/nnUNet_raw/Dataset001_MP2RAGE/imagesTs \
    -o /predictions \
    -d 001 \
    -c 3d_fullres \
    -f all \
    -p ptPlans__MySegmentation_SSL3D__Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z_Z \
    -tr PretrainedTrainer_15ep \
    -npp 12 \
    -nps 12
```

Predictions will be saved in the `predictions/` directory.

## Tips and Best Practices

1. **Training Name**: Choose a descriptive name for `-n` in preprocessing to easily identify your experiments
2. **GPU Memory**: Adjust batch size in plans if you encounter out-of-memory errors
3. **Plans Identifier**: After preprocessing, check `nnUNet_preprocessed/DatasetXXX/` for the exact plans filename
4. **Cross-Validation**: Train on different folds (0-4) or use `all` for final model
5. **Monitoring**: Training logs are saved in `nnUNet_results/` - monitor progress there

## Troubleshooting

**Issue:** Cannot find plans file  
**Solution:** Check exact filename in `nnUNet_preprocessed/DatasetXXX/` and use complete name with `-p`

**Issue:** GPU out of memory  
**Solution:** Reduce patch size or batch size in preprocessing plans

**Issue:** Preprocessing fails  
**Solution:** Ensure data is in correct nnUNet format with proper `dataset.json`

## References

This segmentation workflow is based on:

1. **nnssl Repository** - [https://github.com/MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl)  
   Self-supervised learning adaptations for nnUNet

2. **nnUNet Framework** - [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)  
   Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021)

3. **SSL3D Challenge** - [https://ssl3d-challenge.dkfz.de/](https://ssl3d-challenge.dkfz.de/)  
   Challenge website and leaderboard

For detailed documentation on nnUNet format and advanced options, please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).