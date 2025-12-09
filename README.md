# BrainFM4Challenges

Foundation models for brain MRI from the SSL3D<sup>w</sup> and FOMO<sup>2</sup> challenges (MICCAI 2025). This repository provides pre-trained models and complete workflows for fine-tuning on segmentation and classification tasks.


## Challenge Results

This repository contains the winning solutions from:

- **[SSL3D Challenge](https://ssl3d-challenge.dkfz.de/home)** (MICCAI 2025) - 1st Place (ResEnc-L)
- **[FOMO Challenge](https://fomo25.github.io/)** (MICCAI 2025) - 1st Place (Open)

Both models were developed using similar self-supervised learning principles but with different architectures tailored to each challenge's specific constraints and requirements.

## Repository Organization

This repository is organized into two independent sections, one for each foundation model, to maintain clarity and ease of use:

### SSL3D Model
Pre-trained ResEnc-based<sup>w2</sup> foundation model for brain MRI. Supports both segmentation and classification fine-tuning.

**Available Applications:**
- **[Fine-tuning for Segmentation](./SSL3D_model/segmentation/)** 
- **[Fine-tuning for Classification](./SSL3D_model/classification/)** 

**Documentation:** [SSL3D_model/README.md](./SSL3D_model/README.md)

### FOMO Model
Foundation model developed for the FOMO challenge. [To be documented by collaborators]

**Documentation:** [FOMO_model/README.md](./FOMO_model/README.md)

## Getting Started for Fine-Tuning Your Own Models

We provide step-by-step instructions for fine-tuning and inference using both models. The examples use an openly available dataset from Zenodo.

### 1. Download Example Data

To follow our examples, download the dataset from **[Zenodo](https://zenodo.org/records/1206163)**

1. Download the ZIP file from the Zenodo link above
2. Save and extract it to `segmentation_data/` in the repository root

### 2. Data Preparation

Convert the downloaded data to nnUNet format:
```bash
cd data_preparation

# Convert MP2RAGE dataset
python convert_mp2rage_to_nnunet.py

# Convert MPRAGE dataset
python convert_mprage_to_nnunet.py
```

This creates the nnUNet directory structure with properly formatted data in `nnUNet_raw/`.

**Note:** [nnUNet format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)<sup>i</sup> is **required** for SSL3D segmentation fine-tuning. For classification, we use nnUNet preprocessed data in our examples for consistency, though the classification workflow is flexible and can work with other data formats.

**See:** [data_preparation/README.md](./data_preparation/README.md) for complete instructions and format details.

### 3. Choose Your Application

Navigate to the appropriate section based on your task:

- **SSL3D Segmentation** → [SSL3D_model/segmentation/](./SSL3D_model/segmentation/)
- **SSL3D Classification** → [SSL3D_model/classification/](./SSL3D_model/classification/)
- **FOMO** → [FOMO_model/](./FOMO_model/)

Each section contains:
- Complete setup instructions
- Docker images with embedded pre-trained weights
- Step-by-step fine-tuning guide
- Inference/prediction instructions
- Example configurations

## 🐳 Docker Images

Pre-built Docker images are available on Docker Hub with pre-trained weights embedded:

| Model | Task | Image | Size |
|-------|------|-------|------|
| SSL3D | Segmentation | `petermcgor/nnunetv2:1.1.0-nnssl` | ~11.02GB |
| SSL3D | Classification | `petermcgor/nnssl-classification` | ~11.1GB |
| FOMO | TBD | TBD | TBD |

**Note:** Models are embedded in Docker images - no separate downloads required.

## 💻 HPC/Cluster Usage

For Singularity users (HPC environments), images can be converted:
```bash
singularity pull ssl3d-segmentation.sif docker://petermcgor/nnunetv2:1.1.0-nnssl
```

Contact us if you need assistance with Singularity conversions.

## Prerequisites

- Docker or Singularity
- NVIDIA GPU (for training and inference)

## 🤝 Contributing / Contact

We welcome contributions and feedback! If you encounter issues or have suggestions:

1. Open an issue on GitHub
2. Contact us directly (see below)
3. Submit a pull request

We plan to add more classification examples in the future based on community needs.


## License

[Under study of the CIBM]

---

**Note:** This repository focuses on fine-tuning our pre-trained foundation models. The code structure is kept similar to the original challenge frameworks to maintain transparency about our winning approach, though it could be optimized from a software engineering perspective.

## Citation

Our work is currently under review.

**Temporary Citation:**
```bibtex
@article{brainfm2025,
  title={TITLE},
  author={[Authors]},
  journal={Under Review},
  year={2025}
}
```

See [CITATION.md](./CITATION.md) for full citation details once published.

## References

This work builds upon and utilizes the following frameworks and challenges:

[w] Wald, T. et al. An OpenMind for 3D medical vision self-supervised learning. Preprint at https://doi.org/10.48550/arXiv.2412.17041 (2025).

2. **[Citation 2 - To be added]**  
   [Placeholder for reference]

[w2] Wald, T. et al. Revisiting MAE pre-training for 3D medical image segmentation. Preprint at https://doi.org/10.48550/arXiv.2410.23132 (2025).

[i] Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J. & Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203–211 (2021).



4. **[Citation 4 - To be added]**  
   [Placeholder for reference]