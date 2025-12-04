# prepare_classification_metadata.py
import json
import pandas as pd
import numpy as np
from pathlib import Path

def prepare_sex_classification_metadata():
    """
    Example 1: Sex classification - just create metadata
    """
    print("\n=== Preparing Sex Classification Metadata ===")
    
    # Read participant info
    tsv_path = "segmentation_data/data_mp2rage/MP2RAGE_participants.tsv"
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Sex to label mapping
    sex_to_label = {'M': 0, 'F': 1}
    
    # Build subject list with labels
    subjects = []
    for _, row in df.iterrows():
        subj_id = row['participant_id']
        
        # Find corresponding case in nnUNet format
        nnunet_dir = Path("nnUNet_raw/Dataset001_MP2RAGE/imagesTr")
        case_files = list(nnunet_dir.glob(f"{subj_id}_*_0000.nii.gz"))
        
        if case_files:
            case_id = case_files[0].stem.replace("_0000", "")
            subjects.append({
                "case_id": case_id,
                "subject_id": subj_id,
                "label": sex_to_label[row['sex']],
                "sex": row['sex']
            })
    
    # Split
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    splits = {
        "train": subjects[:2],
        "val": subjects[2:3],
        "test": subjects[3:4]
    }
    
    # Save metadata
    output_dir = Path("classification_data/sex_mp2rage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    dataset_info = {
        "task": "sex_classification",
        "data_source": "nnUNet_raw/Dataset001_MP2RAGE",
        "image_folder": "imagesTr",
        "classes": {"Male": 0, "Female": 1},
        "num_classes": 2,
        "num_modalities": 5,
        "modalities": ["INV1", "INV2", "T1", "T2star", "UNI"],
        "modality_indices": [0, 1, 2, 3, 4]
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Metadata saved to {output_dir}")
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


def prepare_sequence_classification_metadata():
    """
    Example 2: Modality classification within MPRAGE dataset
    Classes: 0=PD, 1=T1w, 2=T2star, 3=T1wDivPD
    """
    print("\n=== Preparing Modality Classification Metadata ===")
    
    # MPRAGE has 4 channels: PD(0), T1w(1), T2star(2), T1wDivPD(3)
    modality_info = {
        0: {"name": "PD", "label": 0},
        1: {"name": "T1w", "label": 1},
        2: {"name": "T2star", "label": 2},
        3: {"name": "T1wDivPD", "label": 3}
    }
    
    train_subjects = []
    test_subjects = []
    
    # Training subjects (imagesTr)
    mprage_train_dir = Path("nnUNet_raw/Dataset002_MPRAGE/imagesTr")
    train_case_ids = set()
    for img_file in sorted(mprage_train_dir.glob("*_0000.nii.gz")):
        case_id = img_file.stem.replace("_0000", "")  # Remove both _0000 and .nii.gz
        train_case_ids.add(case_id)
    
    # Create samples for training subjects
    for case_id in sorted(train_case_ids):
        for mod_idx, mod_data in modality_info.items():
            train_subjects.append({
                "case_id": case_id,
                "modality_index": mod_idx,
                "modality_name": mod_data["name"],
                "label": mod_data["label"],
                "data_source": "nnUNet_raw/Dataset002_MPRAGE",
                "image_folder": "imagesTr"
            })
    
    # Test subject (imagesTs) - sub-07_000
    mprage_test_dir = Path("nnUNet_raw/Dataset002_MPRAGE/imagesTs")
    test_case_ids = set()
    for img_file in sorted(mprage_test_dir.glob("*_0000.nii.gz")):
        case_id = img_file.stem.replace("_0000", "")
        test_case_ids.add(case_id)
    
    # Create samples for test subject
    for case_id in sorted(test_case_ids):
        for mod_idx, mod_data in modality_info.items():
            test_subjects.append({
                "case_id": case_id,
                "modality_index": mod_idx,
                "modality_name": mod_data["name"],
                "label": mod_data["label"],
                "data_source": "nnUNet_raw/Dataset002_MPRAGE",
                "image_folder": "imagesTs"
            })
    
    # Split training into train/val (80/20)
    np.random.seed(42)
    np.random.shuffle(train_subjects)
    
    n_train = len(train_subjects)
    n_val = int(n_train * 0.2)
    
    splits = {
        "train": train_subjects[:-n_val],
        "val": train_subjects[-n_val:],
        "test": test_subjects
    }
    
    # Save metadata
    output_dir = Path("classification_data/modality_mprage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    dataset_info = {
        "task": "modality_classification",
        "dataset": "MPRAGE",
        "classes": {"PD": 0, "T1w": 1, "T2star": 2, "T1wDivPD": 3},
        "num_classes": 4,
        "num_modalities": 1,
        "description": "Multi-class modality type classification using MPRAGE dataset",
        "total_samples": {
            "train": len(splits['train']),
            "val": len(splits['val']),
            "test": len(splits['test'])
        }
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Metadata saved to {output_dir}")
    print(f"  Train subjects: 4 (16 samples = 4 subjects × 4 modalities)")
    print(f"  Test subject: 1 (4 samples = sub-07 × 4 modalities)")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")


if __name__ == "__main__":
    prepare_sex_classification_metadata()
    prepare_sequence_classification_metadata()
    print("\n✓✓✓ All metadata prepared!")