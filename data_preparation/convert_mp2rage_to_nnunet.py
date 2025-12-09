import os
import shutil
from pathlib import Path
import json

def convert_mp2rage_to_nnunet(
    source_dir="segmentation_data/data_mp2rage",
    output_dir="nnUNet_raw/Dataset001_MP2RAGE",
    modalities=["inv1", "inv2", "t1", "t2s", "uni"]
):
    """
    Convert MP2RAGE data to nnUNet format
    
    Modality mapping:
    0: inv1_unbiased
    1: inv2_unbiased
    2: t1_unbiased
    3: t2s_unbiased
    4: uni_unbiased
    """
    
    # Create output directories
    images_tr = Path(output_dir) / "imagesTr"
    labels_tr = Path(output_dir) / "labelsTr"
    images_ts = Path(output_dir) / "imagesTs"
    labels_ts = Path(output_dir) / "labelsTs"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)
    
    # Get all subjects
    derivatives_dir = Path(source_dir) / "derivatives"
    subjects = [d.name for d in derivatives_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    subjects.sort()
    
    # Split: last subject for test, rest for training
    train_subjects = subjects[:-1]
    test_subjects = subjects[-1:]
    
    print(f"Found {len(subjects)} subjects")
    print(f"Training: {train_subjects}")
    print(f"Test: {test_subjects}")
    
    # Process training subjects
    for idx, subject in enumerate(train_subjects):
        case_id = f"{subject}_{idx:03d}"
        print(f"\n[TRAIN] Processing {subject} -> {case_id}")
        
        # Copy each modality
        for mod_idx, modality in enumerate(modalities):
            src_file = derivatives_dir / subject / "unbiased" / f"{subject}_{modality}_unbiased.nii.gz"
            dst_file = images_tr / f"{case_id}_{mod_idx:04d}.nii.gz"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {modality} -> {dst_file.name}")
            else:
                print(f"  WARNING: Missing {src_file}")
        
        # Copy label
        src_label = derivatives_dir / subject / "labeled" / f"{subject}_labels_v01.nii.gz"
        dst_label = labels_tr / f"{case_id}.nii.gz"
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
            print(f"  Copied label -> {dst_label.name}")
        else:
            print(f"  WARNING: Missing {src_label}")
    
    # Process test subjects
    for idx, subject in enumerate(test_subjects):
        case_id = f"{subject}_{idx:03d}"
        print(f"\n[TEST] Processing {subject} -> {case_id}")
        
        # Copy each modality
        for mod_idx, modality in enumerate(modalities):
            src_file = derivatives_dir / subject / "unbiased" / f"{subject}_{modality}_unbiased.nii.gz"
            dst_file = images_ts / f"{case_id}_{mod_idx:04d}.nii.gz"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {modality} -> {dst_file.name}")
            else:
                print(f"  WARNING: Missing {src_file}")
        
        # Copy label
        src_label = derivatives_dir / subject / "labeled" / f"{subject}_labels_v01.nii.gz"
        dst_label = labels_ts / f"{case_id}.nii.gz"
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
            print(f"  Copied label -> {dst_label.name}")
        else:
            print(f"  WARNING: Missing {src_label}")
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "INV1",
            "1": "INV2",
            "2": "T1",
            "3": "T2star",
            "4": "UNI"
        },
        "labels": {
            "background": 0,
            "white_matter": 1,
            "grey_matter": 2,
            "cerebrospinal_fluid": 3,
            "ventricles": 4,
            "subcortical": 5,
            "vessels": 6,
            "sagittal_sinus": 7
        },
        "numTraining": len(train_subjects),
        "numTest": len(test_subjects),
        "file_ending": ".nii.gz",
        "name": "MP2RAGE",
        "description": "MP2RAGE brain segmentation with 7 tissue classes"
    }
    
    with open(Path(output_dir) / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✓ Conversion complete! Dataset saved to {output_dir}")
    print(f"  Training cases: {len(train_subjects)}")
    print(f"  Test cases: {len(test_subjects)}")

if __name__ == "__main__":
    convert_mp2rage_to_nnunet()