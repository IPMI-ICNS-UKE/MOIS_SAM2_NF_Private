# Re-import necessary libraries after kernel reset
import SimpleITK as sitk
import os
from pathlib import Path

# Define base directory
base_dir = Path("./")

# Define target folders and prediction suffixes
folders = ["TS1", "TS2", "TS3", "TS4", "DiffTs1", "DiffTs2"]
suffixes = ["_nnunet", "_dins", "_sam2", "_mois"]
label_values = {"tp": 1, "fp": 2, "fn": 6}

# Process each folder
for folder in folders:
    folder_path = base_dir / folder
    for file in folder_path.glob("*_gt.nii.gz"):
        base_name = file.name.replace("_gt.nii.gz", "")
        gt = sitk.ReadImage(str(file))
        gt_arr = sitk.GetArrayFromImage(gt) > 0

        for suffix in suffixes:
            pred_path = folder_path / f"{base_name}{suffix}.nii.gz"
            print(pred_path)
            if pred_path.exists():
                pred = sitk.ReadImage(str(pred_path))
                pred_arr = sitk.GetArrayFromImage(pred) > 0

                # Create overlap label mask
                tp = (gt_arr & pred_arr)
                fp = (~gt_arr & pred_arr)
                fn = (gt_arr & ~pred_arr)
                label_arr = (
                    tp.astype(int) * label_values["tp"] +
                    fp.astype(int) * label_values["fp"] +
                    fn.astype(int) * label_values["fn"]
                )

                label_img = sitk.GetImageFromArray(label_arr.astype("uint8"))
                label_img.CopyInformation(gt)

                out_path = folder_path / f"{base_name}{suffix}_overlap.nii.gz"
                sitk.WriteImage(label_img, str(out_path))
