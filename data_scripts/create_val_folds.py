import os
import shutil


# Define base paths
base_dir = "/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data"
raw_images_dir = os.path.join(base_dir, "raw/imagesTr")
raw_labels_dir = os.path.join(base_dir, "raw/labelsTr")
splits_dir = os.path.join(base_dir, "splits")

# Loop through each fold
for fold_num in range(1, 4):
    fold_name = f"fold_{fold_num}"
    val_set_path = os.path.join(splits_dir, fold_name, "val_set.txt")

    # Read the validation set file
    with open(val_set_path, "r") as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # Define target directories
    images_val_dir = os.path.join(base_dir, f"raw/imagesVal_{fold_num}")
    labels_val_dir = os.path.join(base_dir, f"raw/labelsVal_{fold_num}")
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Copy files
    for val_id in val_ids:
        image_file = f"{val_id}.nii.gz"
        label_file = f"{val_id}.nii.gz"
        
        src_image_path = os.path.join(raw_images_dir, image_file)
        src_label_path = os.path.join(raw_labels_dir, label_file)
        
        dst_image_path = os.path.join(images_val_dir, image_file)
        dst_label_path = os.path.join(labels_val_dir, label_file)

        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"Image file not found: {src_image_path}")

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"Label file not found: {src_label_path}")
