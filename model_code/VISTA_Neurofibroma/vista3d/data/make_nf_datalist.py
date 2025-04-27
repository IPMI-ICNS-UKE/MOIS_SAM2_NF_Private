import json
from pathlib import Path

def read_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def create_datalist(train_ids, val_ids, label_suffix="_label.nii.gz"):
    def make_entries(sample_ids):
        return [
            {
                "image": f"nifti/imagesTr/{sample_id}.nii.gz",
                "label": f"nifti/labelsTr/{sample_id}.nii.gz",
            }
            for sample_id in sample_ids
        ]

    return {
        "training": make_entries(train_ids),
        "validation": make_entries(val_ids),
        "testing": make_entries(val_ids)
    }

def generate_datalists(base_split_dir="data/splits", output_dir="data/datalists"):
    split_root = Path(base_split_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(split_root.glob("fold_*"))

    for fold_dir in fold_dirs:
        fold_number = fold_dir.name.split("_")[1]  # '1', '2', ...
        train_file = fold_dir / "train_set.txt"
        val_file = fold_dir / "val_set.txt"
        
        if not train_file.exists() or not val_file.exists():
            print(f"Skipping {fold_dir.name}, missing train or val file.")
            continue

        train_ids = read_ids(train_file)
        val_ids = read_ids(val_file)

        datalist = create_datalist(train_ids, val_ids)

        output_path = output_root / f"Neurofibroma_fold_{int(fold_number) - 1}.json"
        with open(output_path, "w") as json_file:
            json.dump(datalist, json_file, indent=4)

        print(f"Saved: {output_path}")

# Run the generation
generate_datalists(base_split_dir="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NF_SAM_Sandbox/data/splits", 
                   output_dir="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/VISTA/vista3d/data/external")