import torch
import os

root_dirs = ["MOIS_SAM_aggregate_first", "MOIS_SAM_only_spatial", "MOIS_SAM_original"]

# This script processes fine-tuned SAM2 model checkpoint files by extracting only the model weights.
for root_dir in root_dirs:
    for fold in os.listdir(root_dir):
        fold_path = os.path.join(root_dir, fold)
        ckpt_path = os.path.join(fold_path, "checkpoint", "checkpoint.pt")
        model_weights_path = os.path.join(fold_path, "checkpoint.pt")
        if os.path.isfile(ckpt_path):
            print(f"Processing: {fold}")
            
            file_finetuned = torch.load(ckpt_path)
            model_only_checkpoint = {'model': file_finetuned['model']}
            torch.save(model_only_checkpoint, model_weights_path)
        
            print(f"Successfully converted {fold} to {model_weights_path}")  
        