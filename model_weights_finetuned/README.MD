# README for `model_weights_finetuned` Directory

This directory contains the final selected **fine-tuned** model weights for various interactive segmentation models, including **DINs, SAM2, and SW-FastEdit**. 

The user should place only the selected final model weights in this directory, after fine-tuning, extracted from the experiment folder. Other artifacts such as tensorboard logs, intermediate checkpoints, and debug files should remain in the experiment folder.

## Directory Structure

```
.
├── DINs/                       # Fine-tuned DINs model weights
│   ├── fold_1/
│   │   └── checkpoint.onnx      # Final selected ONNX model for fold 1
│   ├── fold_2/
│   │   └── checkpoint.onnx
│   ├── fold_3/
│   │   └── checkpoint.onnx
│
├── SAM2/                        # Fine-tuned SAM2 model weights
│   ├── fold_1/
│   │   ├── checkpoint.pt        # Final selected PyTorch checkpoint for fold 1
│   │   └── sam2.1_hiera_b+.yaml # Configuration file for SAM2
│   ├── fold_2/
│   │   ├── checkpoint.pt
│   │   └── sam2.1_hiera_b+.yaml
│   ├── fold_3/
│   │   ├── checkpoint.pt
│   │   └── sam2.1_hiera_b+.yaml
│
├── SW-FastEdit/                 # Fine-tuned SW-FastEdit model weights
│   ├── fold_1/
│   │   └── checkpoint.pt        # Final selected checkpoint for fold 1
│   ├── fold_2/
│   │   └── checkpoint.pt
│   ├── fold_3/
│   │   └── checkpoint.pt
│
└── test_onnxruntime.py          # Script for testing ONNX models
```

## Usage

- **Place only the final selected weights** for each model in this directory.
- Each model should be stored under its respective folder (`DINs`, `SAM2`, `SW-FastEdit`).
- Models should be structured according to training folds (e.g., `fold_1`, `fold_2`, `fold_3`).
- Keep configuration files (e.g., `.yaml`) alongside the respective model weights when necessary.
