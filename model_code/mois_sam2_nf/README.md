# MOIS-SAM2: Multi-Object Interactive Segmentation for Neurofibromas

This repository extends the [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) by Meta AI for the task of **multi-object interactive segmentation** of neurofibromas in whole-body MRI scans of NF1 patients.

---

## Highlights

- Based on SAM2, a foundation model for promptable segmentation in images and videos.
- Extended for multi-lesion medical image segmentation with:
  - Exemplar-based interaction modules
  - Multi-object handling in medical video-like MRI sequences
  - Integration with NF datasets for training and evaluation
- Custom launchers for various fine-tuning and training scenarios
- Medical-specific transformations, sampling, and data loading
- Evaluation scripts tailored to the NF segmentation use case

---

## Key Modifications Compared to the Original SAM2 Repository

### Root Directory
- `launchers/` — Shell scripts to launch various training and evaluation modes
- `*.sh` — Specific training and finetuning scripts
- `NF_sample_train_list.txt` — Example training list for NF dataset

### Model Extensions
- `sam2/modeling/exemplar_attention.py` — Attention module for exemplar-based semantic propagation
- `sam2/modeling/exemplar_encoder.py` — Encoder logic for exemplar features
- `sam2/modeling/mois_sam2_base.py` — MOIS-SAM2 base model integrating exemplar attention

### Configuration Files
- `sam2/config/NF_finetuning/` — Configs for fine-tuning original SAM2 on NF task
- `sam2/config/mois_sam2.1/exp_*/` — Experiment-specific configurations for MOIS-SAM2

### Inference and Utility
- `sam2/mois_sam2_predictor.py` — Predictor for MOIS-SAM2 inference
- `sam2/build_mois_sam.py` — Build function for model construction
- `sam2/utils/mois_misc.py` — Miscellaneous utilities

### Tests
- `tests/` — Unit tests for added modules

### Training Pipeline
- `training/dataset/` — Dataset loading, sampling, and transforms for NF data
- `training/model/` — Model training logic (`mois_sam2_trainer.py`)
- `training/utils/` — Data utility functions (`mois_data_utils.py`)
- `training/mois_loss_fns.py` — Extended loss function
- `training/mois_trainer.py` — Training loop and engine

---

## Original SAM2 Documentation

The original SAM2 documentation from Meta AI is available in [`README_ORIGINAL.md`](README_ORIGINAL.md).

Please refer to it for general usage of SAM2.

