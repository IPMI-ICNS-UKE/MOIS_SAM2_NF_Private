# README for `model_weights` Directory

This directory should contain pre-trained and fine-tuned model weights for various interactive segmentation models, including **DINs, SAM2, STCN, and SW-FastEdit**. These models are structured into subdirectories for better organization. Please place downloaded pretrained weights following the directory structure:

## Directory Structure

```
.
├── convert_tf_to_onnx.py        # Script for converting DINs TensorFlow models to ONNX
├── export_sam2_model_weights.py # Script for exporting SAM2 model weights
├── DINs_pretrained/             # Pre-trained DINs model weights
│   └── best_ckpt/               # Best checkpoint for DINs model
│       ├── checkpoint           # Model checkpoint file
│       ├── ckpt.data-00000-of-00002
│       ├── ckpt.data-00001-of-00002
│       ├── ckpt.index
├── SAM2_pretrained/             # Pre-trained SAM2 model weights
│   ├── sam2.1_hiera_base_plus.pt
│   ├── sam2.1_hiera_b+.yaml
├── SW_FastEdit_pretrained/      # Pre-trained SW-FastEdit model
│   └── 151_best_0.8534.pt
```

## Model Descriptions

### **DINs (Deep Interactive Networks)**
- **Pre-trained** (`DINs_pretrained/`): Contains the best checkpoint of the DINs model.

### **SAM2 (Segment Anything Model 2)**
- **Pre-trained** (`SAM2_pretrained/`): Contains the original Hiera Base Plus transformer weights used in SAM2.

### **SW-FastEdit**
- **Pre-trained** (`SW_FastEdit_pretrained/`): The best pre-trained SW-FastEdit checkpoint.

## Model Conversion and Export Scripts
- `convert_tf_to_onnx.py`: Converts DINs TensorFlow models to ONNX format.
- `export_sam2_model_weights.py`: Extracts and saves the SAM2 model weights.
