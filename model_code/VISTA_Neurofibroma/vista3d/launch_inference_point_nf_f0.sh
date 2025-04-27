#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 -m scripts.validation.val_multigpu_point_patch run --config_file "['configs/supported_eval/infer_patch_point_nf.yaml']"