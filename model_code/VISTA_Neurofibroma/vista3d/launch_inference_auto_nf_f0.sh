#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 -m scripts.validation.val_multigpu_point_patch run --config_file "['configs/supported_eval/infer_patch_auto_nf.yaml']"