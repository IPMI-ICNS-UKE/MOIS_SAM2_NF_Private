#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 -m scripts.train_finetune run --config_file "['configs/finetune/train_finetune_Neurofibroma_f1.yaml']"