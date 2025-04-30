#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 -m scripts.train_finetune run --config_file "['configs/finetune/train_finetune_Neurofibroma_prompt_and_auto_branch_f2.yaml']"