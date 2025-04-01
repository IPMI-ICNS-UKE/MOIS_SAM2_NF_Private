#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
python /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NF_SAM_Sandbox/mois_sam2_nf/training/train.py \
-c configs/NF_finetuning/sam2.1_hiera_b+_NF_finetune.yaml \
--use-cluster 0 \
--num-gpus 1