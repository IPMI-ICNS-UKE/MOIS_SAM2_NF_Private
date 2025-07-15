#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
monailabel start_server --app nf_segmentation_app --studies data_folder --conf models mois_sam_interaction
