#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH

NETWORK="MOIS_SAM2"
EVAL_MODE="lesion_wise_corrective"
TEST_SET_IDS=(1 2 3)


# Please choose parameters
FOLD=1
NUM_INTERACTIONS_PER_LESION=1
NUM_LESIONS=20


USE_COM_DECODER=true
USE_ONLY_PROMPTED_EXEMPLARS=false
INFERENCE_ALL_SLICES=true
NUMBER_OF_EXEMPLARS=8


echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for TEST_SET_ID in "${TEST_SET_IDS[@]}"; do
    echo "Processing test set $TEST_SET_ID..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --exemplar_use_com $USE_COM_DECODER --exemplar_use_only_prompted $USE_ONLY_PROMPTED_EXEMPLARS --exemplar_inference_all_slices $INFERENCE_ALL_SLICES --exemplar_num $NUMBER_OF_EXEMPLARS --fold $FOLD --evaluation_mode $EVAL_MODE --num_lesions $NUM_LESIONS --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu
done

echo "Completed $NETWORK with $EVAL_MODE"
