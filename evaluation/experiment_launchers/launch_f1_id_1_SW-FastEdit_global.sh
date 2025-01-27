#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH

NETWORK="SW-FastEdit"
EVAL_MODE="global_corrective"
TEST_SET_IDS=(1 2 3)


# Please choose parameters
FOLD=1
NUM_INTERACTIONS_PER_LESION=60


echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for TEST_SET_ID in "${TEST_SET_IDS[@]}"; do
    echo "Processing test set $TEST_SET_ID..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --fold $FOLD --evaluation_mode $EVAL_MODE --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu
done

echo "Completed $NETWORK with $EVAL_MODE"
