#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH
export PYTHONPATH="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarkingPrivate/model_code/VISTA_Neurofibroma/vista3d:$PYTHONPATH"

NETWORK="VISTA"
AUTOMATIC_INFERENCE=False
EVAL_MODE="lesion_wise_corrective"
TEST_SET_IDS=(1 2 3 4)

# Please choose parameters
FOLD=2
NUM_INTERACTIONS_PER_LESION=3
NUM_LESIONS=20

echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for TEST_SET_ID in "${TEST_SET_IDS[@]}"; do
    echo "Processing test set $TEST_SET_ID..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --fold $FOLD --use_automatic_vista_inference $AUTOMATIC_INFERENCE --evaluation_mode $EVAL_MODE --num_lesions $NUM_LESIONS --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu
done

echo "Completed $NETWORK with $EVAL_MODE"
