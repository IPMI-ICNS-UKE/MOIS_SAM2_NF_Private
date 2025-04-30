#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH
export PYTHONPATH="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarkingPrivate/model_code/VISTA_Neurofibroma/vista3d:$PYTHONPATH"

TEST_SET_IDS=(1 2 3 4)

NETWORK="MOIS_SAM2"
EVAL_MODE="global_corrective"
USE_COM_DECODER=false
INFERENCE_ALL_SLICES=true

# Please choose parameters
FOLD=1
NUM_INTERACTIONS_PER_LESION=8

NUMBER_OF_EXEMPLARS=8
USE_ONLY_PROMPTED_EXEMPLARS=false
FILTER_EXEMPLAR_PREDICTION=true
MIN_LESION_AREA=40

RESULTS_DIR="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarkingPrivate/evaluation/results/id_9"

echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for TEST_SET_ID in "${TEST_SET_IDS[@]}"; do
    echo "Processing test set $TEST_SET_ID..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --results_dir $RESULTS_DIR --exemplar_use_com $USE_COM_DECODER --min_lesion_area_threshold $MIN_LESION_AREA --filter_prev_prediction_components $FILTER_EXEMPLAR_PREDICTION --exemplar_use_only_prompted $USE_ONLY_PROMPTED_EXEMPLARS --exemplar_inference_all_slices $INFERENCE_ALL_SLICES --exemplar_num $NUMBER_OF_EXEMPLARS --fold $FOLD --evaluation_mode $EVAL_MODE --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu
done

echo "Completed $NETWORK with $EVAL_MODE"
