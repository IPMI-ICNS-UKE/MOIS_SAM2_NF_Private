#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH
export PYTHONPATH="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarkingPrivate/model_code/VISTA_Neurofibroma/vista3d:$PYTHONPATH"


FOLDS=(3) # In different files: 1, 2, 3
MOIS_SAM_AGGREGATE_EXEMPLARS_VALIDATION="stream_exemplars"
NUM_LESIONS_AND_EXEMPLARS=(7 8 9) # Should be equal to the number of exemplars

NETWORK="MOIS_SAM2" 
NUM_INTERACTIONS_PER_LESION=1
EVAL_MODE="lesion_wise_corrective"
USE_COM_DECODER=false
INFERENCE_ALL_SLICES=true
USE_ONLY_PROMPTED_EXEMPLARS=true # Only interacted exemplars

FILTER_EXEMPLAR_PREDICTION=true
MIN_LESION_AREA=10
NO_PROP_BEYOND_LESIONS=true

echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for FOLD in "${FOLDS[@]}"; do
    for NUM_EXEMPLARS in "${NUM_LESIONS_AND_EXEMPLARS[@]}"; do
        echo "Fold $FOLD | Num Exemplars: $NUM_EXEMPLARS"

        python ../pipelines/evaluation_pipeline.py \
            --network_type $NETWORK \
            --no_prop_beyond_lesions $NO_PROP_BEYOND_LESIONS \
            --exemplar_use_com $USE_COM_DECODER \
            --min_lesion_area_threshold $MIN_LESION_AREA \
            --filter_prev_prediction_components $FILTER_EXEMPLAR_PREDICTION \
            --exemplar_use_only_prompted $USE_ONLY_PROMPTED_EXEMPLARS \
            --exemplar_inference_all_slices $INFERENCE_ALL_SLICES \
            --exemplar_num $NUM_EXEMPLARS \
            --fold $FOLD \
            --test_set_id $FOLD \
            --evaluation_mode $EVAL_MODE \
            --num_lesions $NUM_EXEMPLARS \
            --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION \
            --save_predictions \
            --use_gpu \
            --aggregate_exemplars_validation $MOIS_SAM_AGGREGATE_EXEMPLARS_VALIDATION
    done
done

echo "Completed $NETWORK with $EVAL_MODE"
