# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ------------------------------------------------------------------------------
# File created by George Kolokolnikov
# Image Processing and Medical Informatics (IPMI)
# University Medical Center Hamburg-Eppendorf (UKE)

# This file is part of an extension of the Segment Anything Model 2 (SAM2),
# aimed at enabling multi-instance object interactive segmentation (MOIS).

# This work builds upon the original SAM2 framework and integrates exemplar-based 
# learning to enhance segmentation propagation across instances of the same class.

# The objective of this extension is to adapt SAM2 for lesion-wise interactive 
# segmentation, with the ability to propagate segmentation results to 
# non-interacted lesions. The final goal is to apply this method to 
# Neurofibroma segmentation in whole-body MRI.
# ------------------------------------------------------------------------------

import logging
import torch
import numpy as np
from sam2.modeling.mois_sam2_base import MOISSAM2Base
from training.model.sam2 import SAM2Train
from training.utils.mois_data_utils import MOISBatchedVideoDatapoint


class MOISSAM2Train(MOISSAM2Base, SAM2Train):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        exemplar_attention,
        freeze_image_encoder=False, # ToDO: Check whether we need to freeze the encoder
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        num_frames_to_correct_for_train=1,  
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        num_init_cond_frames_for_train=1,  
        num_init_cond_frames_for_eval=1,
        rand_init_cond_frames_for_train=True,
        rand_init_cond_frames_for_eval=False,
        add_all_frames_to_correct_as_cond=False,
        num_correction_pt_per_frame=7,
        pt_sampling_for_eval="center",
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        forward_backbone_per_frame_for_eval=False,
        reset_exemplars_dict_each_time=True,
        aggregate_exemplars_first=True,
        num_max_exemplars_to_store=16,
        **kwargs,
    ):
        super().__init__(image_encoder, 
                         memory_attention, 
                         memory_encoder, 
                         exemplar_attention,
                         **kwargs)
        
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        
        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_train=prob_to_use_box_input_for_train
        self.prob_to_use_box_input_for_eval=prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        self.exemplars_dict = {}  # Initialize dictionary to store exemplars
        self.reset_exemplars_dict_each_time = reset_exemplars_dict_each_time
        self.aggregate_exemplars_first = aggregate_exemplars_first
        self.num_max_exemplars_to_store = num_max_exemplars_to_store
        self.exemplars_activated = None # Flag to calculate exemplars
    
    
    def forward(self, input: MOISBatchedVideoDatapoint, exemplars_activated=False):
        self.exemplars_activated = exemplars_activated
        if self.reset_exemplars_dict_each_time:
            self.exemplars_dict = {}
        previous_stages_out = super().forward(input)
        return previous_stages_out
    
    
    def forward_tracking(
        self, backbone_out, input: MOISBatchedVideoDatapoint, return_dict=False):
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)
        
        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        
        # Helper function for getting vision features
        def get_vision_features(stage_id, vision_feats, vision_pos_embeds):
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                return [x[:, img_ids] for x in vision_feats], [x[:, img_ids] for x in vision_pos_embeds]
            else:
                _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
                return vision_feats, vision_pos_embeds
        
        # First Pass: Process Frames
        for stage_id in processing_order:
            current_vision_feats, current_vision_pos_embeds = get_vision_features(stage_id, vision_feats, vision_pos_embeds)
            
            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                # If True, calculate semantic object mask right away
                exemplar_based_inference=(not self.aggregate_exemplars_first) and self.exemplars_activated
            )
            
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out
        
        # Second Pass: If semantic masks of objects segmented using the exemplar bank was not calculated during the first loop,
        # then let's get segmentation of all objects in each frame given the complete (aggregated) exemplar bank.                
        if self.aggregate_exemplars_first and self.exemplars_activated:
            for stage_id in processing_order:
                # Same as in the previous loop: getting features
                current_vision_feats, current_vision_pos_embeds = get_vision_features(stage_id, vision_feats, vision_pos_embeds)
                
                # Extract single instance features (semantic segmentation uses shared features)
                current_vision_feats_semantic = [feat[:, 0, :].unsqueeze(1) for feat in current_vision_feats]
                current_vision_pos_embeds_semantic = [pos_enc[:, 0, :].unsqueeze(1) for pos_enc in current_vision_pos_embeds]
                    
                # Segment all objects in the current frame given all aggregated exemplars
                (semantic_pred_high_res, 
                 semantic_pred_ious) = self.segment_exemplars_in_slice(
                     stage_id,
                     current_vision_feats_semantic,
                     current_vision_pos_embeds_semantic,
                     feat_sizes)
                
                # Append the output, depending on whether it's a conditioning frame
                add_output_as_cond_frame = stage_id in init_cond_frames or (
                    self.add_all_frames_to_correct_as_cond
                    and stage_id in frames_to_add_correction_pt
                )
                # Insert two new key-values to the corresponding dictionary
                if add_output_as_cond_frame:
                    output_dict["cond_frame_outputs"][stage_id]["semantic_pred_high_res"] = [semantic_pred_high_res]
                    output_dict["cond_frame_outputs"][stage_id]["semantic_pred_ious"] = [semantic_pred_ious]
                else:
                    output_dict["non_cond_frame_outputs"][stage_id]["semantic_pred_high_res"] = [semantic_pred_high_res]
                    output_dict["non_cond_frame_outputs"][stage_id]["semantic_pred_ious"] = [semantic_pred_ious]
        
        if not self.exemplars_activated:
            for stage_id in processing_order:
                # Append the output, depending on whether it's a conditioning frame
                add_output_as_cond_frame = stage_id in init_cond_frames or (
                    self.add_all_frames_to_correct_as_cond
                    and stage_id in frames_to_add_correction_pt
                )
                # Insert two new key-values to the corresponding dictionary
                if add_output_as_cond_frame:
                    output_dict["cond_frame_outputs"][stage_id]["semantic_pred_high_res"] = None
                    output_dict["cond_frame_outputs"][stage_id]["semantic_pred_ious"] = None
                else:
                    output_dict["non_cond_frame_outputs"][stage_id]["semantic_pred_high_res"] = None
                    output_dict["non_cond_frame_outputs"][stage_id]["semantic_pred_ious"] = None
            
        if return_dict:
            return output_dict
        
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]
        return all_frame_outputs
    
    
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
        exemplar_based_inference=False, # Flag to run exemplar-based inference right away without full memory bank aggregation
    ):
        
        # print("POINT INPUT "*5)
        # print("Frame: ", frame_idx)
        # print("Points: ", point_inputs)
        
        # Reuse the original inference logic to perform prompt-based or memory-based segmentation
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
            
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        
        # Store predictions in current_out
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Encode memory features for future tracking
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        
        # Use new inference logic to perform exemplar-based semantic segmentation   
        # First, get new exemplars
        batch_size = current_out["obj_ptr"].shape[0]  # Get batch size
                
        for i in range(batch_size):
            # Check whether an exemplar is positive
            binary_mask = torch.sigmoid(current_out["pred_masks_high_res"][i]) > 0.5
            if binary_mask.sum() == 0:
                continue  # Skip adding this exemplar if the mask is empty
    
            current_exemplar = {
                "frame_idx": frame_idx,  # Frame index where this exemplar was extracted
                "is_prompted": is_init_cond_frame,  # True if the exemplar comes from user interaction
                "obj_ptr": current_out["obj_ptr"][i].unsqueeze(0).to("cpu").detach(),  # Convert to batch=1
                "maskmem_features": current_out["maskmem_features"][i].unsqueeze(0).to("cpu").detach(),  # Convert to batch=1
                "maskmem_pos_enc": self.move_tensor_list_to_cpu([t[i].unsqueeze(0) for t in current_out["maskmem_pos_enc"]]),  # Convert each tensor
            }
            
            prompted_str = "prompted" if is_init_cond_frame else "nonprompted"
            exemplar_id = f"{len(self.exemplars_dict)}_{frame_idx}_{i}_{prompted_str}"
            
            # Manage Exemplar Dictionary Size
            if len(self.exemplars_dict) >= self.num_max_exemplars_to_store:
                self._replace_oldest_exemplar(is_init_cond_frame, exemplar_id)

            # Store the new exemplar
            self.exemplars_dict[exemplar_id] = current_exemplar
            
        # Second, extract only single copy of features for the current frame, 
        # since in this case we are working with semantic masks, 
        # but image features were duplicated for N instances.
        current_vision_feats_semantic = [feat[:, 0, :].unsqueeze(1) for feat in current_vision_feats]
        current_vision_pos_embeds_semantic = [pos_enc[:, 0, :].unsqueeze(1) for pos_enc in current_vision_pos_embeds]
        
        if exemplar_based_inference:
            # Perform exemplar-based segmentation
            (semantic_pred_high_res, 
             semantic_pred_ious) = self.segment_exemplars_in_slice(
                 frame_idx,
                 current_vision_feats_semantic,
                 current_vision_pos_embeds_semantic,
                 feat_sizes
                 )
            # Insert the updated result to current_out
            current_out["semantic_pred_high_res"] = [semantic_pred_high_res]
            current_out["semantic_pred_ious"] = [semantic_pred_ious]
            
        return current_out
    
    
    def segment_exemplars_in_slice(self,
                                   frame_idx,
                                   current_vision_feats,
                                   current_vision_pos_embeds,
                                   feat_sizes                               
                                   ):
        exemplars_dict = self.exemplars_dict
        
        # print("Segmenting Exemplars "*5)
        # for exemplar_key in exemplars_dict.keys():
        #     print(exemplar_key)
        
        # Perform semantic segmentation using exemplars
        semantic_current_out = self.track_semantic_exemplars(exemplars_dict,
                                                    frame_idx,
                                                    current_vision_feats,
                                                    current_vision_pos_embeds,
                                                    feat_sizes
                                                    )
        semantic_pred_high_res = semantic_current_out["pred_masks_high_res"]
        semantic_pred_ious = semantic_current_out["ious"]
        return semantic_pred_high_res, semantic_pred_ious   
    
    
    def move_tensor_list_to_cpu(self, tensor_list):
        return [t.to("cpu").detach() for t in tensor_list]
    
    def _replace_oldest_exemplar(self, is_prompted, new_exemplar_id):
        """
        Ensures that the exemplar dictionary does not exceed `num_max_exemplars_to_store`.

        - If the new exemplar is prompted, replace the oldest non-prompted exemplar first.
        - If the new exemplar is non-prompted, replace the oldest non-prompted exemplar first.
        - If there are only prompted exemplars, replace the oldest prompted exemplar.
        - If all exemplars are prompted and a new non-prompted exemplar arrives, do not add it.
        """
        
        # Separate prompted and non-prompted exemplars
        prompted_exemplars = {key: val for key, val in self.exemplars_dict.items() if val["is_prompted"]}
        nonprompted_exemplars = {key: val for key, val in self.exemplars_dict.items() if not val["is_prompted"]}

        # Sort exemplars by `frame_idx` (oldest first)
        sorted_prompted = sorted(prompted_exemplars.items(), key=lambda x: x[1]["frame_idx"])
        sorted_nonprompted = sorted(nonprompted_exemplars.items(), key=lambda x: x[1]["frame_idx"])
        
        # Case 1: New exemplar is prompted
        if is_prompted:
            if sorted_nonprompted:
                # Remove the oldest non-prompted exemplar
                oldest_nonprompted_key, _ = sorted_nonprompted[0]
                del self.exemplars_dict[oldest_nonprompted_key]
            elif sorted_prompted:
                # Remove the oldest prompted exemplar if no non-prompted exist
                oldest_prompted_key, _ = sorted_prompted[0]
                del self.exemplars_dict[oldest_prompted_key]

        # Case 2: New exemplar is non-prompted
        else:
            if sorted_nonprompted:
                # Remove the oldest non-prompted exemplar
                oldest_nonprompted_key, _ = sorted_nonprompted[0]
                del self.exemplars_dict[oldest_nonprompted_key]
            else:
                # If only prompted exemplars exist, do not add new non-prompted exemplar
                return
    