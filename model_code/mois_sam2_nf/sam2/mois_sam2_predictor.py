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

import torch
import numpy as np
import cv2
from tqdm import tqdm
from collections import OrderedDict

from sam2.modeling.mois_sam2_base import MOISSAM2Base
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores
from sam2.utils.mois_misc import load_scan_slices


class MOISSAM2Predictor(MOISSAM2Base, SAM2VideoPredictor):
    def __init__(
        self, 
        use_low_res_masks_for_com_detection=True,
        **kwargs,
        ):
        super().__init__(**kwargs)
        # Initialize exemplars dictionary
        self.exemplars_dict = {}  # Stores all exemplars
        self.training = False
        self.use_low_res_masks_for_com_detection = use_low_res_masks_for_com_detection
        if self.use_low_res_masks_for_com_detection:
            # Set the scale factor to transform the center of a lesion mass 
            # from low res 256x256 mask to the original 1024x1024.
            self.com_scale_coefficient = 4
        else:
            self.com_scale_coefficient = 1
    
    def init_state(self, video_path, 
                   offload_video_to_cpu=False,
                   offload_state_to_cpu=False,
                   async_loading_frames=False, 
                   reset_exemplars=True, 
                   **kwargs):
        """Initialize an inference state for a new video."""
        compute_device = self.device  # device of the model
        # Replaced original video frame loading with scan slices loader
        images, video_height, video_width = load_scan_slices(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        
        if reset_exemplars:
            self.exemplars_dict = {}  # Reset exemplars if needed

        inference_state["exemplars"] = self.exemplars_dict  # Reference exemplars in state
        return inference_state
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        raise NotImplementedError(
            "MOIS SAM 2 does not support getting model from Hugging Face"
        )
        
    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """
        Run tracking on a single frame based on current inputs and previous memory.
        Updated output with current exemplar placeholder.
        The methods, that call _run_single_frame_inference, 
        should include exemplar handling logic. 
        It was not included here to prevent duplicaiton of data, 
        like maskmem_features, pred_masks, and other.
        """
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        
        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        # Extract "current_exemplar" from current_out (added in MOISSAM2Base)
        current_exemplar = current_out.get("current_exemplar", None)
        
        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )  
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        
        if current_exemplar is not None:
            compact_current_out["current_exemplar"] = current_exemplar  # Include exemplar in the returned output
        return compact_current_out, pred_masks_gpu
    
    def create_exemplar(self, 
                        inference_state, 
                        frame_idx, 
                        obj_idx, 
                        current_exemplar,
                        obj_ptr,
                        maskmem_features,
                        maskmem_pos_enc,
                        pred_masks,
                        object_score_logits,
                        is_prompted,
                        ):
        """Form and store an exemplar using segmentation results."""
        if maskmem_features is None:
            high_res_masks = torch.nn.functional.interpolate(
                pred_masks.to(inference_state["device"]),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=1,
                high_res_masks=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=True,  # User provided input points
            )
        else:
            maskmem_features = maskmem_features
            maskmem_pos_enc = maskmem_pos_enc

        # Update exemplar dictionary
        current_exemplar.update({
            "obj_idx": obj_idx,
            "frame_idx": frame_idx,
            "is_prompted": is_prompted,
            "obj_ptr": obj_ptr,
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
        })
        return current_exemplar

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        
        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        
        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if not is_com:
            if normalize_coords:
                video_H = inference_state["video_height"]
                video_W = inference_state["video_width"]
                points = points / torch.tensor([video_W, video_H]).to(points.device)
            # scale the (normalized) coordinates by the model's internal image size
            points = points * self.image_size
        else:
            if normalize_coords:
                video_H = inference_state["video_height"]
                video_W = inference_state["video_width"]
                points = points / torch.tensor([video_W, video_H]).to(points.device)
                points = points * self.image_size
            else:
                points = points * self.com_scale_coefficient

        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])
        
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        
        # Determine if this is an initial conditioning frame
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        # Get previous segmentation masks
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        
        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False, # Do not encode results as memories
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        # Extract and remove `current_exemplar` from `current_out`
        current_exemplar = current_out.pop("current_exemplar", None)
        
        # Store the output in temporary storage (without `current_exemplar`)
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        
        if add_exemplar:
            if current_exemplar is not None:
                current_exemplar = self.create_exemplar(
                    inference_state, 
                    frame_idx, 
                    obj_id, 
                    current_exemplar,
                    current_out["obj_ptr"],
                    current_out["maskmem_features"],
                    current_out["maskmem_pos_enc"],
                    current_out["pred_masks"],
                    current_out["object_score_logits"],
                    is_prompted=True)
                
                # Store the exemplar with a unique ID
                exemplar_id = f"{obj_id}_{frame_idx}_prompted"
                self.exemplars_dict[exemplar_id] = current_exemplar
        
        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks
    
    def add_new_mask(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        raise NotImplementedError(
            "MOIS SAM 2 does not support mask-prompting yet"
        )
    
    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        obj_id=None,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        add_exemplar=True
    ):
        """
        Propagate the input points across frames to track in the entire video.
        Update functionality includes exemplars management.
        """
        if add_exemplar:
            assert obj_id is not None, "Error: obj_id must not be None if saving exemplar"
        
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"): # Iterate over slices
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size): # Iterate over objects
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else: # Propagating only slice with no interactions for this object
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    # Iterate over objects - on non-interacted slices
                    # Extract and remove `current_exemplar` from `current_out`
                    current_exemplar = current_out.pop("current_exemplar", None)
                    
                    # Store the output in temporary storage (without `current_exemplar`)
                    obj_output_dict[storage_key][frame_idx] = current_out
                    
                    if add_exemplar:
                        if current_exemplar is not None:
                            current_exemplar = self.create_exemplar(
                                inference_state, 
                                frame_idx, 
                                obj_id, 
                                current_exemplar,
                                current_out["obj_ptr"],
                                current_out["maskmem_features"],
                                current_out["maskmem_pos_enc"],
                                current_out["pred_masks"],
                                current_out["object_score_logits"],
                                is_prompted=False)
                            # Store the exemplar with a unique ID
                            # ToDo: Make sure exemplar_id is not overwritten!
                            exemplar_id = f"{obj_id}_{frame_idx}_nonprompted"
                            self.exemplars_dict[exemplar_id] = current_exemplar
                
                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks
    
    @torch.inference_mode()
    def find_exemplars_in_slice(self,
                                inference_state,
                                frame_idx,
                                binarization_threshold=0.5,
                                min_area_threshold=1, # ToDo: Need to experiment with that
                                ):
        """
        Entry point for finding and segmenting objects 
        similar to exemplars in a given frame.
        
        Args:
            inference_state (dict): The current inference state.
            frame_idx (int): The frame index to process.
        
        The steps include:
        1. Perform semantic segmentation of all objects in a given slice with a set of exemplars.
        2. Define discrete objects in the semantic maske by performing connected component analysis.
        3. Define center of mass (CoM) for each object.
        """
        # Step 1. Perform semantic segmentation.
        current_out, _ = self._segment_exemplars_in_slice(
            inference_state=inference_state,
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            )
        
        # Step 2. Convert the predicted mask to binary
        # pred_mask: torch.Size([1, 1, 256, 256])
        # pred_mask_high_res: torch.Size([1, 1, 1024, 1024])
        if self.use_low_res_masks_for_com_detection:
            pred_mask_logits = current_out["pred_masks"]
        else:
            pred_mask_logits = current_out["pred_masks_high_res"]
        
        pred_mask_semantic = torch.sigmoid(pred_mask_logits)
        pred_mask_semantic_bin = (pred_mask_semantic >= binarization_threshold).float()
        objects_dict = self._extract_objects(pred_mask_semantic_bin, min_area_threshold)
        
        # The dictionary contains:
        # {local_obj_id:
        # {"mask": binary_mask_for_object,
        #  "com_point": center_of_mass_point}}
        # Points should be np.array([[com_x, com_y]], dtype=np.float32)
        # Center of mass is positive click with label np.array([1], np.int32)
        
        obj_ids = inference_state["obj_ids"]
        
        # Restore original resolution for the semantic segmentation mask
        # ToDo: Check whether it works as expected
        _, video_res_semantic_mask = self._get_orig_video_res_output(
            inference_state, current_out["pred_masks"]
        )
                
        return frame_idx, obj_ids, video_res_semantic_mask, objects_dict
    
    def _segment_exemplars_in_slice(self,
                                    inference_state,
                                    frame_idx,
                                    batch_size                                  
                                    ):
        """
        Generate a semantic segmentation mask for all objects in a given frame 
        that are similar to provided exemplars.

        This function:
        1. Retrieves image features from the model.
        2. Uses stored exemplars to track similar objects in the frame.
        3. Generates a **semantic mask** where all found objects are labeled as the same class.
        4. Returns a compact dictionary containing segmentation results.

        Args:
            inference_state (dict): The current inference state of the model.
            frame_idx (int): The index of the frame to process.
            batch_size (int): Number of frames processed in a batch.

        Returns:
            tuple:
                - compact_current_out (dict): A dictionary containing:
                    - `"pred_masks"`: Segmented mask for similar objects (semantic mask).
                    - `"pred_masks_high_res"`: High-resolution version of the mask.
                    - `"obj_ptr"`: Object pointer information.
                    - `"object_score_logits"`: Logits for object presence confidence.
                - pred_masks_gpu (torch.Tensor): Segmentation mask stored on GPU.

        Raises:
            ValueError: If no exemplars are found.
        """
        # Step 1. Retrieve image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        
        # Step 2. Get exemplars and make sure they exist
        exemplars_dict = self.exemplars_dict
        if not exemplars_dict:
            raise ValueError("No exemplars found! Cannot perform segmentation.")
        
        # Step 3. Perform semantic segmentation using exemplars
        current_out = self.track_semantic_exemplars(exemplars_dict,
                                                    frame_idx,
                                                    current_vision_feats,
                                                    current_vision_pos_embeds,
                                                    feat_sizes
                                                    )
        
        # Step 4. Retrieve and post-process outputs
        storage_device = inference_state["storage_device"]
        pred_masks_gpu = current_out["pred_masks"]
        pred_masks_high_res_gpu = current_out["pred_masks_high_res"]
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        
        # ToDo: Check filling holes - may not work properly with semantic mask.
        
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        pred_masks_high_res = pred_masks_high_res_gpu.to(storage_device, non_blocking=True)
        
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "pred_masks": pred_masks,
            "pred_masks_high_res": pred_masks_high_res,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu
    
    def _extract_objects(self, 
                         mask_semantic_bin, 
                         min_area_threshold=5):
        """
        Perform connected component analysis to separate discrete objects
        from a single-class segmentation mask.

        Args:
            mask_semantic_bin (torch.Tensor): Segmented mask (1, 1, 256, 256), where all objects
                                            belong to the same class.
            min_area_threshold (int): Minimum size of components to keep (default: 5 pixels).

        Returns:
            dict: A dictionary where each key is an object ID (local ID), and the value is a dictionary:
                - `"mask"` (torch.Tensor): Binary mask of the object.
                - `"com_point"` (tuple): Center of mass (x, y) of the object.
        """
        binary_mask = mask_semantic_bin.squeeze().cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        
        segmented_objects_dict = {}
        for local_obj_id in range(1, num_labels):  # Ignore background (label 0)
            obj_mask = (labels == local_obj_id).astype(np.uint8)  # Extract object
            area = obj_mask.sum()

            # Remove small noise components
            if area < min_area_threshold:
                continue
            
            com_point = self._define_center_of_mass(obj_mask)
            
            segmented_object = {
                "mask": torch.tensor(obj_mask, dtype=torch.uint8, device="cpu"),
                "com_point": com_point,
            }
            segmented_objects_dict[local_obj_id] = segmented_object
            
        return segmented_objects_dict
    
    def _define_center_of_mass(self, obj_mask):
        """
        Compute the center of mass (CoM) for a given binary object mask
        using distance transform.

        Args:
            obj_mask (np.ndarray): A binary mask of shape (H, W), where the object pixels are 1.

        Returns:
            tuple: (x, y) coordinates of the center of mass.
        """
        if obj_mask.sum() == 0:
            return None  # No valid object

        # Step 1. Compute distance transform (L2 norm) inside the object
        obj_mask_dt = cv2.distanceTransform(obj_mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Step 2. Find the point **furthest from the boundary** (center of mass)
        max_dist_idx = np.argmax(obj_mask_dt)  # Flattened index of max distance
        h, w = obj_mask.shape
        com_x, com_y = max_dist_idx % w, max_dist_idx // w  # Convert to (x, y)
        
        return np.array([[com_x, com_y]], dtype=np.float32)
    
    @torch.inference_mode()
    def propagate_exemplars_in_video(
        self, 
        inference_state,
        frame_idx,
        binarization_threshold=0.5,
        min_area_threshold=1,
        max_frame_num_to_track=None,
        bidirectional=False,
        ):
        
        # Step 1: Find similar objects in the specified slice
        (_, 
         _, 
         _, 
         filtered_objects_dict) = self.find_exemplars_in_slice(
             inference_state, 
             frame_idx, 
             binarization_threshold=binarization_threshold,
             min_area_threshold=min_area_threshold)
                  
        # Step 2: Interact with each found object using its center of mass as a positive point
        for obj_idx, obj_data in filtered_objects_dict.items():
            # ToDo: Check the format of the interaction point: np.array([[210, 350]], dtype=np.float32)
            com_point = obj_data["com_point"]  # Center of mass point 
            com_label = np.array([1], np.int32)
 
            # Add interaction points (positive label: 1)
            _ = self.add_new_points_or_box(
                inference_state,
                frame_idx,
                obj_idx,
                points=com_point,
                labels=com_label,
                is_com=True,
                clear_old_points=True,
                normalize_coords=False, # ToDo: Make sure this is correct
                box=None,
                add_exemplar=False
                )
        
        # Step 3: Propagate all newly segmented objects through video
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.propagate_in_video(
            inference_state=inference_state, 
            start_frame_idx=frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            add_exemplar=False
            ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        if bidirectional:
            # ToDo: Make sure that I am not overwriting the segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.propagate_in_video(
                inference_state=inference_state, 
                start_frame_idx=frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=True,
                add_exemplar=False
                ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        return video_segments
    
    @torch.inference_mode()
    def reset_state(self, inference_state, reset_exemplars=True):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        
        if reset_exemplars:
            self.exemplars_dict = {}  # Reset exemplars if needed
            inference_state["exemplars"].clear()
        
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()
