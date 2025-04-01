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
import torch.distributed
from torch.nn.init import trunc_normal_
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP
from sam2.modeling.sam2_base import SAM2Base


class MOISSAM2Base(SAM2Base):
    """
    Modified SAM2 model for Multi-Object Instance Segmentation (MOIS).
    Extends SAM2Base class to enable exemplar-based segmentation propagation.
    
    Enhancements:
    - Utilizes all interacted and inter-frame propagated segmentation results as exemplars.
    - Encodes exemplars using the existing memory encoder.
    - Introduces an exemplar attention layer to retrieve objects based on encoded exemplars.
    - Generates a binary semantic mask to segment multiple instances of the same category.            
    """
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        exemplar_attention, # New component for exemplar attention
        num_max_exemplars=10,
        use_exemplar_obj_ptrs_in_encoder=False,
        max_exemplar_obj_ptrs_in_encoder=16,
        use_signed_tpos_enc_to_exemplar_obj_ptrs=True,
        add_tpos_enc_to_exemplar_obj_ptrs=True,
        use_mlp_for_exemplar_obj_ptr_proj = False,
        proj_tpos_enc_in_exemplar_obj_ptrs=False,
        directly_add_no_exemplar_embed=False,        
        **kwargs
    ):
        """
        Initialize the MOISSAM2Base model with the provided configuration.            
        """
        self.use_exemplar_obj_ptrs_in_encoder = use_exemplar_obj_ptrs_in_encoder
        self.max_exemplar_obj_ptrs_in_encoder = max_exemplar_obj_ptrs_in_encoder
        self.use_signed_tpos_enc_to_exemplar_obj_ptrs = use_signed_tpos_enc_to_exemplar_obj_ptrs
        self.add_tpos_enc_to_exemplar_obj_ptrs = add_tpos_enc_to_exemplar_obj_ptrs
        
        self.use_mlp_for_exemplar_obj_ptr_proj = use_mlp_for_exemplar_obj_ptr_proj
        self.proj_tpos_enc_in_exemplar_obj_ptrs = proj_tpos_enc_in_exemplar_obj_ptrs
        self.directly_add_no_exemplar_embed = directly_add_no_exemplar_embed
        
        # Call parent constructor
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            **kwargs
        )
        # Initialize exemplar components
        self.exemplar_attention = exemplar_attention
        # Maximum number of exemplars
        self.num_max_exemplars = num_max_exemplars 
        
        # 2 stands for two types of exemplars - conditioning and non-conditioning
        self.exemplar_type_enc = torch.nn.Parameter(
            torch.zeros(2, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.exemplar_type_enc, std=0.02)
        self.no_exemplar_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_exemplar_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_exemplar_embed, std=0.02)
        trunc_normal_(self.no_exemplar_pos_enc, std=0.02)
            
    def _build_sam_heads(self):
        """
        Extended the original method with exemplar-specific module.
        
        Components:
        - **Exemplar Object Pointer Projection (`exemplar_obj_ptr_proj`)**:
            - Converts SAM2 output tokens into object pointers for retrieval.
            - Uses an MLP if `use_mlp_for_exemplar_obj_ptr_proj` is True; otherwise, applies a linear layer.
        - **Temporal Positional Encoding Projection (`exemplar_obj_ptr_tpos_proj`)**:
            - Projects temporal positional encoding separately to avoid interference with spatial encoding.
        """
        super()._build_sam_heads()
        
        # Exemplar object pointer projection
        if self.use_exemplar_obj_ptrs_in_encoder:
            # A linear projection on SAM output tokens to turn them into object pointers
            self.exemplar_obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_exemplar_obj_ptr_proj:
                self.exemplar_obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.exemplar_obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_exemplar_obj_ptrs:
            # A linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.exemplar_obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.exemplar_obj_ptr_tpos_proj = torch.nn.Identity()

    def _prepare_exemplars_conditioned_features(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        exemplars_dict,
    ):
        """
        Fuse the current frame's visual feature map with available exemplars.
        
         This method conditions the feature map of the current frame on stored exemplars 
        (previously segmented objects) to propagate segmentation across similar instances 
        within the same video. The goal is to identify all occurrences of the same object 
        category in the current frame using existing exemplars.

        Args:
            frame_idx (int): Current frame index.
            current_vision_feats (list of tensors): Feature maps of the current frame from different levels.
            current_vision_pos_embeds (list of tensors): Positional embeddings corresponding to the current frame's features.
            feat_sizes (list of tuples): Spatial resolutions of feature maps at different levels.
            exemplars_dict (dict): Dictionary of stored exemplars with segmentation features.

        Returns:
            torch.Tensor: Conditioned pixel features for segmentation.
        """
        B = current_vision_feats[-1].size(1)  # Batch size
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # Lowest resolution feature size
        device = current_vision_feats[-1].device

        # The case of `self.num_max_exemplars == 0` stands for not using exemplar propagation
        # In this case, we skip the fusion with any exemplars.
        if (self.num_max_exemplars == 0):  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        
        num_obj_ptr_tokens = 0 # Number of object pointer tokens
        
        # If we already have exemplars
        if exemplars_dict:
            
            to_cat_exemplars_features = []
            to_cat_exemplars_pos_embed = []
            exemplar_pos_and_ptrs = []
                        
            # Separate prompted and non-prompted exemplars
            exemplars_all = exemplars_dict.values()
            prompted_exemplars = [ex for ex in exemplars_all if ex.get("is_prompted", False)]
            non_prompted_exemplars = [ex for ex in exemplars_all if not ex.get("is_prompted", False)]

            # Prioritize prompted exemplars, then fill remaining slots with non-prompted ones
            selected_exemplars = prompted_exemplars[:self.num_max_exemplars]
            num_remaining = self.num_max_exemplars - len(selected_exemplars)
            if num_remaining > 0:
                selected_exemplars.extend(non_prompted_exemplars[:num_remaining])
            
            # Extract and process information from selected exemplars         
            for exemplar in selected_exemplars:
                # Condition type
                condtype = 1 if exemplar["is_prompted"] else 0
                ex_condtype = self.exemplar_type_enc[condtype] # Encoding for prompted vs non-prompted
                
                ex_frame_idx = exemplar["frame_idx"]  # Slice index of the exemplar
                
                # Exemplar features
                ex_feat = exemplar["maskmem_features"].to(device, non_blocking=True)
                ex_feat = ex_feat.flatten(2).permute(2, 0, 1)  # Convert BCHW -> (HW)BC
                to_cat_exemplars_features.append(ex_feat)
                
                # Spatial position embedding
                pos_embed = exemplar["maskmem_pos_enc"][-1].to(device)
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                pos_embed = (pos_embed + ex_condtype)
                to_cat_exemplars_pos_embed.append(pos_embed)
                
                # Exemplar relative position and object pointer
                ex_pos = frame_idx - ex_frame_idx
                ex_obj_ptr = exemplar["obj_ptr"].to(device)
                exemplar_pos_and_ptrs.append((ex_pos, ex_obj_ptr))
            
            # Form object pointers
            if len(exemplar_pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*exemplar_pos_and_ptrs)
                
                # Stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                
                # Add temporal positional embedding based on how far an exemplar is
                if self.add_tpos_enc_to_exemplar_obj_ptrs:
                    
                    t_diff_max = max(1, max(pos_list) - min(pos_list))
                    tpos_dim = tpos_dim = C if self.proj_tpos_enc_in_exemplar_obj_ptrs else self.mem_dim
                    obj_pos = torch.tensor(pos_list).to(
                                device=device, non_blocking=True
                            )
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self.exemplar_obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                
                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                
                to_cat_exemplars_features.append(obj_ptrs)
                to_cat_exemplars_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        
        else: # No exemplars yet
            if self.directly_add_no_exemplar_embed:
                # directly add no-exemplar embedding (instead of using the transformer encoder)
                pix_feat_with_exemplar = current_vision_feats[-1] + self.no_exemplar_embed
                pix_feat_with_exemplar = pix_feat_with_exemplar.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_exemplar
            
            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_exemplars_features = [self.no_exemplar_embed.expand(1, B, self.mem_dim)]
            to_cat_exemplars_pos_embed = [self.no_exemplar_pos_enc.expand(1, B, self.mem_dim)]
        
        # Step 2: Concatenate the exemplars and forward through the transformer encoder
        memory = torch.cat(to_cat_exemplars_features, dim=0)
        memory_pos_embed = torch.cat(to_cat_exemplars_pos_embed, dim=0)
        
        # Process feature map with exemplar attention mechanism
        pix_feat_with_exemplar = self.exemplar_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            exemplars=memory,
            exemplar_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_exemplar = pix_feat_with_exemplar.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_exemplar
    
    def track_semantic_exemplars(
        self,
        exemplars_dict, # MOIS-specific: Stores exemplar information
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
    ):
        """
        Tracks and propagates exemplars stored in `exemplars_dict` to segment all same-class objects 
        in the current frame.

        This method performs exemplar-based semantic segmentation by identifying multiple instances 
        of an object class in a frame, leveraging stored exemplars. The obtained segmentation mask 
        represents all detected objects of the same class. However, since the mask corresponds to 
        multiple objects, feature representation is **not** extracted. Instead, the resulting mask 
        should be processed via connected component analysis to determine the center of mass for 
        each object. These centers serve as interaction points for interactive segmentation.

        Assumptions:
        - No explicit interaction points are provided, as this function strictly propagates exemplars.
        - The current frame is a conditioning frame (i.e., prompted frame).

        Args:
            exemplars_dict (dict): Dictionary containing previously saved exemplars for tracking.
            frame_idx (int): Index of the current frame being processed.
            current_vision_feats (list of tensors): Feature maps from different levels for the current frame.
            current_vision_pos_embeds (list of tensors): Positional embeddings corresponding to the current frame.
            feat_sizes (list of tuples): Spatial resolutions of feature maps at different levels.

        Returns:
            dict: Dictionary containing:
                - `frame_idx` (int): The current frame index.
                - `pred_masks` (Tensor): Low-resolution semantic segmentation mask.
                - `pred_masks_high_res` (Tensor): High-resolution semantic segmentation mask.
                - `obj_ptr` (Tensor): Object pointer representation (for further processing).
                - `object_score_logits` (Tensor, optional): Object confidence scores (only in inference mode).
        """
        # Track exemplars in the current frame and obtain segmentation outputs        
        current_out, sam_outputs, _, _ = self._track_exemplars(
            exemplars_dict,
            frame_idx,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        )
        
        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        
        # Store segmentation results in the output dictionary
        current_out["frame_idx"] = frame_idx
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["ious"] = ious
        
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits
        
        return current_out
    
    def _track_exemplars(
        self,
        exemplars_dict,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
    ):
        """
        Tracks and segments all same-looking objects in the current frame using previously stored exemplars.

        This method conditions the feature representation of the current frame on the stored exemplars
        to propagate segmentation across objects of the same class. It processes the frame's features,
        fuses them with memory features from exemplars, and forwards them through the segmentation heads.

        Args:
            exemplars_dict (dict): Dictionary containing exemplar features and metadata.
            frame_idx (int): The index of the current frame being processed.
            current_vision_feats (list of tensors): Multi-scale feature maps extracted from the vision backbone.
            current_vision_pos_embeds (list of tensors): Corresponding positional embeddings for the feature maps.
            feat_sizes (list of tuples): Spatial dimensions of the feature maps at different scales.

        Returns:
            tuple:
                - `current_out` (dict): Output dictionary containing:
                    - `"point_inputs"` (None): Placeholder for point-based interaction.
                    - `"mask_inputs"` (None): Placeholder for mask-based interaction.
                - `sam_outputs` (tuple): Output from the SAM head containing:
                    - Segmentation masks and object tracking outputs.
                - `high_res_features` (list or None): High-resolution feature maps for refining segmentation.
                - `pix_feat` (tensor): Processed feature map conditioned on exemplars.
        """
        # Initialize output dictionary with placeholders for interactive segmentation inputs
        current_out = {"point_inputs": None, "mask_inputs": None}
        
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # Fuse the current frame's features with features from exemplars
        pix_feat = self._prepare_exemplars_conditioned_features(
            frame_idx=frame_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            exemplars_dict=exemplars_dict
        )
        
        # Forward SAM heads - update object tracking
        sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=None, # No explicit point-based interactions
                mask_inputs=None, # No explicit mask-based interactions
                high_res_features=high_res_features,
                multimask_output=False, # Single-mask output mode
            )
        
        return current_out, sam_outputs, high_res_features, pix_feat
    
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
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        ):
        """
        Extends the original `track_step` method to include a newly segmented exemplar 
        as a dedicated key-value pair in the output dictionary.

        This method first calls the superclass `track_step` to perform standard tracking 
        and segmentation. After obtaining the segmentation results, it extracts and 
        stores a newly segmented exemplar, which will later be added to the Exemplar 
        Dictionary in the `Predictor` class.
            
        Args:
            frame_idx (int): Index of the current frame being processed.
            is_init_cond_frame (bool): Indicates whether the current frame is a conditioning (prompted) frame.
            current_vision_feats (list of tensors): Multi-scale feature maps extracted from the vision model.
            current_vision_pos_embeds (list of tensors): Positional embeddings corresponding to the feature maps.
            feat_sizes (list of tuples): Spatial resolutions of the feature maps at different levels.
            point_inputs (Tensor or None): Interaction points for segmentation (if available).
            mask_inputs (Tensor or None): Mask inputs for segmentation (if available).
            output_dict (dict): Dictionary to store output results of segmentation.
            num_frames (int): Total number of frames in the video sequence.
            track_in_reverse (bool, optional): If True, tracking propagates in reverse order. Defaults to False.
            run_mem_encoder (bool, optional): Whether to run the memory encoder to update stored memory. Defaults to True.
            prev_sam_mask_logits (Tensor or None, optional): Previous segmentation mask logits (if available).

        Returns:
            dict: Updated output dictionary containing:
                - Standard tracking results from `super().track_step()`
                - `"current_exemplar"`: Newly segmented exemplar to be added to the Exemplar Dictionary.
        """
        # Call the original method to perform standard tracking
        current_out = super().track_step(
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
            run_mem_encoder,
            prev_sam_mask_logits,
        )
        
        # Check whether an exemplar is positive
        binary_mask = torch.sigmoid(current_out["pred_masks_high_res"]) > 0.5
        if binary_mask.sum() != 0:
            # Extract and save the newly segmented exemplar
            current_out = self._create_exemplar_placeholder_in_current_out(
                frame_idx,
                point_inputs,
                current_out,
            )
        # The output now may contain "current_exemplar" as a key with the new exemplar as a value.
        # This exemplar will later be stored in the Exemplar Dictionary in the Predictor class.
        return current_out
    
    def _create_exemplar_placeholder_in_current_out(
        self,
        frame_idx,
        point_inputs,
        current_out,
    ):
        """
        Creates and adds an exemplar entry to the `current_out` dictionary.

        This helper method extracts key features from the current segmentation output 
        and structures them into an exemplar format. The exemplar contains relevant 
        metadata and feature representations necessary for tracking and segmentation 
        propagation.

        Args:
            frame_idx (int): The index of the current frame where the exemplar is extracted.
            point_inputs (Tensor or None): Interaction points for segmentation; 
                if provided, the exemplar is marked as prompted.
            current_out (dict): The dictionary containing the current segmentation outputs.

        Returns:
            dict: Updated `current_out` dictionary, now including:
                - `"current_exemplar"` (dict): The newly extracted exemplar with:
                    - `"frame_idx"` (int): Frame index of the exemplar.
                    - `"is_prompted"` (bool): Whether the exemplar was derived from user input.
                    - `"obj_ptr"` (Tensor): Placeholder for object pointer representation.
                    - `"maskmem_features"` (Tensor): Placeholder for feature representation of the exemplar.
                    - `"maskmem_pos_enc"` (Tensor): Placeholder for positional encoding of the exemplar.
        """
        
        # Initialize a dictionary for the current exemplar
        current_exemplar = {
            "frame_idx": frame_idx,  # Frame index where this exemplar was extracted
            "is_prompted": point_inputs is not None,  # True if the exemplar comes from user interaction
            "obj_ptr": None,  # Object pointer representation
            "maskmem_features": None,  # Extracted feature map
            "maskmem_pos_enc": None,  # Corresponding positional encoding
        }

        # Add the newly formed exemplar to the output dictionary
        current_out["current_exemplar"] = current_exemplar

        return current_out
