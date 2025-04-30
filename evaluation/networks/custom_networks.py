from typing import Any
from time import time
from tqdm import tqdm
import logging
import os

import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from functools import partial
from scipy.ndimage import label
import torch.nn.functional as F
from PIL import Image, ImageDraw
from monai.inferers import SlidingWindowInferer

from evaluation.utils.image_cache import ImageCache
# Importing SAM2 dependencies
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_mois_sam import build_mois_sam2_predictor

from model_code.VISTA_Neurofibroma.vista3d.scripts.sliding_window import sliding_window_inference
from model_code.VISTA_Neurofibroma.vista3d.scripts.train import CONFIG, infer_wrapper
from model_code.VISTA_Neurofibroma.vista3d.vista3d.build_vista3d import vista_model_registry

logger = logging.getLogger("evaluation_pipeline_logger")


class DINsNetwork(nn.Module):
    """
    Wrapper class for deploying a Deep Interactive Network (DINs) using an ONNX runtime inference session.

    This class loads a pre-trained ONNX model and provides a PyTorch-compatible forward method 
    to handle input processing and inference.

    Attributes:
        network (onnxruntime.InferenceSession): ONNX inference session for executing the DINs model.
        device (torch.device): The device (CPU/GPU) to which the output tensor is mapped.

    Args:
        model_path (str): Path to the pre-trained ONNX model file.
        providers (list): List of ONNX execution providers (e.g., `["CUDAExecutionProvider", "CPUExecutionProvider"]`).
        device (torch.device): PyTorch device where the output tensor will be stored (e.g., `torch.device("cuda")`).
    """
    def __init__(self, model_path, providers, device):
        """
        Initializes the DINs ONNX-based inference model.
        """
        super().__init__()
        self.network = ort.InferenceSession(
            model_path, 
            providers=providers
            )
        self.device = device
    
    def forward(self, x):
        """
        Performs forward pass using ONNX inference, handling input processing and tensor conversion.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width, depth, channels).
                              Expected format: (B, H, W, D, C), where:
                              - `B` = batch size
                              - `H` = height
                              - `W` = width
                              - `D` = depth
                              - `C` = channels (image + guidance)

        Returns:
            torch.Tensor: Output logits tensor of shape (B, height, width, depth, 2),
                          with values mapped to the specified `device`.

        Processing Steps:
            1. Reorders the input tensor dimensions to match ONNX expected format (B, C, D, H, W).
            2. Splits input into `image` and `guide` tensors.
            3. Converts the tensors to NumPy format for ONNX execution.
            4. Runs ONNX inference and retrieves the output logits.
            5. Converts the output back to a PyTorch tensor and reorders the dimensions.
            6. Transfers the output tensor to the specified device.
        """
        # Reorder input tensor to match ONNX input format (B, C, D, H, W)
        input_tensor_onnx = x.permute(0, 4, 2, 3, 1)
        
        # Extract image and guidance channels
        image_tensor_onnx = input_tensor_onnx[..., :1].cpu().numpy() # Extracts first channel (image)
        guide_tensor_onnx = input_tensor_onnx[..., 1:].cpu().numpy() # Extracts remaining channels (guidance)
        
        # Prepare ONNX input dictionary
        input_onnx = {
            "image": image_tensor_onnx,
            "guide": guide_tensor_onnx
            }
        
        # Perform ONNX inference
        output_onnx = self.network.run(None, input_onnx)[0]
        # Convert ONNX output to PyTorch tensor
        output_tensor = torch.from_numpy(output_onnx)
        # Reorder tensor back to (B, D, H, W, 2) and transfer to correct device
        output_tensor = output_tensor.permute(0, 4, 2, 3, 1).to(dtype=torch.float32)
        return output_tensor.to(self.device)


class SAM2Network(nn.Module):
    """
    Implements the SAM2 model for 3D interactive segmentation using video-based inference.

    This class manages caching, model initialization, and bidirectional propagation of segmentation 
    predictions in a 3D volume. It uses an inference state to maintain consistency across frames.

    Attributes:
        image_cache (ImageCache): Cache manager for storing image slices used in inference.
        model_path (str): Path to the pre-trained model file.
        config_name (str): Name of the configuration file.
        config_path (str): Path to the configuration file.
        device (str): Computation device (`cuda` or `cpu`).
        predictors (dict): Stores initialized predictors for different devices.
        inference_state (Any): Holds the inference state used for propagation.
    
    Args:
        model_path (str): Path to the ONNX or PyTorch model.
        config_path (str): Path to the model configuration file.
        cache_path (str): Directory path for caching input images.
        device (str): Computation device (`cuda` or `cpu`).
    """

    def __init__(self, model_path, config_path, cache_path, device):
        """
        Initializes the SAM2Network model for interactive segmentation.

        Args:
            model_path (str): Path to the model file.
            config_path (str): Path to the configuration YAML file.
            cache_path (str): Directory for caching images used in inference.
            device (str): Computation device (`cuda` or `cpu`).
        """
        super().__init__()
        self.image_cache = ImageCache(cache_path)
        self.image_cache.monitor()
        model_dir = os.path.dirname(model_path)
        self.model_path = model_path
        self.config_name = os.path.basename(config_path)
        self.config_path = config_path 
        self.device = device
        
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=model_dir)
        
        self.predictors = {}
        self.inference_state = None
    
    def run_3d(self, reset_state, image_tensor, guidance, case_name):
        """
        Runs 3D interactive segmentation with bidirectional propagation.

        Args:
            reset_state (bool): Whether to reset the inference state before processing.
            image_tensor (torch.Tensor): 3D image volume tensor of shape `(H, W, D)`.
            guidance (Dict[str, torch.Tensor]): Dictionary containing lesion/background interaction points.
            case_name (str): Unique identifier for the current case.

        Returns:
            np.ndarray: 3D binary segmentation mask of shape `(H, W, D)`.
        """
        predictor = self.predictors.get(self.device)
        
        if predictor is None:
            logger.info(f"Using Device: {self.device}")
            device_t = torch.device(self.device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            predictor = build_sam2_video_predictor(self.config_name, self.model_path, device=self.device)
            self.predictors[self.device] = predictor
        
        # Prepare input image directory
        video_dir = os.path.join(
            self.image_cache.cache_path, case_name
        ) 
        
        logger.info(f"Image: {image_tensor.shape}")
        
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            for slice_idx in tqdm(range(image_tensor.shape[-1])):
                slice_np = image_tensor[:, :, slice_idx].numpy()
                slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")
                Image.fromarray(slice_np).convert("RGB").save(slice_file)
            logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices; {video_dir}")
        
        # Set expiry time for cached images
        self.image_cache.cached_dirs[video_dir] = time() + self.image_cache.cache_expiry_sec
        
        # Initialize inference state if required
        if reset_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
            self.inference_state = predictor.init_state(video_path=video_dir)
        
        # Extract interaction points from guidance
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        
        for key in {"lesion", "background"}:
            point_tensor = np.array(guidance[key].cpu())
            logger.info(f"point tensor: {point_tensor}")
            if point_tensor.size == 0:
                continue # Skip if no interaction points
            else:
                for point_id in range(point_tensor.shape[1]):
                    point = point_tensor[:, point_id, :][0][1:] # Extract (x, y, slice_index)
                    logger.info(f"p: {point}")
                    sid = point[2]
                    
                    sids.add(sid)
                    kps = fps if key == "lesion" else bps
                    if kps.get(sid):
                        kps[sid].append([point[0], point[1]])
                    else:
                        kps[sid] = [[point[0], point[1]]]

        # Forward propagation
        pred = np.zeros(tuple(image_tensor.shape))
        for sid in sorted(sids):
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])
            
            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x
            point_labels = [1] * len(fp) + [0] * len(bp)
            logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}")
            
            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=0,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                box=None,
            )
            pred[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()
        
        pred_forward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_forward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # Backward propagation
        pred_backward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, reverse=True):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_backward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # Merge forward and backward propagation
        pred_prop = np.logical_or(pred_forward, pred_backward) 
        pred = np.logical_or(pred, pred_prop) 
        return pred
    
    def forward(self, x):
        """
        Performs segmentation inference using SAM2 model.

        Args:
            x (Dict[str, Any]): Dictionary containing:
                - `"image"`: 3D image tensor.
                - `"guidance"`: Interaction guidance points.
                - `"case_name"`: Case identifier.
                - `"reset_state"`: Boolean flag to reset the inference state.

        Returns:
            torch.Tensor: Segmentation prediction of shape `(1, 1, H, W, D)`.
        """
        image = torch.squeeze(x["image"].cpu())[0]
        guidance = x["guidance"]
        case_name = x["case_name"]
        reset_state = x["reset_state"]
        output = self.run_3d(reset_state, image, guidance, case_name)
        output = torch.Tensor(output).unsqueeze(0).unsqueeze(0)
        return output.to(self.device)


class MOIS_SAM2Network(nn.Module):
    def __init__(self, 
                 model_path, 
                 config_path, 
                 cache_path, 
                 device, 
                 exemplar_use_com,
                 exemplar_inference_all_slices,
                 exemplar_num,
                 exemplar_use_only_prompted,
                 filter_prev_prediction_components,
                 overlap_threshold=0.5,
                 use_low_res_masks_for_com_detection=True,
                 default_image_size = 1024,
                 min_lesion_area_threshold = 40 # For filtering exemplar-based segmentation
                 ):
        super().__init__()
        self.image_cache = ImageCache(cache_path)
        self.image_cache.monitor()
        model_dir = os.path.dirname(model_path)
        self.model_path = model_path
        self.config_name = os.path.basename(config_path)
        self.config_path = config_path 
        self.device = device
        
        self.exemplar_use_com = exemplar_use_com
        self.exemplar_inference_all_slices = exemplar_inference_all_slices
        self.exemplar_num = exemplar_num
        self.exemplar_use_only_prompted = exemplar_use_only_prompted
        
        self.filter_prev_prediction_components = filter_prev_prediction_components
        self.overlap_threshold = overlap_threshold
        self.use_low_res_masks_for_com_detection = use_low_res_masks_for_com_detection
        self.default_image_size = default_image_size
        
        if self.use_low_res_masks_for_com_detection:
            # Set the scale factor to transform the center of a lesion mass 
            # from low res 256x256 mask to the original 1024x1024.
            self.com_scale_coefficient = 4
        else:
            self.com_scale_coefficient = 1
        
        self.min_lesion_area_threshold = min_lesion_area_threshold * (self.com_scale_coefficient ** 2)
        
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=model_dir)
        
        self.predictors = {}
        self.inference_state = None
        self.prompted_slice_accumulator = set()
    
    def forward(self, x):
        """
        Performs segmentation inference using SAM2 model.

        Args:
        x (Dict[str, Any]): A dictionary containing:
            - "image" (torch.Tensor): 3D image tensor of shape (1, 1, H, W, D).
            - "guidance" (Any): User interaction guidance (e.g., clicks, boxes).
            - "case_name" (str): Identifier of the current case.
            - "reset_state" (bool): If True, resets the internal memory/state before inference.
            - "reset_exemplars" (bool): If True, resets the exemplar memory bank before inference.
            - "evaluation_mode" (str): Inference mode; one of:
                - "global_corrective": Uses memory-based correction with exemplar refinement.
                - "lesion_wise_non_corrective": Lesion-wise segmentation without post-hoc exemplar inference.
                - "lesion_wise_corrective": Lesion-wise segmentation with potential post-hoc exemplar inference.
            - "current_instance_id" (int): ID of the current lesion/object being segmented.
            - "call_exemplar_post_inference" (bool): If True, performs final inference using exemplar-based refinement.
            - "previous_global_prediction" (np.ndarray or torch.Tensor): Prior segmentation result used as input
              for final exemplar inference (only relevant if call_exemplar_post_inference is True).

        Returns:
            torch.Tensor: Segmentation prediction of shape `(1, 1, H, W, D)`.
        """
        evaluation_mode = x["evaluation_mode"]
        
        image = torch.squeeze(x["image"].cpu())[0]
        guidance = x["guidance"]
        case_name = x["case_name"]
        reset_state = x["reset_state"]
        reset_exemplars = x["reset_exemplars"]
        
        current_instance_id = x["current_instance_id"]
        call_exemplar_post_inference = x["call_exemplar_post_inference"]
        previous_global_prediction = x["previous_global_prediction"]
                
        if evaluation_mode == "global_corrective":
            # Perform operations iteratively:
            # - prompt-based + memory-based segmentation + accumulate exemplars
            # - use exemplars to get new positive click prompts
            # - extended prompt-based + memory-based segmentation (no accumulation of exemplars)
            output = self.run_3d_global_correction_mode(reset_state, reset_exemplars, image, guidance, 
                                                        case_name, current_instance_id)
        
        elif (evaluation_mode == "lesion_wise_non_corrective") or (evaluation_mode == "lesion_wise_corrective"):
            if not call_exemplar_post_inference:
                # Normal prompt-based and memory-based segmentation with exemplar bank accumulation
                output = self.run_3d_local_correction_mode(reset_state, reset_exemplars, image, guidance, 
                                                           case_name, current_instance_id)
            else:
                # Apply accumulated exemplar bank to segment objects as the final step after all interactions
                output = self.run_3d_local_exemplar_post_inference(image, case_name, previous_global_prediction)
        else:
            raise ValueError("Evaluation mode is not supported")
        output = torch.Tensor(output).unsqueeze(0).unsqueeze(0)
        return output.to(self.device)
    
    def get_predictor(self):
        predictor = self.predictors.get(self.device)
        
        if predictor is None:
            logger.info(f"Using Device: {self.device}")
            device_t = torch.device(self.device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            predictor = build_mois_sam2_predictor(self.config_name, self.model_path, device=self.device)
            self.predictors[self.device] = predictor
            predictor.num_max_exemplars = self.exemplar_num
            predictor.exemplar_use_only_prompted = self.exemplar_use_only_prompted
            predictor.use_low_res_masks_for_com_detection = self.use_low_res_masks_for_com_detection
            predictor.com_scale_coefficient = self.com_scale_coefficient
        return predictor
    
    def prepare_input_image_directory(self, case_name, image_tensor):
        video_dir = os.path.join(
            self.image_cache.cache_path, case_name
        ) 
        
        logger.info(f"Image: {image_tensor.shape}")
        
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            for slice_idx in tqdm(range(image_tensor.shape[-1])):
                slice_np = image_tensor[:, :, slice_idx].numpy()
                slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")
                Image.fromarray(slice_np).convert("RGB").save(slice_file)
            logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices; {video_dir}")
        
        # Set expiry time for cached images
        self.image_cache.cached_dirs[video_dir] = time() + self.image_cache.cache_expiry_sec
        return video_dir
    
    def extract_interaction_points_from_guidance(self, guidance):
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        
        for key in {"lesion", "background"}:
            point_tensor = np.array(guidance[key].cpu())
            logger.info(f"point tensor: {point_tensor}")
            if point_tensor.size == 0:
                continue # Skip if no interaction points
            else:
                for point_id in range(point_tensor.shape[1]):
                    point = point_tensor[:, point_id, :][0][1:] # Extract (x, y, slice_index)
                    logger.info(f"p: {point}")
                    sid = point[2]
                    
                    sids.add(sid)
                    kps = fps if key == "lesion" else bps
                    if kps.get(sid):
                        kps[sid].append([point[0], point[1]])
                    else:
                        kps[sid] = [[point[0], point[1]]]
        return fps, bps, sids
    
    
    def remove_overlapping_lesions(self, current_pred, previous_pred):
        # Label connected components in both masks
        prev_labels, num_prev = label(previous_pred)
        curr_labels, num_curr = label(current_pred)
        
        # Prepare a copy of current prediction for modification
        refined_pred = current_pred.copy()

        for prev_idx in range(1, num_prev + 1):
            prev_mask = (prev_labels == prev_idx)

            # Find overlapping current labels
            overlapping_curr_labels = np.unique(curr_labels[prev_mask])
            overlapping_curr_labels = overlapping_curr_labels[overlapping_curr_labels != 0]  # exclude background

            for curr_idx in overlapping_curr_labels:
                curr_mask = (curr_labels == curr_idx)
                overlap_area = np.logical_and(prev_mask, curr_mask).sum()
                curr_area = curr_mask.sum()
                
                if curr_area > 0 and (overlap_area / curr_area) >= self.overlap_threshold:
                    refined_pred[curr_mask] = 0  # Remove lesion from current prediction
        return refined_pred
    
    
    def overlapping_coms(self, current_com, previous_pred):
        full_size_com = current_com * self.com_scale_coefficient
        full_size_com = full_size_com.astype(np.int32)
        com_x, com_y = full_size_com[0][0], full_size_com[0][1]
                
        if previous_pred[com_y, com_x].item() > 0:
            return True
        return False
    
    def run_3d_local_correction_mode(self,
                                     reset_state, 
                                     reset_exemplars,
                                     image_tensor, 
                                     guidance, 
                                     case_name,
                                     current_instance_id):
        """
        In the local correction mode the interactions are performed on a lesion level.
        With each interaction an exemplar for a given lesion is being updated.
        With this method we aggregate exemplars.
        """
        predictor = self.get_predictor()
        video_dir = self.prepare_input_image_directory(case_name, image_tensor)
        
        # Initialize inference state if required
        if reset_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state, reset_exemplars=reset_exemplars)
                
            self.inference_state = predictor.init_state(video_path=video_dir, reset_exemplars=reset_exemplars)
            if reset_exemplars:
                self.prompted_slice_accumulator = set()
        
        fps, bps, sids = self.extract_interaction_points_from_guidance(guidance)
        
        # Add interaction points, perform instance prompt-based segmentation in a given slice 
        # and add prompted exemplars (if it is not empty)
        pred = np.zeros(tuple(image_tensor.shape))
        
        for sid in sorted(sids):
            self.prompted_slice_accumulator.add(sid)
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])
            
            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x
            point_labels = [1] * len(fp) + [0] * len(bp)
            logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}")
            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=current_instance_id,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                box=None,
                is_com=False, # In this case the interaction points are not obtained via exemplars
                add_exemplar=True 
            )
            pred[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()
            
        # Forward-slice memory-based propagation + add non-prompted exemplars
        pred_forward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, current_instance_id, add_exemplar=True):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_forward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        #  Backward-slice memory-based propagation + add non-prompted exemplars
        pred_backward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, current_instance_id, reverse=True, add_exemplar=True):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_backward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
                
        # Merge forward and backward propagation
        pred_prop = np.logical_or(pred_forward, pred_backward) 
        pred = np.logical_or(pred, pred_prop) 
        return pred
    
    def run_3d_local_exemplar_post_inference(self, image_tensor, case_name, previous_prediction):
        """
        Applies the exemplar bank to extrapolate segmentation to other lesions after
        all interactions on a lesion have been completed.

        Supports three inference strategies:
        - COM detection on all slices
        - COM detection only on prompted slices followed by temporal propagation
        - Semantic mask-based inference when COM is not used
        """
        # Initialization
        predictor = self.get_predictor()
        video_dir = self.prepare_input_image_directory(case_name, image_tensor)
        prediction = np.zeros(tuple(image_tensor.shape))
        
        previous_prediction = previous_prediction.squeeze().cpu().numpy()
            
            
        def apply_com_masking(filtered_objects_dict, slice_idx, obj_counter):
            """Apply COM points to predictor for a given slice and return mask."""
            slice_pred = np.zeros_like(prediction[:, :, slice_idx])
            prev_pred_slice = previous_prediction[:, :, slice_idx]
            for obj_idx, obj_data in filtered_objects_dict.items():
                com_point = obj_data["com_point"]
                com_label = np.array([1], np.int32)

                if self.filter_prev_prediction_components and self.overlapping_coms(com_point, prev_pred_slice):
                    continue

                obj_counter += 1
                logger.info(f"{slice_idx} - {obj_counter} - Coords: {com_point}; Labels: {com_label}")
                _, _, mask_logits = predictor.add_new_points_or_box(
                    self.inference_state, 
                    slice_idx, 
                    obj_counter,
                    points=com_point, 
                    labels=com_label,
                    clear_old_points=True, # Need to remove all previous points
                    is_com=True, 
                    normalize_coords=False, # Do not need to normalize, since COM is either in 1024x1024 or 256x256
                    add_exemplar=False,
                    box=None,
                )
                mask = (mask_logits[-1][0] > 0.0).cpu().numpy()
                slice_pred = np.maximum(slice_pred, mask)
            return slice_pred, obj_counter
        
        if self.exemplar_use_com: 
            # Use the center of mass definition
            predictor.reset_state(self.inference_state, reset_exemplars=False)
            self.inference_state = predictor.init_state(video_path=video_dir, reset_exemplars=False)
            object_counter = 0 # Initialize the counter that will be used to provide unique object IDs

            if self.exemplar_inference_all_slices: 
                # Definition of CoMs on all slices independently
                for slice_idx in range(image_tensor.shape[-1]):
                    _, _, _, filtered_objs = predictor.find_exemplars_in_slice(
                        self.inference_state, 
                        slice_idx,
                        binarization_threshold=0.5,
                        min_area_threshold=self.min_lesion_area_threshold
                    )
                    pred_slice, object_counter = apply_com_masking(filtered_objs, slice_idx, object_counter)
                    prediction[:, :, slice_idx] = np.maximum(prediction[:, :, slice_idx], pred_slice)
                            
            else: 
                # Definition of CoMs only on the prompted slices + propagation                                
                # Find objects in the prompted slices
                for slice_idx in sorted(self.prompted_slice_accumulator):
                    _, _, _, filtered_objs = predictor.find_exemplars_in_slice(
                        self.inference_state, 
                        slice_idx,
                        binarization_threshold=0.5,
                        min_area_threshold=self.min_lesion_area_threshold
                    )
                    pred_slice, object_counter = apply_com_masking(filtered_objs, slice_idx, object_counter)
                    prediction[:, :, slice_idx] = np.maximum(prediction[:, :, slice_idx], pred_slice)
                                                        
                # Propagate in both directions
                def propagate(reverse=False):
                    pred_temp = np.zeros(image_tensor.shape)
                    for idx, _, logits in predictor.propagate_in_video(
                        self.inference_state, 
                        reverse=reverse, 
                        add_exemplar=False
                    ):
                        pred_temp[:, :, idx] = (logits > 0.0).any(dim=0).squeeze(0).cpu().numpy()
                    return pred_temp

                # Merge forward and backward propagation
                pred_forward = propagate(reverse=False)
                pred_backward = propagate(reverse=True)
                pred_propagated = np.logical_or(pred_forward, pred_backward)
                prediction = np.maximum(prediction, pred_propagated)

        else: 
            # Use the binary prediction directly and independently on each slice
            if not self.exemplar_inference_all_slices:
                raise ValueError("Use case without center of mass detection assumes exemplar inference on all slices!")

            for slice_idx in range(image_tensor.shape[-1]):
                _, _, semantic_logits, _ = predictor.find_exemplars_in_slice(self.inference_state, slice_idx)
                pred_slice = (semantic_logits[0][0] > 0.0).cpu().numpy()

                if self.filter_prev_prediction_components:
                    prev_pred_slice = previous_prediction[:, :, slice_idx]
                    pred_slice = self.remove_overlapping_lesions(current_pred=pred_slice, previous_pred=prev_pred_slice)

                prediction[:, :, slice_idx] = pred_slice
            
        return prediction

    
    def run_3d_global_correction_mode(self,
                                      reset_state, 
                                      reset_exemplars,
                                      image_tensor, 
                                      guidance, 
                                      case_name,
                                      current_instance_id):
        """
        Global corrective segmentation mode.

        This mode supports three segmentation strategies:
        - Prompt-based segmentation with memory-based propagation (forward + backward).
        - Optional global inference via exemplar-based COM (center of mass) detection.
        - Optional semantic segmentation using exemplar masks directly (without COM).

        Args:
            reset_state (bool): Whether to reset the inference state.
            reset_exemplars (bool): Whether to reset the exemplar memory bank.
            image_tensor (torch.Tensor): The 3D image of shape (H, W, D).
            guidance (Dict[str, torch.Tensor]): Click-based or box-based prompts for segmentation.
            case_name (str): Identifier of the current case.
            current_instance_id (int): ID of the current object being segmented.

        Returns:
            np.ndarray: 3D segmentation mask of shape (H, W, D).
        """
        predictor = self.get_predictor()
        video_dir = self.prepare_input_image_directory(case_name, image_tensor)
        
        if self.inference_state:
            predictor.reset_state(self.inference_state, reset_exemplars=reset_exemplars)
                
        self.inference_state = predictor.init_state(video_path=video_dir, reset_exemplars=reset_exemplars)
        if reset_exemplars:
            self.prompted_slice_accumulator = set()
        
        fps, bps, sids = self.extract_interaction_points_from_guidance(guidance)
        prediction = np.zeros(tuple(image_tensor.shape))
        
        # === Prompted slice segmentation ===        
        for sid in sorted(sids):
            self.prompted_slice_accumulator.add(sid)
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])
            point_coords = [[p[1], p[0]] for p in (fp + bp)]
            point_labels = [1] * len(fp) + [0] * len(bp)
            
            logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}")
            _, _, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=current_instance_id,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                is_com=False,
                add_exemplar=True,
                box=None,
            )
            prediction[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # === Forward and backward memory propagation ===
        def propagate_and_merge(reverse=False):
            pred = np.zeros(image_tensor.shape)
            for idx, _, logits in predictor.propagate_in_video(self.inference_state,
                                                            obj_id=current_instance_id,
                                                            reverse=reverse,
                                                            add_exemplar=True):
                pred[:, :, idx] = (logits[0][0] > 0.0).cpu().numpy()
            return pred
        
        prediction = np.logical_or(prediction,
                                   propagate_and_merge(reverse=False))
        prediction = np.logical_or(prediction,
                                   propagate_and_merge(reverse=True))
        
        # === Global inference with exemplars ===
        if self.exemplar_use_com: 
            # Use the center of mass definition
            predictor.reset_state(self.inference_state, reset_exemplars=False)
            self.inference_state = predictor.init_state(video_path=video_dir, reset_exemplars=False)
            object_counter = 0
            
            def collect_com_inference(slice_indices):
                nonlocal object_counter
                for slice_idx in slice_indices:
                    _, _, _, objects = predictor.find_exemplars_in_slice(
                        self.inference_state, slice_idx,
                        binarization_threshold=0.5,
                        min_area_threshold=self.min_lesion_area_threshold)

                    pred_slice = np.zeros(image_tensor.shape[:2])
                    bp = bps.get(slice_idx, [])
                    if bp:
                        scale_x = self.default_image_size / self.inference_state["video_width"]
                        scale_y = self.default_image_size / self.inference_state["video_height"]
                        bp = [[int(p[1] * scale_x / self.com_scale_coefficient),
                            int(p[0] * scale_y / self.com_scale_coefficient)] for p in bp]

                    for _, obj_data in objects.items():
                        com_point = obj_data["com_point"][0].astype(int) * self.com_scale_coefficient
                        fp = [[com_point[0], com_point[1]]]
                        points = fp + bp
                        labels = [1] * len(fp) + [0] * len(bp)

                        if self.filter_prev_prediction_components:
                            prev_slice = prediction[:, :, slice_idx]
                            if self.overlapping_coms(current_com=com_point, previous_pred=prev_slice):
                                continue

                        object_counter += 1
                        logger.info(f"{slice_idx} - {object_counter} - Coords: {points}; Labels: {labels}")
                        _, _, logits = predictor.add_new_points_or_box(
                            self.inference_state, slice_idx, object_counter,
                            points=points, labels=labels,
                            is_com=True, clear_old_points=True,
                            normalize_coords=False, box=None, add_exemplar=False)
                        current_mask = (logits[-1][0] > 0.0).cpu().numpy()
                        pred_slice = np.maximum(pred_slice, current_mask)

                    prediction[:, :, slice_idx] = np.maximum(prediction[:, :, slice_idx], pred_slice)

            
            if self.exemplar_inference_all_slices: 
                # Definition of CoMs on all slices independently
                collect_com_inference(range(image_tensor.shape[-1]))

            else: 
                # Definition of CoMs only on the prompted slices + propagation
                collect_com_inference(sorted(self.prompted_slice_accumulator))
                
                # Final propagation
                prediction = np.maximum(
                    prediction,
                    np.logical_or(
                        propagate_and_merge(reverse=False),
                        propagate_and_merge(reverse=True)
                    )
                )

        else: 
            # === Direct semantic inference from masks ===
            if not self.exemplar_inference_all_slices:
                raise ValueError("Use case without center of mass detection assumes exemplar inference on all slices!")

            for slice_idx in range(image_tensor.shape[-1]):
                _, _, logits, _ = predictor.find_exemplars_in_slice(self.inference_state, slice_idx)
                pred_slice = (logits[0][0] > 0.0).cpu().numpy()

                if self.filter_prev_prediction_components:
                    prev_slice = prediction[:, :, slice_idx]
                    pred_slice = self.remove_overlapping_lesions(current_pred=pred_slice, previous_pred=prev_slice)

                prediction[:, :, slice_idx] = np.maximum(prediction[:, :, slice_idx], pred_slice)

        return prediction


class VISTANetwork(nn.Module):
    def __init__(self, 
                 model_path, 
                 device,
                 device_type = "cuda",
                 automatic_inference = False,
                 model_registry = "vista3d_segresnet_d",
                 input_channels = 1,
                 patch_size = [256, 256, 64],
                 overlap = 0.5,
                 sw_batch_size = 1,
                 label_set = [0, 1],
                 mapped_label_set = [0, 133],
                 amp = True
                ):
        super().__init__()  
        self.device = device
        self.device_type = device_type
        self.automatic_inference = automatic_inference  
        self.patch_size = patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.label_set = label_set
        self.mapped_label_set = mapped_label_set
        self.amp = amp
                    
        model = vista_model_registry[model_registry](
            in_channels=input_channels, 
            image_size=patch_size
            )
        model = model.to(device)        
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.model_inferer = partial(infer_wrapper, model=model)

    @staticmethod
    def guidance_to_point_and_label_vista_fast(guidance):
        """
        Fast vectorized version to convert guidance dict to VISTA-style point and point_label.
        
        Args:
            guidance (dict): 'background' and 'lesion' tensors of shape [B, *, 4].
        
        Returns:
            point (torch.Tensor): [B, N, 3]
            point_label (torch.Tensor): [B, N]
        """
        device = guidance['lesion'].device

        # Extract (x,y,z) coordinates, ignoring the time/frame (index 0)
        bg_points_raw = guidance['background'][..., 1:] if guidance['background'].shape[1] > 0 else None
        lesion_points_raw = guidance['lesion'][..., 1:] if guidance['lesion'].shape[1] > 0 else None

        # Get number of background and lesion points
        n_bg = guidance['background'].shape[1] if guidance['background'].shape[1] > 0 else 0
        n_lesion = guidance['lesion'].shape[1] if guidance['lesion'].shape[1] > 0 else 0
        max_points = n_bg + n_lesion
        
        # Prepare lesion
        if lesion_points_raw is not None:
            
            if bg_points_raw is not None:
                lesion_points_raw = torch.cat([lesion_points_raw, bg_points_raw], dim=1)
                lesion_labels = torch.cat([
                    torch.ones((1, n_lesion), device=device, dtype=torch.int32),
                    torch.zeros((1, n_bg), device=device, dtype=torch.int32)
                ], dim=1)
            else:
                lesion_labels = torch.ones((1, n_lesion), device=device, dtype=torch.int32)
        else:
            lesion_points_raw = torch.zeros((1, max_points, 3), device=device, dtype=torch.int32)
            lesion_labels = torch.full((1, max_points), -1, device=device, dtype=torch.int32)
        
        # Prepare background
        bg_points_raw = torch.zeros((1, max_points, 3), device=device, dtype=torch.int32)
        bg_labels = torch.full((1, max_points), -1, device=device, dtype=torch.int32)
            
        # Concatenate points and labels
        points = torch.cat([bg_points_raw, lesion_points_raw], dim=0)  # [B, total_points, 3]
        labels = torch.cat([bg_labels, lesion_labels], dim=0)          # [B, total_points]

        return points, labels


    def forward(self, x):
        image = torch.unsqueeze(x["image"][:, 0, :], dim=1)
        guidance = x["guidance"]
        
        with torch.no_grad():
            with autocast(device_type=self.device_type, enabled=self.amp):
                torch.cuda.empty_cache()
                if self.automatic_inference:
                    # Inference without interaction points                       
                    outputs = sliding_window_inference(
                        inputs=image.to(self.device), # Should be: torch.Size([1, 1, 801, 2853, 64])
                        roi_size=self.patch_size,
                        sw_batch_size=self.sw_batch_size,
                        predictor=self.model_inferer,
                        mode="gaussian",
                        overlap=self.overlap,
                        sw_device=self.device,
                        device=self.device,
                        point_coords=None,
                        point_labels=None,
                        class_vector=torch.tensor(self.mapped_label_set) # Should be tensor([  0, 133])
                        .to(self.device),
                        prompt_class=None,
                        labels=None,
                        label_set=None,
                        use_cfp=True,
                        brush_radius=None,
                        val_point_sampler=None,
                    )
                                                                    
                else:
                    # Inference with interaction points
                    # Convert the interactive points to 
                    point, point_label = self.guidance_to_point_and_label_vista_fast(guidance)
                    prompt_class = torch.tensor([[label] for label in self.label_set], device=self.device)
                    outputs = sliding_window_inference(
                        inputs=image,
                        roi_size=self.patch_size,
                        sw_batch_size=1,
                        predictor=self.model_inferer,
                        mode="gaussian",
                        overlap=self.overlap,
                        sw_device=self.device,
                        device=self.device,
                        point_coords=point, 
                        point_labels=point_label,
                        class_vector = None,
                        prompt_class = prompt_class,
                        labels=None,
                        label_set=self.label_set,
                        use_cfp=True,
                        brush_radius=None,
                        prev_mask=None,
                        val_point_sampler=None,
                    )        
        return outputs           
       