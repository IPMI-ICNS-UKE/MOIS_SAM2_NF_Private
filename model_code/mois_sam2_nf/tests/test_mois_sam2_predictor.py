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

import os
import sys
import pytest
import torch
import numpy as np

# Go one folder up to include 'sam2' as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.exemplar_attention import ExemplarAttention
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.mois_sam2_predictor import MOISSAM2Predictor


@pytest.fixture
def model():
    """Fixture to create an instance of MOISSAM2Base."""
    
    # Instantiate model components
    trunk = Hiera(embed_dim=112, num_heads=2)
    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, 
                                              scale=None, temperature=10000)
    neck = FpnNeck(position_encoding=position_encoding, d_model=256, 
                   backbone_channel_list=[896, 448, 224, 112], 
                   fpn_top_down_levels=[2, 3], fpn_interp_model='nearest')
    image_encoder = ImageEncoder(scalp=1, trunk=trunk, neck=neck)

    self_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], 
                                   embedding_dim=256, num_heads=1, 
                                   downsample_rate=1, dropout=0.1)
    cross_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], 
                                    rope_k_repeat=True, embedding_dim=256, 
                                    num_heads=1, downsample_rate=1, 
                                    dropout=0.1, kv_in_dim=64)
    memory_attention_layer = MemoryAttentionLayer(activation='relu', 
                                                  dim_feedforward=2048, 
                                                  dropout=0.1, 
                                                  pos_enc_at_attn=False, 
                                                  self_attention=self_attention, 
                                                  d_model=256, 
                                                  pos_enc_at_cross_attn_keys=True, 
                                                  pos_enc_at_cross_attn_queries=False, 
                                                  cross_attention=cross_attention)
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, 
                                       layer=memory_attention_layer, num_layers=4)
    exemplar_attention = ExemplarAttention(d_model=256, pos_enc_at_input=True, 
                                           layer=memory_attention_layer, num_layers=4)

    memory_encoder_position_encoding = PositionEmbeddingSine(num_pos_feats=64, 
                                                             normalize=True, 
                                                             scale=None, 
                                                             temperature=10000)
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser_layer = CXBlock(dim=256, kernel_size=7, padding=3, 
                          layer_scale_init_value=1e-6, use_dwconv=True)
    fuser = Fuser(layer=fuser_layer, num_layers=2)
    memory_encoder = MemoryEncoder(out_dim=64, 
                                   position_encoding=memory_encoder_position_encoding, 
                                   mask_downsampler=mask_downsampler, 
                                   fuser=fuser)
    
    # Instantiate the MOISSAM2Predictor model
    model = MOISSAM2Predictor(
        exemplar_attention=exemplar_attention,
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False
    )
        
    return model


def run_add_new_points_or_box_test(
    model, video_path, interactions, expected_exemplars, device="cuda"
):
    """
    Helper function to test add_new_points_or_box() method.

    Args:
        model: The segmentation model.
        video_path (str): Path to the video.
        interactions (list of dict): Each dictionary contains:
            - "frame_idx": Frame index where interaction occurs.
            - "obj_id": Unique object ID.
            - "points": np.array of shape (N, 2) for click coordinates.
            - "labels": np.array of shape (N,) for click labels.
        expected_exemplars (list of str): Expected exemplar keys in `model.exemplars_dict`.
        device (str, optional): Device to run the test (default: "cuda").
    """
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU-enabled machine."
    device = torch.device(device)
    model.to(device)

    # Step 1: Initialize inference state
    inference_state = model.init_state(video_path, reset_exemplars=True)

    # Step 2: Iterate through interactions and add points
    for interaction in interactions:
        ann_frame_idx = interaction["frame_idx"]
        ann_obj_id = interaction["obj_id"]
        points = interaction["points"]
        labels = interaction["labels"]

        model.add_new_points_or_box(
            inference_state,
            ann_frame_idx,
            ann_obj_id,
            points=points,
            labels=labels,
            is_com=False,
            clear_old_points=True,
            normalize_coords=True,
            box=None,
            add_exemplar=True,
        )

    # Step 3: Assertions
    assert len(model.exemplars_dict) == len(expected_exemplars), \
        f"Expected {len(expected_exemplars)} exemplars, but found {len(model.exemplars_dict)}"

    for exemplar_name in expected_exemplars:
        assert exemplar_name in model.exemplars_dict, f"Missing exemplar {exemplar_name} in model.exemplars_dict"


# Test initialization
def test_init(model):
    """Test if the model initializes correctly."""
    assert hasattr(model, 'hidden_dim')
    assert hasattr(model, 'mem_dim')
    assert hasattr(model, 'num_max_exemplars')
    
    
def test_init_state(model):
    """Test the init_state method."""
    video_path = "../notebooks/videos/bedroom"
    inference_state = model.init_state(video_path, reset_exemplars=True)
    
    assert isinstance(inference_state, dict), "init_state should return a dictionary."
    assert "exemplars" in inference_state, "init_state dictionary should contain 'exemplars' key."
    assert inference_state["exemplars"] == {}, "Exemplars should be reset to an empty dictionary when reset_exemplars is True."
    
    
# Test Case 1: One Object, One Point
def test_add_new_points_or_box_one_object_one_point(model, device="cuda"):
    video_path = "../notebooks/videos/bedroom"
    interactions = [
        {
            "frame_idx": 0,
            "obj_id": 0,
            "points": np.array([[210, 350]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        }
    ]
    expected_exemplars = ["0_0_prompted"]
    run_add_new_points_or_box_test(model, video_path, interactions, expected_exemplars, device)


# Test Case 2: One Object, Two Points
def test_add_new_points_or_box_one_object_two_points(model, device="cuda"):
    video_path = "../notebooks/videos/bedroom"
    interactions = [
        {
            "frame_idx": 0,
            "obj_id": 0,
            "points": np.array([[210, 350], [250, 220]], dtype=np.float32),
            "labels": np.array([1, 0], dtype=np.int32),
        }
    ]
    expected_exemplars = ["0_0_prompted"]
    run_add_new_points_or_box_test(model, video_path, interactions, expected_exemplars, device)


# Test Case 3: One Object, Two Points, Extra Frame
def test_add_new_points_or_box_one_object_two_points_extra_frame(model, device="cuda"):
    video_path = "../notebooks/videos/bedroom"
    interactions = [
        {
            "frame_idx": 0,
            "obj_id": 0,
            "points": np.array([[210, 350], [250, 220]], dtype=np.float32),
            "labels": np.array([1, 0], dtype=np.int32),
        },
        {
            "frame_idx": 10,
            "obj_id": 0,
            "points": np.array([[110, 110]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
    ]
    expected_exemplars = ["0_0_prompted", "0_10_prompted"]
    run_add_new_points_or_box_test(model, video_path, interactions, expected_exemplars, device)


# Test Case 4: Two Objects, Two Points
def test_add_new_points_or_box_two_object_two_points(model, device="cuda"):
    video_path = "../notebooks/videos/bedroom"
    interactions = [
        {
            "frame_idx": 0,
            "obj_id": 0,
            "points": np.array([[210, 350], [250, 220]], dtype=np.float32),
            "labels": np.array([1, 0], dtype=np.int32),
        },
        {
            "frame_idx": 0,
            "obj_id": 1,
            "points": np.array([[220, 360], [240, 210]], dtype=np.float32),
            "labels": np.array([1, 0], dtype=np.int32),
        },
    ]
    expected_exemplars = ["0_0_prompted", "1_0_prompted"]
    run_add_new_points_or_box_test(model, video_path, interactions, expected_exemplars, device)
    
    
def test_propagate_in_video(model, device="cuda"):
    """Test the propagate_in_video method."""
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU-enabled machine."
    device = torch.device(device)  # Set device (GPU if available)
    model.to(device)
    
    video_path = "../notebooks/videos/bedroom"
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    
    # Step 1: Add an interaction point
    points = torch.tensor([[210, 350]], dtype=torch.float32, device=device)  # Positive Click
    labels = torch.tensor([1], dtype=torch.int32, device=device)  # 1 = Positive Click

    # Step 2: Initialize inference state (on GPU)
    inference_state = model.init_state(video_path, reset_exemplars=True)

    model.add_new_points_or_box(
        inference_state,
        ann_frame_idx,
        ann_obj_id,
        points=points,
        labels=labels,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    )

    # Step 3: Run propagation
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(inference_state):
        # Move mask logits to GPU before thresholding
        out_mask_logits = out_mask_logits.to(device)

        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()  # Convert to binary mask (CPU for validation)
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Step 4: Assertions for Propagation Validity
    assert video_segments, "No frames were processed in propagation."
    
    # Check that multiple frames were processed
    assert len(video_segments) > 1, "Propagation did not extend beyond the first frame."


def test_find_exemplars_in_slice_two_objects_two_points(model, device="cuda"):
    """Test the find_exemplars_in_slice method with two objects and two points."""
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU-enabled machine."
    device = torch.device(device)
    model.to(device)

    video_path = "../notebooks/videos/bedroom"
    inference_state = model.init_state(video_path, reset_exemplars=True)

    # Interaction 1
    ann_frame_idx = 0  # Frame index we interact with
    ann_obj_id_1 = 0  # First object ID
    points_1 = np.array([[210, 350], [200, 300]], dtype=np.float32)  # Click points
    labels_1 = np.array([1, 0], np.int32)  # Positive & Negative Clicks

    model.add_new_points_or_box(
        inference_state,
        ann_frame_idx,
        ann_obj_id_1,
        points=points_1,
        labels=labels_1,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    )

    # Interaction 2
    ann_obj_id_2 = 2  # Second object ID
    points_2 = np.array([[110, 250], [100, 200]], dtype=np.float32)  # Click points
    labels_2 = np.array([1, 0], np.int32)  # Positive & Negative Clicks

    model.add_new_points_or_box(
        inference_state,
        ann_frame_idx,
        ann_obj_id_2,
        points=points_2,
        labels=labels_2,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    )
    
    # Find new objects
    frame_idx_return, _, semantic_mask, object_dict = model.find_exemplars_in_slice(
        inference_state,
        ann_frame_idx,
        min_area_threshold=3 # Since model is not trained, many false triggers may occur
        )
    
    assert frame_idx_return == ann_frame_idx, f"Frame index mismatch: expected {ann_frame_idx}, got {frame_idx_return}"
    assert isinstance(semantic_mask, torch.Tensor), "Semantic mask is not a PyTorch tensor."
    assert semantic_mask.shape == torch.Size([1, 1, 540, 960]), f"Unexpected mask shape: {semantic_mask.shape}"
    

def test_propagate_exemplars_in_video(model, device="cuda"):
    """Test the propagate_in_video method."""
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU-enabled machine."
    device = torch.device(device)  # Set device (GPU if available)
    model.to(device)
    
    video_path = "../notebooks/videos/bedroom"
    inference_state = model.init_state(video_path, reset_exemplars=True)
    
    # Propagation parameters
    min_area_threshold = 2
    binarization_threshold = 0.1
    max_frame_num_to_track = 5
    bidirectional = False

    # Interaction 1 (Object 0)
    ann_frame_idx = 0  # Frame index for interaction
    ann_obj_id_1 = 0  # First object ID
    points_1 = np.array([[210, 350], [200, 300]], dtype=np.float32)  # Click points
    labels_1 = np.array([1, 0], np.int32)  # Positive & Negative Clicks

    model.add_new_points_or_box(
        inference_state,
        ann_frame_idx,
        ann_obj_id_1,
        points=points_1,
        labels=labels_1,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    )

    # Interaction 2 (Object 1)
    ann_obj_id_2 = 1  # Second object ID
    points_2 = np.array([[110, 250], [100, 200]], dtype=np.float32)  # Click points
    labels_2 = np.array([1, 0], np.int32)  # Positive & Negative Clicks

    model.add_new_points_or_box(
        inference_state,
        ann_frame_idx,
        ann_obj_id_2,
        points=points_2,
        labels=labels_2,
        is_com=False,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        add_exemplar=True
    )

    # Run propagation
    video_segments = model.propagate_exemplars_in_video(
        inference_state,
        ann_frame_idx,
        binarization_threshold,
        min_area_threshold,
        max_frame_num_to_track,
        bidirectional
    )
    
    # Check whether video was segmented
    assert video_segments, "No frames were processed in propagation." 
    # Check that the current frame and max_frame_num_to_track are segmented
    assert len(video_segments) == max_frame_num_to_track + 1, "Expected {max_frame_num_to_track + 1} frames, but got {len(video_segments)}."
    # Make sure that no exemplars are added except for prompted
    assert len(model.exemplars_dict) == 2, "Exemplar dictionary is affected, although it shouldn't"
    