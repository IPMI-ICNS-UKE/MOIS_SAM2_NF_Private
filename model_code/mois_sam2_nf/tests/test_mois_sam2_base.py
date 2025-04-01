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
import random

# Go one folder up to include 'sam2' as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.modeling.mois_sam2_base import MOISSAM2Base
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
    
    # Instantiate the MOISSAM2Base model
    model = MOISSAM2Base(
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


def _generate_mock_output(num_entries, feature_shape, mask_shape, 
                          obj_ptr_shape, score_shape, 
                          is_prompted=None, use_frame_idx=False):
    """Helper function to generate mock memory-conditioned outputs for testing."""
    return {
        i: {
            "maskmem_features": torch.randn(*feature_shape),
            "maskmem_pos_enc": [torch.randn(*feature_shape)],
            "pred_masks": torch.randn(*mask_shape),
            "obj_ptr": torch.randn(*obj_ptr_shape),
            "object_score_logits": torch.randn(*score_shape),
            "is_prompted": is_prompted,
            "frame_idx": random.randint(0, 5) if use_frame_idx else None
        }
        for i in range(num_entries)
    }


# Test initialization
def test_init(model):
    """Test if the model initializes correctly."""
    assert hasattr(model, 'hidden_dim')
    assert hasattr(model, 'mem_dim')
    assert hasattr(model, 'num_max_exemplars')


# Test _build_sam_heads
def test_build_sam_heads(model):
    """Test if SAM heads are built correctly."""
    model._build_sam_heads()
    assert hasattr(model, 'exemplar_obj_ptr_proj')
    assert hasattr(model, 'exemplar_obj_ptr_tpos_proj')


# # Test _prepare_memory_conditioned_features
def test_prepare_memory_conditioned_features_non_conditional_frame(model):
    """Test feature fusion with memories."""
    # Mock memory-conditioned outputs
    output_dict = {
        "cond_frame_outputs": _generate_mock_output(
            num_entries=3, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1)
        ),
        "non_cond_frame_outputs": _generate_mock_output(
            num_entries=3, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1)
        ),
    }

    # Define test frame index and vision features
    frame_idx = 7
    current_vision_feats = [torch.randn(4096, 1, 256)]
    current_vision_pos_embeds = [torch.randn(4096, 1, 256)]
    feat_sizes = [(64, 64)]

    # Call the function under test
    output = model._prepare_memory_conditioned_features(
        frame_idx, 
        False,  # Non-conditional frame
        current_vision_feats, 
        current_vision_pos_embeds, 
        feat_sizes,
        output_dict,
        1  # Number of frames
    )
    # Assertions: Ensure output is valid
    assert isinstance(output, torch.Tensor), "Output should be a PyTorch tensor"


# Test _prepare_exemplars_conditioned_features
def test_prepare_exemplars_conditioned_features(model):
    """Test feature fusion with exemplars."""
    # Mock memory-conditioned outputs
    prompted_exemplars = _generate_mock_output(
            num_entries=3, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1), is_prompted=True, use_frame_idx=True
        )
    non_prompted_exemplars = _generate_mock_output(
            num_entries=6, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1), is_prompted=False, use_frame_idx=True
        )
    exemplars_dict = prompted_exemplars | non_prompted_exemplars

    # Define test frame index and vision features
    frame_idx = 0
    current_vision_feats = [torch.randn(4096, 1, 256)]
    current_vision_pos_embeds = [torch.randn(4096, 1, 256)]
    feat_sizes = [(64, 64)]

    output = model._prepare_exemplars_conditioned_features(
        frame_idx, 
        current_vision_feats, 
        current_vision_pos_embeds, 
        feat_sizes, 
        exemplars_dict)
    
    # Assertions: Ensure output is valid
    assert isinstance(output, torch.Tensor), "Output should be a PyTorch tensor"


# Test _form_exemplar_in_current_out
def test_form_exemplar_in_current_out(model):
    """Test exemplar extraction."""
    frame_idx = 0
    point_inputs = None
    current_out = {
        "obj_ptr": torch.randn(1, 256),
        "maskmem_features": torch.randn(1, 64, 64, 64),
        "maskmem_pos_enc": [torch.randn(1, 64, 64, 64),]
    }
    result = model._create_exemplar_placeholder_in_current_out(frame_idx, point_inputs, current_out)
    assert "current_exemplar" in result
    assert result["current_exemplar"]["frame_idx"] == frame_idx


# Test track_semantic_exemplars
def test_track_semantic_exemplars(model):
    """Test semantic exemplar tracking."""
    prompted_exemplars = _generate_mock_output(
            num_entries=3, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1), is_prompted=True, use_frame_idx=True
        )
    non_prompted_exemplars = _generate_mock_output(
            num_entries=6, feature_shape=(1, 64, 64, 64),
            mask_shape=(1, 1, 256, 256), obj_ptr_shape=(1, 256), 
            score_shape=(1, 1), is_prompted=False, use_frame_idx=True
        )
    exemplars_dict = prompted_exemplars | non_prompted_exemplars
    
     # Define test frame index and vision features
    frame_idx = 0
    current_vision_feats = [torch.randn(65536, 1, 32),
                            torch.randn(16384, 1, 64),
                            torch.randn(4096, 1, 256)]
    current_vision_pos_embeds = [torch.randn(65536, 1, 32),
                                 torch.randn(16384, 1, 64),
                                 torch.randn(4096, 1, 256)]
    feat_sizes = [(256, 256),
                  (128, 128),
                  (64, 64),]
    
    output = model.track_semantic_exemplars(
        exemplars_dict, 
        frame_idx, 
        current_vision_feats, 
        current_vision_pos_embeds, 
        feat_sizes)
    assert isinstance(output, dict)
    assert "pred_masks" in output
