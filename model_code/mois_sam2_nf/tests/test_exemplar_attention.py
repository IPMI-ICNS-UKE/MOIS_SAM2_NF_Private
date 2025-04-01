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
import torch
import pytest

# Go one folder up to include 'sam2' as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.modeling.exemplar_attention import ExemplarAttention
from sam2.modeling.memory_attention import MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_clones


@pytest.fixture
def exemplar_attention():
    """Fixture to initialize ExemplarAttention with sample parameters."""
    self_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1
    )
    
    cross_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64
    )
    
    # Define a simple layer with self-attention and cross-attention
    layer = MemoryAttentionLayer(
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=self_attention,
        
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    return ExemplarAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=4
    )
    

def test_exemplar_attention_forward(exemplar_attention):
    """Tests the forward pass of ExemplarAttention with random inputs."""
    batch_size = 2
    height_embed, width_embed = 64, 64
    seq_len = 64  # Number of spatial tokens
    d_model = 256
    
    # Create random tensors simulating feature maps and positional encoding
    curr = torch.randn(height_embed*width_embed, batch_size, d_model)
    curr_pos = torch.randn(height_embed*width_embed, batch_size, d_model)
    
    # Test forward pass with and without exemplar tokens excluded from RoPE
    for num_exemplar_tokens in [0, 5]: 
        
        exemplars = torch.randn(height_embed*width_embed + num_exemplar_tokens, batch_size, seq_len)
        exemplar_pos = torch.randn(height_embed*width_embed + num_exemplar_tokens, batch_size, seq_len)
        
        output = exemplar_attention(
            curr=curr,
            exemplars=exemplars,
            curr_pos=curr_pos,
            exemplar_pos=exemplar_pos,
            num_obj_ptr_tokens=num_exemplar_tokens,
        )

        # Check output shape
        assert output.shape == curr.shape, "Output shape mismatch!"
        
        # Check that no NaNs are present
        assert not torch.isnan(output).any(), "NaN detected in output tensor!"

    print("ExemplarAttention forward pass test passed.")
