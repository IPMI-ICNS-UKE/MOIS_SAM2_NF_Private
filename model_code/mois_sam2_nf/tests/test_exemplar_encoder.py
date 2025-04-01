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

from sam2.modeling.exemplar_encoder import ExemplarEncoder
from sam2.modeling.memory_encoder import MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine


@pytest.fixture
def exemplar_encoder():
    """Fixture to initialize ExemplarEncoder with real components."""
    embed_dim = 256
    stride = 2
    padding_md = 1
    padding_cxb = 3
    kernel_size_md = 3
    kernel_size_cxb = 7
    layer_scale_init_vale = 1e-6
    use_dwconv = True
    num_layers = 2

    # Initialize required modules
    mask_downsampler = MaskDownSampler(kernel_size=kernel_size_md, 
                                       stride=stride, 
                                       padding=padding_md)
    fuser = Fuser(
        layer=CXBlock(
            dim=embed_dim,
            kernel_size=kernel_size_cxb,
            padding=padding_cxb,
            layer_scale_init_value=layer_scale_init_vale,
            use_dwconv=use_dwconv
            ), 
        num_layers=num_layers) 
    position_encoding = PositionEmbeddingSine(embed_dim, normalize=True)

    return ExemplarEncoder(
        out_dim=embed_dim,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        position_encoding=position_encoding,
    )

def test_exemplar_encoder_forward(exemplar_encoder):
    """Tests the forward pass of ExemplarEncoder."""
    batch_size = 2
    embed_dim = 256
    height_emb, width_emb = 64, 64
    height_im, width_im = 1024, 1024

    # Generate dummy inputs
    pix_feat = torch.randn(batch_size, embed_dim, height_emb, width_emb)  # Random feature map
    masks = torch.randint(0, 2, (batch_size, 1, height_im, width_im)).float()  # Binary mask

    # Run forward pass
    output = exemplar_encoder(pix_feat, masks)

    # Assertions
    assert "vision_features" in output, "Output dictionary missing 'vision_features'"
    assert "vision_pos_enc" in output, "Output dictionary missing 'vision_pos_enc'"

    # Check shapes
    assert output["vision_features"].shape == pix_feat.shape, "Mismatch in vision features shape!"
    assert output["vision_pos_enc"][0].shape == (batch_size, embed_dim, height_emb, width_emb), "Mismatch in positional encoding shape!"

    # Check for NaNs
    assert not torch.isnan(output["vision_features"]).any(), "NaN detected in vision features!"
    assert not torch.isnan(output["vision_pos_enc"][0]).any(), "NaN detected in position encoding!"
