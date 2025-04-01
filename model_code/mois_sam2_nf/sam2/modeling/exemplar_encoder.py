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

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExemplarEncoder(nn.Module):
    """Encodes exemplars (segmentation results) for object retrieval and 
    segmentation propagation.

    The Exemplar Encoder takes a segmented mask and the corresponding visual 
    features, processes them, and generates a compact representation for 
    retrieval in later frames or instances.

    Args:
        out_dim (int): Output dimensionality of the encoded exemplar.
        mask_downsampler (nn.Module): Module for downsampling the segmentation mask.
        fuser (nn.Module): Module for fusing visual features and mask embeddings.
        position_encoding (nn.Module): Module for generating position encodings.
        in_dim (int, optional): Dimensionality of the input feature map. Defaults to 256.
    """
    def __init__(
        self,
        out_dim: int,
        mask_downsampler: nn.Module,
        fuser: nn.Module,
        position_encoding: nn.Module,
        in_dim=256,  # Default input feature dimensionality
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler # Compresses mask information

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser # Fuses downsampled mask and features
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        
        # If output dimensionality differs, apply a projection layer
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes input feature maps and segmentation masks to generate exemplar embeddings.

        Args:
            pix_feat (torch.Tensor): Feature map from the vision backbone (B, C, H, W).
            masks (torch.Tensor): Binary segmentation masks (B, 1, H, W).
            skip_mask_sigmoid (bool, optional): Whether to skip sigmoid activation 
                                                on input masks. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "vision_features": Processed feature map fused with mask information.
                - "vision_pos_enc": Positional encoding of the output.
        """
        # Process masks using sigmoid (helps align with network training)
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)              
        masks = self.mask_downsampler(masks)

        # Ensure feature tensor is on the same device as masks
        pix_feat = pix_feat.to(masks.device)
        
        # Fuse the downsampled mask with visual features
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        
        # Compute positional encoding
        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
