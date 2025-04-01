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

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from training.dataset.vos_segment_loader import PalettisedPNGSegmentLoader


class MOISPNGSegmentLoader(PalettisedPNGSegmentLoader):
    """
    Extended PNG Segment Loader that returns both binary segmentation masks and semantic masks.
    """

    def load(self, frame_id):
        """
        Loads segmentation masks from a palettised PNG image.

        Args:
            frame_id (int): The ID of the frame to load.

        Returns:
            binary_segments (dict): A dictionary where keys are object IDs and values are binary masks.
            semantic_mask (torch.Tensor): A tensor of the same size as the original mask image,
                                          where 0 = background, 1 = foreground.
        """
        # Get the mask path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # Load the mask
        masks = PILImage.open(mask_path).convert("P")  # Convert to palettised mode
        masks = np.array(masks)

        # Get unique object IDs (ignoring background, which is 0)
        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # Remove background (0)

        # Generate binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i  # Create binary mask for each object
            binary_segments[i] = torch.from_numpy(bs)

        # Generate semantic mask: Background = 0, Foreground = 1
        semantic_mask = torch.from_numpy((masks > 0).astype(np.uint8))

        return binary_segments, semantic_mask
