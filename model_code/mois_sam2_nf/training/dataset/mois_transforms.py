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

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.mois_data_utils import MOISVideoDatapoint
from training.dataset.transforms import get_size_with_aspect_ratio, RandomAffine


def mois_hflip(datapoint, index):
    """
    Horizontally flips the image, segment masks, and semantic masks for the given frame index.
    
    Args:
        datapoint (VideoDatapoint): The video data containing frames.
        index (int): The frame index to apply horizontal flip.

    Returns:
        VideoDatapoint: The horizontally flipped datapoint.
    """
    # Flip the image
    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)

    # Flip all available masks
    for obj in datapoint.frames[index].objects:
        for mask_attr in ["segment", "semantic_mask"]:  # Process both masks
            mask = getattr(obj, mask_attr, None)  # Get the attribute safely
            if mask is not None:
                setattr(obj, mask_attr, F.hflip(mask))  # Set the flipped mask

    return datapoint


def mois_resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    # Helper function to resize masks
    def resize_mask(mask):
        return F.resize(mask[None, None], size).squeeze() if mask is not None else None

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    
    # Resize segment and semantic masks
    for obj in datapoint.frames[index].objects:
        for mask_attr in ["segment", "semantic_mask"]:
            mask = getattr(obj, mask_attr, None)  # Get mask safely
            if mask is not None:
                setattr(obj, mask_attr, resize_mask(mask))  # Resize and set back
            
    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def mois_pad(datapoint, index, padding, v2=False):
    
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w

    # Define padding format based on tuple length
    pad_params = (0, 0, padding[0], padding[1]) if len(padding) == 2 else tuple(padding)

    # Pad image
    if v2:
        datapoint.frames[index].data = Fv2.pad(datapoint.frames[index].data, pad_params)
    else:
        datapoint.frames[index].data = F.pad(datapoint.frames[index].data, pad_params)

    # Update new height & width
    h += padding[1] if len(padding) == 2 else padding[1] + padding[3]
    w += padding[0] if len(padding) == 2 else padding[0] + padding[2]
    datapoint.frames[index].size = (h, w)

    # Helper function for padding masks
    def pad_mask(mask):
        if mask is None:
            return None
        return Fv2.pad(mask, pad_params) if v2 else F.pad(mask, pad_params)

    # Pad segment and semantic masks
    for obj in datapoint.frames[index].objects:
        for mask_attr in ["segment", "semantic_mask"]:
            mask = getattr(obj, mask_attr, None)  # Get mask safely
            setattr(obj, mask_attr, pad_mask(mask))  # Apply padding and set back

    return datapoint


class MOISRandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = mois_hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = mois_hflip(datapoint, i)
        return datapoint


class MOISRandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = mois_resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = mois_resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class MOISRandomAffine(RandomAffine):
    """
    Extends `RandomAffine` to also process `semantic_mask` while keeping other transformations unchanged.
    """
    
    def transform_datapoint(self, datapoint: MOISVideoDatapoint):
        """
        Applies the same affine transformation to both `segment` and `semantic_mask`.
        """
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            # Extract both segment and semantic masks
            masks_dict = {
                "segment": [obj.segment.unsqueeze(0) if obj.segment is not None else None for obj in img.objects],
                "semantic_mask": [obj.semantic_mask.unsqueeze(0) if obj.semantic_mask is not None else None for obj in img.objects]
            }

            if not self.consistent_transform:
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_masks = {key: [] for key in masks_dict.keys()}

            for i in range(len(img.objects)):
                for key in masks_dict:
                    mask = masks_dict[key][i]
                    if mask is not None:
                        transformed_mask = F.affine(
                            mask,
                            *affine_params,
                            interpolation=InterpolationMode.NEAREST,
                            fill=0.0,
                        )
                        transformed_masks[key].append(transformed_mask.squeeze())
                    else:
                        transformed_masks[key].append(None)

                if img_idx == 0 and transformed_masks["segment"][-1].max() == 0:
                    return None  # Skip transformation if segment disappears

            # Assign transformed masks back to objects
            for i, obj in enumerate(img.objects):
                for key in transformed_masks:
                    setattr(obj, key, transformed_masks[key][i])

            # Transform the image itself
            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )

        return datapoint


def mois_random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: Downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        # Handle `segment` mask safely
        if obj.segment is not None:
            assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
            segment_output = torch.zeros_like(obj.segment)

            target_y_offset_b = target_grid_y * H_im // grid_h
            target_x_offset_b = target_grid_x * W_im // grid_w
            target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
            target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
            target_H_im_downsize = target_y_offset_e - target_y_offset_b
            target_W_im_downsize = target_x_offset_e - target_x_offset_b

            segment_downsize = F.resize(
                obj.segment[None, None],
                size=(target_H_im_downsize, target_W_im_downsize),
                interpolation=InterpolationMode.NEAREST,
                antialias=True,
            )[0, 0]
            if should_hflip[target_grid_y, target_grid_x].item():
                segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

            segment_output[
                target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
            ] = segment_downsize
            obj.segment = segment_output

        # Handle `semantic_mask` safely
        if obj.semantic_mask is not None:
            assert obj.semantic_mask.shape == (H_im, W_im) and obj.semantic_mask.dtype == torch.uint8
            semantic_mask_output = torch.zeros_like(obj.semantic_mask)

            semantic_mask_downsize = F.resize(
                obj.semantic_mask[None, None],
                size=(target_H_im_downsize, target_W_im_downsize),
                interpolation=InterpolationMode.NEAREST,
                antialias=True,
            )[0, 0]
            if should_hflip[target_grid_y, target_grid_x].item():
                semantic_mask_downsize = F.hflip(semantic_mask_downsize[None, None])[0, 0]

            semantic_mask_output[
                target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
            ] = semantic_mask_downsize
            obj.semantic_mask = semantic_mask_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = mois_random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint
