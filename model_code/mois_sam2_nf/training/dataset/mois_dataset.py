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

import glob
import os
import torch

from training.dataset.vos_raw_dataset import PNGRawDataset, VOSFrame, VOSVideo
from training.dataset.mois_segment_loader import MOISPNGSegmentLoader
from training.dataset.vos_dataset import load_images, VOSDataset
from training.dataset.vos_sampler import VOSSampler
from training.utils.mois_data_utils import MOISFrame, MOISObject, MOISVideoDatapoint

MAX_RETRIES = 100


class MOISPNGRawDataset(PNGRawDataset):
    """
    Extension of PNGRawDataset that exclusively uses MOISPNGSegmentLoader.
    """

    def get_video(self, idx):
        """
        Override get_video() to use MOISPNGSegmentLoader instead of 
        PalettisedPNGSegmentLoader or MultiplePNGSegmentLoader.
        """
        video_name = self.video_names[idx]

        # Determine frame root path
        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        # Define mask root path
        video_mask_root = os.path.join(self.gt_folder, video_name)

        # Always use MOISPNGSegmentLoader
        segment_loader = MOISPNGSegmentLoader(video_mask_root)

        # Load all frames
        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]

        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))

        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader


class MOISVOSDataset(VOSDataset):
    """
    MOIS extension of VOSDataset to ensure:
    - video_dataset is of type MOISPNGRawDataset.
    - segment_loader is MOISPNGSegmentLoader.
    - semantic_mask is included in each object.
    """
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: MOISPNGRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        # Ensure video_dataset is an instance of MOISPNGRawDataset
        if not isinstance(video_dataset, MOISPNGRawDataset):
            raise TypeError(
                f"Expected `video_dataset` to be an instance of MOISPNGRawDataset, "
                f"but got {type(video_dataset).__name__} instead."
            )
        
        super().__init__(transforms, 
                         training, 
                         video_dataset, 
                         sampler, 
                         multiplier, 
                         always_target, 
                         target_segments_available)
        self.target_segments_available = target_segments_available
    
    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
         Constructs a MOISVideoDatapoint sample with semantic segmentation support.
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)
        
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                MOISFrame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            
            # Ensure segment_loader is of type MOISPNGSegmentLoader
            if isinstance(segment_loader, MOISPNGSegmentLoader):
                # We load the gt segments associated with the current frame
                segments, semantic_mask = segment_loader.load(frame.frame_idx)
            else:
                raise TypeError(
                f"Expected `segment_loader` to be an instance of MOISPNGSegmentLoader, "
                f"but got {type(segment_loader).__name__} instead."
                )
                
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                images[frame_idx].objects.append(
                    MOISObject(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                        semantic_mask=semantic_mask
                    )
                )
        return MOISVideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)
