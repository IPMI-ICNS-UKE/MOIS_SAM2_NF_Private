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

import torch.nn as nn
import os
from typing import Optional

from training.trainer import Trainer
from training.utils.mois_data_utils import MOISBatchedVideoDatapoint


class MOISTrainer(Trainer):
    """
    Trainer for MOIS SAM2 with exemplar-based learning for multi-instance object segmentation.
    """
    
    def __init__(
        self,
        *,
        add_exemplars_after_epoch: Optional[int] = None,
        add_exemplars_after_plateau: bool = False,
        add_exemplars_gradually: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.add_exemplars_after_epoch = add_exemplars_after_epoch
        self.add_exemplars_after_plateau = add_exemplars_after_plateau
        self.add_exemplars_gradually = add_exemplars_gradually

        # Validation: only one of the modes can be active
        active_flags = [
            bool(add_exemplars_after_epoch is not None),
            add_exemplars_after_plateau,
            add_exemplars_gradually,
        ]

        if sum(active_flags) > 1:
            raise ValueError(
                "Only one of 'add_exemplars_after_epoch', "
                "'add_exemplars_after_plateau', or 'add_exemplars_gradually' "
                "can be set. Set only one or none."
            )
    
    def _step(
        self,
        batch: MOISBatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        """
        Batch contains:
            img_batch: torch.FloatTensor
            obj_to_frame_idx: torch.IntTensor
            masks: torch.BoolTensor
            semantic_masks: torch.BoolTensor
            metadata: BatchedVideoMetaData
            dict_key: str
        """
         # Ensure batch is an instance of MOISBatchedVideoDatapoint
        assert isinstance(batch, MOISBatchedVideoDatapoint), \
            f"Expected batch to be of type MOISBatchedVideoDatapoint, got {type(batch)}"
        if self.epoch >= self.add_exemplars_after_epoch:
            exemplars_activated = True
        else:
            exemplars_activated = False
        outputs = model(batch, exemplars_activated) # Output is a dict
        targets = batch.masks
        semantic_targets = batch.semantic_masks
        batch_size = len(batch.img_batch)
        
        key = batch.dict_key  # key for dataset
        # Expect the loss to be MultiStepMultiMasksAndIousAndSemantic
        loss = self.loss[key](outputs, targets, semantic_targets) 
        loss_str = f"Losses/{phase}_{key}_loss"
        
        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple
