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

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
from training.loss_fns import dice_loss, sigmoid_focal_loss, iou_loss, MultiStepMultiMasksAndIous

import os
import torchvision.transforms as transforms
from PIL import Image
import datetime

class MultiStepMultiMasksAndIousAndSemantic(nn.Module):
    def __init__(
        self,
        weight_dict, # Should inlcude weights for semantic masks as well
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):        
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # Validate required loss keys
        required_keys = ["loss_mask", 
                         "loss_dice", 
                         "loss_iou", 
                         "loss_semantic_mask", 
                         "loss_semantic_dice", 
                         "loss_semantic_iou"]
        for key in required_keys:
            if key not in self.weight_dict:
                raise ValueError(f"Missing required loss weight: '{key}' in weight_dict.")

        # Ensure loss_class has a default value
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
    
    
    def forward(self, outs_batch: List[Dict], 
                targets_batch: torch.Tensor, 
                targets_semantic_batch: torch.Tensor):
        """
        Compute losses for segmentation, IoU, and semantic segmentation.
        
        Args:
            outs_batch (List[Dict]): Model predictions for each batch item.
            targets_batch (torch.Tensor): Ground-truth segmentation masks.
            targets_semantic_batch (torch.Tensor): Ground-truth semantic segmentation masks.
            
        Returns:
            Dict[str, torch.Tensor]: Loss dictionary containing segmentation and semantic segmentation losses.
        """
        # print("STARTED_"*5)
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        num_frames = targets_batch.shape[0]
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets, targets_semantic in zip(outs_batch, targets_batch, targets_semantic_batch):
            cur_losses = self._forward(outs, targets, targets_semantic, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        
        # print("---"*100)
        # print("---"*100)
        # print("LOSSES BEFORE AVERAGING")
        # print(losses)
        # print("---"*100)
        
        # if num_frames > 0:
        #     losses["loss_semantic_mask"] /= num_frames
        #     losses["loss_semantic_dice"] /= num_frames
        #     losses["loss_semantic_iou"] /= num_frames
            
        # losses[CORE_LOSS_KEY] += (losses["loss_semantic_mask"] * self.weight_dict["loss_semantic_mask"] + 
        #                           losses["loss_semantic_dice"] * self.weight_dict["loss_semantic_dice"] + 
        #                           losses["loss_semantic_iou"] * self.weight_dict["loss_semantic_iou"])
        
        # print("LOSSES AFTER AVERAGING")
        # print(losses)
        # print("---"*100)
        # print("---"*100)

        return losses
    
    
    def _forward(self, outputs: Dict, targets: torch.Tensor, targets_semantic: torch.Tensor, num_objects):
        """
        Compute losses for instance segmentation and semantic segmentation.
        """
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        
        target_semantic_masks = targets_semantic.unsqueeze(1).float()
        assert target_semantic_masks.dim() == 4  # torch.Size([2, 1, 1024, 1024])
        
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        
        src_semantic_masks_list = outputs["semantic_pred_high_res"] # [torch.Size([1, 1, 1024, 1024])]
        ious_semantic_list = outputs["semantic_pred_ious"] # [torch.Size([1, 1])]
                
        if (src_semantic_masks_list == None) or (ious_semantic_list == None):
            # Skip semantic loss if missing
            src_semantic_masks_list = [None]
            ious_semantic_list = [None]
            compute_semantic_loss = False
        else:
            compute_semantic_loss = True
                
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        if compute_semantic_loss:
            assert len(src_semantic_masks_list) == len(ious_semantic_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, 
                  "loss_semantic_mask": 0, "loss_semantic_dice": 0, "loss_semantic_iou": 0,
                  "loss_class": 0}
        for src_masks, ious, src_semantic_masks, ious_semantic, object_score_logits in zip(
            src_masks_list, ious_list, src_semantic_masks_list, ious_semantic_list, object_score_logits_list
        ):
            self._update_losses(
                losses, 
                src_masks, target_masks, ious, 
                src_semantic_masks, target_semantic_masks, ious_semantic, 
                num_objects, object_score_logits, compute_semantic_loss
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses
    
    
    def _save_images(self, tensor_masks, name_prefix, apply_activation=False):
        """ Saves the mask tensors as PNG images with timestamps. 
            If apply_activation is True, applies sigmoid and binarization.
        """
        save_dir = "saved_masks"
        os.makedirs(save_dir, exist_ok=True)
        
        batch_size = tensor_masks.shape[0]
        num_masks = tensor_masks.shape[1]

        to_pil = transforms.ToPILImage()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{datetime.datetime.now().microsecond // 1000:03d}"
        
        for i in range(batch_size):
            for j in range(num_masks):
                img = tensor_masks[i, j].cpu().detach()
                
                if apply_activation:
                    img = torch.sigmoid(img)  # Apply sigmoid activation
                    img = (img > 0.5).float()  # Binarize using threshold 0.5
                
                img = (img * 255).byte()  # Scale to 0-255 for PNG
                img_pil = to_pil(img)
                
                filename = f"{save_dir}/{timestamp}_{name_prefix}_b{i}_m{j}.png"
                img_pil.save(filename)
                print(f"Saved: {filename}")
                
    
    def _update_losses(self, losses,
                       src_masks, target_masks, 
                       ious, src_semantic_masks, 
                       target_semantic_masks, 
                       ious_semantic, num_objects, 
                       object_score_logits,
                       compute_semantic_loss
                       ):
        
        # Instance segmentation
        target_masks = target_masks.expand_as(src_masks)
        
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        
        # Semantic segmentation
        # Extract a single ground truth semantic mask
        if compute_semantic_loss:
            target_semantic_masks = target_semantic_masks[0].unsqueeze(0)
            target_semantic_masks = target_semantic_masks.expand_as(src_semantic_masks)

            loss_semantic_mask = sigmoid_focal_loss(
                src_semantic_masks,
                target_semantic_masks,
                num_objects=1, # We should consider it as a single object
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                loss_on_multimask=True,
            ).squeeze()
                
            loss_semantic_dice = dice_loss(
                src_semantic_masks, 
                target_semantic_masks, 
                num_objects=1, 
                loss_on_multimask=True
            ).squeeze()
            loss_semantic_iou = iou_loss(
                src_semantic_masks,
                target_semantic_masks,
                ious_semantic,
                num_objects=1,
                loss_on_multimask=True,
                use_l1_loss=self.iou_use_l1_loss,
            ).squeeze()

            losses["loss_semantic_mask"] += loss_semantic_mask
            losses["loss_semantic_dice"] += loss_semantic_dice
            losses["loss_semantic_iou"] += loss_semantic_iou
        else:
            losses["loss_semantic_mask"] += torch.tensor(0.0, device=src_masks.device)
            losses["loss_semantic_dice"] += torch.tensor(0.0, device=src_masks.device)
            losses["loss_semantic_iou"] += torch.tensor(0.0, device=src_masks.device)
        
        # print('Predictions:_________')
        # print("src_masks: ", 
        #       src_masks.shape, 
        #       torch.min(src_masks), 
        #       torch.max(src_masks))
        # print("src_semantic_masks: ", 
        #       src_semantic_masks.shape, 
        #       torch.min(src_semantic_masks), 
        #       torch.max(src_semantic_masks))
        
        # print('Targets:_________')
        # print("target_masks: ", 
        #       target_masks.shape, 
        #       torch.min(target_masks), 
        #       torch.max(target_masks))
        # print("target_semantic_masks: ", 
        #       target_semantic_masks.shape, 
        #       torch.min(target_semantic_masks), 
        #       torch.max(target_semantic_masks))
        
        # self._save_images(src_masks, "src_masks", apply_activation=True)
        # self._save_images(src_semantic_masks, "src_semantic_masks", apply_activation=True)
        # self._save_images(target_masks, "target_masks", apply_activation=False)
        # self._save_images(target_semantic_masks, "target_semantic_masks", apply_activation=False)
        
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
            
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        # print("loss_mask vs loss_semantic_mask")
        # print(loss_mask.shape, loss_semantic_mask.shape, loss_semantic_mask)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        
        # There should be only single semantic mask
        losses["loss_class"] += loss_class
    
    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
                # if not "semantic" in loss_key:
                #     reduced_loss += losses[loss_key] * weight
                # else:
                #     print("semantic")

        return reduced_loss
        