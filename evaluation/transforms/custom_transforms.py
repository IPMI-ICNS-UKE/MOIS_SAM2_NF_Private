# This file contains code from the SW-FastEdit repository:
# https://github.com/Zrrr1997/SW-FastEdit
#
# The original implementation accompanies the following research paper:
#
# M. Hadlich, Z. Marinov, M. Kim, E. Nasca, J. Kleesiek, and R. Stiefelhagen,
# "Sliding Window Fastedit: A Framework for Lesion Annotation in Whole-Body PET Images,"
# 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, 
# pp. 1-5, doi: 10.1109/ISBI56570.2024.10635459.
#
# Keywords: Training; Image segmentation; Solid modeling; Annotations; Memory management;
# Whole-body PET; Manuals; Interactive Segmentation; PET; Sliding Window; 
# Lung Cancer; Melanoma; Lymphoma
#
# The majority of the classes in this file were taken without modification from the 
# original SW-FastEdit repository.

from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Tuple
from enum import IntEnum
import logging
import gc

from scipy.ndimage import label
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor, PatchIterd
from monai.losses import DiceLoss
from monai.utils.enums import CommonKeys
from monai.networks.layers import GaussianFilter
from monai.transforms.utils import distance_transform_edt
from monai.transforms import (
    AsDiscreted,
    Compose,
    MapTransform,
    Randomizable,
)
from evaluation.utils.distance_transform import get_random_choice_from_tensor
from evaluation.utils.helper import (
    get_tensor_at_coordinates, 
    get_global_coordinates_from_patch_coordinates
)

logger = logging.getLogger("evaluation_pipeline_logger")
LABELS_KEY = "label_names"


class ClickGenerationStrategy(IntEnum):
    # Sample a click randomly based on the label, so no correction based on the prediction
    GLOBAL_NON_CORRECTIVE = 1
    # Sample a click based on the discrepancy between label and predition
    # Thus generate corrective clicks where the networks predicts incorrectly so far
    GLOBAL_CORRECTIVE = 2
    # Subdivide volume into patches of size train_crop_size, calculate the dice score for each, then sample click on the worst one
    PATCH_BASED_CORRECTIVE = 3
    # At each iteration sample from the probability and don't add a click if it yields False
    DEEPGROW_GLOBAL_CORRECTIVE = 4
    

class AddEmptySignalChannels(MapTransform):
    def __init__(self, device, keys: KeysCollection = None):
        """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
        """
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data
        in_channels = 1 + len(data[LABELS_KEY])
        tmp_image = data[CommonKeys.IMAGE][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0]
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"

        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )
            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "image":
                image = data[key]
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                # e.g. {'spleen': '[[1, 202, 190, 192], [2, 224, 212, 192], [1, 242, 202, 192], [1, 256, 184, 192], [2.0, 258, 198, 118]]',
                # 'background': '[[257, 0, 98, 118], [1.0, 223, 303, 86]]'}

                for _, (label_key, _) in enumerate(data[LABELS_KEY].items()):
                    # label_guidance = data[label_key]
                    label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                    logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")

                    if label_guidance is not None and label_guidance.numel():
                        signal = self._get_corrective_signal(
                            image,
                            label_guidance.to(device=self.device),
                            key_label=label_key,
                        )
                        assert torch.sum(signal) > 0
                    else:
                        # TODO can speed this up here
                        signal = self._get_corrective_signal(
                            image,
                            torch.Tensor([]).to(device=self.device),
                            key_label=label_key,
                        )

                    tmp_image = torch.cat([tmp_image, signal], dim=0)
                    if isinstance(data[key], MetaTensor):
                        data[key].array = tmp_image
                    else:
                        data[key] = tmp_image
                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")


class FindDiscrepancyRegions(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred_key: key to prediction source.
        discrepancy_key: key to store discrepancies found between label and prediction.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred_key: str = "pred",
        discrepancy_key: str = "discrepancy",
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred_key = pred_key
        self.discrepancy_key = discrepancy_key
        self.device = device

    def disparity(self, label, pred):
        disparity = label - pred
        # +1 means predicted label is not part of the ground truth
        # -1 means predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).to(dtype=torch.float32, device=self.device)  # FN
        neg_disparity = (disparity < 0).to(dtype=torch.float32, device=self.device)  # FP
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key in self.keys:
                assert (
                    (type(data[key]) is torch.Tensor)
                    or (type(data[key]) is MetaTensor)
                    and (type(data[self.pred_key]) is torch.Tensor or type(data[self.pred_key]) is MetaTensor)
                )
                all_discrepancies = {}

                for _, (label_key, label_value) in enumerate(data[LABELS_KEY].items()):
                    if label_key != "background":
                        label = torch.clone(data[key].detach())
                        # Label should be represented in 1
                        label[label != label_value] = 0
                        label = (label > 0.5).to(dtype=torch.float32)

                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    else:
                        # Taking single label
                        label = torch.clone(data[key].detach())
                        label[label != label_value] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).to(dtype=torch.float32)
                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    all_discrepancies[label_key] = self._apply(label, pred)
                data[self.discrepancy_key] = all_discrepancies
                return data
            else:
                logger.error("This transform only applies to 'label' key")
        raise UserWarning


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui


class AddGuidance(Randomizable, MapTransform):
    """
    Add guidance based on different click generation strategies.

    Args:
        discrepancy_key: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability_key: key to click/interaction probability, shape (1)
        device: device this transform shall run on.
        click_generation_strategy_key: sets the used ClickGenerationStrategy.
        patch_size: Only relevant for the patch-based click generation strategy. Sets the size of the cropped patches
        on which then further analysis is run.
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str = "label",
        pred_key: str = "pred",
        discrepancy_key: str = "discrepancy",
        probability_key: str = "probability",
        allow_missing_keys: bool = False,
        device=None,
        click_generation_strategy_key: str = "click_generation_strategy",
        patch_size: Tuple[int] = (128, 128, 128),
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred_key = pred_key
        self.label_key = label_key
        self.discrepancy_key = discrepancy_key
        self.probability_key = probability_key
        self._will_interact = None
        self.is_other = None
        self.default_guidance = None
        self.device = device
        self.click_generation_strategy_key = click_generation_strategy_key
        self.patch_size = patch_size

    def randomize(self, data: Mapping[Hashable, torch.Tensor]):
        probability = data[self.probability_key]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy) -> List[int | List[int]] | None:
        distance = distance_transform_edt(discrepancy)
        t_index, t_value = get_random_choice_from_tensor(distance)
        return t_index

    def add_guidance_based_on_discrepancy(
        self,
        data: Dict,
        guidance: torch.Tensor,
        key_label: str,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert guidance.dtype == torch.int32
        # Positive clicks of the segment in the iteration
        discrepancy = data[self.discrepancy_key][key_label]
        # idx 0 is positive discrepancy and idx 1 is negative discrepancy
        pos_discr = discrepancy[0]

        if coordinates is None:
            # Add guidance to the current key label
            if torch.sum(pos_discr) > 0:
                tmp_gui = self.find_guidance(pos_discr)
                self.check_guidance_length(data, tmp_gui)
                if tmp_gui is not None:
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
        else:
            pos_discr = get_tensor_at_coordinates(pos_discr, coordinates=coordinates)
            if torch.sum(pos_discr) > 0:
                # TODO Add suport for 2d
                tmp_gui = self.find_guidance(pos_discr)
                if tmp_gui is not None:
                    tmp_gui = get_global_coordinates_from_patch_coordinates(tmp_gui, coordinates)
                    self.check_guidance_length(data, tmp_gui)
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
        return guidance

    def add_guidance_based_on_label(self, data, guidance, label):
        assert guidance.dtype == torch.int32
        # Add guidance to the current key label
        if torch.sum(label) > 0:
            # generate a random sample
            tmp_gui_index, tmp_gui_value = get_random_choice_from_tensor(label)
            if tmp_gui_index is not None:
                self.check_guidance_length(data, tmp_gui_index)
                guidance = torch.cat(
                    (
                        guidance,
                        torch.tensor([tmp_gui_index], dtype=torch.int32, device=guidance.device),
                    ),
                    0,
                )
        return guidance

    def check_guidance_length(self, data, new_guidance):
        dimensions = 3 if len(data[CommonKeys.IMAGE].shape) > 3 else 2
        if dimensions == 3:
            assert len(new_guidance) == 4, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"
        else:
            assert len(new_guidance) == 3, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        click_generation_strategy = data[self.click_generation_strategy_key]

        if click_generation_strategy == ClickGenerationStrategy.GLOBAL_NON_CORRECTIVE:
            # uniform random sampling on label
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                data[key_label] = self.add_guidance_based_on_label(
                    data, tmp_gui, data["label"].eq(idx).to(dtype=torch.int32)
                )
        elif (
            click_generation_strategy == ClickGenerationStrategy.GLOBAL_CORRECTIVE
            or click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE
        ):
            if click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE:
                # sets self._will_interact
                self.randomize(data)
            else:
                self._will_interact = True

            if self._will_interact:
                for key_label in data[LABELS_KEY].keys():
                    tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)

                    # Add guidance based on discrepancy
                    data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label)
        elif click_generation_strategy == ClickGenerationStrategy.PATCH_BASED_CORRECTIVE:
            assert data[self.pred_key].shape == data[self.label_key].shape
            
            t = [
                AsDiscreted(
                    keys=[self.pred_key, self.label_key],
                    argmax=(True, False),
                    to_onehot=(len(data[LABELS_KEY]), len(data[LABELS_KEY])),
                ),
            ]
            post_transform = Compose(t)
            t_data = post_transform(data)

            # Split the data into patches of size self.patch_size
            # TODO not working for 2d data yet!
            new_data = PatchIterd(keys=[self.pred_key, self.label_key], patch_size=self.patch_size)(t_data)
            pred_list = []
            label_list = []
            coordinate_list = []

            for patch in new_data:
                actual_patch = patch[0]
                pred_list.append(actual_patch[self.pred_key])
                label_list.append(actual_patch[self.label_key])
                coordinate_list.append(actual_patch["patch_coords"])

            label_stack = torch.stack(label_list, 0)
            pred_stack = torch.stack(pred_list, 0)

            dice_loss = DiceLoss(include_background=True, reduction="none")
            
            
            with torch.no_grad():
                loss_per_label = dice_loss.forward(input=pred_stack, target=label_stack).squeeze()
                assert len(loss_per_label.shape) == 2
                # 1. dim: patch number, 2. dim: number of labels, e.g. [27,2]
                max_loss_position_per_label = torch.argmax(loss_per_label, dim=0)
                assert len(max_loss_position_per_label) == len(data[LABELS_KEY])

            # We now have the worst patches for each label, now sample clicks on them
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                patch_number = max_loss_position_per_label[idx]
                coordinates = coordinate_list[patch_number]

                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                # Add guidance based on discrepancy
                data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label, coordinates)

            gc.collect()
        else:
            raise UserWarning("Unknown click strategy")

        return data


class ConnectedComponentAnalysisd(MapTransform):
    """
    Custom MONAI dictionary-based transform to perform connected component analysis on binary masks.
    """
    def __init__(self, keys, allow_missing_keys=False):
        """
        Args:
            keys (list): List of keys to apply the transform (e.g., ["label"])
            allow_missing_keys (bool): Whether to skip keys that are not found in the data dictionary.
        """
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        """
        Apply the transform to the dictionary data.

        Args:
            data (dict): Dictionary containing image and label data.

        Returns:
            dict: Updated dictionary with instance-labeled masks.
        """
        for key in self.keys:
            # Retrieve metadata from the input tensor
            meta = data[key].meta
            
            binary_mask = data[key].cpu().numpy()  # Convert tensor to numpy
            labeled_mask, _ = label(binary_mask)  # Perform connected component labeling
            instance_mask = torch.tensor(labeled_mask, dtype=torch.int32, device=data[key].device)  # Convert back to tensor
            data[key] = MetaTensor(instance_mask, meta=meta)        
        return data
