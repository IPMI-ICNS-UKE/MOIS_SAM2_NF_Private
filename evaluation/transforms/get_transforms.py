import logging
import os
import numpy as np
import torch

from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.data.folder_layout import FolderLayout
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Identityd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    NormalizeIntensityd,
    KeepLargestConnectedComponentd,
    Resized,
    Flipd,
    SpatialPadd,    
)

from evaluation.transforms.custom_transforms import (
    AddEmptySignalChannels, 
    FindDiscrepancyRegions,
    AddGuidance,
    AddGuidanceSignal,
    ConnectedComponentAnalysisd,
    ScaleIntensityRangePercentilesIgnoreZerod
)

logger = logging.getLogger("evaluation_pipeline_logger")

SPACING_FOR_DINS = (1.7, 1.7, 7.8)
ORIENTATION_FOR_DINS = ("SRA")
SPACING_FOR_SW_FASTEDIT = (0.625, 0.625, 7.8)
ORIENTATION_FOR_SW_FASTEDIT = ("RSA")
SPACING_FOR_SAM2 = (-1, -1, -1)
ORIENTATION_FOR_SAM2 = ("RSA")
SPACING_FOR_MOISSAM2 = (-1, -1, -1)
ORIENTATION_FOR_MOISSAM2 = ("SRA")
TARGET_SIZE_FOR_MOISSAM2 = (1024, 1024, -1)
SPACING_FOR_VISTA = (0.625, 0.625, 7.8)
ORIENTATION_FOR_VISTA = ("RSA")


def get_pre_transforms(args, 
                       device="cpu", 
                       input_keys=("image", "label", "connected_component_label")):  
    """
    Constructs a pre-processing transformation pipeline based on the selected segmentation model.

    This function applies a series of transformations such as loading images, normalizing labels, 
    applying orientation and spacing adjustments, and scaling intensity values. The transformations 
    vary based on the selected `network_type` in `args`.

    Args:
        args (Any): Parsed command-line arguments containing the model configuration.
            - `labels` (Dict): Dictionary mapping label names to numerical values.
            - `evaluation_mode` (str): Specifies whether to use a corrective segmentation mode.
            - `network_type` (str): Type of network being used (`SW-FastEdit`, `DINs`, or `SAM2`).
            - `num_lesions` (int): Maximum number of lesion instances to retain.
        device (str, optional): Computation device (`cpu` or `cuda`). Default is "cpu".
        input_keys (Tuple[str, ...], optional): Keys representing different input data types. Default is `("image", "label", "connected_component_label")`.

    Returns:
        Compose: A MONAI `Compose` object containing the specified transformations.

    Raises:
        ValueError: If an unsupported network type is provided.
    """  
    # Unified transforms for all models
    transforms = [
        LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=input_keys),
        NormalizeLabelsInDatasetd(keys=["label", "connected_component_label"], 
                                  label_names=args.labels),
        KeepLargestConnectedComponentd(keys="connected_component_label", 
                                       num_components=args.num_lesions) if args.evaluation_mode not in ["global_corrective", "global_non_corrective"]
        else Identityd(keys="connected_component_label"),
        ConnectedComponentAnalysisd(keys="connected_component_label") if args.evaluation_mode not in ["global_corrective", "global_non_corrective"]
        else Identityd(keys="connected_component_label"),
    ]
    
    # Add model-specific transforms
    if args.network_type == "SW-FastEdit":
        spacing = SPACING_FOR_SW_FASTEDIT
        orientation = ORIENTATION_FOR_SW_FASTEDIT
        
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Spacingd(keys='image', pixdim=spacing),
                Spacingd(keys=['label', 'connected_component_label'], 
                         pixdim=spacing, mode="nearest"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=input_keys, k=32, value=0),
                AddEmptySignalChannels(keys='image', device=device) 
            ]
        )
        
    elif args.network_type == "DINs":
        spacing = SPACING_FOR_DINS
        orientation = ORIENTATION_FOR_DINS
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Spacingd(keys='image', pixdim=spacing),
                Spacingd(keys=['label', 'connected_component_label'], 
                         pixdim=spacing, mode="nearest"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        
    elif args.network_type == "SAM2":
        spacing = SPACING_FOR_SAM2
        orientation = ORIENTATION_FOR_SAM2
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, 
                                                b_min=0.0, b_max=255.0, clip=True),
            ]
        )

    elif args.network_type == "MOIS_SAM2":
        spacing = SPACING_FOR_MOISSAM2
        orientation = ORIENTATION_FOR_MOISSAM2
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Flipd(input_keys, spatial_axis=0),
                Flipd(input_keys, spatial_axis=1),
                ScaleIntensityRangePercentilesIgnoreZerod(keys="image", lower=0.5, upper=99.5, 
                                                          out_min=0, out_max=255),
            ]
        )
        
    elif args.network_type == "VISTA":
        spacing = SPACING_FOR_VISTA
        orientation = ORIENTATION_FOR_VISTA
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Spacingd(keys='image', pixdim=spacing),
                Spacingd(keys=['label', 'connected_component_label'], 
                         pixdim=spacing, mode="nearest"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                SpatialPadd(keys=input_keys, spatial_size=args.patch_size_inference),
            ]
        )
        
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}")    
    
    return Compose(transforms)


def get_interaction_pre_transforms(args, device="cpu"):
    """
    Constructs a sequence of preprocessing transformations used in the Interaction Class.

    This function applies a series of operations that prepare the image and label data
    for interaction-based corrections during segmentation. It ensures data types,
    identifies discrepancy regions, and augments input with guidance signals.

    Args:
        args (Any): Parsed command-line arguments containing:
            - `patch_size_discrepancy` (tuple): Size of the patch used for discrepancy calculation.
            - `sigma` (float or tuple): Standard deviation for the Gaussian smoothing filter.
            - `no_disks` (bool): Flag to disable disk-based guidance signals.
        device (str, optional): Computation device (`cpu` or `cuda`). Default is "cpu".

    Returns:
        Compose: A MONAI `Compose` object containing the specified transformations.
    """
    transforms = [
        EnsureTyped(keys=["image", "connected_component_label_local", "pred_local"], 
                    device="cuda"),
        FindDiscrepancyRegions(keys="connected_component_label_local", 
                               pred_key="pred_local", 
                               discrepancy_key="discrepancy", 
                               device="cuda"),
        AddGuidance(keys="NA", pred_key="pred_local", 
                    label_key="connected_component_label_local", 
                    discrepancy_key="discrepancy", 
                    probability_key="probability", device="cuda", 
                    patch_size=(args.patch_size_discrepancy)),
        AddGuidanceSignal(keys="image", sigma=args.sigma, 
                          disks=(not args.no_disks), device="cuda",),
        EnsureTyped(keys=["image", "connected_component_label_local", "pred_local"], 
                    device="cpu"),
    ]
    return Compose(transforms)


def get_interaction_post_transforms(args, device="cpu"):
    """
    Constructs post-processing transformations used in the Interaction Class.

    Depending on the selected network type, this function applies activation functions,
    discretization, and ensures that the prediction tensor is properly formatted.

    Args:
        args (Any): Parsed command-line arguments containing:
            - `network_type` (str): Specifies which segmentation model is being used.
        device (str, optional): Computation device (`cpu` or `cuda`). Default is "cpu".

    Returns:
        Compose: A MONAI `Compose` object containing the specified transformations.

    Raises:
        ValueError: If an unsupported network type is provided.
    """
    # Expects "current_label" containing only a binary mask 
    if ((args.network_type == "SW-FastEdit") or 
        (args.network_type == "DINs")):
        transforms = [
            Activationsd(keys="pred_local", softmax=True),
            AsDiscreted(keys="pred_local", argmax=True),
            EnsureTyped(keys="pred_local", device=device),
        ]
    elif ((args.network_type == "SAM2") or 
          (args.network_type == "MOIS_SAM2")):
        transforms = [
            EnsureTyped(keys="pred_local", device=device)
        ]
    elif (args.network_type == "VISTA"):
        transforms = [
            AsDiscreted(keys="pred_local", threshold=0.0, dtype=torch.uint8),
            AsDiscreted(keys="pred_local", argmax=True),
            EnsureTyped(keys="pred_local", device=device),
        ]
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}") 
        
    return Compose(transforms)


def get_post_transforms(args, pre_transforms, device="cpu"):
    """
    Constructs post-processing transformations for saving and restoring segmentation results.

    This function applies inverse transformations to restore the original image orientation
    and spacing after inference. Additionally, it allows saving predictions as NIfTI files
    if `save_predictions` is enabled.

    Args:
        args (Any): Parsed command-line arguments containing:
            - `save_predictions` (bool): Whether to save predictions to disk.
            - `results_dir` (str): Root directory for storing results.
            - `network_type` (str): The segmentation network being used.
            - `evaluation_mode` (str): Mode of evaluation.
            - `test_set_id` (int): Identifier for the test dataset.
            - `fold` (int): Cross-validation fold number.
        pre_transforms (Compose): The set of preprocessing transformations applied before inference.
        device (str, optional): Computation device (`cpu` or `cuda`). Default is "cpu".

    Returns:
        Compose: A MONAI `Compose` object containing the specified transformations.
    """
    
    if args.save_predictions:
        predictions_output_dir = os.path.join(
            args.results_dir, "predictions", args.network_type, args.evaluation_mode,
            f"TestSet_{args.test_set_id}", f"fold_{args.fold}")
        
        if not os.path.exists(predictions_output_dir):
            os.makedirs(predictions_output_dir)
        
        nii_layout = FolderLayout(output_dir=predictions_output_dir, 
                                  postfix="", 
                                  extension=".nii.gz", 
                                  makedirs=False)
    else:
        nii_layout = None
    
    transforms = [
        Invertd(
            keys=["pred", "label"],
            orig_keys="image",
            nearest_interp=False,
            transform=pre_transforms,
        ),
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ) if args.save_predictions else Identityd(keys="pred")
    ]
    return Compose(transforms)
    