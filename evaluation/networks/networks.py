import logging
import os
from monai.networks.nets.dynunet import DynUNet
import torch

from evaluation.networks.custom_networks import DINsNetwork, SAM2Network, MOIS_SAM2Network, VISTANetwork

logger = logging.getLogger("evaluation_pipeline_logger")


def get_network(args, device):
    """
    Loads and returns the appropriate segmentation network based on the specified network type.

    Args:
        args (Any): Parsed command-line arguments containing network configuration details.
            - `model_dir` (str): Directory where the model checkpoint is stored.
            - `checkpoint_name` (str): Name of the model checkpoint file.
            - `network_type` (str): Type of network to load (`SW-FastEdit`, `DINs`, or `SAM2`).
            - `labels` (list): List of label classes for segmentation.
            - `config_name` (str, optional): Configuration file name for SAM2.
            - `cache_dir` (str, optional): Directory for caching in SAM2.
        device (torch.device): The computation device (`cuda` or `cpu`).

    Returns:
        nn.Module: The selected deep learning model ready for inference.

    Raises:
        FileNotFoundError: If the model checkpoint file is not found.
        ValueError: If an unsupported network type is specified.
    """
    model_path = os.path.join(args.model_dir, args.checkpoint_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    if args.network_type == "SW-FastEdit":
        in_channels = 1 + len(args.labels)
        out_channels = len(args.labels)
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
        
        network.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)["net"]
        )
        network.to(device)
        
    elif args.network_type == "DINs":
        # The original DINs model was trained with Tensorflow 2.8.
        # For reusability, it was exported as ONNX and is launched as an ONNX runtime.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        network = DINsNetwork(model_path, providers, device)
        
    elif args.network_type == "SAM2":
        config_path = os.path.join(args.model_dir, args.config_name)
        network = SAM2Network(model_path=model_path,
                              config_path=config_path,
                              cache_path=args.cache_dir,
                              device=device
                              )
    
    elif args.network_type == "MOIS_SAM2":
        config_path = os.path.join(args.model_dir, args.config_name)
        network = MOIS_SAM2Network(model_path=model_path,
                                   config_path=config_path,
                                   cache_path=args.cache_dir,
                                   device=device,
                                   exemplar_use_com=args.exemplar_use_com,
                                   exemplar_inference_all_slices=args.exemplar_inference_all_slices,
                                   exemplar_num=args.exemplar_num,
                                   exemplar_use_only_prompted=args.exemplar_use_only_prompted,
                                   filter_prev_prediction_components=args.filter_prev_prediction_components,
                                   use_low_res_masks_for_com_detection=args.use_low_res_masks_for_com_detection,
                                   min_lesion_area_threshold=args.min_lesion_area_threshold
                                   )
    
    elif args.network_type == "VISTA":
        model_path = os.path.join(args.model_dir, args.checkpoint_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
       
        network = VISTANetwork(
            model_path=model_path,
            device=device,
            automatic_inference=args.use_automatic_vista_inference,
            model_registry = args.model_registry,
            input_channels = args.input_channels,
            patch_size=args.patch_size_inference,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            label_set=args.label_set,
            mapped_label_set=args.mapped_label_set,
            amp=args.amp
        )
        
    else:
        raise ValueError(f"Unsupported network: {args.network_type}")

    logger.info(f"Selected network: {args.network_type}")

    return network
