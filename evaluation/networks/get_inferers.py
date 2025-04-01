from collections.abc import Callable, Sequence
from typing import Any
import logging

from monai.inferers import SlidingWindowInferer, SimpleInferer
import torch

logger = logging.getLogger("evaluation_pipeline_logger")

ROI_SIZE_FOR_SW_FASTEDIT = (128, 128, 64)
SW_OVERLAP_FOR_SW_FASTEDIT = 0.25
ROI_SIZE_FOR_DINS = (512, 160, 10)
SW_OVERLAP_FOR_DINS = 0.25


def get_inferer(args, device, network=None):
    """
    Returns the appropriate inferer based on the network type specified in `args`.

    Args:
        args (Any): Parsed command-line arguments containing `network_type`.
        device (str): Computation device (e.g., 'cuda', 'cpu').
        network (Any, optional): The model network, if required. Default is None.

    Returns:
        inferer (Any): The corresponding inference method for the specified network type.

    Raises:
        ValueError: If an unsupported `network_type` is provided.
    """
    inferer = None
    if args.network_type == "SW-FastEdit":
        inferer = configure_sliding_window_inferer(
            args, device, ROI_SIZE_FOR_SW_FASTEDIT, SW_OVERLAP_FOR_SW_FASTEDIT)
    elif args.network_type == "DINs":
        inferer = configure_sliding_window_inferer(
            args, device, ROI_SIZE_FOR_DINS, SW_OVERLAP_FOR_DINS)
    elif args.network_type in ["SimpleClick", "SAM2", "MOIS_SAM2"]:
        inferer = SimpleInferer()
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}")    
    return inferer


def configure_sliding_window_inferer(args, device, roi_size, sw_overlap, cache_roi_weight_map=True):
    """
    Configures and returns a sliding window inferer for 3D segmentation models.

    Args:
        args (Any): Parsed command-line arguments containing `sw_batch_size`.
        device (str): Computation device (e.g., 'cuda', 'cpu').
        roi_size (tuple): Size of the region of interest (ROI) for inference.
        sw_overlap (float): Overlap ratio for the sliding window.
        cache_roi_weight_map (bool, optional): Whether to cache the ROI weight map. Default is True.

    Returns:
        inferer (Conv3DBasedModelInferer): Configured sliding window inferer.
    """
    batch_size = args.sw_batch_size
    logger.info(f"{batch_size=}")
    
    sw_params = {
        "roi_size": roi_size,
        "mode": "gaussian",
        "cache_roi_weight_map": cache_roi_weight_map,
        }
    inferer = Conv3DBasedModelInferer(sw_batch_size=batch_size, overlap=sw_overlap, sw_device=device, **sw_params)
    return inferer


class Conv3DBasedModelInferer(SlidingWindowInferer):
    """
    A specialized 3D model inferer that extends MONAI's `SlidingWindowInferer`.
    
    This inferer extracts only the "image" key from the input dictionary and applies
    the inference model on it. It supports dynamic device assignment and optional
    buffering mechanisms to handle large image volumes efficiently.
    
    Attributes:
        device (str, optional): Device on which the model inference is performed.
        buffer_steps (int, optional): Number of buffer steps for memory-efficient inference.
        buffer_dim (int, optional): Dimension along which buffering is applied.
        cpu_thresh (int, optional): Threshold for switching to CPU-based inference if image
                                    size exceeds this limit.
    """
    def __call__(
        self,
        inputs: dict[str, torch.Tensor],
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Overridden `__call__` method to extract and process only the "image" key from inputs.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary containing model input data.
                - "image" (torch.Tensor): The primary input tensor for the model.
            network (Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]]):
                The target model used for inference.
            *args (Any): Additional arguments to pass to `network`.
            **kwargs (Any): Additional keyword arguments to pass to `network`.
                - "device" (str, optional): Target device for inference.
                - "buffer_steps" (int, optional): Number of buffer steps for processing.
                - "buffer_dim" (int, optional): Dimension along which buffering is applied.
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]: Model inference output.
            
        Raises:
            KeyError: If "image" key is missing from the input dictionary.
        """
        if "image" not in inputs:
            raise KeyError("The input dictionary must contain an 'image' key.")

        image_input = inputs["image"]  # Extracting only the image data

        device = kwargs.pop("device", self.device)
        buffer_steps = kwargs.pop("buffer_steps", self.buffer_steps)
        buffer_dim = kwargs.pop("buffer_dim", self.buffer_dim)

        if device is None and self.cpu_thresh is not None and image_input.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in CPU memory if image is too large

        return super().__call__(
            image_input,
            network,
            *args,
            device=device,
            buffer_steps=buffer_steps,
            buffer_dim=buffer_dim,
            **kwargs
        )
