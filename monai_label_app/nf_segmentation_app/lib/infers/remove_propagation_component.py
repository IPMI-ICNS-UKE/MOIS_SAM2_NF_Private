from typing import Callable, Sequence, Union, Any
from time import time
import logging
import copy
import torch
from scipy.ndimage import label as cc_label
import numpy as np

from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Flipd,
    EnsureTyped,
    ToNumpyd,
    SqueezeDimd
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from lib.transforms.transforms import (ReorientToOriginald,
                                       TransformPointsd)

logger = logging.getLogger(__name__)


class RemovePropagationComponent(BasicInferTask):
    """
    Task for removing unwanted connected components in propagated label masks,
    typically after automatic segmentation.

    Args:
        path (str): Path to model weights (not used, inherited from BasicInferTask).
        network: Optional network object (not used here).
        type (InferType): MONAI label inference type (typically DEEPGROW).
        labels (list): List of label names.
        dimension (int): Input dimension (3D).
        orientation (tuple): Orientation for loading image.
        description (str): Description for logging/UI.
    """
    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        orientation=("SRA"),
        description="MOIS-SAM2 in the interaction mode",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.orientation = orientation

    def pre_transforms(self, data=None) -> Sequence[Callable]: 
        """Define the sequence of preprocessing transforms for input data."""
        transforms = [
            LoadImaged(keys=["label_mask"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["label_mask"]),
            Orientationd(keys=["label_mask"], axcodes=self.orientation),
            TransformPointsd(keys=["foreground", "background"], 
                            ref_image_key="label_mask", 
                            orientation=self.orientation, 
                            flip_axes=[0, 1]),
            Flipd(["label_mask"], spatial_axis=0),
            Flipd(["label_mask"], spatial_axis=1),
            SqueezeDimd(keys="label_mask", dim=0),
        ]
        self.add_cache_transform(transforms, data)
        transforms.append(EnsureTyped(keys=["label_mask"], device="cpu"))
        return transforms
    
    def inferer(self, data=None) -> Inferer:
        """Define the inference method."""
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        """Run no pre-transforms in reverse mode."""
        return None
    
    def post_transforms(self, data=None) -> Sequence[Callable]:
        """Define the sequence of postprocessing transforms applied to predictions."""
        transforms = [
            EnsureTyped(keys="pred", device="cuda"),
            EnsureChannelFirstd(keys=["pred"]),
            Flipd(["label_mask"], spatial_axis=1),
            Flipd(["label_mask"], spatial_axis=0),
            ReorientToOriginald(keys="pred", ref_image="label_mask"),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
        ]
        return transforms
    
    def remove_propagation_components(self, data, convert_to_batch=True, device="cuda"):
        """
        Remove connected components that intersect with user-provided background points.

        Args:
            data (dict): Dictionary containing label_mask and background points.
            convert_to_batch (bool): Unused here.
            device (str): Device to place final prediction (default: 'cuda').

        Returns:
            dict: Updated data with cleaned prediction in `data["pred"]`.
        """
        label_mask_tensor = data["label_mask"]
        
        label_mask = label_mask_tensor.cpu().numpy().astype(np.uint8)
        cleaned_mask = np.copy(label_mask)
        
        bps: dict[int, list] = {}
        for p in data.get("background", []):
            sid = p[2]
            if sid in bps:
                bps[sid].append((p[0], p[1]))  # (x, y)
            else:
                bps[sid] = [(p[0], p[1])]
                
        for sid, points in bps.items():
            if sid < 0 or sid >= cleaned_mask.shape[2]:
                continue

            mask_slice = cleaned_mask[:, :, sid]
            cc_mask, _ = cc_label(mask_slice)

            # Collect unique component IDs touched by any point
            cc_ids_to_remove = set()
            for x, y in points:
                cc_id = cc_mask[x, y]
                if cc_id > 0:
                    cc_ids_to_remove.add(cc_id)

            # Zero out all collected components
            for cc_id in cc_ids_to_remove:
                mask_slice[cc_mask == cc_id] = 0

            cleaned_mask[:, :, sid] = mask_slice

        pred = torch.from_numpy(cleaned_mask.astype(np.uint8))
        meta = label_mask_tensor.meta
        pred = MetaTensor(pred, meta=meta)
        data["pred"] = pred
        return data
            
    def __call__(self, request):
        """ Main inference call entrypoint."""
        request["save_label"] = True
        request["label_tag"] = "final"
        begin = time()
        req = copy.deepcopy(self._config)
        req.update(request)
        data = copy.deepcopy(req)
        
        start = time()        
        pre_transforms = self.pre_transforms(data)
        data = self.run_pre_transforms(data, pre_transforms)
        latency_pre = time() - start
        
        start = time()        
        logger.info(f"Infer Request: {request}")                          
        data = self.remove_propagation_components(data)
        latency_inferer = time() - start

        start = time()        
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms(data))
        latency_invert = time() - start

        start = time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        latency_post = time() - start
        
        start = time()
        result_file_name, result_json = self.writer(data)
        latency_write = time() - start

        latency_total = time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Inferer: {:.4f}; Invert: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_inferer,
                latency_invert,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "infer": round(latency_inferer, 2),
            "invert": round(latency_invert, 2),
            "post": round(latency_post, 2),
            "write": round(latency_write, 2),
            "total": round(latency_total, 2),
            "transform": data.get("latencies"),
        }

        if result_file_name is not None and isinstance(result_file_name, str):
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        return result_file_name, result_json     
    
    def writer(self, data, extension=None, dtype=None):
        return super().writer(data, extension=".nii.gz")
