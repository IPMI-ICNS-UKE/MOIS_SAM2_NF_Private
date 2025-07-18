from typing import Callable, Sequence, Union, Any
from tqdm import tqdm
from time import time
import logging
import copy
import os
import torch
import numpy as np
from PIL import Image
import pylab
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer, SimpleInferer
from monai.utils import set_determinism
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
from monailabel.utils.others.generic import (
    get_basename_no_ext,
    md5_digest,
    name_to_device,
    strtobool,
)
from sam2.build_mois_sam import build_mois_sam2_predictor
from lib.cache_logic.cache_logic import ImageCache
from lib.transforms.transforms import (ScaleIntensityRangePercentilesIgnoreZerod,
                                       ReorientToOriginald,
                                       TransformPointsd)

logger = logging.getLogger(__name__)


image_cache = ImageCache()
image_cache.monitor()


class MOISSAM_Interaction(BasicInferTask):
    """
    Interactive segmentation task using MOIS-SAM2. 
    It supports multi-slice interactive image segmentation and uses forward/backward
    memory-based propagation.

    Args:
        path (str): Path to model weights.
        config_name (str): Name of configuration YAML file.
        network: Optional initialized model (if not using path-based init).
        type (InferType): Type of inference task.
        labels (list): List of label names.
        dimension (int): Dimensionality of the task (typically 3).
        spacing (tuple): Desired spacing for image resampling.
        orientation (tuple): Orientation code (e.g., 'SRA').
        exemplar_num (int): Maximum number of exemplars to use.
        exemplar_use_only_prompted (bool): Restrict exemplar memory to user-provided prompts.
        filter_prev_prediction_components (bool): Optionally remove spurious previous predictions.
        overlap_threshold (float): Threshold for region overlap filtering.
        use_low_res_masks_for_com_detection (bool): Use low-res masks to determine center-of-mass.
        default_image_size (int): Default image size used by the model.
        min_lesion_area_threshold (int): Minimum area for considering lesion regions.
        description (str): Description of this inference task.
    """
    def __init__(
        self,
        path,
        config_name,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        spacing=(-1, -1, -1),
        orientation=("SRA"),
        exemplar_num=10,
        exemplar_use_only_prompted=True,
        filter_prev_prediction_components = False,
        overlap_threshold=0.5,
        use_low_res_masks_for_com_detection=True,
        default_image_size=1024,
        min_lesion_area_threshold=40,
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
        
        set_determinism(42)
        self.spacing = spacing
        self.orientation = orientation
        self.exemplar_num = exemplar_num
        self.exemplar_use_only_prompted = exemplar_use_only_prompted
        self.filter_prev_prediction_components = filter_prev_prediction_components
        self.overlap_threshold = overlap_threshold
        self.use_low_res_masks_for_com_detection = use_low_res_masks_for_com_detection
        self.default_image_size = default_image_size
        self.min_lesion_area_threshold = min_lesion_area_threshold
        
        if self.use_low_res_masks_for_com_detection:
            # Set the scale factor to transform the center of a lesion mass 
            # from low res 256x256 mask to the original 1024x1024.
            self.com_scale_coefficient = 4
        else:
            self.com_scale_coefficient = 1

        self.path = path
        model_dir = os.path.dirname(path)
        self.config_name = config_name
        
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=model_dir)
        
        self.predictors = {}
        self.image_cache = {}
        self.inference_state = None
        
    def pre_transforms(self, data=None) -> Sequence[Callable]: 
        """Define the sequence of preprocessing transforms for input data."""
        transforms = [
            LoadImaged(keys=["image"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=self.orientation),
            TransformPointsd(keys=["foreground", "background"], 
                            ref_image_key="image", 
                            orientation=self.orientation, 
                            flip_axes=[0, 1]),
            Flipd(["image"], spatial_axis=0),
            Flipd(["image"], spatial_axis=1),
            ScaleIntensityRangePercentilesIgnoreZerod(keys="image", lower=0.5, upper=99.5, 
                                                      out_min=0, out_max=255),
            SqueezeDimd(keys="image", dim=0)
        ]
        self.add_cache_transform(transforms, data)
        transforms.append(EnsureTyped(keys=["image"], device="cpu"))
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
            Flipd(["image"], spatial_axis=1),
            Flipd(["image"], spatial_axis=0),
            ReorientToOriginald(keys="pred", ref_image="image"),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
        ]
        return transforms
    
    def get_predictor(self):
        """
        Initialize and return the MOIS-SAM2 predictor. Caches per device.
        Applies optimization settings for CUDA-capable GPUs.
        """
        predictor = self.predictors.get(self.device)
        
        if predictor is None:
            logger.info(f"Using Device: {self.device}")
            device_t = torch.device(self.device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            predictor = build_mois_sam2_predictor(self.config_name, self.path, device=self.device)
            self.predictors[self.device] = predictor
            predictor.num_max_exemplars = self.exemplar_num
            predictor.exemplar_use_only_prompted = self.exemplar_use_only_prompted
            predictor.use_low_res_masks_for_com_detection = self.use_low_res_masks_for_com_detection
            predictor.com_scale_coefficient = self.com_scale_coefficient
        return predictor   

    def move_to_cpu(self, obj):
        """Recursively move tensors in nested structure to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self.move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.move_to_cpu(v) for v in obj)
        else:
            return obj
    
    def move_to_device(self, obj, device):
        """Recursively move tensors in nested structure to a given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_device(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.move_to_device(v, device) for v in obj)
        else:
            return obj
    
    def run_inferer(self, image_tensor, set_image_state, request, data, debug=False):
        """
        Perform the main inference logic:
        - Load/prepare video directory
        - Restore or initialize inference state
        - Run initial segmentation from user interactions
        - Apply forward and backward propagation
        - Merge outputs and update the data dictionary

        Args:
            image_tensor (MetaTensor): Input 3D image.
            set_image_state (bool): Whether to reset model state.
            request (dict): MONAI Label interaction request.
            data (dict): Internal data state.
            debug (bool): Save intermediate images for debugging.

        Returns:
            dict: Updated data with predicted mask and metadata.
        """
        self.device = name_to_device(request.get("device", "cuda"))
        reset_state = strtobool(request.get("reset_state", "false"))
        
        instance_label = request.get("label", "lesion_1")
        if "lesion" in instance_label:
            current_instance_id = int(instance_label.split("_")[-1])
        else:
            current_instance_id = 0
        
        predictor = self.get_predictor()
        
        image_path = request["image"]
        video_dir = os.path.join(
            image_cache.cache_path, get_basename_no_ext(image_path) if debug else md5_digest(image_path)
        )
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            for slice_idx in tqdm(range(image_tensor.shape[-1])):
                slice_np = image_tensor[:, :, slice_idx].numpy()
                slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")

                if strtobool(request.get("pylab")):
                    pylab.imsave(slice_file, slice_np, format="jpg", cmap="Greys_r")
                else:
                    Image.fromarray(slice_np).convert("RGB").save(slice_file)
            logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices; {video_dir}")
        
        # Set Expiry Time
        image_cache.cached_dirs[video_dir] = time() + image_cache.cache_expiry_sec
        
        if reset_state or set_image_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
            self.inference_state = predictor.init_state(video_path=video_dir)
        
        exemplars_path = os.path.join(video_dir, "exemplars.pt")
        if os.path.exists(exemplars_path):
            try:
                exemplars = torch.load(exemplars_path, map_location="cpu")
                exemplars = self.move_to_device(exemplars, self.device)
                self.inference_state["exemplars"] = exemplars
                predictor.exemplars_dict = exemplars
                logger.info(f"Restored exemplars from: {exemplars_path}")
            except Exception as e:
                logger.info(f"Failed to load exemplars: {e}")
            
        logger.info(f"Image Shape: {image_tensor.shape}")
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        for key in {"foreground", "background"}:
            for p in request[key]:
                sid = p[2] 
                sids.add(sid)
                kps = fps if key == "foreground" else bps
                if kps.get(sid):
                    kps[sid].append([p[0], p[1]]) 
                else:
                    kps[sid] = [[p[0], p[1]]]

        pred = np.zeros(tuple(image_tensor.shape))
        for sid in sorted(sids):
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])

            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords] # Flip x,y => y,x
            point_labels = [1] * len(fp) + [0] * len(bp)
            
            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=current_instance_id,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                box=None,
                is_com=False, # In this case the interaction points are not obtained via exemplars
                add_exemplar=True 
            )
            pred[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # Forward-slice memory-based propagation + add non-prompted exemplars
        pred_forward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, 
                                                                                        current_instance_id, 
                                                                                        max_frame_num_to_track=3,
                                                                                        add_exemplar=True):
            pred_forward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        #  Backward-slice memory-based propagation + add non-prompted exemplars
        pred_backward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, 
                                                                                        current_instance_id, 
                                                                                        max_frame_num_to_track=3,
                                                                                        reverse=True, add_exemplar=True):
            pred_backward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
                
        # Merge forward and backward propagation
        pred_prop = np.logical_or(pred_forward, pred_backward).astype(np.uint8)
        pred = np.logical_or(pred, pred_prop).astype(np.uint8)        
        pred = torch.from_numpy(pred)
        meta = image_tensor.meta
        pred = MetaTensor(pred, meta=meta)

        data["pred"] = pred
        data["image_path"] = request["image"]
        data["image"] = image_tensor
        
        exemplar_path = os.path.join(video_dir, "exemplars.pt")
        cpu_state = self.move_to_cpu(self.inference_state["exemplars"])
        torch.save(cpu_state, exemplar_path)
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
        image_path = request["image"]
        image_tensor = self.image_cache.get(image_path)
        set_image_state = False
        cache_image = request.get("cache_image", True)
        
        if "foreground" not in request:
            request["foreground"] = []
        else:
            request["foreground"] = data["foreground"]
        if "background" not in request:
            request["background"] = []
        else:
            request["background"] = data["background"]
        if "roi" not in request:
            request["roi"] = []
        
        if not cache_image or image_tensor is None:
            # TODO:: Fix this to cache more than one image session
            self.image_cache.clear()
            image_tensor = data["image"]
            self.image_cache[image_path] = image_tensor
            set_image_state = True
                            
        data = self.run_inferer(image_tensor, set_image_state, request, data)
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
