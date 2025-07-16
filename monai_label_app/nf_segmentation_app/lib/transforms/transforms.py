from __future__ import annotations

from typing import Hashable, Mapping
import logging

import torch
import numpy as np
from warnings import warn
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.transforms.transform import Transform
from monai.utils import convert_to_tensor, convert_data_type, convert_to_dst_type
from monai.data.meta_obj import get_track_meta
from monai.config import DtypeLike, NdarrayOrTensor
from monai.utils.enums import TransformBackends

import logging
from typing import Optional, Sequence, Union, Mapping, Hashable
from collections import OrderedDict
from scipy.ndimage import binary_dilation
import nibabel as nib
import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, Orientation, Transform
from monai.utils import InterpolateMode, ensure_tuple_rep, TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

logger = logging.getLogger("evaluation_pipeline_logger")
LABELS_KEY = "label_names"

class ScaleIntensityRangePercentilesIgnoreZero(Transform):
    """
    Scale intensities based on the lower and upper percentiles of non-zero values.

    The intensity range between the lower and upper percentiles is linearly scaled to [out_min, out_max].
    Zero-valued voxels are ignored in percentile computation and restored to zero in the final output.

    Args:
        lower: lower percentile (e.g., 0.5).
        upper: upper percentile (e.g., 99.5).
        out_min: target minimum intensity (default: 0.0).
        out_max: target maximum intensity (default: 255.0).
        dtype: output data type (default: np.float32).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        lower: float = 0.5,
        upper: float = 99.5,
        out_min: float = 0.0,
        out_max: float = 255.0,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.out_min = out_min
        self.out_max = out_max
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        ret: NdarrayOrTensor
        
        input_dtype = self.dtype or img.dtype

        # Convert to numpy for percentile calculations
        img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
        nonzero = img_np[img_np > 0]

        if nonzero.size == 0:
            warn("Image has no non-zero voxels. Returning zero image.", UserWarning)
            return convert_data_type(np.zeros_like(img_np), dtype=input_dtype)[0]

        lower_val = np.percentile(nonzero, self.lower)
        upper_val = np.percentile(nonzero, self.upper)

        if upper_val - lower_val == 0.0:
            warn("Divide by zero (percentile range is zero)", UserWarning)
            scaled = img_np - lower_val
        else:
            img_clipped = np.clip(img_np, lower_val, upper_val)
            scaled = (img_clipped - lower_val) / (upper_val - lower_val)
            scaled = scaled * (self.out_max - self.out_min) + self.out_min

        # Keep zeros as zeros
        scaled[img_np == 0] = 0

        # Convert back to MetaTensor if needed
        scaled_tensor = torch.tensor(scaled, dtype=torch.float32, device=img.device if isinstance(img, torch.Tensor) else "cpu")

        ret = convert_to_dst_type(scaled_tensor, dst=img, dtype=self.dtype or img_t.dtype)[0]
        return ret

class ScaleIntensityRangePercentilesIgnoreZerod(MapTransform):
    """
    Dictionary-based wrapper of `ScaleIntensityRangePercentilesIgnoreZero`.

    Args:
        keys: keys of the corresponding items to be transformed.
        lower: lower percentile (e.g., 0.5).
        upper: upper percentile (e.g., 99.5).
        out_min: target minimum intensity.
        out_max: target maximum intensity.
        dtype: output data type. If None, same as input image.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRangePercentilesIgnoreZero.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float = 0.5,
        upper: float = 99.5,
        out_min: float = 0.0,
        out_max: float = 255.0,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRangePercentilesIgnoreZero(lower, upper, out_min, out_max, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d

class ReorientToOriginald(MapTransform):
    """
    A MONAI MapTransform that reorients an image back to its original orientation using metadata
    from a reference image. This is useful for restoring images that have been resampled or reoriented
    during preprocessing steps.

    Args:
        keys (KeysCollection): Keys of the items to be transformed.
        ref_image (str): The key for the reference image used to restore the original orientation.
        has_channel (bool): Whether the image has a channel dimension (default: True).
        invert_orient (bool): Whether to invert the orientation (default: False).
        mode (str): Interpolation mode for reorientation (default: 'nearest').
        config_labels (Optional[dict]): Optional dictionary to map config labels (default: None).
        align_corners (Optional[Union[Sequence[bool], bool]]): Alignment option for interpolation.
        meta_key_postfix (str): The postfix used for the metadata key (default: 'meta_dict').
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        invert_orient: bool = False,
        mode: str = InterpolateMode.NEAREST,
        config_labels=None,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.invert_orient = invert_orient
        self.config_labels = config_labels
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        """
        Reorient the image to its original orientation using 
        the affine transformation stored in the reference image's metadata.

        Args:
            data (dict): A dictionary containing the image and metadata.

        Returns:
            dict: The input dictionary with the reoriented image and updated metadata.
        """
        d = dict(data)

        # Extract the metadata from the reference image
        meta_dict = (
            d[self.ref_image].meta
            if d.get(self.ref_image) is not None and isinstance(d[self.ref_image], MetaTensor)
            else d.get(f"{self.ref_image}_{self.meta_key_postfix}", {})
        )

        # Loop through each key (image) to apply the inverse transformation
        for _, key in enumerate(self.keys):
            result = d[key]
            
            # Retrieve the original affine matrix for the inverse transformation
            orig_affine = meta_dict.get("original_affine", None)
            if orig_affine is not None:
                orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
                inverse_transform = Orientation(axcodes=orig_axcodes)

                # Apply inverse reorientation
                with inverse_transform.trace_transform(False):
                    result = inverse_transform(result)
            else:
                logger.info("Failed to invert orientation - 'original_affine' not found in image metadata.")

            d[key] = result

            # Update the metadata with the affine of the original image
            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
        return d



class TransformPointsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image_key: str = "image",
        orientation: str = "SRA",  # Target orientation
        flip_axes: list[int] = [0, 1],
    ):
        super().__init__(keys)
        self.ref_image_key = ref_image_key
        self.orientation = orientation
        self.flip_axes = flip_axes

    def __call__(self, data):
        d = dict(data)
        img = d[self.ref_image_key]
        print("LABEL"*10)
        print(img)
        meta = img.meta if hasattr(img, "meta") else img.get("meta", {})

        orig_affine = meta.get("original_affine", meta.get("affine", None))
        spatial_shape = np.array(meta["spatial_shape"])  # shape: (X, Y, Z)
        shape_dict = dict(zip(["x", "y", "z"], spatial_shape))

        # Build orientation transform
        orig_ornt = nib.aff2axcodes(orig_affine)
        tgt_ornt = tuple(self.orientation)
        ornt_transform = nib.orientations.ornt_transform(
            nib.orientations.axcodes2ornt(orig_ornt),
            nib.orientations.axcodes2ornt(tgt_ornt)
        )

        # Example: [[1, -1], [0, 1], [2, -1]]
        axes = ornt_transform[:, 0].astype(int)
        flips = ornt_transform[:, 1].astype(int)

        for key in self.keys:
            transformed = []
            for pt in d.get(key, []):
                pt = np.array(pt, dtype=float)

                # Step 1: reorder axes
                pt = pt[axes]

                # Step 2: apply image flips
                if 0 in self.flip_axes: # Flip Z
                    pt[2] = spatial_shape[2] - 1 - pt[2]
                if 1 in self.flip_axes:
                    pt[0] = spatial_shape[1] - 1 - pt[0]
                if 2 in self.flip_axes:
                    pt[1] = spatial_shape[0] - 1 - pt[1]

                pt = pt.astype(int)
                transformed.append(pt.tolist())
            d[key] = transformed
        return d