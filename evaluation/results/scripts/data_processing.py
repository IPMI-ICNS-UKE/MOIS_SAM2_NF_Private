from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from skimage.measure import regionprops
from joblib import Parallel, delayed
from skimage.morphology import remove_small_objects


def load_binary_mask(file_path, threshold=0.5, min_volume_cm3=0.0):
    """
    Load a binary mask from NIfTI and filter out small connected components by volume (in cm³).

    Args:
        file_path (str or Path): Path to the .nii.gz file.
        threshold (float): Binarization threshold.
        min_volume_cm3 (float): Minimum component volume to keep (in cm³).

    Returns:
        np.ndarray: Filtered binary mask (dtype=uint8).
    """
    img = nib.load(str(file_path))
    data = img.get_fdata()
    header = img.header

    binary = (data >= threshold).astype(bool)

    if min_volume_cm3 > 0.0:
        # Voxel volume in mm³
        voxel_spacing = header.get_zooms()
        voxel_volume_mm3 = np.prod(voxel_spacing)
        volume_threshold_voxels = int((min_volume_cm3 * 1000) / voxel_volume_mm3)

        # Label connected components
        labeled_mask, _ = label(binary)

        # Remove small objects
        filtered_labeled = remove_small_objects(
            labeled_mask, min_size=volume_threshold_voxels, connectivity=1
        )

        return (filtered_labeled > 0).astype(np.uint8)

    return binary.astype(np.uint8)


def extract_connected_components(binary_mask):
    """
    Extract 3D connected components (lesions) from a binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask (3D).

    Returns:
        labeled_mask (np.ndarray): Mask with labeled components (0 = background).
        num_components (int): Number of detected components.
    """
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labeled_mask, num_components = label(binary_mask, structure=structure)
    return labeled_mask, num_components


def match_lesions(gt_mask_labeled, pred_mask_labeled, iou_threshold=0.1):
    """
    Match lesions between ground truth and prediction based on IoU.

    Args:
        gt_mask_labeled (np.ndarray): Labeled ground truth mask.
        pred_mask_labeled (np.ndarray): Labeled predicted mask.
        iou_threshold (float): IoU threshold for detection.

    Returns:
        matched_pairs (List[Tuple[int, int]]): List of (gt_label, pred_label).
        unmatched_gt (Set[int]): Unmatched ground truth labels.
        unmatched_pred (Set[int]): Unmatched prediction labels.
    """
    matched_pairs = []
    candidate_matches = []

    gt_labels = set(np.unique(gt_mask_labeled)) - {0}
    pred_labels = set(np.unique(pred_mask_labeled)) - {0}
    
    print(len(gt_labels))
    print(len(pred_labels))

    for gt_label in gt_labels:
        gt_mask = (gt_mask_labeled == gt_label)
        for pred_label in pred_labels:
            pred_mask = (pred_mask_labeled == pred_label)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union
            if iou >= iou_threshold:
                candidate_matches.append((iou, gt_label, pred_label))

    # Sort by IoU descending for greedy matching
    candidate_matches.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for iou, gt_label, pred_label in candidate_matches:
        if gt_label not in matched_gt and pred_label not in matched_pred:
            matched_pairs.append((gt_label, pred_label))
            matched_gt.add(gt_label)
            matched_pred.add(pred_label)

    unmatched_gt = gt_labels - matched_gt
    unmatched_pred = pred_labels - matched_pred

    return matched_pairs, unmatched_gt, unmatched_pred
