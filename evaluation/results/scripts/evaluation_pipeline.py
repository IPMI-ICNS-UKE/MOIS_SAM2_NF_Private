from pathlib import Path
import numpy as np
from data_processing import (load_binary_mask, extract_connected_components, match_lesions)
from metrics import (compute_detection_and_lesion_metrics, compute_scan_dice)
from joblib import Parallel, delayed
from scipy.ndimage import find_objects

def process_tumor(i, instance_mask_src, ref_mask, src_mask, gt_flag):
    """
    Check if lesion i in instance_mask_src overlaps with ref_mask.

    If gt_flag: compute Dice between ground truth lesion and prediction.

    Returns:
        overlap (bool), lesion size, dice score (or None)
    """
    instance = instance_mask_src == i
    bbox = find_objects(instance)[0]
    src_crop = src_mask[bbox]
    ref_crop = ref_mask[bbox] > 0

    overlap = np.any(instance[bbox] & ref_crop)
    
    if gt_flag:
        gt_local = instance[bbox]
        pred_local = ref_mask[bbox]
        masked_pred = gt_local * pred_local
        intersection = masked_pred.sum()
        union = gt_local.sum() + pred_local.sum()
        dice = (2 * intersection / union) if union > 0 else 0.0
        tumor_size = gt_local.sum()
        return overlap, tumor_size, dice
    else:
        return overlap

def evaluate_case_fast(gt_mask, pred_mask):
    """
    Fast lesion-wise evaluation without anatomical information.

    Args:
        gt_mask (np.ndarray): Binary ground truth mask.
        pred_mask (np.ndarray): Binary predicted mask.

    Returns:
        dict: {
            'num_gt_lesions', 'num_pred_lesions',
            'tp', 'fn', 'fp',
            'lesion_dice': List[float],
            'lesion_sizes': List[int]
        }
    """
    instance_mask_gt, num_gt = extract_connected_components(gt_mask)
    instance_mask_pred, num_pred = extract_connected_components(pred_mask)

    # Ground truth lesions (check which are detected by prediction)
    results_gt = Parallel(n_jobs=-1)(
        delayed(process_tumor)(i, instance_mask_gt, pred_mask, gt_mask, gt_flag=True)
        for i in range(1, num_gt + 1)
    )
    tp_gt = np.array([r[0] for r in results_gt])
    dice_scores = [r[2] for r in results_gt]
    tumor_sizes = [r[1] for r in results_gt]

    # Prediction lesions (check which are false positives)
    results_pred = Parallel(n_jobs=-1)(
        delayed(process_tumor)(i, instance_mask_pred, gt_mask, pred_mask, gt_flag=False)
        for i in range(1, num_pred + 1)
    )
    tp_pred = np.array(results_pred)

    tp = int(np.sum(tp_gt))
    fn = int(num_gt - tp)
    fp = int(num_pred - np.sum(tp_pred))

    return {
        'num_gt_lesions': num_gt,
        'num_pred_lesions': num_pred,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'lesion_dice': dice_scores,
        'lesion_sizes': tumor_sizes
    }

def evaluate_case(gt_path, pred_path, size_threshold=10):
    """
    Evaluate a single case: scan-wise Dice, lesion detection, and lesion-wise Dice.

    Args:
        gt_path (str or Path): Path to ground truth .nii.gz file.
        pred_path (str or Path): Path to predicted .nii.gz file.
        iou_threshold (float): (Unused in fast pipeline) — kept for compatibility.

    Returns:
        dict: Evaluation results.
    """
    gt_mask = load_binary_mask(gt_path, min_volume_cm3=size_threshold)
    pred_mask = load_binary_mask(pred_path, min_volume_cm3=size_threshold)

    scan_dice = compute_scan_dice(gt_mask, pred_mask)

    print("--- --- --- Fast lesion-wise evaluation ...")
    lesion_eval = evaluate_case_fast(gt_mask, pred_mask)

    tp = lesion_eval['tp']
    fn = lesion_eval['fn']
    fp = lesion_eval['fp']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    detection_metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return {
        'scan_dice': scan_dice,
        'detection_metrics': detection_metrics,
        'lesion_dscs': lesion_eval['lesion_dice']
    }
    
    
def evaluate_fold(pred_dir, gt_dir, size_threshold=10):
    """
    Evaluate all cases in a fold directory.

    Args:
        pred_dir (str or Path): Directory with predicted .nii.gz files.
        gt_dir (str or Path): Directory with ground truth .nii.gz files.
        iou_threshold (float): IoU threshold for lesion detection.

    Returns:
        List[dict]: Evaluation results for each case.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    results = []
    for pred_file in pred_dir.glob("*.nii.gz"):
        case_name = pred_file.name
        print("--- --- Case: ", case_name)
        gt_file = gt_dir / case_name
        if not gt_file.exists():
            print(f"Warning: Ground truth for {case_name} not found.")
            continue

        case_result = evaluate_case(gt_file, pred_file, 
                                    size_threshold=size_threshold)
        case_result["case_name"] = case_name
        results.append(case_result)

    return results


def aggregate_fold_results(case_results):
    """
    Aggregate metrics from all cases in a fold.

    Args:
        case_results (List[dict]): List of case-level results.

    Returns:
        dict: Aggregated metrics (mean ± std).
    """
    scan_dices = [res['scan_dice'] for res in case_results]
    f1_scores = [res['detection_metrics']['f1_score'] for res in case_results]

    lesion_dscs_all = []
    for res in case_results:
        lesion_dscs_all.extend(res['lesion_dscs'])

    def safe_mean_std(values):
        if len(values) == 0:
            return {'mean': None, 'std': None}
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    return {
        'scan_dice': safe_mean_std(scan_dices),
        'lesion_detection_f1': safe_mean_std(f1_scores),
        'lesion_dice': safe_mean_std(lesion_dscs_all),
        'num_cases': len(case_results),
        'num_lesions': len(lesion_dscs_all)
    }
    

def evaluate_experiment(pred_root, 
                        gt_root, 
                        sets_and_folds, 
                        gt_set_map, 
                        size_threshold=10):
    """
    Evaluate an experiment across multiple test/val sets and folds.

    Args:
        pred_root (str or Path): Path to model prediction root.
        gt_root (str or Path): Path to ground truth root.
        sets_and_folds (dict): Dict of {dataset_name: [fold_ids]}.
        gt_set_map (dict): Maps dataset name to GT folder name.
        iou_threshold (float): IoU threshold for detection.

    Returns:
        dict: Evaluation results (per-case + per-set aggregation).
    """
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    all_results = {}

    for dataset_name, fold_ids in sets_and_folds.items():
        print("Evaluating dataset: ", dataset_name)
        gt_folder = gt_root / gt_set_map[dataset_name]

        set_case_results = []
        for fold in fold_ids:
            print("---fold: ", fold)
            pred_folder = pred_root / dataset_name / f"fold_{fold}"
            fold_case_results = evaluate_fold(pred_folder, gt_folder, 
                                              size_threshold=size_threshold)

            set_case_results.extend(fold_case_results)

        aggregated = aggregate_fold_results(set_case_results)

        all_results[dataset_name] = {
            'cases': set_case_results,
            'aggregated': aggregated
        }

    return all_results