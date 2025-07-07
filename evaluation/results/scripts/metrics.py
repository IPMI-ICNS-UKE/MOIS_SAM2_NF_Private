import numpy as np

def compute_detection_and_lesion_metrics(gt_labeled, pred_labeled, matched_pairs):
    """
    Compute lesion detection and lesion-level Dice metrics.

    Args:
        gt_labeled (np.ndarray): Labeled ground truth mask.
        pred_labeled (np.ndarray): Labeled predicted mask.
        matched_pairs (List[Tuple[int, int]]): Matched lesion label pairs.

    Returns:
        detection_metrics (dict): Precision, recall, F1, TP, FP, FN.
        lesion_dscs (List[float]): Dice scores of matched lesions.
    """
    tp = len(matched_pairs)
    gt_labels = set(np.unique(gt_labeled)) - {0}
    pred_labels = set(np.unique(pred_labeled)) - {0}
    fn = len(gt_labels) - tp
    fp = len(pred_labels) - tp

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

    lesion_dscs = []
    for gt_label, pred_label in matched_pairs:
        gt_mask = (gt_labeled == gt_label)
        pred_mask = (pred_labeled == pred_label)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = gt_mask.sum() + pred_mask.sum()
        dice = (2 * intersection) / union if union > 0 else 0.0
        lesion_dscs.append(dice)

    return detection_metrics, lesion_dscs


def compute_scan_dice(gt_mask, pred_mask):
    """
    Compute Dice similarity coefficient over the full scan.

    Args:
        gt_mask (np.ndarray): Binary ground truth mask.
        pred_mask (np.ndarray): Binary predicted mask.

    Returns:
        float: Dice coefficient.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()
    union = gt_sum + pred_sum

    if union == 0:
        return 1.0  # Both masks empty
    else:
        return (2 * intersection) / union