import csv
from pathlib import Path
import numpy as np


def export_case_results_to_csv(experiment_results, output_csv_path, include_lesion_dscs=False):
    """
    Export all per-case results to a CSV file.

    Args:
        experiment_results (dict): Output from evaluate_experiment.
        output_csv_path (str or Path): Where to save CSV.
        include_lesion_dscs (bool): Whether to include full lesion DSC lists.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'dataset', 'case_name', 'scan_dice', 'f1_score',
        'true_positives', 'false_positives', 'false_negatives',
        'num_matched_lesions', 'mean_lesion_dice'
    ]
    if include_lesion_dscs:
        fieldnames.append('lesion_dice_list')

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name, dataset_info in experiment_results.items():
            for case in dataset_info['cases']:
                lesion_dscs = case['lesion_dscs']
                row = {
                    'dataset': dataset_name,
                    'case_name': case['case_name'],
                    'scan_dice': case['scan_dice'],
                    'f1_score': case['detection_metrics']['f1_score'],
                    'true_positives': case['detection_metrics']['true_positives'],
                    'false_positives': case['detection_metrics']['false_positives'],
                    'false_negatives': case['detection_metrics']['false_negatives'],
                    'num_matched_lesions': len(lesion_dscs),
                    'mean_lesion_dice': np.mean(lesion_dscs) if lesion_dscs else None
                }
                if include_lesion_dscs:
                    row['lesion_dice_list'] = str(lesion_dscs)
                writer.writerow(row)
