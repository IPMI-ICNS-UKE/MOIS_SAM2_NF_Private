# Interactive Segmentation Model Evaluation

This directory contains the evaluation framework for interactive segmentation models. The developed unified evaluation pipeline supports multiple models and interaction scenarios. It also provides a set of bash scripts for automated evaluation of different models across multiple test cases.

## Directory Structure

```
.
├── cache                  # Stores temporary cached data
├── config                 # Configuration files and argument parser
├── data                   # Data loading scripts
├── experiment_launchers   # Bash scripts to launch evaluations
├── interaction            # Code for interaction modeling
├── logs                   # Logging directory
├── networks               # Model architecture definitions and inferer setup
├── pipelines              # Evaluation pipeline implementation
├── results                # Output results, metrics, and logs
└── README.MD              # This documentation file
```


## Evaluation Pipeline Overview

The evaluation pipeline supports two interaction scenarios:

1. **Lesion-wise Corrective**: Interactive segmentation is performed with a focus on individual lesions, applying corrections iteratively.
2. **Global (Scan-wise) Corrective**: Interactive segmentation is applied to the entire scan, incorporating global interactions.


### Implemented Models

The following interactive segmentation models are supported:
- **DINs** (Deep Interactive Networks)
- **SW-FastEdit** (Sliding Window FastEdit)
- **SAM2** (Segment Anything Model v2)

# Running Evaluations

### 1. Bash Scripts for Automated Evaluation

The `experiment_launchers/` directory contains multiple shell scripts to trigger evaluation runs for all implemented models across the two interaction scenarios.

#### Example:
To launch the evaluation for **DINs** using the lesion-wise approach:
```bash
bash experiment_launchers/launch_DINs_lesion_wise_corrective.sh
```

To launch the evaluation for **SW-FastEdit** in the global scenario:
```bash
bash experiment_launchers/launch_SW-FastEdit_global_corrective.sh
```

### 2. Unified Evaluation Pipeline

The main evaluation pipeline script is:
```
pipelines/evaluation_pipeline.py
```
It integrates dataset preprocessing, interaction simulation, inference, and performance metric computation.

### 3. Results and Metrics

Evaluation outputs are stored in:
```
results/metrics/
```
Each test case is saved within its respective model folder, following this structure:
```
results/metrics/{model}/{interaction_scenario}/TestSet_{id}/fold_{n}
```
- **global_metrics.xlsx**: Contains scan-wise segmentation performance scores.
- **lesion_metrics.xlsx**: Contains per-lesion segmentation performance scores.

## How to Add a New Model

1. Define the model in `networks/custom_networks.py`.
2. Implement an inferer in `networks/get_inferers.py`.
3. Add the model name to the argument parser in `config/argparser.py`.
4. Create a corresponding launch script in `experiment_launchers/`.
