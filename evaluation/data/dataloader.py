from monai.data import DataLoader, Dataset
from typing import List
import logging
import glob
import os

logger = logging.getLogger("evaluation_pipeline_logger")


def get_evaluation_datalist(args):
    """
    Retrieves a list of test cases for evaluation.

    This function searches for image and label files in the `imagesTs_<test_set_id>` and `labelsTs_<test_set_id>`
    directories inside the `args.input_dir`. The function then pairs them and creates a datalist containing
    dictionary entries for each case.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
           
    Returns:
        List[dict]: A list where each element is a dictionary containing:
            - `"image"` (str): Path to the test image file.
            - `"label"` (str): Path to the corresponding label file.
            - `"connected_component_label"` (str): Path to the label file (same as `"label"` for consistency).
    """
    logger.info(f"Getting data list from: {os.path.join(args.input_dir, f'images{args.data_folder_prefix}_{args.test_set_id}')}")
    
    # Get all available images and labels
    image_files = sorted(glob.glob(os.path.join(args.input_dir, f"images{args.data_folder_prefix}_{args.test_set_id}", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(args.input_dir, f"labels{args.data_folder_prefix}_{args.test_set_id}", "*.nii.gz")))
    
    # Build the train and validation datasets
    datalist = [
        {"image": image_name, "label": label_name, "connected_component_label": label_name}
        for image_name, label_name in zip(image_files, label_files)
    ]
    logger.info(f"Number of evaluation cases: {len(datalist)}")
    
    if args.limit:
        logger.info(f"Limiting the evaluation cases to: {args.limit}")
        datalist = datalist[0 : args.limit]
        
    return datalist


def get_evaluation_data_loader(args, pre_transforms):
    """
    Creates a DataLoader for the evaluation dataset.

    This function retrieves the evaluation datalist using `get_evaluation_datalist()`, applies the necessary
    preprocessing transformations, and loads the dataset into a MONAI `DataLoader` for evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        pre_transforms (monai.transforms.Compose): Preprocessing transformations to be applied to the dataset.

    Returns:
        monai.data.DataLoader: A DataLoader object for iterating over the evaluation dataset.
    """
    datalist = get_evaluation_datalist(args)
    
    if not len(datalist):
        raise ValueError("No valid data found..")

    dataset = Dataset(datalist, pre_transforms)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    return data_loader
