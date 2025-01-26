import logging
import os
from typing import Callable, Dict, Sequence, Union

import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.utils.enums import CommonKeys
from monai.utils.enums import CommonKeys as Keys
from monai.metrics import compute_dice

from evaluation.transforms.custom_transforms import ClickGenerationStrategy

logger = logging.getLogger("evaluation_pipeline_logger")


def create_dataframe_from_dict(data_dict):
    max_length = max(len(lst) for lst in data_dict.values())
    for key, value in data_dict.items():
        data_dict[key] = value + [np.nan] * (max_length - len(value))
    return pd.DataFrame(data_dict)

class Interaction:
    '''
    ToDO: Add documentation
    '''
    def __init__(
        self,
        args,
        pre_transforms: Union[Sequence[Callable], Callable],
        post_transforms: Union[Sequence[Callable], Callable],
        train: bool,
        num_instances_to_correct: int = None,
        num_interactions_local_max: int = None,
        num_interactions_total_max: int = None,
        dsc_local_max: float = None,
        dsc_global_max: float = None,
        label_names: Union[None, Dict[str, int]] = None,
        click_probability_key: str = "probability",
        click_generation_strategy_key: str = "click_generation_strategy",
        deepgrow_probability: float = 1.0,
        click_generation_strategy: ClickGenerationStrategy = ClickGenerationStrategy.PATCH_BASED_CORRECTIVE,
    ):
        self.args = args
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.train = train
        self.label_names = label_names
        self.deepgrow_probability = deepgrow_probability
        
        self.click_probability_key = click_probability_key
        self.click_generation_strategy_key = click_generation_strategy_key
        self.click_generation_strategy = click_generation_strategy
        
        self.num_instances_to_correct = num_instances_to_correct
        self.num_interactions_local_max = num_interactions_local_max
        self.num_interactions_total_max = num_interactions_total_max
        
        self.dsc_local_max = dsc_local_max
        self.dsc_global_max = dsc_global_max
    
    def __call__(
        self,
        engine: Union[SupervisedTrainer, SupervisedEvaluator],
        batchdata: Dict[str, torch.Tensor],
    ):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        
        device = engine.state.device
      
        # Get data "image", "label", "connected_component_label"
        case_name = os.path.basename(batchdata["image"].meta["filename_or_obj"][0])
        label = batchdata["label"].detach().clone()
        cc_label = batchdata["connected_component_label"].detach().clone()
                
        # Create a global prediction placeholder
        prediction_global = torch.zeros_like(cc_label)
        prediction_global_after_single_interaction = torch.zeros_like(cc_label)
        
        # Initialize global counters
        dsc_global = [0, ]
        dsc_global_after_single_interaction = [0, ]
        num_interactions_total = [0, ]
        dsc_instance_dict = {}
        max_number_lesions_achieved = False
        
        logger.info(f"Starting a new case: {case_name}")  
        while ((dsc_global[-1] < self.dsc_global_max) and 
               (num_interactions_total[-1] < self.num_interactions_total_max) and
               (not max_number_lesions_achieved)
               ):
            
            logger.info(f"> Starting global interaction, current total number of interactions: {num_interactions_total[-1]}")
            num_interactions_per_instance = 0
            
            # Perform operation per instance, instance_id == 0 stands for background.
            # In case of a semantic mask there is a single instance representing all lesions.
            for instance_id in range(1, self.num_instances_to_correct+1): # Start a new lesion
                new_image_flag = True
                
                
                logger.info(f">> Starting local interaction cycle for instance: {instance_id}")
                
                # Initialize local counters
                dsc_local = [0, ]
                num_interactions_local = [0, ]
                
                # Clear guidances for this instance
                for key in self.args.labels.keys():
                    if key in batchdata:
                        del batchdata[key]
                
                # Get instance mask
                # If there is no instance_id lesion in the cc_label, then we skip this instance
                if not torch.any(cc_label == instance_id):
                    logger.info(f">> Skipping instance {instance_id} as it is not present in cc_label.")
                    max_number_lesions_achieved = True
                    continue
                
                cc_label_local = (cc_label == instance_id).int()
                batchdata["connected_component_label_local"] = cc_label_local
                
                # Create a local prediction placeholder
                prediction_local = torch.zeros_like(cc_label_local)
                prediction_after_single_interaction = torch.zeros_like(cc_label_local)
                batchdata["pred_local"] = prediction_local
                
                while ((dsc_local[-1] < self.dsc_local_max) and 
                       (num_interactions_local[-1] < self.num_interactions_local_max)): # Start a new interaction
                    # Get discrepancy mask and interaction point
                    batchdata_list = decollate_batch(batchdata)
                    for i in range(len(batchdata_list)):
                        batchdata_list[i][self.click_probability_key] = self.deepgrow_probability
                        batchdata_list[i][self.click_generation_strategy_key] = self.click_generation_strategy.value
                        batchdata_list[i] = self.pre_transforms(batchdata_list[i])
                    batchdata = list_data_collate(batchdata_list)
                    
                    # Interaction was done - increase the counter
                    num_interactions_local.append(num_interactions_local[-1] + 1)
                    
                    # Build input dictionary for the inferer
                    logger.info(f">>> Guidance: {batchdata['lesion'], batchdata['background']}")
                    inputs = {
                        "image": batchdata["image"].to(device),
                        "guidance": {key: batchdata[key] for key in self.args.labels.keys()},
                        "previous_prediction": batchdata["pred_local"].to(device),
                        "case_name": case_name,
                        "reset_state": new_image_flag
                    }
                    
                    # Perform inference and get the local prediction
                    engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                    if not self.args.non_standard_network:
                        engine.network.eval()
                    
                    # Forward Pass
                    with torch.no_grad():
                        prediction_local = engine.inferer(inputs, engine.network)
                    batchdata["pred_local"] = prediction_local
                    new_image_flag = False
                    
                    # Apply post-processing for the local prediction
                    batchdata_list = decollate_batch(batchdata)
                    for i in range(len(batchdata_list)):
                        batchdata_list[i] = self.post_transforms(batchdata_list[i])
                    batchdata = list_data_collate(batchdata_list)
                    
                    # Save the prediction after a single interaction
                    if num_interactions_local[-1] == 1:
                        prediction_after_single_interaction = batchdata["pred_local"].detach().clone()
                    
                    # Calculate local DSC
                    dsc_local_current = compute_dice(y_pred=batchdata["pred_local"],
                                                     y=batchdata["connected_component_label_local"],
                                                     include_background=self.args.include_background_in_metric
                                                     )
                    
                    # Append local DSC to the list of local DSC
                    dsc_local.append(dsc_local_current.item())
                
                # Finished local corrections
                logger.info(f">> Finished local interaction cycle: local DSC {dsc_local[-1]}, {num_interactions_local[-1]} interactions")
                
                # Add number of local interactions for this counter to the num_interactions_total
                num_interactions_per_instance = num_interactions_per_instance + num_interactions_local[-1]
                
                # Add dice_local to the dictionary
                dsc_instance_dict[instance_id] = dsc_local
                
                # Insert the local prediction to global prediction, if the evaluation was done lesion-wise
                if self.args.evaluation_mode != "global_corrective":
                    prediction_global = torch.max(prediction_global, batchdata["pred_local"])
                    prediction_global_after_single_interaction = torch.max(prediction_global_after_single_interaction, prediction_after_single_interaction)
            
            # If the evaluation was done globally, consider the last local interaction result as global
            if self.args.evaluation_mode == "global_corrective":
                prediction_global = batchdata["pred_local"]
                prediction_global_after_single_interaction = prediction_after_single_interaction
                
            # Get global DSC and append it
            dsc_global_current_after_single_interaction = compute_dice(
                y_pred=prediction_global_after_single_interaction,
                y=label,
                include_background=self.args.include_background_in_metric
                )
            dsc_global_current = compute_dice(
                y_pred=prediction_global,
                y=label,
                include_background=self.args.include_background_in_metric
                )
            dsc_global.append(dsc_global_current.item())
            dsc_global_after_single_interaction.append(dsc_global_current_after_single_interaction.item())
            
            # Increase the total number of interactions accordingly
            num_interactions_total.append(num_interactions_total[-1] + num_interactions_per_instance)
            logger.info(f"> Finished global interaction cycle: global DSC {dsc_global[-1]}, total interactions {num_interactions_total[-1]}")
        
        batchdata[CommonKeys.PRED] = prediction_global
        logger.info(f"Finished case. Resulting global DSC: {dsc_global[-1]}. Total interactions {num_interactions_total[-1]}.")
        
        # Prepare directory for saving the reports
        metrics_output_directory = os.path.join(
            self.args.results_dir, "metrics", self.args.network_type, self.args.evaluation_mode,
            f"TestSet_{self.args.test_set_id}", f"fold_{self.args.fold}", case_name
        )
        if not os.path.exists(metrics_output_directory):
            os.makedirs(metrics_output_directory)
        
        # Create pandas dataframe with dsc_local and save it to the specified location as an excel file
        df_lesion_level = create_dataframe_from_dict(dsc_instance_dict)
        df_lesion_level.to_excel(
            os.path.join(metrics_output_directory, "lesion_metrics.xlsx"),
            index=False
        )
        
        # Create pandas dataframe with dsc_global and save it to the specified location as an excel file
        df_global_level_after_single_interaction = pd.DataFrame({"interaction_total": [0, 1], "dsc_global": dsc_global_after_single_interaction})
        df_global_level_after_single_interaction.to_excel(
            os.path.join(metrics_output_directory, "global_metrics_after_single_interaction.xlsx"),
            index=False
        )
        
        # Create pandas dataframe with dsc_global and save it to the specified location as an excel file
        df_global_level = pd.DataFrame({"interaction_total": num_interactions_total, "dsc_global": dsc_global})
        df_global_level.to_excel(
            os.path.join(metrics_output_directory, "global_metrics.xlsx"),
            index=False
        )
        
        engine.state.batch = batchdata
        engine.state.output = {
            Keys.IMAGE: batchdata["image"], 
            Keys.LABEL: batchdata["label"],
            Keys.PRED: batchdata[CommonKeys.PRED],
            }
        
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        
        return engine.state.output
    