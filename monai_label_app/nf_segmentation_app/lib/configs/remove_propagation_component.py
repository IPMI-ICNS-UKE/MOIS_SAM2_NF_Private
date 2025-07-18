import logging
import os
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
import lib.infers.remove_propagation_component 

logger = logging.getLogger(__name__)


class RemovePropagationComponent(TaskConfig):
    """
    Task configuration for post-processing to remove propagation components.

    This class configures a MONAI Label task that does not use a model or network,
    but instead applies logic to remove certain components (e.g. false positives or artifacts)
    from propagated neurofibroma segmentations.

    Attributes:
        labels (dict): Label mapping with only background and propagated neurofibromas.
        path (str): Dummy path (no model used).
        network (Any): Set to None, as no neural network is used.
        dimension (int): Indicates that data is 3D.
        orientation (tuple): Image orientation, e.g., ("SRA").
    """
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """
        Initialize the task configuration.

        Args:
            name (str): Name of the task.
            model_dir (str): Directory used for context (no model file required).
            conf (dict): Configuration dictionary.
            planner (Any): Task planner used by MONAI Label.
            **kwargs: Additional keyword arguments.
        """
        super().init(name, model_dir, conf, planner, **kwargs)
        self.labels = {"background": 0, "propagated_neurofibromas": 1}
        self.path = self.model_dir # No model path needed; just using base directory
        self.network = None # No network involved in this task
        self.dimension = 3
        self.orientation=("SRA")
        
        logger = logging.getLogger(__name__)
        
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create and return the infer task for removing propagation components.

        Returns:
            dict: Dictionary mapping task name to RemovePropagationComponent inferer.
        """
        inferer = lib.infers.remove_propagation_component.RemovePropagationComponent(
            path=self.path,
            network=self.network,
            labels=self.labels,
            dimension=self.dimension, # Defines the propagation in 3D
            orientation=self.orientation,
        )
        return {
            self.name: inferer,
        }
    
    def trainer(self) -> Optional[TrainTask]:
        return None
