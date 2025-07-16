import logging
import os
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
import lib.infers.remove_propagation_component 

logger = logging.getLogger(__name__)


class RemovePropagationComponent(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        self.labels = {"background": 0, "propagated_neurofibromas": 1}
        self.path = self.model_dir # No model path
        self.network = None # No network
        self.dimension = 3
        self.orientation=("SRA")
        
        logger = logging.getLogger(__name__)
        
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
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