import logging
import os
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
import lib.infers.mois_sam_interaction

logger = logging.getLogger(__name__)


class MOISSAM_Interaction(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        
        # Parameters of the MOIS-SAM2 interactive model
        num_lesion_instances = 100
        self.labels = {"background": 0}
        
        for lesion_id in range(1, num_lesion_instances+1):
            self.labels[f"lesion_{lesion_id}"] = lesion_id
            
        self.path = os.path.join(self.model_dir, "MOIS_SAM2", "checkpoint.pt")
        self.config_name = "mois_sam2.1_hiera_b+.yaml"
        
        self.network = None # The MOIS-SAM2 model is loaded in the inferer
        self.dimension=3
        self.spacing=(-1, -1, -1)
        self.orientation=("SRA")
        self.exemplar_num=10
        self.exemplar_use_only_prompted=True
        self.filter_prev_prediction_components = False
        self.overlap_threshold=0.5
        self.use_low_res_masks_for_com_detection=True
        self.default_image_size=1024
        self.min_lesion_area_threshold = 40
        
        logger.info("Initialized MOISSAM_Interaction")
        
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        inferer = lib.infers.mois_sam_interaction.MOISSAM_Interaction(
            path=self.path,
            config_name=self.config_name,
            network=self.network,
            labels=self.labels,
            dimension=self.dimension,
            spacing=self.spacing,
            orientation=self.orientation,
            exemplar_num=self.exemplar_num,
            exemplar_use_only_prompted=self.exemplar_use_only_prompted,
            filter_prev_prediction_components = self.filter_prev_prediction_components,
            overlap_threshold=self.overlap_threshold,
            use_low_res_masks_for_com_detection=self.use_low_res_masks_for_com_detection,
            default_image_size=self.default_image_size,
            min_lesion_area_threshold=self.min_lesion_area_threshold,
            
        )
        return {
            self.name: inferer,
        }
    
    def trainer(self) -> Optional[TrainTask]:
        return None
        