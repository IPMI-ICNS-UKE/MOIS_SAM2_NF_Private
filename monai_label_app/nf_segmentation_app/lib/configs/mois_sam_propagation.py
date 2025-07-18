import logging
import os
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
import lib.infers.mois_sam_propagation

logger = logging.getLogger(__name__)


class MOISSAM_Propagation3D(TaskConfig):
    """
        Task configuration for MOIS-SAM2 propagation model in 3D.

        This task uses a propagation-based inference strategy for neurofibroma segmentation,
        leveraging the exemplars.

        Attributes:
            path (str): Path to the model checkpoint file.
            config_name (str): YAML configuration file for the network.
            labels (dict): Mapping of label names to integer class IDs.
            network (Any): Placeholder for the model network, loaded during inference.
            dimension (int): Specifies the data dimensionality (3D).
            spacing (tuple): Image spacing to be used for inference (-1 means use original).
            orientation (tuple): Image orientation, e.g., ("SRA").
            exemplar_num (int): Number of exemplars to keep in memory.
            exemplar_use_only_prompted (bool): Whether to use only user-specified exemplars.
            filter_prev_prediction_components (bool): Flag to filter disconnected predictions.
            overlap_threshold (float): Overlap threshold for filtering predictions.
            use_low_res_masks_for_com_detection (bool): Use downsampled masks to detect centers of mass.
            default_image_size (int): Default image size for input resizing.
            min_lesion_area_threshold (int): Minimum area (in pixels) to consider a region as a lesion.
        """
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """
        Initialize the propagation task configuration.

        Args:
            name (str): Task name.
            model_dir (str): Path to the directory containing model files.
            conf (dict): Dictionary of configuration parameters.
            planner (Any): Task planner object.
            **kwargs: Additional keyword arguments.
        """
        super().init(name, model_dir, conf, planner, **kwargs)
        
        # Parameters of the MOIS-SAM2 interactive model
        # Only one foreground class: propagated neurofibromas
        self.labels = {"background": 0, "propagated_neurofibromas": 1}
        
        # Model checkpoint and configuration
        self.path = os.path.join(self.model_dir, "MOIS_SAM2", "checkpoint.pt")
        self.config_name = "mois_sam2.1_hiera_b+.yaml"
        
         # Model and inference configuration
        self.network = None # Will be instantiated in the inferer
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
        
        logger.info("Initialized MOISSAM_Propagation3D")
    
    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        """
        Create and return the inference task for propagation.

        Returns:
            dict: Dictionary mapping task name to MOISSAM_Propagation inferer instance.
        """
        inferer = lib.infers.mois_sam_propagation.MOISSAM_Propagation(
            path=self.path,
            config_name=self.config_name,
            network=self.network,
            labels=self.labels,
            dimension=self.dimension, # Defines the propagation in 3D
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
