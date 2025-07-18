import logging
import os
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
import lib.infers.mois_sam_interaction

logger = logging.getLogger(__name__)


class MOISSAM_Interaction(TaskConfig):
    """
    Task configuration for MOIS-SAM2 interactive segmentation model.

    This class configures and initializes the parameters for the MOIS-SAM2 model,
    an exemplar-based interactive segmentation model designed for multi-lesion detection.

    Attributes:
        path (str): Filesystem path to the model checkpoint.
        config_name (str): Configuration file name used to initialize the model.
        labels (dict): Dictionary mapping label names to integer IDs.
        network (Any): Placeholder for the model network, which is loaded in the inferer.
        dimension (int): Image dimensionality (3D).
        spacing (tuple): Spacing to be used; (-1, -1, -1) implies using original image spacing.
        orientation (tuple): Orientation string (e.g., "SRA").
        exemplar_num (int): Number of exemplars to use for memory.
        exemplar_use_only_prompted (bool): Whether to use only user-prompted exemplars.
        filter_prev_prediction_components (bool): Whether to filter previous prediction components.
        overlap_threshold (float): Threshold for overlap-based filtering.
        use_low_res_masks_for_com_detection (bool): Whether to use low-resolution masks to find lesion centers.
        default_image_size (int): Default image size for resizing during inference.
        min_lesion_area_threshold (int): Minimum lesion area (in pixels) to be considered valid.
    """
    
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        """
        Initialize the task configuration with custom parameters for MOIS-SAM2.

        Args:
            name (str): Name of the task.
            model_dir (str): Directory where the model and checkpoints are stored.
            conf (dict): Configuration dictionary.
            planner (Any): Task planner.
            **kwargs: Additional keyword arguments.
        """
        super().init(name, model_dir, conf, planner, **kwargs)
        
        # Parameters of the MOIS-SAM2 interactive model
        num_lesion_instances = 100 # Number of lesions to support as labels
        self.labels = {"background": 0}
        for lesion_id in range(1, num_lesion_instances+1):
            self.labels[f"lesion_{lesion_id}"] = lesion_id
        
        # Path to the model checkpoint and configuration
        self.path = os.path.join(self.model_dir, "MOIS_SAM2", "checkpoint.pt")
        self.config_name = "mois_sam2.1_hiera_b+.yaml"
        
        # Inference configuration
        self.network = None # Will be set by the inferer
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
        """
        Construct and return the inference task for the MOIS-SAM2 model.

        Returns:
            dict: Dictionary mapping task name to an instance of MOISSAM_Interaction inferer.
        """
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
        