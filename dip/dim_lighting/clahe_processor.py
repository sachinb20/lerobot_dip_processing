#!/usr/bin/env python3
"""
CLAHE Processor for LeRobot observation pipeline.

This processor applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
to enhance dim lighting images during robot data recording.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from lerobot.processor import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


@dataclass
@ProcessorStepRegistry.register(name="clahe_processor")
class CLAHEProcessorStep(ObservationProcessorStep):
    """
    Processor step that applies CLAHE enhancement to camera images.
    
    This processor loads camera-specific CLAHE parameters from a JSON file
    and applies them to enhance images from dim lighting conditions.
    """
    
    params_file: str | Path | None = None
    camera_suffix: str | None = None
    enabled: bool = True
    
    def __post_init__(self):
        """
        Initialize the CLAHE processor after dataclass initialization.
        """
        self.camera_suffix = self.camera_suffix if self.camera_suffix else ""
        
        # Load CLAHE parameters
        if self.params_file is None:
            params_file = Path(__file__).parent / "output" / "clahe_params.json"
        else:
            params_file = Path(self.params_file)
        
        self.params = {}
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.params = json.load(f)
            logging.info(f"✓ Loaded CLAHE parameters from {params_file}")
            for camera_name, params in self.params.items():
                logging.info(f"  {camera_name}: clip_limit={params['clip_limit']}, "
                           f"tile_grid_size={params['tile_grid_size']}")
            if self.camera_suffix:
                logging.info(f"  Will process cameras with suffix: '{self.camera_suffix}'")
            else:
                logging.info(f"  Will process cameras matching parameter names directly")
        else:
            logging.warning(f"CLAHE parameters file not found: {params_file}")
            logging.warning("CLAHE enhancement will be disabled")
            self.enabled = False
    
    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE to an image.
        
        Args:
            image: Input image (BGR format)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def get_camera_base_name(self, camera_name: str) -> str:
        """
        Extract the base camera name by removing the suffix.
        
        Args:
            camera_name: Full camera name (e.g., "side_dim", "front_dim")
        
        Returns:
            Base camera name (e.g., "side", "front")
        """
        if self.camera_suffix and camera_name.endswith(self.camera_suffix):
            return camera_name[:-len(self.camera_suffix)]
        return camera_name
    
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Process robot observation by applying CLAHE to camera images.
        
        Args:
            observation: Robot observation containing camera images
        
        Returns:
            Observation with enhanced images
        """
        if not self.enabled:
            return observation
        
        # Process each camera image
        for key, value in observation.items():
            # Check if this is an image
            if isinstance(value, np.ndarray) and value.ndim == 3:
                # Determine if this camera should be processed
                should_process = False
                base_name = key
                
                if self.camera_suffix:
                    # If suffix is specified, only process cameras with that suffix
                    if key.endswith(self.camera_suffix):
                        should_process = True
                        base_name = self.get_camera_base_name(key)
                else:
                    # If no suffix, process cameras that match parameter names directly
                    if key in self.params:
                        should_process = True
                        base_name = key
                
                if should_process and base_name in self.params:
                    params = self.params[base_name]
                    clip_limit = params['clip_limit']
                    tile_grid_size = tuple(params['tile_grid_size'])
                    
                    # Apply CLAHE enhancement
                    enhanced = self.apply_clahe(value, clip_limit, tile_grid_size)
                    observation[key] = enhanced
                    
                    logging.debug(f"Applied CLAHE to {key} using {base_name} parameters")
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform features (no transformation needed for CLAHE).
        
        CLAHE doesn't change the structure of features, only enhances image values.
        
        Args:
            features: The policy features dictionary
        
        Returns:
            The same features dictionary unchanged
        """
        return features
    
    def reset(self):
        """Reset the processor state (no state to reset for CLAHE)."""
        pass
