#!/usr/bin/env python3
"""
Gamma Correction Processor for LeRobot observation pipeline.

This processor applies gamma correction to enhance dim lighting images during robot data recording.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from lerobot.processor import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


@dataclass
@ProcessorStepRegistry.register(name="gamma_processor")
class GammaProcessorStep(ObservationProcessorStep):
    """
    Processor step that applies gamma correction to camera images.
    
    This processor loads camera-specific gamma parameters from a JSON file
    and applies them to enhance images from dim lighting conditions.
    """
    
    params_file: str | Path | None = None
    camera_suffix: str | None = None
    enabled: bool = True
    
    def __post_init__(self):
        """
        Initialize the gamma processor after dataclass initialization.
        """
        self.camera_suffix = self.camera_suffix if self.camera_suffix else ""
        
        # Load gamma parameters
        if self.params_file is None:
            params_file = Path(__file__).parent / "output" / "gamma_params.json"
        else:
            params_file = Path(self.params_file)
        
        self.params = {}
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.params = json.load(f)
            logging.info(f"✓ Loaded gamma parameters from {params_file}")
            for camera_name, params in self.params.items():
                logging.info(f"  {camera_name}: gamma={params['gamma']}")
            if self.camera_suffix:
                logging.info(f"  Will process cameras with suffix: '{self.camera_suffix}'")
            else:
                logging.info(f"  Will process cameras matching parameter names directly")
        else:
            logging.warning(f"Gamma parameters file not found: {params_file}")
            logging.warning("Gamma enhancement will be disabled")
            self.enabled = False
    
    def apply_gamma_correction(
        self,
        image: np.ndarray,
        gamma: float = 1.0
    ) -> np.ndarray:
        """
        Apply gamma correction to an image.
        
        Args:
            image: Input image (BGR format)
            gamma: Gamma value (< 1 brightens, > 1 darkens)
        
        Returns:
            Gamma corrected image
        """
        # Normalize to [0, 1]
        img_norm = image / 255.0
        
        # Apply gamma correction
        img_gamma = np.power(img_norm, gamma)
        
        # Convert back to [0, 255]
        return (img_gamma * 255).astype(np.uint8)
    
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
        Process robot observation by applying gamma correction to camera images.
        
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
                    gamma = params['gamma']
                    
                    # Apply gamma correction
                    enhanced = self.apply_gamma_correction(value, gamma)
                    observation[key] = enhanced
                    
                    logging.debug(f"Applied gamma correction to {key} using {base_name} parameters")
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform features (no transformation needed for gamma correction).
        
        Gamma correction doesn't change the structure of features, only enhances image values.
        
        Args:
            features: The policy features dictionary
        
        Returns:
            The same features dictionary unchanged
        """
        return features
    
    def reset(self):
        """Reset the processor state (no state to reset for gamma correction)."""
        pass
