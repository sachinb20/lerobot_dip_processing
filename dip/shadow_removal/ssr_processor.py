#!/usr/bin/env python3
"""
SSR (Single Scale Retinex) Processor for LeRobot observation pipeline.

This processor applies SSR for shadow removal to camera images during robot data recording.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

from lerobot.processor import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


@dataclass
@ProcessorStepRegistry.register(name="ssr_processor")
class SSRProcessorStep(ObservationProcessorStep):
    """
    Processor step that applies SSR (Single Scale Retinex) to camera images.
    
    This processor applies SSR for shadow removal using fixed parameters.
    """
    
    sigma: float = 50.0
    gain: float = 0.6
    enabled: bool = True
    camera_suffix: str | None = None
    
    def __post_init__(self):
        """
        Initialize the SSR processor after dataclass initialization.
        """
        self.camera_suffix = self.camera_suffix if self.camera_suffix else ""
        
        logging.info(f"✓ SSR processor initialized with sigma={self.sigma}, gain={self.gain}")
        if self.camera_suffix:
            logging.info(f"  Will process cameras with suffix: '{self.camera_suffix}'")
        else:
            logging.info(f"  Will process all camera images")
    
    def single_scale_retinex(
        self,
        img: np.ndarray,
        sigma: float = 50,
        gain: float = 0.6
    ) -> np.ndarray:
        """
        Apply Single Scale Retinex for shadow removal.
        
        Args:
            img: Input image (BGR format)
            sigma: Gaussian blur sigma for illumination estimation
            gain: Output gain factor (0-1), lower values reduce brightness
            
        Returns:
            SSR-enhanced image (BGR format)
        """
        # Convert to float
        img_float = img.astype(np.float32) + 1.0
        
        # Gaussian surround (illumination estimate)
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        
        # SSR formula: log(I) - log(blur(I))
        retinex = np.log(img_float) - np.log(blur)
        
        # Normalize to 0–255 for display
        retinex = retinex - np.min(retinex)
        retinex = retinex / (np.max(retinex) + 1e-6)
        
        # Apply gain to control brightness
        retinex = (retinex * 255 * gain).astype(np.uint8)
        
        return retinex
    
    def get_camera_base_name(self, camera_name: str) -> str:
        """
        Extract the base camera name by removing the suffix.
        
        Args:
            camera_name: Full camera name (e.g., "side_shadow", "front_shadow")
        
        Returns:
            Base camera name (e.g., "side", "front")
        """
        if self.camera_suffix and camera_name.endswith(self.camera_suffix):
            return camera_name[:-len(self.camera_suffix)]
        return camera_name
    
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Process robot observation by applying SSR to camera images.
        
        Args:
            observation: Robot observation containing camera images
        
        Returns:
            Observation with shadow-removed images
        """
        if not self.enabled:
            return observation
        
        # Process each camera image
        for key, value in observation.items():
            # Check if this is an image
            if isinstance(value, np.ndarray) and value.ndim == 3:
                # Determine if this camera should be processed
                should_process = False
                
                if self.camera_suffix:
                    # If suffix is specified, only process cameras with that suffix
                    if key.endswith(self.camera_suffix):
                        should_process = True
                else:
                    # If no suffix, process all images
                    should_process = True
                
                if should_process:
                    # Apply SSR enhancement
                    enhanced = self.single_scale_retinex(value, sigma=self.sigma, gain=self.gain)
                    observation[key] = enhanced
                    
                    logging.debug(f"Applied SSR to {key}")
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform features (no transformation needed for SSR).
        
        SSR doesn't change the structure of features, only enhances image values.
        
        Args:
            features: The policy features dictionary
        
        Returns:
            The same features dictionary unchanged
        """
        return features
    
    def reset(self):
        """Reset the processor state (no state to reset for SSR)."""
        pass
