#!/usr/bin/env python3
"""
Richardson-Lucy Deconvolution Processor for LeRobot observation pipeline.

This processor applies Richardson-Lucy deconvolution to deblur camera images during robot data recording.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

from lerobot.processor import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


def make_gaussian_psf(size, sigma):
    """
    Creates a Gaussian kernel to serve as the Point Spread Function (PSF).
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g / g.sum()


@dataclass
@ProcessorStepRegistry.register(name="richardson_lucy_processor")
class RichardsonLucyProcessorStep(ObservationProcessorStep):
    """
    Processor step that applies Richardson-Lucy deconvolution to camera images.
    
    This processor applies Richardson-Lucy deconvolution for blind image deblurring.
    """
    
    kernel_size: int = 15
    sigma: float = 5.0
    iterations: int = 30
    enabled: bool = True
    camera_suffix: str | None = None
    
    def __post_init__(self):
        """
        Initialize the Richardson-Lucy processor after dataclass initialization.
        """
        self.camera_suffix = self.camera_suffix if self.camera_suffix else ""
        
        # Pre-compute the PSF
        self.psf = make_gaussian_psf(self.kernel_size, self.sigma)
        
        # Check if scikit-image is available
        try:
            from skimage import restoration
            self.restoration = restoration
            self.available = True
            logging.info(f"✓ Richardson-Lucy deconvolution processor initialized")
            logging.info(f"  Parameters: kernel_size={self.kernel_size}, sigma={self.sigma}, iterations={self.iterations}")
            if self.camera_suffix:
                logging.info(f"  Will process cameras with suffix: '{self.camera_suffix}'")
            else:
                logging.info(f"  Will process all camera images")
        except ImportError:
            logging.warning("Richardson-Lucy processor disabled: scikit-image not installed")
            logging.warning("Install with: pip install scikit-image")
            self.available = False
            self.enabled = False
    
    def apply_richardson_lucy(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Richardson-Lucy deconvolution to an image.
        
        Args:
            image: Input image (BGR format, uint8)
        
        Returns:
            Deblurred image (BGR format, uint8)
        """
        if not self.available:
            return image
        
        # Convert to float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Process each channel separately
        deblurred_channels = []
        for i in range(3):  # B, G, R channels
            channel = img_float[:, :, i]
            
            # Apply Richardson-Lucy deconvolution
            restored = self.restoration.richardson_lucy(
                channel, 
                self.psf, 
                num_iter=self.iterations, 
                clip=False
            )
            restored = np.clip(restored, 0, 1)
            deblurred_channels.append(restored)
        
        # Merge channels back
        deblurred = np.dstack(deblurred_channels)
        deblurred = (deblurred * 255).astype(np.uint8)
        
        return deblurred
    
    def get_camera_base_name(self, camera_name: str) -> str:
        """
        Extract the base camera name by removing the suffix.
        
        Args:
            camera_name: Full camera name (e.g., "side_blurred", "front_blurred")
        
        Returns:
            Base camera name (e.g., "side", "front")
        """
        if self.camera_suffix and camera_name.endswith(self.camera_suffix):
            return camera_name[:-len(self.camera_suffix)]
        return camera_name
    
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Process robot observation by applying Richardson-Lucy deconvolution to camera images.
        
        Args:
            observation: Robot observation containing camera images
        
        Returns:
            Observation with deblurred images
        """
        if not self.enabled or not self.available:
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
                    # Apply Richardson-Lucy deconvolution
                    deblurred = self.apply_richardson_lucy(value)
                    observation[key] = deblurred
                    
                    logging.debug(f"Applied Richardson-Lucy deconvolution to {key}")
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform features (no transformation needed for Richardson-Lucy deconvolution).
        
        Richardson-Lucy deconvolution doesn't change the structure of features, only enhances image values.
        
        Args:
            features: The policy features dictionary
        
        Returns:
            The same features dictionary unchanged
        """
        return features
    
    def reset(self):
        """Reset the processor state (no state to reset for Richardson-Lucy deconvolution)."""
        pass
