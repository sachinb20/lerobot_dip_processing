#!/usr/bin/env python3
"""
Wiener Deconvolution Processor for LeRobot observation pipeline.

This processor applies Wiener deconvolution to deblur camera images during robot data recording.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

from lerobot.processor import RobotObservation
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def wiener_deconvolution(img, kernel, K):
    """
    Perform Wiener deconvolution.

    img    : blurred image (grayscale, float32)
    kernel : blur kernel (PSF)
    K      : noise-to-signal power ratio
    """
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape

    # Pad kernel to image size
    kernel_padded = np.zeros_like(img)
    kernel_padded[:ker_h, :ker_w] = kernel

    # Shift kernel center to (0,0)
    kernel_padded = np.roll(kernel_padded, -ker_h // 2, axis=0)
    kernel_padded = np.roll(kernel_padded, -ker_w // 2, axis=1)

    # FFTs
    IMG = np.fft.fft2(img)
    KER = np.fft.fft2(kernel_padded)

    # Wiener filter
    KER_conj = np.conj(KER)
    wiener_filter = KER_conj / (np.abs(KER)**2 + K)

    # Apply filter
    deblurred = np.fft.ifft2(IMG * wiener_filter)
    deblurred = np.real(deblurred)

    return np.clip(deblurred, 0, 1)


@dataclass
@ProcessorStepRegistry.register(name="wiener_processor")
class WienerProcessorStep(ObservationProcessorStep):
    """
    Processor step that applies Wiener deconvolution to camera images.
    
    This processor applies Wiener deconvolution for image deblurring using fixed parameters.
    """
    
    kernel_size: int = 15
    sigma: float = 3.0
    k: float = 0.01  # noise-to-signal ratio
    enabled: bool = True
    camera_suffix: str | None = None
    
    def __post_init__(self):
        """
        Initialize the Wiener processor after dataclass initialization.
        """
        self.camera_suffix = self.camera_suffix if self.camera_suffix else ""
        
        # Pre-compute the blur kernel
        self.kernel = gaussian_kernel(self.kernel_size, self.sigma)
        
        logging.info(f"✓ Wiener deconvolution processor initialized")
        logging.info(f"  Parameters: kernel_size={self.kernel_size}, sigma={self.sigma}, k={self.k}")
        if self.camera_suffix:
            logging.info(f"  Will process cameras with suffix: '{self.camera_suffix}'")
        else:
            logging.info(f"  Will process all camera images")
    
    def apply_wiener_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Wiener deconvolution to an image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Deblurred image (BGR format)
        """
        # Process each channel separately
        deblurred_channels = []
        for i in range(3):  # B, G, R channels
            channel = image[:, :, i].astype(np.float32) / 255.0
            deblurred_channel = wiener_deconvolution(channel, self.kernel, self.k)
            deblurred_channels.append((deblurred_channel * 255).astype(np.uint8))
        
        # Merge channels back
        deblurred = cv2.merge(deblurred_channels)
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
        Process robot observation by applying Wiener deconvolution to camera images.
        
        Args:
            observation: Robot observation containing camera images
        
        Returns:
            Observation with deblurred images
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
                    # Apply Wiener deconvolution
                    deblurred = self.apply_wiener_deblur(value)
                    observation[key] = deblurred
                    
                    logging.debug(f"Applied Wiener deconvolution to {key}")
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform features (no transformation needed for Wiener deconvolution).
        
        Wiener deconvolution doesn't change the structure of features, only enhances image values.
        
        Args:
            features: The policy features dictionary
        
        Returns:
            The same features dictionary unchanged
        """
        return features
    
    def reset(self):
        """Reset the processor state (no state to reset for Wiener deconvolution)."""
        pass
