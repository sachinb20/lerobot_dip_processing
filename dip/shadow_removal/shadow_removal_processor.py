"""
Shadow removal processor for camera images.
Applies adaptive illumination correction to remove shadows from camera observations.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


def learn_illumination_params(shadow_img, reference_img):
    """Learn illumination correction parameters from shadow and reference image pair."""
    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    ref_l = ref_lab[:, :, 0].astype(np.float32)
    
    params = {
        'target_mean_luminance': np.mean(ref_l),
    }
    
    return params


def load_learned_parameters(reference_dir="/home/aditya/lerobot_alt/dip"):
    """
    Load and learn parameters from reference images.
    Returns the target luminance value.
    """
    ref_dir = Path(reference_dir)
    
    training_pairs = [
        ('side_shadow.jpg', 'side.jpg'),
        ('front_shadow.jpg', 'front.jpg'),
    ]
    
    learned_params = []
    
    for shadow_name, ref_name in training_pairs:
        shadow_path = ref_dir / shadow_name
        ref_path = ref_dir / ref_name
        
        if not shadow_path.exists() or not ref_path.exists():
            continue
        
        shadow_img = cv2.imread(str(shadow_path))
        ref_img = cv2.imread(str(ref_path))
        
        if shadow_img is not None and ref_img is not None:
            params = learn_illumination_params(shadow_img, ref_img)
            learned_params.append(params['target_mean_luminance'])
    
    # Calculate average target luminance
    if learned_params:
        avg_target_lum = np.mean(learned_params)
        return avg_target_lum
    
    # Default fallback
    return 132.0


def adaptive_illumination_correction(shadow_img, target_mean=132.0, sigma=50):
    """
    Adaptive illumination correction - Method 2 from dip.py.
    
    Args:
        shadow_img: Image with shadows (RGB format, uint8 numpy array)
        target_mean: Target mean luminance (learned from reference)
        sigma: Gaussian blur sigma for illumination estimation
        
    Returns:
        Shadow-corrected image (RGB format, uint8 numpy array)
    """
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(shadow_img, cv2.COLOR_RGB2BGR)
    
    # Convert to LAB
    shadow_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    
    # Estimate illumination
    shadow_blur = cv2.GaussianBlur(shadow_l, (0, 0), sigma)
    
    # Normalize to target mean
    current_mean = np.mean(shadow_blur)
    global_correction = target_mean / (current_mean + 1e-6)
    global_correction = np.clip(global_correction, 0.7, 1.5)
    
    # Local correction based on illumination variation
    mean_blur = np.mean(shadow_blur)
    local_correction = mean_blur / (shadow_blur + 1e-6)
    local_correction = np.clip(local_correction, 0.8, 1.5)
    
    # Combine global and local correction
    total_correction = global_correction * local_correction
    
    # Apply correction
    corrected_l = shadow_l * total_correction
    corrected_l = np.clip(corrected_l, 0, 255).astype(np.uint8)
    
    # Merge back
    corrected_lab = cv2.merge([corrected_l, shadow_lab[:, :, 1], shadow_lab[:, :, 2]])
    result_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return result_rgb


@dataclass
@ProcessorStepRegistry.register(name="shadow_removal_processor")
class ShadowRemovalProcessorStep(ObservationProcessorStep):
    """
    Processor that applies shadow removal to camera images in robot observations.
    Uses adaptive illumination correction learned from reference images.
    """
    
    reference_dir: str = "/home/aditya/lerobot_alt/dip"
    target_luminance: float = None
    
    def __post_init__(self):
        """Initialize and load learned parameters."""
        if self.target_luminance is None:
            self.target_luminance = load_learned_parameters(self.reference_dir)
            print(f"ShadowRemovalProcessorStep initialized with target luminance: {self.target_luminance:.2f}")
    
    def observation(self, observation):
        """
        Apply shadow removal to camera images in the observation.
        
        Args:
            observation: Dictionary containing robot observations
            
        Returns:
            observation: Dictionary with shadow-removed camera images
        """
        # Process all keys that end with 'image' (e.g., 'front_image', 'side_image')
        for key in list(observation.keys()):
            if key.endswith('_image') or key == 'image':
                img = observation[key]
                
                # Apply shadow removal
                # Input is expected to be RGB uint8 numpy array
                if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                    corrected = adaptive_illumination_correction(
                        img, 
                        target_mean=self.target_luminance
                    )
                    observation[key] = corrected
        
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Shadow removal doesn't change feature structure, so return features unchanged.
        """
        return features
