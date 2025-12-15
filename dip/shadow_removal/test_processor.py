"""
Test script to verify shadow removal processor works correctly.
"""

import sys
sys.path.insert(0, '/home/aditya/lerobot_alt/dip')

import cv2
import numpy as np
from pathlib import Path
from shadow_removal_processor import ShadowRemovalProcessorStep

def test_shadow_removal_processor():
    print("="*80)
    print("Testing Shadow Removal Processor")
    print("="*80)
    
    # Load test image
    test_img_path = Path("/home/aditya/lerobot_alt/dip/side_shadow.jpg")
    if not test_img_path.exists():
        print(f"Error: Test image not found at {test_img_path}")
        return
    
    img_bgr = cv2.imread(str(test_img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"\nLoaded test image: {test_img_path}")
    print(f"Image shape: {img_rgb.shape}")
    
    # Create processor
    processor = ShadowRemovalProcessorStep()
    
    # Create mock observation
    observation = {
        'front_image': img_rgb.copy(),
        'side_image': img_rgb.copy(),
        'state': np.array([1.0, 2.0, 3.0])  # Mock state
    }
    
    print(f"\nCreated mock observation with keys: {list(observation.keys())}")
    
    # Process observation
    print("\nProcessing observation...")
    processed_obs = processor.observation(observation)
    
    print(f"Processed observation keys: {list(processed_obs.keys())}")
    
    # Save results
    output_dir = Path("/home/aditya/lerobot_alt/dip/output")
    output_dir.mkdir(exist_ok=True)
    
    for key in ['front_image', 'side_image']:
        if key in processed_obs:
            # Convert back to BGR for saving
            result_bgr = cv2.cvtColor(processed_obs[key], cv2.COLOR_RGB2BGR)
            output_path = output_dir / f"processor_test_{key}.png"
            cv2.imwrite(str(output_path), result_bgr)
            print(f"Saved processed {key} to: {output_path}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    test_shadow_removal_processor()
