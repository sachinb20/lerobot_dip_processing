#!/usr/bin/env python3
"""
Shadow removal comparison script.
Creates side-by-side comparison images similar to CLAHE enhancement.
"""

import cv2
import numpy as np
from pathlib import Path


def adaptive_illumination_correction(shadow_img, target_mean=132.0, sigma=50):
    """
    Adaptive illumination correction - Method 2 from dip.py.
    
    Args:
        shadow_img: Image with shadows (BGR format)
        target_mean: Target mean luminance (learned from reference)
        sigma: Gaussian blur sigma for illumination estimation
        
    Returns:
        Shadow-corrected image (BGR format)
    """
    # Convert to LAB
    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    
    # Estimate illumination
    shadow_blur = cv2.GaussianBlur(shadow_l, (0, 0), sigma)
    
    # Global correction
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
    result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    return result


def learn_target_luminance(shadow_img, reference_img):
    """Learn target luminance from reference image."""
    ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    ref_l = ref_lab[:, :, 0].astype(np.float32)
    return np.mean(ref_l)


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main():
    """Main function to create shadow removal comparisons."""
    # Set up paths
    data_dir = Path(__file__).parent
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    side_shadow_path = data_dir / "side_shadow.jpg"
    side_path = data_dir / "side.jpg"
    front_shadow_path = data_dir / "front_shadow.jpg"
    front_path = data_dir / "front.jpg"
    
    # Load images
    print("Loading images...")
    side_shadow = cv2.imread(str(side_shadow_path))
    side = cv2.imread(str(side_path))
    front_shadow = cv2.imread(str(front_shadow_path))
    front = cv2.imread(str(front_path))
    
    if side_shadow is None or side is None:
        raise FileNotFoundError("Could not load side_shadow.jpg or side.jpg")
    if front_shadow is None or front is None:
        raise FileNotFoundError("Could not load front_shadow.jpg or front.jpg")
    
    print(f"side_shadow shape: {side_shadow.shape}")
    print(f"side shape: {side.shape}")
    print(f"front_shadow shape: {front_shadow.shape}")
    print(f"front shape: {front.shape}")
    
    # Learn target luminance from both cameras
    print("\n" + "="*60)
    print("LEARNING TARGET LUMINANCE")
    print("="*60)
    
    side_target_lum = learn_target_luminance(side_shadow, side)
    front_target_lum = learn_target_luminance(front_shadow, front)
    avg_target_lum = (side_target_lum + front_target_lum) / 2
    
    print(f"Side camera target luminance: {side_target_lum:.2f}")
    print(f"Front camera target luminance: {front_target_lum:.2f}")
    print(f"Average target luminance: {avg_target_lum:.2f}")
    
    # Apply shadow removal to both cameras
    print("\n" + "="*60)
    print("APPLYING SHADOW REMOVAL")
    print("="*60)
    
    # Side camera
    side_enhanced = adaptive_illumination_correction(side_shadow, target_mean=avg_target_lum)
    side_psnr = calculate_psnr(side_enhanced, side)
    output_path = output_dir / "side_shadow_enhanced.jpg"
    cv2.imwrite(str(output_path), side_enhanced)
    print(f"Enhanced side image saved to: {output_path}")
    print(f"  PSNR: {side_psnr:.2f} dB")
    
    # Front camera
    front_enhanced = adaptive_illumination_correction(front_shadow, target_mean=avg_target_lum)
    front_psnr = calculate_psnr(front_enhanced, front)
    front_output_path = output_dir / "front_shadow_enhanced.jpg"
    cv2.imwrite(str(front_output_path), front_enhanced)
    print(f"Enhanced front image saved to: {front_output_path}")
    print(f"  PSNR: {front_psnr:.2f} dB")
    
    # Create comparison images
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Side camera comparison
    comparison_side = np.hstack([side_shadow, side_enhanced, side])
    comparison_path = output_dir / "comparison_side.jpg"
    cv2.imwrite(str(comparison_path), comparison_side)
    print(f"Side comparison saved to: {comparison_path}")
    print("  Layout: [With Shadow | Shadow Removed | Normal Lighting]")
    
    # Front camera comparison
    comparison_front = np.hstack([front_shadow, front_enhanced, front])
    comparison_front_path = output_dir / "comparison_front.jpg"
    cv2.imwrite(str(comparison_front_path), comparison_front)
    print(f"Front comparison saved to: {comparison_front_path}")
    print("  Layout: [With Shadow | Shadow Removed | Normal Lighting]")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Side Camera  - PSNR: {side_psnr:.2f} dB")
    print(f"Front Camera - PSNR: {front_psnr:.2f} dB")
    print(f"Average target luminance: {avg_target_lum:.2f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nAt inference time, use the learned target luminance parameter")
    print("to remove shadows without needing reference images.")


if __name__ == "__main__":
    main()
