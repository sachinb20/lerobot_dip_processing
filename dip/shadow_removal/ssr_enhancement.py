#!/usr/bin/env python3
"""
SSR (Single Scale Retinex) comparison script.
Creates side-by-side comparison images similar to CLAHE and gamma enhancement.
"""

import cv2
import numpy as np
from pathlib import Path


def single_scale_retinex(img, sigma=50, gain=0.6):
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


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main():
    """Main function to create SSR comparisons."""
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
    
    # SSR parameters (from ssr.py, with gain adjustment)
    sigma = 50
    gain = 0.6  # Reduce brightness
    
    # Apply SSR to both cameras
    print("\n" + "="*60)
    print("APPLYING SINGLE SCALE RETINEX (SSR)")
    print("="*60)
    print(f"Using sigma = {sigma}, gain = {gain}")
    
    # Side camera
    side_enhanced = single_scale_retinex(side_shadow, sigma=sigma, gain=gain)
    side_psnr = calculate_psnr(side_enhanced, side)
    output_path = output_dir / "side_shadow_ssr_enhanced.jpg"
    cv2.imwrite(str(output_path), side_enhanced)
    print(f"\nEnhanced side image saved to: {output_path}")
    print(f"  PSNR: {side_psnr:.2f} dB")
    
    # Front camera
    front_enhanced = single_scale_retinex(front_shadow, sigma=sigma, gain=gain)
    front_psnr = calculate_psnr(front_enhanced, front)
    front_output_path = output_dir / "front_shadow_ssr_enhanced.jpg"
    cv2.imwrite(str(front_output_path), front_enhanced)
    print(f"Enhanced front image saved to: {front_output_path}")
    print(f"  PSNR: {front_psnr:.2f} dB")
    
    # Create comparison images
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Side camera comparison
    comparison_side = np.hstack([side_shadow, side_enhanced, side])
    comparison_path = output_dir / "comparison_side_ssr.jpg"
    cv2.imwrite(str(comparison_path), comparison_side)
    print(f"Side comparison saved to: {comparison_path}")
    print("  Layout: [With Shadow | SSR Enhanced | Normal Lighting]")
    
    # Front camera comparison
    comparison_front = np.hstack([front_shadow, front_enhanced, front])
    comparison_front_path = output_dir / "comparison_front_ssr.jpg"
    cv2.imwrite(str(comparison_front_path), comparison_front)
    print(f"Front comparison saved to: {comparison_front_path}")
    print("  Layout: [With Shadow | SSR Enhanced | Normal Lighting]")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SSR Parameters: sigma = {sigma}, gain = {gain}")
    print(f"Side Camera  - PSNR: {side_psnr:.2f} dB")
    print(f"Front Camera - PSNR: {front_psnr:.2f} dB")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nSSR uses a fixed sigma parameter for shadow removal.")


if __name__ == "__main__":
    main()
