#!/usr/bin/env python3
"""
Gamma correction-based enhancement for dim lighting conditions.

This script optimizes gamma correction parameters using ground truth images (side and side_dim)
and then applies the learned parameters to enhance dim images at inference time.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Args:
        image: Input image (BGR)
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


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images (simplified version)."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean
    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def optimize_gamma_parameters(dim_image: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Optimize gamma correction parameter to make dim_image look like ground_truth.
    
    Args:
        dim_image: The dim input image
        ground_truth: The ground truth well-lit image
    
    Returns:
        Dictionary with best parameters and metrics
    """
    best_psnr = 0
    best_params = {}
    
    # Grid search over gamma values
    # Values < 1 brighten the image, values > 1 darken it
    gamma_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    print("Optimizing gamma correction parameters...")
    print(f"Testing {len(gamma_values)} gamma values")
    
    for gamma in gamma_values:
        # Apply gamma correction with current parameter
        enhanced = apply_gamma_correction(dim_image, gamma)
        
        # Calculate metrics
        psnr = calculate_psnr(enhanced, ground_truth)
        ssim = calculate_ssim(enhanced, ground_truth)
        
        # Update best parameters
        if psnr > best_psnr:
            best_psnr = psnr
            best_params = {
                'gamma': gamma,
                'psnr': psnr,
                'ssim': ssim
            }
            print(f"  New best: gamma={gamma:.2f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    return best_params


def main():
    """Main function to optimize and apply gamma correction."""
    # Set up paths
    data_dir = Path(__file__).parent
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    side_dim_path = data_dir / "side_dim.jpg"
    side_path = data_dir / "side.jpg"
    front_dim_path = data_dir / "front_dim.jpg"
    front_path = data_dir / "front.jpg"
    
    # Load images
    print("Loading images...")
    side_dim = cv2.imread(str(side_dim_path))
    side = cv2.imread(str(side_path))
    front_dim = cv2.imread(str(front_dim_path))
    front = cv2.imread(str(front_path))
    
    if side_dim is None or side is None:
        raise FileNotFoundError("Could not load side_dim.jpg or side.jpg")
    if front_dim is None or front is None:
        raise FileNotFoundError("Could not load front_dim.jpg or front.jpg")
    
    print(f"side_dim shape: {side_dim.shape}")
    print(f"side shape: {side.shape}")
    print(f"front_dim shape: {front_dim.shape}")
    print(f"front shape: {front.shape}")
    
    # Dictionary to store all parameters
    all_params = {}
    
    # Optimize parameters for SIDE camera
    print("\n" + "="*60)
    print("OPTIMIZING GAMMA CORRECTION FOR SIDE CAMERA")
    print("="*60)
    side_params = optimize_gamma_parameters(side_dim, side)
    
    print("\n" + "="*60)
    print("BEST SIDE CAMERA PARAMETERS:")
    print("="*60)
    print(f"Gamma: {side_params['gamma']:.2f}")
    print(f"PSNR: {side_params['psnr']:.2f} dB")
    print(f"SSIM: {side_params['ssim']:.4f}")
    
    all_params['side'] = {
        'gamma': side_params['gamma'],
        'psnr': side_params['psnr'],
        'ssim': side_params['ssim']
    }
    
    # Optimize parameters for FRONT camera
    print("\n" + "="*60)
    print("OPTIMIZING GAMMA CORRECTION FOR FRONT CAMERA")
    print("="*60)
    front_params = optimize_gamma_parameters(front_dim, front)
    
    print("\n" + "="*60)
    print("BEST FRONT CAMERA PARAMETERS:")
    print("="*60)
    print(f"Gamma: {front_params['gamma']:.2f}")
    print(f"PSNR: {front_params['psnr']:.2f} dB")
    print(f"SSIM: {front_params['ssim']:.4f}")
    
    all_params['front'] = {
        'gamma': front_params['gamma'],
        'psnr': front_params['psnr'],
        'ssim': front_params['ssim']
    }
    
    # Save all parameters to JSON
    import json
    params_file = output_dir / "gamma_params.json"
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    print(f"\nAll parameters saved to: {params_file}")
    
    # Apply optimized parameters
    print("\n" + "="*60)
    print("APPLYING GAMMA CORRECTION TO IMAGES")
    print("="*60)
    
    # Apply to side camera
    side_enhanced = apply_gamma_correction(
        side_dim,
        gamma=side_params['gamma']
    )
    output_path = output_dir / "side_dim_gamma_enhanced.jpg"
    cv2.imwrite(str(output_path), side_enhanced)
    print(f"Enhanced side image saved to: {output_path}")
    
    # Apply to front camera
    front_enhanced = apply_gamma_correction(
        front_dim,
        gamma=front_params['gamma']
    )
    front_output_path = output_dir / "front_dim_gamma_enhanced.jpg"
    cv2.imwrite(str(front_output_path), front_enhanced)
    print(f"Enhanced front image saved to: {front_output_path}")
    
    # Create comparison images
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Side camera comparison
    comparison_side = np.hstack([side_dim, side_enhanced, side])
    comparison_path = output_dir / "comparison_side_gamma.jpg"
    cv2.imwrite(str(comparison_path), comparison_side)
    print(f"Side comparison saved to: {comparison_path}")
    print("  Layout: [Dim Lighting | Gamma Enhanced | Normal Lighting]")
    
    # Front camera comparison
    comparison_front = np.hstack([front_dim, front_enhanced, front])
    comparison_front_path = output_dir / "comparison_front_gamma.jpg"
    cv2.imwrite(str(comparison_front_path), comparison_front)
    print(f"Front comparison saved to: {comparison_front_path}")
    print("  Layout: [Dim Lighting | Gamma Enhanced | Normal Lighting]")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Side Camera  - Gamma: {side_params['gamma']:.2f}, PSNR: {side_params['psnr']:.2f} dB, SSIM: {side_params['ssim']:.4f}")
    print(f"Front Camera - Gamma: {front_params['gamma']:.2f}, PSNR: {front_params['psnr']:.2f} dB, SSIM: {front_params['ssim']:.4f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nAt inference time, use the saved camera-specific gamma values from gamma_params.json")
    print("to enhance dim images without needing ground truth.")


if __name__ == "__main__":
    main()
