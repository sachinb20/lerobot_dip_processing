#!/usr/bin/env python3
"""
CLAHE-based enhancement for dim lighting conditions.

This script optimizes CLAHE parameters using ground truth images (side and side_dim)
and then applies the learned parameters to enhance dim images at inference time.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        image: Input image (BGR or grayscale)
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


def optimize_clahe_parameters(dim_image: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Optimize CLAHE parameters to make dim_image look like ground_truth.
    
    Args:
        dim_image: The dim input image
        ground_truth: The ground truth well-lit image
    
    Returns:
        Dictionary with best parameters and metrics
    """
    best_psnr = 0
    best_params = {}
    
    # Grid search over CLAHE parameters
    clip_limits = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    tile_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    
    print("Optimizing CLAHE parameters...")
    print(f"Testing {len(clip_limits)} clip limits × {len(tile_sizes)} tile sizes = {len(clip_limits) * len(tile_sizes)} combinations")
    
    for clip_limit in clip_limits:
        for tile_size in tile_sizes:
            # Apply CLAHE with current parameters
            enhanced = apply_clahe(dim_image, clip_limit, tile_size)
            
            # Calculate metrics
            psnr = calculate_psnr(enhanced, ground_truth)
            ssim = calculate_ssim(enhanced, ground_truth)
            
            # Update best parameters
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = {
                    'clip_limit': clip_limit,
                    'tile_grid_size': tile_size,
                    'psnr': psnr,
                    'ssim': ssim
                }
                print(f"  New best: clip_limit={clip_limit}, tile_size={tile_size}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    return best_params


def main():
    """Main function to optimize and apply CLAHE."""
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
    print("OPTIMIZING CLAHE PARAMETERS FOR SIDE CAMERA")
    print("="*60)
    side_params = optimize_clahe_parameters(side_dim, side)
    
    print("\n" + "="*60)
    print("BEST SIDE CAMERA PARAMETERS:")
    print("="*60)
    print(f"Clip Limit: {side_params['clip_limit']}")
    print(f"Tile Grid Size: {side_params['tile_grid_size']}")
    print(f"PSNR: {side_params['psnr']:.2f} dB")
    print(f"SSIM: {side_params['ssim']:.4f}")
    
    all_params['side'] = {
        'clip_limit': side_params['clip_limit'],
        'tile_grid_size': list(side_params['tile_grid_size']),
        'psnr': side_params['psnr'],
        'ssim': side_params['ssim']
    }
    
    # Optimize parameters for FRONT camera
    print("\n" + "="*60)
    print("OPTIMIZING CLAHE PARAMETERS FOR FRONT CAMERA")
    print("="*60)
    front_params = optimize_clahe_parameters(front_dim, front)
    
    print("\n" + "="*60)
    print("BEST FRONT CAMERA PARAMETERS:")
    print("="*60)
    print(f"Clip Limit: {front_params['clip_limit']}")
    print(f"Tile Grid Size: {front_params['tile_grid_size']}")
    print(f"PSNR: {front_params['psnr']:.2f} dB")
    print(f"SSIM: {front_params['ssim']:.4f}")
    
    all_params['front'] = {
        'clip_limit': front_params['clip_limit'],
        'tile_grid_size': list(front_params['tile_grid_size']),
        'psnr': front_params['psnr'],
        'ssim': front_params['ssim']
    }
    
    # Save all parameters to JSON
    params_file = output_dir / "clahe_params.json"
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    print(f"\nAll parameters saved to: {params_file}")
    
    # Apply optimized parameters
    print("\n" + "="*60)
    print("APPLYING CLAHE TO IMAGES")
    print("="*60)
    
    # Apply to side camera
    side_enhanced = apply_clahe(
        side_dim,
        clip_limit=side_params['clip_limit'],
        tile_grid_size=side_params['tile_grid_size']
    )
    output_path = output_dir / "side_dim_enhanced.jpg"
    cv2.imwrite(str(output_path), side_enhanced)
    print(f"Enhanced side image saved to: {output_path}")
    
    # Apply to front camera
    front_enhanced = apply_clahe(
        front_dim,
        clip_limit=front_params['clip_limit'],
        tile_grid_size=front_params['tile_grid_size']
    )
    front_output_path = output_dir / "front_dim_enhanced.jpg"
    cv2.imwrite(str(front_output_path), front_enhanced)
    print(f"Enhanced front image saved to: {front_output_path}")
    
    # Create comparison images
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Side camera comparison
    comparison_side = np.hstack([side_dim, side_enhanced, side])
    comparison_path = output_dir / "comparison_side.jpg"
    cv2.imwrite(str(comparison_path), comparison_side)
    print(f"Side comparison saved to: {comparison_path}")
    print("  Layout: [Dim Lighting | CLAHE Enhanced | Normal Lighting]")
    
    # Front camera comparison
    comparison_front = np.hstack([front_dim, front_enhanced, front])
    comparison_front_path = output_dir / "comparison_front.jpg"
    cv2.imwrite(str(comparison_front_path), comparison_front)
    print(f"Front comparison saved to: {comparison_front_path}")
    print("  Layout: [Dim Lighting | CLAHE Enhanced | Normal Lighting]")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Side Camera  - PSNR: {side_params['psnr']:.2f} dB, SSIM: {side_params['ssim']:.4f}")
    print(f"Front Camera - PSNR: {front_params['psnr']:.2f} dB, SSIM: {front_params['ssim']:.4f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nAt inference time, use the saved camera-specific parameters from clahe_params.json")
    print("to enhance dim images without needing ground truth.")


if __name__ == "__main__":
    main()
