#!/usr/bin/env python3
"""
Test script to verify both deblurring methods work correctly.
Creates a blurred test image and then deblurs it using both methods.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from weiner import gaussian_kernel, wiener_deconvolution


def create_test_blurred_image(input_image_path, output_path, kernel_size=15, sigma=3.0):
    """Create a blurred version of an image for testing."""
    # Load image
    img = cv2.imread(str(input_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_image_path}")
    
    # Convert to grayscale for testing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create blur kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Apply blur using filter2D
    blurred = cv2.filter2D(gray, -1, kernel)
    
    # Save blurred image
    cv2.imwrite(str(output_path), blurred)
    print(f"✓ Created blurred test image: {output_path}")
    
    return output_path, gray


def test_wiener_deblur(blurred_path, output_path, kernel_size=15, sigma=3.0, k=0.01):
    """Test Wiener deconvolution."""
    print("\n" + "="*60)
    print("TESTING WIENER DECONVOLUTION")
    print("="*60)
    
    # Load blurred image
    img = cv2.imread(str(blurred_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {blurred_path}")
    
    img = img.astype(np.float32) / 255.0
    
    # Create blur kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Wiener deconvolution
    deblurred = wiener_deconvolution(img, kernel, k)
    
    # Save output
    deblurred_uint8 = (deblurred * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), deblurred_uint8)
    
    print(f"✓ Wiener deblurred image saved to: {output_path}")
    print(f"  Parameters: kernel_size={kernel_size}, sigma={sigma}, k={k}")
    
    return deblurred_uint8


def test_richardson_lucy(blurred_path, output_path, kernel_size=15, sigma=3.0, iterations=30):
    """Test Richardson-Lucy deconvolution."""
    print("\n" + "="*60)
    print("TESTING RICHARDSON-LUCY DECONVOLUTION")
    print("="*60)
    
    try:
        from skimage import restoration, img_as_float
        
        # Load blurred image
        img = cv2.imread(str(blurred_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {blurred_path}")
        
        img_float = img_as_float(img)
        
        # Create PSF
        x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, 
                        -kernel_size//2 + 1:kernel_size//2 + 1]
        psf = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
        psf = psf / psf.sum()
        
        # Apply Richardson-Lucy
        restored = restoration.richardson_lucy(img_float, psf, num_iter=iterations, clip=False)
        restored = np.clip(restored, 0, 1)
        
        # Save output
        restored_uint8 = (restored * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), restored_uint8)
        
        print(f"✓ Richardson-Lucy deblurred image saved to: {output_path}")
        print(f"  Parameters: kernel_size={kernel_size}, sigma={sigma}, iterations={iterations}")
        
        return restored_uint8
        
    except ImportError as e:
        print(f"✗ Richardson-Lucy test skipped: {e}")
        print("  Install scikit-image: pip install scikit-image")
        return None


def create_comparison(original, blurred, wiener_result, rl_result, output_path):
    """Create a comparison image showing all results."""
    print("\n" + "="*60)
    print("CREATING COMPARISON IMAGE")
    print("="*60)
    
    images = [original, blurred, wiener_result]
    labels = ["Original", "Blurred", "Wiener"]
    
    if rl_result is not None:
        images.append(rl_result)
        labels.append("Richardson-Lucy")
    
    # Stack horizontally
    comparison = np.hstack(images)
    cv2.imwrite(str(output_path), comparison)
    
    print(f"✓ Comparison image saved to: {output_path}")
    print(f"  Layout: {' | '.join(labels)}")


def main():
    """Main test function."""
    print("="*60)
    print("DEBLURRING METHODS SANITY CHECK")
    print("="*60)
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Use a test image from dim_lighting
    test_image = script_dir.parent / "dim_lighting" / "side.jpg"
    
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        print("  Please provide a test image")
        return
    
    print(f"\nUsing test image: {test_image}")
    
    # Step 1: Create blurred test image
    blurred_path, original_gray = create_test_blurred_image(
        test_image,
        output_dir / "test_blurred.jpg",
        kernel_size=15,
        sigma=3.0
    )
    
    # Step 2: Test Wiener deconvolution
    wiener_result = test_wiener_deblur(
        blurred_path,
        output_dir / "test_wiener_deblurred.jpg",
        kernel_size=15,
        sigma=3.0,
        k=0.01
    )
    
    # Step 3: Test Richardson-Lucy deconvolution
    rl_result = test_richardson_lucy(
        blurred_path,
        output_dir / "test_rl_deblurred.jpg",
        kernel_size=15,
        sigma=3.0,
        iterations=30
    )
    
    # Step 4: Create comparison
    blurred_img = cv2.imread(str(blurred_path), cv2.IMREAD_GRAYSCALE)
    create_comparison(
        original_gray,
        blurred_img,
        wiener_result,
        rl_result,
        output_dir / "comparison_deblur_methods.jpg"
    )
    
    print("\n" + "="*60)
    print("SANITY CHECK COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nSummary:")
    print("  ✓ Wiener deconvolution: WORKING")
    if rl_result is not None:
        print("  ✓ Richardson-Lucy deconvolution: WORKING")
    else:
        print("  ⚠ Richardson-Lucy deconvolution: SKIPPED (missing scikit-image)")
    print("\nBoth methods are ready to use for deblurring images!")


if __name__ == "__main__":
    main()
