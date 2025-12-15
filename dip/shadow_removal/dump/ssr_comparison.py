import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def SSR(img, sigma=80):
    """
    Single Scale Retinex (SSR) implementation.
    
    Args:
        img: Input image (grayscale or color)
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        Enhanced image using SSR
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    img_f = img.astype(np.float32) + 1e-6
    blur_f = blur.astype(np.float32) + 1e-6
    retinex = np.log(img_f) - np.log(blur_f)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def process_color_image(img, sigma=80):
    """
    Apply SSR to color image by processing each channel separately.
    
    Args:
        img: Input BGR color image
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        Enhanced color image
    """
    # Split into channels
    b, g, r = cv2.split(img)
    
    # Apply SSR to each channel
    b_ssr = SSR(b, sigma)
    g_ssr = SSR(g, sigma)
    r_ssr = SSR(r, sigma)
    
    # Merge channels back
    result = cv2.merge([b_ssr, g_ssr, r_ssr])
    return result


def compare_shadow_removal():
    """
    Compare side.jpg (no shadow) with side_shadow.jpg (with shadow) 
    and show how SSR removes the shadow.
    """
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    side_img = cv2.imread(str(script_dir / "side.jpg"))
    side_shadow_img = cv2.imread(str(script_dir / "side_shadow.jpg"))
    
    if side_img is None or side_shadow_img is None:
        print("Error: Could not load images!")
        return
    
    print("Loaded images successfully")
    print(f"side.jpg shape: {side_img.shape}")
    print(f"side_shadow.jpg shape: {side_shadow_img.shape}")
    
    # Apply SSR to shadow image with different sigma values
    sigma_values = [15, 80, 150]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, len(sigma_values) + 2, figsize=(20, 8))
    
    # Row 1: Original images
    axes[0, 0].imshow(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Reference: side.jpg\n(No Shadow)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(side_shadow_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Input: side_shadow.jpg\n(With Shadow)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Apply SSR with different sigmas
    for idx, sigma in enumerate(sigma_values):
        print(f"\nProcessing with sigma={sigma}...")
        result = process_color_image(side_shadow_img, sigma=sigma)
        
        # Show result in row 1
        axes[0, idx + 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[0, idx + 2].set_title(f"SSR Result\n(σ={sigma})", fontsize=12, fontweight='bold')
        axes[0, idx + 2].axis('off')
        
        # Save individual result
        result_path = output_dir / f"side_shadow_ssr_sigma_{sigma}.png"
        cv2.imwrite(str(result_path), result)
        print(f"Saved: {result_path}")
    
    # Row 2: Show difference maps
    axes[1, 0].text(0.5, 0.5, "Difference Maps\n(vs Reference)", 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference between shadow and reference
    diff_shadow = cv2.absdiff(side_img, side_shadow_img)
    axes[1, 1].imshow(cv2.cvtColor(diff_shadow, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Diff: Shadow vs Ref", fontsize=10)
    axes[1, 1].axis('off')
    
    # Difference between SSR results and reference
    for idx, sigma in enumerate(sigma_values):
        result = process_color_image(side_shadow_img, sigma=sigma)
        diff = cv2.absdiff(side_img, result)
        axes[1, idx + 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        axes[1, idx + 2].set_title(f"Diff: SSR(σ={sigma}) vs Ref", fontsize=10)
        axes[1, idx + 2].axis('off')
    
    plt.suptitle("SSR Shadow Removal Comparison: Making side_shadow.jpg look like side.jpg", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save comparison
    comparison_path = output_dir / "shadow_removal_comparison.png"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Saved comparison to: {comparison_path}")
    print(f"{'='*60}")
    
    # Also create a simple before/after comparison
    create_before_after_comparison(side_img, side_shadow_img, output_dir)


def create_before_after_comparison(side_img, side_shadow_img, output_dir):
    """
    Create a simple before/after comparison with the best sigma value.
    """
    # Use sigma=80 as it typically gives good results
    best_sigma = 80
    result = process_color_image(side_shadow_img, sigma=best_sigma)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Target: side.jpg (No Shadow)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(side_shadow_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Before: side_shadow.jpg (With Shadow)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"After: SSR Applied (σ={best_sigma})", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle("Shadow Removal: Before and After SSR", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    before_after_path = output_dir / "before_after_ssr.png"
    plt.savefig(str(before_after_path), dpi=150, bbox_inches='tight')
    print(f"Saved before/after comparison to: {before_after_path}")


def main():
    print("=" * 60)
    print("SSR Shadow Removal Comparison")
    print("Goal: Make side_shadow.jpg look like side.jpg")
    print("=" * 60)
    
    compare_shadow_removal()


if __name__ == "__main__":
    main()
