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


def create_test_shadow_image():
    """
    Create a test image with side-by-side shadow regions.
    
    Returns:
        Test image with shadows
    """
    # Create a 600x800 image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # Add some colored rectangles
    cv2.rectangle(img, (50, 50), (350, 250), (100, 150, 200), -1)
    cv2.rectangle(img, (450, 50), (750, 250), (200, 100, 150), -1)
    cv2.rectangle(img, (50, 350), (350, 550), (150, 200, 100), -1)
    cv2.rectangle(img, (450, 350), (750, 550), (100, 200, 150), -1)
    
    # Add shadow on left half (darken)
    shadow_mask = np.zeros_like(img, dtype=np.float32)
    shadow_mask[:, :400] = 0.4  # Left side darker
    shadow_mask[:, 400:] = 1.0  # Right side normal
    
    img = (img * shadow_mask).astype(np.uint8)
    
    # Add some text
    cv2.putText(img, "Shadow Side", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    cv2.putText(img, "Normal Side", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    
    return img


def test_ssr_on_shadow_image(input_path=None, sigma_values=[15, 80, 150]):
    """
    Test SSR on shadow image with different sigma values.
    
    Args:
        input_path: Path to input image (if None, creates test image)
        sigma_values: List of sigma values to test
    """
    # Load or create test image
    if input_path and Path(input_path).exists():
        img = cv2.imread(input_path)
        print(f"Loaded image from: {input_path}")
    else:
        img = create_test_shadow_image()
        print("Created synthetic shadow test image")
        # Save the test image
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "test_shadow_image.png"), img)
        print(f"Saved test image to: {output_dir / 'test_shadow_image.png'}")
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure for comparison
    num_results = len(sigma_values) + 1
    fig, axes = plt.subplots(1, num_results, figsize=(5 * num_results, 5))
    
    # Show original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (with shadow)")
    axes[0].axis('off')
    
    # Apply SSR with different sigma values
    for idx, sigma in enumerate(sigma_values, 1):
        print(f"Processing with sigma={sigma}...")
        result = process_color_image(img, sigma=sigma)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(result_rgb)
        axes[idx].set_title(f"SSR (σ={sigma})")
        axes[idx].axis('off')
        
        # Save result
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"ssr_result_sigma_{sigma}.png"
        cv2.imwrite(str(output_path), result)
        print(f"Saved result to: {output_path}")
    
    plt.tight_layout()
    
    # Save comparison figure
    comparison_path = output_dir / "ssr_comparison.png"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {comparison_path}")


def main():
    """
    Main function to run SSR tests.
    """
    print("=" * 60)
    print("Single Scale Retinex (SSR) Test")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Test on side.jpg
    side_image = script_dir / "side.jpg"
    if side_image.exists():
        print("\nTest 1: side.jpg")
        test_ssr_on_shadow_image(input_path=str(side_image))
    else:
        print(f"\nWarning: {side_image} not found")
    
    # Test on side_shadow.jpg
    side_shadow_image = script_dir / "side_shadow.jpg"
    if side_shadow_image.exists():
        print("\nTest 2: side_shadow.jpg")
        test_ssr_on_shadow_image(input_path=str(side_shadow_image))
    else:
        print(f"\nWarning: {side_shadow_image} not found")


if __name__ == "__main__":
    main()
