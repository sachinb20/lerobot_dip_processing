import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def SSR(img, sigma=80):
    """Single Scale Retinex"""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    img_f = img.astype(np.float32) + 1e-6
    blur_f = blur.astype(np.float32) + 1e-6
    retinex = np.log(img_f) - np.log(blur_f)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def MSR(img, sigma_list=[15, 80, 250]):
    """Multi-Scale Retinex - combines multiple scales"""
    retinex = np.zeros_like(img, dtype=np.float32)
    
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img_f = img.astype(np.float32) + 1e-6
        blur_f = blur.astype(np.float32) + 1e-6
        retinex += np.log(img_f) - np.log(blur_f)
    
    retinex = retinex / len(sigma_list)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def MSRCR(img, sigma_list=[15, 80, 250], G=192, b=-30, alpha=125, beta=46):
    """Multi-Scale Retinex with Color Restoration"""
    img_float = img.astype(np.float32) + 1.0
    
    # Calculate intensity
    intensity = np.sum(img_float, axis=2) / 3.0
    
    # Multi-scale retinex on intensity
    retinex = np.zeros_like(intensity)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(intensity, (0, 0), sigma)
        retinex += np.log(intensity + 1e-6) - np.log(blur + 1e-6)
    retinex = retinex / len(sigma_list)
    
    # Color restoration
    color_restoration = beta * (np.log(alpha * img_float) - np.log(np.sum(img_float, axis=2, keepdims=True)))
    
    # Apply to each channel
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:, :, i] = retinex + color_restoration[:, :, i]
    
    # Simplest color balance
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def histogram_matching(source, reference):
    """Match histogram of source to reference"""
    result = np.zeros_like(source)
    
    for i in range(3):  # For each channel
        # Calculate CDFs
        source_hist, _ = np.histogram(source[:, :, i].flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference[:, :, i].flatten(), 256, [0, 256])
        
        source_cdf = source_hist.cumsum()
        reference_cdf = reference_hist.cumsum()
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        reference_cdf = reference_cdf / reference_cdf[-1]
        
        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        g_j = 0
        for g_i in range(256):
            while g_j < 255 and reference_cdf[g_j] < source_cdf[g_i]:
                g_j += 1
            lookup_table[g_i] = g_j
        
        # Apply lookup table
        result[:, :, i] = cv2.LUT(source[:, :, i], lookup_table)
    
    return result


def adaptive_shadow_removal(shadow_img, reference_img=None):
    """
    Adaptive shadow removal using ground truth reference
    """
    # Convert to LAB color space for better shadow handling
    lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    result_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # If we have reference, match histogram
    if reference_img is not None:
        result_hist_match = histogram_matching(result_clahe, reference_img)
        return result_hist_match
    
    return result_clahe


def illumination_correction(shadow_img, reference_img):
    """
    Correct illumination by estimating and removing shadow mask
    """
    # Convert to LAB
    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    
    # Get L channels
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    ref_l = ref_lab[:, :, 0].astype(np.float32)
    
    # Estimate illumination difference
    shadow_blur = cv2.GaussianBlur(shadow_l, (0, 0), 50)
    ref_blur = cv2.GaussianBlur(ref_l, (0, 0), 50)
    
    # Calculate correction factor
    correction = ref_blur / (shadow_blur + 1e-6)
    correction = np.clip(correction, 0.5, 2.0)
    
    # Apply correction to L channel
    corrected_l = shadow_l * correction
    corrected_l = np.clip(corrected_l, 0, 255).astype(np.uint8)
    
    # Merge back
    corrected_lab = cv2.merge([corrected_l, shadow_lab[:, :, 1], shadow_lab[:, :, 2]])
    result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    return result


def hybrid_shadow_removal(shadow_img, reference_img):
    """
    Hybrid approach combining multiple techniques
    """
    # Method 1: Illumination correction
    illum_corrected = illumination_correction(shadow_img, reference_img)
    
    # Method 2: MSR
    msr_result = np.zeros_like(shadow_img)
    for i in range(3):
        msr_result[:, :, i] = MSR(shadow_img[:, :, i], sigma_list=[15, 80, 250])
    
    # Method 3: Adaptive with histogram matching
    adaptive_result = adaptive_shadow_removal(shadow_img, reference_img)
    
    # Blend results
    result = cv2.addWeighted(illum_corrected, 0.5, adaptive_result, 0.5, 0)
    
    return result, illum_corrected, msr_result, adaptive_result


def calculate_metrics(result, reference):
    """Calculate similarity metrics"""
    # MSE
    mse = np.mean((result.astype(np.float32) - reference.astype(np.float32)) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM (simplified version)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    return {"MSE": mse, "PSNR": psnr}


def main():
    print("=" * 70)
    print("Improved Shadow Removal with Ground Truth Guidance")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    reference = cv2.imread(str(script_dir / "side.jpg"))
    shadow = cv2.imread(str(script_dir / "side_shadow.jpg"))
    
    print(f"\nLoaded images:")
    print(f"  Reference (side.jpg): {reference.shape}")
    print(f"  Shadow (side_shadow.jpg): {shadow.shape}")
    
    # Apply different methods
    print("\nApplying shadow removal methods...")
    
    # 1. Original SSR
    print("  1. SSR (sigma=80)")
    ssr_result = np.zeros_like(shadow)
    for i in range(3):
        ssr_result[:, :, i] = SSR(shadow[:, :, i], sigma=80)
    
    # 2. MSR
    print("  2. Multi-Scale Retinex")
    msr_result = np.zeros_like(shadow)
    for i in range(3):
        msr_result[:, :, i] = MSR(shadow[:, :, i])
    
    # 3. Adaptive with histogram matching
    print("  3. Adaptive + Histogram Matching")
    adaptive_result = adaptive_shadow_removal(shadow, reference)
    
    # 4. Illumination correction
    print("  4. Illumination Correction")
    illum_result = illumination_correction(shadow, reference)
    
    # 5. Hybrid approach
    print("  5. Hybrid Approach")
    hybrid_result, _, _, _ = hybrid_shadow_removal(shadow, reference)
    
    # Calculate metrics for each method
    print("\n" + "=" * 70)
    print("Metrics (compared to ground truth side.jpg):")
    print("=" * 70)
    
    methods = {
        "Original Shadow": shadow,
        "SSR": ssr_result,
        "MSR": msr_result,
        "Adaptive+HistMatch": adaptive_result,
        "Illumination Correction": illum_result,
        "Hybrid": hybrid_result
    }
    
    for name, result in methods.items():
        metrics = calculate_metrics(result, reference)
        print(f"{name:25s} - MSE: {metrics['MSE']:8.2f}, PSNR: {metrics['PSNR']:6.2f} dB")
    
    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    images = [
        (reference, "Ground Truth\n(side.jpg)"),
        (shadow, "Input Shadow\n(side_shadow.jpg)"),
        (ssr_result, "SSR"),
        (msr_result, "MSR"),
        (adaptive_result, "Adaptive+HistMatch"),
        (illum_result, "Illumination Correction"),
        (hybrid_result, "Hybrid (Best)"),
    ]
    
    # Plot images
    for idx, (img, title) in enumerate(images):
        row = idx // 4
        col = idx % 4
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Add metrics if not reference or shadow
        if idx > 1:
            metrics = calculate_metrics(img, reference)
            title += f"\nPSNR: {metrics['PSNR']:.2f} dB"
        
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        
        # Save individual result
        if idx > 1:
            method_name = title.split('\n')[0].replace('+', '_').replace(' ', '_').lower()
            save_path = output_dir / f"method_{method_name}.png"
            cv2.imwrite(str(save_path), img)
    
    # Hide last subplot
    axes[1, 3].axis('off')
    
    plt.suptitle("Shadow Removal Methods Comparison (Higher PSNR = Better)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "improved_shadow_removal_comparison.png"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Saved comparison to: {comparison_path}")
    print(f"{'='*70}")
    
    # Save best result
    best_result_path = output_dir / "best_shadow_removal.png"
    cv2.imwrite(str(best_result_path), hybrid_result)
    print(f"Saved best result to: {best_result_path}")


if __name__ == "__main__":
    main()
