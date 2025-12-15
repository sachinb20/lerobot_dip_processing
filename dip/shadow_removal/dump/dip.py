import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def learn_illumination_params(shadow_img, reference_img):
    """
    Learn illumination correction parameters from shadow and reference image pair.
    
    Returns:
        Dictionary with learned parameters
    """
    # Convert to LAB
    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    
    # Extract L channels
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    ref_l = ref_lab[:, :, 0].astype(np.float32)
    
    # Estimate illumination
    shadow_blur = cv2.GaussianBlur(shadow_l, (0, 0), 50)
    ref_blur = cv2.GaussianBlur(ref_l, (0, 0), 50)
    
    # Calculate average correction factor
    correction = ref_blur / (shadow_blur + 1e-6)
    
    # Get statistics
    params = {
        'mean_correction': np.mean(correction),
        'std_correction': np.std(correction),
        'median_correction': np.median(correction),
        'min_correction': np.min(correction),
        'max_correction': np.max(correction),
        'target_mean_luminance': np.mean(ref_l),
        'target_std_luminance': np.std(ref_l),
    }
    
    return params


def illumination_correction_no_reference(shadow_img, correction_factor=1.3, sigma=50):
    """
    Correct illumination WITHOUT needing a reference image.
    Uses learned correction factor from training.
    
    Args:
        shadow_img: Image with shadows (BGR format)
        correction_factor: Learned correction factor (default 1.3)
        sigma: Gaussian blur sigma for illumination estimation
        
    Returns:
        Shadow-corrected image (BGR format)
    """
    # Convert to LAB color space
    shadow_lab = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2LAB)
    
    # Extract L (luminance) channel
    shadow_l = shadow_lab[:, :, 0].astype(np.float32)
    
    # Estimate illumination using Gaussian blur
    shadow_blur = cv2.GaussianBlur(shadow_l, (0, 0), sigma)
    
    # Apply learned correction factor
    # Areas with low illumination (shadows) get boosted more
    correction = correction_factor * (1.0 - shadow_blur / 255.0) + 1.0
    correction = np.clip(correction, 0.8, 2.0)
    
    # Apply correction to L channel
    corrected_l = shadow_l * correction
    corrected_l = np.clip(corrected_l, 0, 255).astype(np.uint8)
    
    # Merge back with original color channels
    corrected_lab = cv2.merge([corrected_l, shadow_lab[:, :, 1], shadow_lab[:, :, 2]])
    
    # Convert back to BGR
    result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    return result


def adaptive_illumination_correction(shadow_img, target_mean=None, sigma=50):
    """
    Adaptive illumination correction that normalizes to target luminance.
    
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
    
    # If target mean is provided, normalize to it
    if target_mean is not None:
        current_mean = np.mean(shadow_blur)
        global_correction = target_mean / (current_mean + 1e-6)
        global_correction = np.clip(global_correction, 0.7, 1.5)
    else:
        global_correction = 1.0
    
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


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def learn_parameters_from_training_data():
    """
    Learn correction parameters from the provided reference images.
    This is done once during training/calibration.
    """
    script_dir = Path(__file__).parent
    
    print("\n" + "="*80)
    print("LEARNING PHASE: Extracting parameters from reference images")
    print("="*80)
    
    training_pairs = [
        ('side_shadow.jpg', 'side.jpg'),
        ('front_shadow.jpg', 'front.jpg'),
    ]
    
    learned_params = {}
    
    for shadow_name, ref_name in training_pairs:
        shadow_path = script_dir / shadow_name
        ref_path = script_dir / ref_name
        
        if not shadow_path.exists() or not ref_path.exists():
            continue
        
        shadow_img = cv2.imread(str(shadow_path))
        ref_img = cv2.imread(str(ref_path))
        
        params = learn_illumination_params(shadow_img, ref_img)
        learned_params[shadow_name] = params
        
        print(f"\n{shadow_name} -> {ref_name}:")
        print(f"  Mean correction factor: {params['mean_correction']:.3f}")
        print(f"  Median correction factor: {params['median_correction']:.3f}")
        print(f"  Target mean luminance: {params['target_mean_luminance']:.2f}")
    
    # Calculate average parameters across all training pairs
    if learned_params:
        avg_correction = np.mean([p['mean_correction'] for p in learned_params.values()])
        avg_target_lum = np.mean([p['target_mean_luminance'] for p in learned_params.values()])
        
        print(f"\n{'='*80}")
        print(f"LEARNED PARAMETERS (averaged):")
        print(f"  Correction factor: {avg_correction:.3f}")
        print(f"  Target luminance: {avg_target_lum:.2f}")
        print(f"{'='*80}")
        
        return {
            'correction_factor': avg_correction,
            'target_mean_luminance': avg_target_lum
        }
    
    # Default fallback parameters
    return {
        'correction_factor': 1.3,
        'target_mean_luminance': 120.0
    }


def test_without_reference(shadow_path, reference_path, learned_params, output_dir):
    """
    Test shadow removal WITHOUT using reference image.
    Uses only learned parameters.
    """
    # Load images
    shadow_img = cv2.imread(str(shadow_path))
    reference_img = cv2.imread(str(reference_path))  # Only for evaluation
    
    if shadow_img is None:
        print(f"Error: Could not load {shadow_path}")
        return None
    
    # Method 1: Using learned correction factor
    start_time = time.time()
    result1 = illumination_correction_no_reference(
        shadow_img, 
        correction_factor=learned_params['correction_factor']
    )
    time1 = (time.time() - start_time) * 1000
    
    # Method 2: Using adaptive correction with target luminance
    start_time = time.time()
    result2 = adaptive_illumination_correction(
        shadow_img,
        target_mean=learned_params['target_mean_luminance']
    )
    time2 = (time.time() - start_time) * 1000
    
    # Calculate metrics (only for evaluation, not used in correction)
    if reference_img is not None:
        psnr_before = calculate_psnr(shadow_img, reference_img)
        psnr_method1 = calculate_psnr(result1, reference_img)
        psnr_method2 = calculate_psnr(result2, reference_img)
    else:
        psnr_before = psnr_method1 = psnr_method2 = 0.0
    
    # Save results
    output_path1 = output_dir / f"corrected_{shadow_path.stem}_method1.png"
    output_path2 = output_dir / f"corrected_{shadow_path.stem}_method2.png"
    cv2.imwrite(str(output_path1), result1)
    cv2.imwrite(str(output_path2), result2)
    
    return {
        'shadow_path': shadow_path,
        'reference_path': reference_path,
        'shadow_img': shadow_img,
        'reference_img': reference_img,
        'result1': result1,
        'result2': result2,
        'time1_ms': time1,
        'time2_ms': time2,
        'psnr_before': psnr_before,
        'psnr_method1': psnr_method1,
        'psnr_method2': psnr_method2,
        'output_path1': output_path1,
        'output_path2': output_path2,
    }


def create_comparison_figure(results_list, output_dir):
    """Create comparison figure showing results."""
    num_tests = len(results_list)
    fig, axes = plt.subplots(num_tests, 4, figsize=(20, 5 * num_tests))
    
    if num_tests == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results_list):
        # Reference
        axes[idx, 0].imshow(cv2.cvtColor(result['reference_img'], cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Ground Truth\n{result['reference_path'].name}", 
                               fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Shadow input
        axes[idx, 1].imshow(cv2.cvtColor(result['shadow_img'], cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"Shadow Input\n{result['shadow_path'].name}\nPSNR: {result['psnr_before']:.2f} dB", 
                               fontsize=11, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Method 1
        axes[idx, 2].imshow(cv2.cvtColor(result['result1'], cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f"Method 1: Fixed Factor\nPSNR: {result['psnr_method1']:.2f} dB\nTime: {result['time1_ms']:.2f} ms", 
                               fontsize=11, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Method 2
        axes[idx, 3].imshow(cv2.cvtColor(result['result2'], cv2.COLOR_BGR2RGB))
        axes[idx, 3].set_title(f"Method 2: Adaptive\nPSNR: {result['psnr_method2']:.2f} dB\nTime: {result['time2_ms']:.2f} ms", 
                               fontsize=11, fontweight='bold', color='green')
        axes[idx, 3].axis('off')
    
    plt.suptitle("Shadow Removal WITHOUT Reference Image (Test Mode)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "test_without_reference.png"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {comparison_path}")


def main():
    """
    Main function:
    1. Learn parameters from reference images (training phase)
    2. Test shadow removal WITHOUT reference images (test phase)
    """
    print("="*80)
    print("Shadow Removal: Learning and Testing WITHOUT Reference")
    print("="*80)
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # PHASE 1: Learn parameters from training data
    learned_params = learn_parameters_from_training_data()
    
    # PHASE 2: Test without reference
    print("\n" + "="*80)
    print("TEST PHASE: Applying shadow removal WITHOUT reference images")
    print("="*80)
    
    test_cases = [
        {
            'shadow': script_dir / "side_shadow.jpg",
            'reference': script_dir / "side.jpg",  # Only for evaluation
            'name': 'Side View'
        },
        {
            'shadow': script_dir / "front_shadow.jpg",
            'reference': script_dir / "front.jpg",  # Only for evaluation
            'name': 'Front View'
        }
    ]
    
    results_list = []
    
    for test in test_cases:
        print(f"\n{'-'*80}")
        print(f"Testing: {test['name']}")
        print(f"Shadow image: {test['shadow'].name}")
        print(f"(Reference {test['reference'].name} used ONLY for evaluation metrics)")
        
        if not test['shadow'].exists():
            print(f"Warning: {test['shadow']} not found, skipping...")
            continue
        
        result = test_without_reference(
            test['shadow'], 
            test['reference'], 
            learned_params, 
            output_dir
        )
        
        if result:
            results_list.append(result)
            
            print(f"\nResults:")
            print(f"  Method 1 (Fixed Factor):")
            print(f"    Time: {result['time1_ms']:.2f} ms")
            print(f"    PSNR: {result['psnr_method1']:.2f} dB (vs ground truth)")
            print(f"    Improvement: +{result['psnr_method1'] - result['psnr_before']:.2f} dB")
            
            print(f"  Method 2 (Adaptive):")
            print(f"    Time: {result['time2_ms']:.2f} ms")
            print(f"    PSNR: {result['psnr_method2']:.2f} dB (vs ground truth)")
            print(f"    Improvement: +{result['psnr_method2'] - result['psnr_before']:.2f} dB")
    
    # Summary
    if results_list:
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY (Test without reference)")
        print(f"{'='*80}")
        print(f"{'Test':<20} {'Method':<15} {'Time (ms)':<12} {'PSNR':<12} {'Improvement':<12}")
        print(f"{'-'*80}")
        
        for result in results_list:
            test_name = result['shadow_path'].stem
            print(f"{test_name:<20} {'Fixed Factor':<15} {result['time1_ms']:<12.2f} "
                  f"{result['psnr_method1']:<12.2f} +{result['psnr_method1']-result['psnr_before']:<11.2f}")
            print(f"{'':<20} {'Adaptive':<15} {result['time2_ms']:<12.2f} "
                  f"{result['psnr_method2']:<12.2f} +{result['psnr_method2']-result['psnr_before']:<11.2f}")
        
        print(f"{'='*80}")
        
        # Create comparison figure
        create_comparison_figure(results_list, output_dir)
    
    print("\n" + "="*80)
    print("IMPORTANT: At test time, NO reference image is needed!")
    print("Only the learned parameters are used for shadow removal.")
    print("="*80)


if __name__ == "__main__":
    main()
