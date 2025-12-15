import cv2
import numpy as np
import time
from pathlib import Path


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
        'target_mean_luminance': np.mean(ref_l),
        'target_std_luminance': np.std(ref_l),
    }
    
    return params


def load_learned_parameters():
    """
    Load and learn parameters from reference images.
    This is done once at startup.
    """
    script_dir = Path(__file__).parent
    
    print("\n" + "="*80)
    print("Loading learned parameters from reference images...")
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
            print(f"Warning: {shadow_name} or {ref_name} not found, skipping...")
            continue
        
        shadow_img = cv2.imread(str(shadow_path))
        ref_img = cv2.imread(str(ref_path))
        
        params = learn_illumination_params(shadow_img, ref_img)
        learned_params[shadow_name] = params
        
        print(f"\n{shadow_name} -> {ref_name}:")
        print(f"  Target mean luminance: {params['target_mean_luminance']:.2f}")
    
    # Calculate average parameters
    if learned_params:
        avg_target_lum = np.mean([p['target_mean_luminance'] for p in learned_params.values()])
        
        print(f"\n{'='*80}")
        print(f"LEARNED PARAMETERS (averaged):")
        print(f"  Target luminance: {avg_target_lum:.2f}")
        print(f"{'='*80}\n")
        
        return avg_target_lum
    
    # Default fallback
    print("Warning: No reference images found, using default target luminance: 132.0")
    return 132.0


def adaptive_illumination_correction(shadow_img, target_mean=132.0, sigma=50):
    """
    Adaptive illumination correction that normalizes to target luminance.
    This is Method 2 from dip.py - the best performing method.
    
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
    
    # Normalize to target mean
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


def main():
    """
    Real-time shadow removal on camera feeds.
    Uses Method 2 (Adaptive) with learned parameters.
    Saves frames to disk since OpenCV GUI is not available.
    """
    print("="*80)
    print("Real-Time Shadow Removal - Method 2 (Adaptive)")
    print("="*80)
    
    # Load learned parameters from reference images
    target_luminance = load_learned_parameters()
    
    # Setup output directory
    output_dir = Path(__file__).parent / "output" / "realtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Camera configuration
    cameras = [
        {'id': '/dev/video5', 'name': 'Camera 1 (video5)'},
        {'id': '/dev/video7', 'name': 'Camera 2 (video7)'},
    ]
    
    # Try to open cameras
    caps = []
    for cam in cameras:
        cap = cv2.VideoCapture(cam['id'])
        if cap.isOpened():
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            caps.append({'cap': cap, 'name': cam['name'], 'id': cam['id'], 'index': len(caps)})
            print(f"✓ Opened {cam['name']}")
        else:
            print(f"✗ Failed to open {cam['name']}")
    
    if not caps:
        print("\nError: No cameras could be opened!")
        print("Available cameras should be at /dev/video4 and /dev/video6")
        return
    
    print(f"\n{'='*80}")
    print(f"Successfully opened {len(caps)} camera(s)")
    print(f"Target luminance: {target_luminance:.2f}")
    print(f"{'='*80}")
    print("\nProcessing frames and saving to disk...")
    print("Press Ctrl+C to stop")
    print(f"{'='*80}\n")
    
    # FPS calculation
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    frame_count = 0
    
    # Adjustable target luminance
    current_target_lum = target_luminance
    
    try:
        while True:
            frames = []
            corrected_frames = []
            
            # Capture frames from all cameras
            for cam_info in caps:
                ret, frame = cam_info['cap'].read()
                if ret:
                    frames.append((frame, cam_info['name'], cam_info['index']))
                else:
                    print(f"Warning: Failed to read from {cam_info['name']}")
                    frames.append((None, cam_info['name'], cam_info['index']))
            
            # Process frames
            for frame, cam_name, cam_idx in frames:
                if frame is None:
                    continue
                
                # Apply shadow removal
                start_time = time.time()
                corrected = adaptive_illumination_correction(frame, target_mean=current_target_lum)
                process_time = (time.time() - start_time) * 1000
                
                corrected_frames.append((frame, corrected, cam_name, cam_idx, process_time))
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
                
                # Print status every second
                print(f"FPS: {fps} | Processing time: {process_time:.1f}ms | Target lum: {current_target_lum:.1f}")
            
            # Save frames periodically (every 30 frames = ~1 second at 30fps)
            if frame_count % 30 == 0:
                for original, corrected, cam_name, cam_idx, process_time in corrected_frames:
                    # Create side-by-side comparison
                    h, w = original.shape[:2]
                    combined = np.hstack([original, corrected])
                    
                    # Add text overlay
                    cv2.putText(combined, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(combined, "Shadow Removed", (w + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f"FPS: {fps}", (10, h - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(combined, f"Process: {process_time:.1f}ms", (10, h - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(combined, f"Target Lum: {current_target_lum:.1f}", (10, h - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Save combined frame
                    filename = f"camera{cam_idx+1}_latest.png"
                    filepath = output_dir / filename
                    cv2.imwrite(str(filepath), combined)
                    
                    # Also save just the corrected frame
                    corrected_filename = f"camera{cam_idx+1}_corrected_latest.png"
                    corrected_filepath = output_dir / corrected_filename
                    cv2.imwrite(str(corrected_filepath), corrected)
            
            frame_count += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Cleanup
    for cam_info in caps:
        cam_info['cap'].release()
    
    print("\nCameras released.")
    print(f"Latest frames saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
