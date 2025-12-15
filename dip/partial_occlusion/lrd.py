import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, img_as_float
from scipy.signal import convolve2d

def make_gaussian_guess(size, sigma):
    """
    Creates a Gaussian kernel to serve as our 'guessed' degradation function.
    If you don't know the blur, a Gaussian is the safest mathematical assumption.
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g / g.sum()

def deblur_blind(image_path, iterations=30, kernel_size=15, sigma=5):
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert BGR (OpenCV) to RGB (Matplotlib)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float (0.0 to 1.0) for processing
    img_float = img_as_float(img)

    # 2. Create a "Guessed" PSF (Point Spread Function)
    # Since we don't know the degradation, we assume a Gaussian blur.
    # You can tweak 'sigma' to match the severity of the blur in your image.
    psf = make_gaussian_guess(kernel_size, sigma)

    print(f"Processing... This may take a moment for {iterations} iterations.")

    # 3. Apply Richardson-Lucy Deconvolution
    # We must process each color channel (R, G, B) separately
    deblurred_channels = []
    for i in range(3): # Loop through R, G, B
        channel = img_float[:, :, i]
        
        # The core restoration algorithm
        restored = restoration.richardson_lucy(channel, psf, num_iter=iterations, clip=False)
        deblurred_channels.append(restored)

    # 4. Merge channels back
    restored_img = np.dstack(deblurred_channels)
    
    # Clip values to stay within valid color range [0, 1]
    restored_img = np.clip(restored_img, 0, 1)

    # 5. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    
    ax[0].imshow(img)
    ax[0].set_title("Original Blurry Input")
    ax[0].axis('off')

    ax[1].imshow(restored_img)
    ax[1].set_title(f"Restored (Richardson-Lucy)\nSigma Guess: {sigma}, Iter: {iterations}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# --- RUN THE SCRIPT ---
# Replace 'blurry_photo.jpg' with your actual file name
# Tweak 'sigma': Higher values = stronger deblurring (but more noise)
if __name__ == "__main__":
    # Create a dummy image if you don't have one, or provide path
    deblur_blind('dcpop.png', iterations=10, sigma=1.75)
    print("Please provide a valid image path in the code above.")