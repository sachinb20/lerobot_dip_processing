import cv2
import numpy as np

def single_scale_retinex(img, sigma=50):
    # Convert to float
    img = img.astype(np.float32) + 1.0

    # Gaussian surround (illumination estimate)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    # SSR formula: log(I) - log(blur(I))
    retinex = np.log(img) - np.log(blur)

    # Normalize to 0–255 for display
    retinex = retinex - np.min(retinex)
    retinex = retinex / np.max(retinex)
    retinex = (retinex * 255).astype(np.uint8)

    return retinex


img = cv2.imread("shadow_removal/side_shadow.jpg")

    # Apply SSR
result = single_scale_retinex(img, sigma=50)

    # Save / Show
cv2.imwrite("shadow_enhanced_side.jpg", result)