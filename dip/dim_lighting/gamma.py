import cv2
import numpy as np

def gamma_correct(img, gamma=0.5):
    img_norm = img / 255.0
    img_gamma = np.power(img_norm, gamma)
    return (img_gamma * 255).astype(np.uint8)

img = cv2.imread("front_dim.jpg")
enhanced = gamma_correct(img, gamma=0.5)

cv2.imwrite("enhanced_front.jpg", enhanced)