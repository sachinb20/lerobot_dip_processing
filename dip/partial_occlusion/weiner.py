"""
Wiener Deconvolution for Image Deblurring

Usage:
    python wiener_deblur.py --image blurred.png --kernel_size 15 --sigma 3 --k 0.01

Author: Sachin
"""

import argparse
import cv2
import numpy as np
import os


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def wiener_deconvolution(img, kernel, K):
    """
    Perform Wiener deconvolution.

    img    : blurred image (grayscale, float32)
    kernel : blur kernel (PSF)
    K      : noise-to-signal power ratio
    """
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape

    # Pad kernel to image size
    kernel_padded = np.zeros_like(img)
    kernel_padded[:ker_h, :ker_w] = kernel

    # Shift kernel center to (0,0)
    kernel_padded = np.roll(kernel_padded, -ker_h // 2, axis=0)
    kernel_padded = np.roll(kernel_padded, -ker_w // 2, axis=1)

    # FFTs
    IMG = np.fft.fft2(img)
    KER = np.fft.fft2(kernel_padded)

    # Wiener filter
    KER_conj = np.conj(KER)
    wiener_filter = KER_conj / (np.abs(KER)**2 + K)

    # Apply filter
    deblurred = np.fft.ifft2(IMG * wiener_filter)
    deblurred = np.real(deblurred)

    return np.clip(deblurred, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Wiener Deconvolution Image Deblurring")
    parser.add_argument("--image", type=str, required=True, help="Path to blurred image")
    parser.add_argument("--kernel_size", type=int, default=15, help="Gaussian kernel size")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian kernel sigma")
    parser.add_argument("--k", type=float, default=0.01, help="Noise-to-signal ratio (K)")
    parser.add_argument("--output", type=str, default="deblurred.png", help="Output filename")

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    img = img.astype(np.float32) / 255.0

    # Create blur kernel
    kernel = gaussian_kernel(args.kernel_size, args.sigma)

    # Wiener deconvolution
    deblurred = wiener_deconvolution(img, kernel, args.k)

    # Save output
    deblurred_uint8 = (deblurred * 255).astype(np.uint8)
    cv2.imwrite(args.output, deblurred_uint8)

    print(f"Deblurred image saved to: {args.output}")


if __name__ == "__main__":
    main()
