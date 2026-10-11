# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a 1D Gaussian kernel as a column vector.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        sigma = float(self.sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        k_size = int(6 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1

        half = k_size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        g = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g[:, np.newaxis]

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        from scipy.ndimage import convolve1d

        kernel = self.gauss_kernel()
        kernel_1d = kernel.ravel()
        k_size = len(kernel_1d)
        pad = k_size // 2

        img = image.astype(np.float64)
        img_padded = np.pad(img, pad, mode='reflect')

        
        step1 = convolve1d(img_padded, kernel_1d, axis=1, mode='constant',cval=0.0)
        step2 = convolve1d(step1, kernel_1d, axis=0, mode='constant',cval=0.0)

        return np.clip(np.round(step2[pad:-pad, pad:-pad]), 0, 255).astype(np.uint8)

