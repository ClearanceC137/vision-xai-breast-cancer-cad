# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:41:10 2025
@author: clear

This script demonstrates a prototype image preprocessing and segmentation pipeline
for breast cancer image analysis. It uses a simplified U-Net-like structure built
manually with NumPy (without deep learning frameworks).
"""

import numpy as np
#from preprocessing import Preprocessing   # Custom preprocessing class for handling dataset
import matplotlib.pyplot as plt
from scipy.ndimage import zoom            # For image resizing
import cv2                                # OpenCV for image manipulation

# ------------------------------------------------------
# Load Dataset and Prepare Label Encoding
# ------------------------------------------------------

# Path to dataset class mapping file (CSV format)
parent_dir = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\Prototype\dicom_class_mapping.csv"

# Initialize Preprocessing class
#BreastCancerProcessor = Preprocessing(parent_dir)

# Fit label encoder for class names
#BreastCancerProcessor.fit_label_encoder()

# ------------------------------------------------------
# Define Image Segmentation Class
# ------------------------------------------------------

class ImageSegmentation:
    """
    Implements a simplified U-Net-like segmentation pipeline using only NumPy.
    This is for educational/experimental purposes (not production ready).
    """

    def __init__(self):
        self.original_image = None         # Raw image input (4D array: batch, H, W, C)
        self.preprocessed_image = None     # Image feature map after encoding
        self.segmented_mask = None         # Placeholder for segmentation output

    def load_image(self, image_data):
        """
        Loads and validates input image data, ensuring it has batch dimension.
        """
        if image_data.ndim == 3:  # Single image without batch dimension
            image_data = np.expand_dims(image_data, axis=0)
        elif image_data.ndim != 4:  # Must be 4D
            raise ValueError("Invalid image array shape.")

        self.original_image = image_data

    def conv2d(self, input, kernel, padding='same'):
        """
        Performs a 2D convolution using a sliding window.
        - input: 4D array (batch, H, W, C)
        - kernel: 4D filter (kh, kw, in_channels, out_channels)
        """
        k = kernel.shape[0]
        if padding == 'same':
            pad = k // 2
            input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

        b, h, w, c = input.shape
        kh, kw, _, filters = kernel.shape
        output = np.zeros((b, h, w, filters))

        # Sliding window convolution
        for i in range(h):
            for j in range(w):
                input_patch = input[:, i:i + kh, j:j + kw, :]
                if input_patch.shape[1] != kh or input_patch.shape[2] != kw:
                    continue
                for f in range(filters):
                    output[:, i, j, f] = np.sum(input_patch * kernel[:, :, :, f], axis=(1, 2, 3))
        return output

    def max_pool(self, input):
        """
        Applies 2x2 max pooling operation to reduce spatial dimensions by half.
        """
        b, h, w, c = input.shape
        h_new = h // 2
        w_new = w // 2
        output = np.zeros((b, h_new, w_new, c))
        for i in range(h_new):
            for j in range(w_new):
                h_start = i * 2
                w_start = j * 2
                patch = input[:, h_start:h_start + 2, w_start:w_start + 2, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
        return output

    def upsample(self, input):
        """
        Performs nearest-neighbor upsampling (x2 enlargement).
        """
        b, h, w, c = input.shape
        output = np.zeros((b, h * 2, w * 2, c))
        for i in range(h):
            for j in range(w):
                output[:, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2, :] = input[:, i:i + 1, j:j + 1, :]
        return output

    def relu(self, x):
        """Applies ReLU activation (max(0, x))."""
        return np.maximum(0, x)

    def sigmoid(self, x):
        """Applies sigmoid activation (1 / (1 + exp(-x)))."""
        return 1 / (1 + np.exp(-x))

    def postprocess_segmented_image(self):
        """
        Resizes images to nearest lower power-of-two dimensions for UNet compatibility.
        """
        if self.preprocessed_image.ndim != 4:
            raise ValueError("Expected image with shape (batch, height, width, channels)")

        batch_size, h, w, c = self.preprocessed_image.shape

        # Helper: find nearest lower power of two
        def nearest_power_of_two(x):
            powers = [2 ** i for i in range(4, 10)]  # from 16 to 512
            powers = [p for p in powers if p <= x]
            return max(powers) if powers else x

        new_h = nearest_power_of_two(h)
        new_w = nearest_power_of_two(w)

        # Resize each image using zoom
        zoom_h = new_h / h
        zoom_w = new_w / w
        resized_batch = np.zeros((batch_size, new_h, new_w, c))

        for i in range(batch_size):
            for ch in range(c):
                resized_batch[i, :, :, ch] = zoom(self.preprocessed_image[i, :, :, ch], (zoom_h, zoom_w), order=1)

        print("Images downscaled :", resized_batch.shape)

    def average_pool(self, input, pool_size=5):
        """
        Applies average pooling to reduce spatial dimensions.
        - pool_size: size of pooling window (default=5)
        """
        b, h, w, c = input.shape
        h_new = h // pool_size
        w_new = w // pool_size
        output = np.zeros((b, h_new, w_new, c))

        for i in range(h_new):
            for j in range(w_new):
                h_start = i * pool_size
                w_start = j * pool_size
                patch = input[:, h_start:h_start + pool_size, w_start:w_start + pool_size, :]
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))
        return output

    def unet(self):
        """
        Simulates a basic U-Net encoder path:
        - Conv -> Pool -> Conv -> Pool -> Bottleneck -> AveragePool
        """
        input_image = self.original_image

        # Encoder block 1
        c1 = self.relu(self.conv2d(input_image, np.random.randn(3, 3, input_image.shape[-1], 16)))
        p1 = self.max_pool(c1)

        # Encoder block 2
        c2 = self.relu(self.conv2d(p1, np.random.randn(3, 3, 16, 32)))
        p2 = self.max_pool(c2)

        # Bottleneck
        bn = self.relu(self.conv2d(p2, np.random.randn(3, 3, 32, 64)))

        # 🔹 Average pool bottleneck for CNN input
        pooled_bn = self.average_pool(bn, pool_size=3)

        # Save downsampled bottleneck for later use
        self.preprocessed_image = pooled_bn
        print("✅ UNet output shape (after avg pooling):", self.preprocessed_image.shape)

        

        
        print("✅ UNet output shape:", self.preprocessed_image.shape)

    def display_segmented_image(self, image_segmented):
        """
        Visualizes feature maps (channels) from intermediate or final output.
        """
        num_channels = image_segmented.shape[-1]
        cols = 8
        rows = num_channels // cols + (num_channels % cols > 0)

        plt.figure(figsize=(15, rows * 2))

        for i in range(num_channels):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image_segmented[:, :, i], cmap='gray')
            plt.axis('off')
            plt.title(f'Ch {i + 1}')

        plt.tight_layout()
        plt.show()

# ------------------------------------------------------
# Step 1: Preprocess Dataset Images
# ------------------------------------------------------

processed_images = []
for i, img in enumerate(BreastCancerProcessor.raw_images):
    try:
        # Resize image to 128x128
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # Convert to float32
        resized = resized.astype(np.float32)

        # Add channel dimension if grayscale
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)

        # Per-channel normalization (zero mean, unit variance)
        mean = np.mean(resized, axis=(0, 1), keepdims=True)
        std  = np.std(resized, axis=(0, 1), keepdims=True) + 1e-8
        resized = (resized - mean) / std

        processed_images.append(resized)
    except Exception as e:
        print(f"⚠️ Failed processing image {i}: {e}")

# Convert list to numpy array (final dataset batch)
processed_images_np = np.array(processed_images)

# ------------------------------------------------------
# Step 2: Run Segmentation Pipeline
# ------------------------------------------------------

# Instantiate segmentation class
segmenter = ImageSegmentation()

# Load preprocessed images into the model
segmenter.load_image(processed_images_np)

# Run the mock U-Net encoder
segmenter.unet()

# ------------------------------------------------------
# Step 3: Combine Results with Labels
# ------------------------------------------------------

# Final dataset combining encoded features with their respective labels
#final_dataset = list(zip(segmenter.preprocessed_image, BreastCancerProcessor.raw_classes))
