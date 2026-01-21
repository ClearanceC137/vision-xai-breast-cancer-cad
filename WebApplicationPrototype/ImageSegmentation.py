# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:41:10 2025
@author: clear
"""
import numpy as np
#from preprocessing import Preprocessing
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
# Define the parent directory
#parent_dir = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\Prototype\dicom_class_mapping.csv"
#BreastCancerProcessor = Preprocessing(parent_dir)
#BreastCancerProcessor.fit_label_encoder()

class ImageSegmentation:
    def __init__(self):
        self.original_image = None            # Raw input image from file
        self.preprocessed_image = None        # Image preprocessed for segmentation
        self.segmented_mask = None            # Output segmentation mask from the model

    def load_image(self, image_data):
        # Ensure image has batch dimension
        if image_data.ndim == 3:
            image_data = np.expand_dims(image_data, axis=0)
        elif image_data.ndim != 4:
            raise ValueError("Invalid image array shape.")

        self.original_image = image_data

    def conv2d(self, input, kernel, padding='same'):
        k = kernel.shape[0]
        if padding == 'same':
            pad = k // 2
            input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

        b, h, w, c = input.shape
        kh, kw, _, filters = kernel.shape
        output = np.zeros((b, h, w, filters))

        for i in range(h):
            for j in range(w):
                input_patch = input[:, i:i + kh, j:j + kw, :]
                if input_patch.shape[1] != kh or input_patch.shape[2] != kw:
                    continue
                for f in range(filters):
                    output[:, i, j, f] = np.sum(input_patch * kernel[:, :, :, f], axis=(1, 2, 3))
        return output

    def max_pool(self, input):
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
        b, h, w, c = input.shape
        output = np.zeros((b, h * 2, w * 2, c))
        for i in range(h):
            for j in range(w):
                output[:, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2, :] = input[:, i:i + 1, j:j + 1, :]
        return output

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def postprocess_segmented_image(self):
        # Ensure image is 4D (batch, height, width, channels)
        if self.preprocessed_image.ndim != 4:
            raise ValueError("Expected image with shape (batch, height, width, channels)")

        batch_size, h, w, c = self.preprocessed_image.shape

        # Find nearest lower power-of-two size
        def nearest_power_of_two(x):
            powers = [2 ** i for i in range(4, 10)]  # from 16 to 512
            powers = [p for p in powers if p <= x]
            return max(powers) if powers else x

        new_h = nearest_power_of_two(h)
        new_w = nearest_power_of_two(w)

        # Compute zoom factors
        zoom_h = new_h / h
        zoom_w = new_w / w

        # Resize each image in the batch
        resized_batch = np.zeros((batch_size, new_h, new_w, c))
        for i in range(batch_size):
            for ch in range(c):
                resized_batch[i, :, :, ch] = zoom(self.preprocessed_image[i, :, :, ch], (zoom_h, zoom_w), order=1)
        print("Images downscaled :",resized_batch.shape)


    def unet(self):
        # Simulates a very basic U-Net encoder path
        input_image = self.original_image

        c1 = self.relu(self.conv2d(input_image, np.random.randn(3, 3, input_image.shape[-1], 16)))
        p1 = self.max_pool(c1)

        c2 = self.relu(self.conv2d(p1, np.random.randn(3, 3, 16, 32)))
        p2 = self.max_pool(c2)

        bn = self.relu(self.conv2d(p2, np.random.randn(3, 3, 32, 64)))

        self.preprocessed_image = bn  # Save the bottleneck layer for visualization
        print("✅ UNet output shape:", self.preprocessed_image.shape)



    def display_segmented_image(self, image_segmented):
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


# Step 1: Resize and normalize all images

#for i, img in enumerate(BreastCancerProcessor.raw_images):
#    try:
#       # Resize image to 128x128
#        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
#
#       # Normalize pixel values, avoid division by zero
#        max_val = np.max(resized)
#        if max_val > 0:
#            resized = resized.astype(np.float32) / max_val
#        else:
#            resized = resized.astype(np.float32)

        # Add channel dimension if grayscale
#        if resized.ndim == 2:
#            resized = np.expand_dims(resized, axis=-1)

#       processed_images.append(resized)
#    except Exception as e:
#       print(f"⚠️ Failed processing image {i}: {e}")



# Visualize

#final_dataset = list(zip(segmenter.preprocessed_image, BreastCancerProcessor.raw_classes))