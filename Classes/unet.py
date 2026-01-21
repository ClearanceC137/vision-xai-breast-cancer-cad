# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:37:54 2025

@author: clear
"""

import numpy as np
from preprocessing import processed_images_np  # Import preprocessed images
import matplotlib.pyplot as plt
# ----------- Layer Functions ------------

def conv2d(input, kernel, padding='same'):
    k = kernel.shape[0]
    if padding == 'same':
        pad = k // 2
        input = np.pad(input, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant')

    b, h, w, c = input.shape
    kh, kw, _, filters = kernel.shape
    output = np.zeros((b, h, w, filters))
    
    for i in range(h):
        for j in range(w):
            input_patch = input[:, i:i+kh, j:j+kw, :]
            if input_patch.shape[1] != kh or input_patch.shape[2] != kw:
                continue  # skip padding edges
            for f in range(filters):
                output[:, i, j, f] = np.sum(input_patch * kernel[:, :, :, f], axis=(1,2,3))
    return output

def max_pool(input):
    b, h, w, c = input.shape
    h_new = h // 2
    w_new = w // 2
    output = np.zeros((b, h_new, w_new, c))
    for i in range(h_new):
        for j in range(w_new):
            h_start = i * 2
            w_start = j * 2
            patch = input[:, h_start:h_start+2, w_start:w_start+2, :]
            output[:, i, j, :] = np.max(patch, axis=(1, 2))
    return output

def upsample(input):
    b, h, w, c = input.shape
    output = np.zeros((b, h * 2, w * 2, c))
    for i in range(h):
        for j in range(w):
            output[:, i*2:(i+1)*2, j*2:(j+1)*2, :] = input[:, i:i+1, j:j+1, :]
    return output

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------- Tiny UNet ------------

def tiny_unet_numpy(input_image):
    c1 = relu(conv2d(input_image, np.random.randn(3, 3, input_image.shape[-1], 16)))
    p1 = max_pool(c1)

    c2 = relu(conv2d(p1, np.random.randn(3, 3, 16, 32)))
    p2 = max_pool(c2)

    bn = relu(conv2d(p2, np.random.randn(3, 3, 32, 64)))

    # u1 = upsample(bn)
    # ... skip decoder ...

    return bn

# ----------- Run UNet on All Images ------------

# Keep a copy of the original first image for display (assuming grayscale with shape (128,128,1))
original_first_image = processed_images_np[0, :, :, 0]  # shape (128,128)

# Ensure batch dimension is correct
if processed_images_np.ndim == 3:
    processed_images_np = np.expand_dims(processed_images_np, axis=0)  # Single image
elif processed_images_np.ndim == 4:
    pass
else:
    raise ValueError("Invalid image array shape.")

# Run UNet
predictions = tiny_unet_numpy(processed_images_np)
print("✅ UNet output shape:", predictions.shape)

first_image = predictions[0]  # shape: (35, 35, 64)
num_channels = first_image.shape[-1]

# Plot original image and all channels side by side
cols = 8
rows = (num_channels // cols) + 1  # plus one for the original image row

#plt.figure(figsize=(20, (rows + 1) * 2))

# Plot original image on top-left, spanning 2 cols
#plt.subplot(rows + 1, cols, 1)
#plt.imshow(original_first_image, cmap='gray')
#plt.title('Original Image')
#plt.axis('off')

# Plot predicted channels starting from the second subplot
for i in range(num_channels):
    plt.subplot(rows + 1, cols, i + 2)  # +2 because original image took position 1
    plt.imshow(first_image[:, :, i], cmap='gray')
    plt.axis('off')
    plt.title(f'Ch {i+1}')

plt.tight_layout()
plt.show()