##https://github.com/lishen/end2end-all-conv/blob/master/dm_preprocess.py
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:41:10 2025
@author: clear

This script demonstrates a prototype image preprocessing and feature extraction
pipeline for breast cancer image analysis using a UNet encoder.
"""
#


import numpy as np
#from preprocessing import Preprocessing   # Custom preprocessing class
import cv2
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Load Dataset and Prepare Label Encoding
# ------------------------------------------------------

parent_dir = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\Prototype\dicom_class_mapping.csv"

'''BreastCancerProcessor = Preprocessing(parent_dir)
BreastCancerProcessor.fit_label_encoder()

processed_images = []
for i, img in enumerate(BreastCancerProcessor.raw_images):
    try:
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32)
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        mean = np.mean(resized, axis=(0, 1), keepdims=True)
        std  = np.std(resized, axis=(0, 1), keepdims=True) + 1e-8
        resized = (resized - mean) / std
        processed_images.append(resized)
    except Exception as e:
        print(f"⚠️ Failed processing image {i}: {e}")

X_Data = np.array(processed_images)[:5]
num_classes = len(np.unique(BreastCancerProcessor.raw_classes))
Y_Data = np.eye(num_classes)[BreastCancerProcessor.raw_classes[:5]]

print("Raw scaled image shape:", X_Data[0].shape)
print("One-hot label shape:", Y_Data[0].shape)'''

# -----------------------------




class DMImagePreprocessor(object):
    '''Class for preprocessing images in the DM challenge'''

    def __init__(self):
        pass

    def select_largest_obj(self, img_bin, lab_val=255, fill_holes=False, 
                           smooth_boundary=False, kernel_size=15):
        n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
            img_bin, connectivity=8, ltype=cv2.CV_32S)
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val

        if fill_holes:
            bkg_locs = np.where(img_labeled == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)
            largest_mask = largest_mask + holes_mask

        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)

        return largest_mask

    @staticmethod
    def max_pix_val(dtype):
        if dtype == np.dtype('uint8'):
            return 255
        elif dtype == np.dtype('uint16'):
            return 65535
        else:
            raise Exception('Unknown dtype found in input image array')

    def suppress_artifacts(self, img, global_threshold=.05, fill_holes=False, 
                           smooth_boundary=True, kernel_size=15):
        maxval = self.max_pix_val(img.dtype)
        if global_threshold < 1.:
            low_th = int(img.max()*global_threshold)
        else:
            low_th = int(global_threshold)

        _, img_bin = cv2.threshold(img, low_th, maxval=maxval, type=cv2.THRESH_BINARY)
        breast_mask = self.select_largest_obj(img_bin, lab_val=maxval, fill_holes=True, 
                                              smooth_boundary=True, kernel_size=kernel_size)
        img_suppr = cv2.bitwise_and(img, breast_mask)
        return (img_suppr, breast_mask)

    @classmethod
    def segment_breast(cls, img, low_int_threshold=.05, crop=True):
        img_8u = (img.astype('float32') / img.max() * 255).astype(np.uint8)
        low_th = int(img_8u.max() * low_int_threshold) if low_int_threshold < 1. else int(low_int_threshold)
        _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)

        # Fixed OpenCV 4.x return values
        contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont_areas = [cv2.contourArea(cont) for cont in contours]
        idx = np.argmax(cont_areas)
        breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1)
        img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
        x, y, w, h = cv2.boundingRect(contours[idx])
        if crop:
            img_breast_only = img_breast_only[y:y+h, x:x+w]
        return img_breast_only, (x, y, w, h)

    def remove_pectoral(self, img, breast_mask, high_int_threshold=.8, 
                        morph_kn_size=3, n_morph_op=7, sm_kn_size=25):
        img_equ = cv2.equalizeHist(img)
        high_th = int(img.max() * high_int_threshold) if high_int_threshold < 1. else int(high_int_threshold)
        maxval = self.max_pix_val(img.dtype)
        _, img_bin = cv2.threshold(img_equ, high_th, maxval=maxval, type=cv2.THRESH_BINARY)

        pect_marker_img = np.zeros(img_bin.shape, dtype=np.int32)
        pect_mask_init = self.select_largest_obj(img_bin, lab_val=maxval, fill_holes=True, smooth_boundary=False)

        kernel_ = np.ones((morph_kn_size, morph_kn_size), dtype=np.uint8)
        pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_morph_op)
        pect_marker_img[pect_mask_eroded > 0] = 255

        pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_morph_op)
        pect_marker_img[pect_mask_dilated == 0] = 128
        pect_marker_img[breast_mask == 0] = 64

        img_equ_3c = cv2.cvtColor(img_equ, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_equ_3c, pect_marker_img)
        img_equ_3c[pect_marker_img == -1] = (0, 0, 255)

        breast_only_mask = pect_marker_img.copy()
        breast_only_mask[breast_only_mask == -1] = 0
        breast_only_mask = breast_only_mask.astype(np.uint8)
        breast_only_mask[breast_only_mask != 128] = 0
        breast_only_mask[breast_only_mask == 128] = 255

        kernel_ = np.ones((sm_kn_size, sm_kn_size), dtype=np.uint8)
        breast_only_mask = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
        img_breast_only = cv2.bitwise_and(img_equ, breast_only_mask)

        return (img_breast_only, img_equ_3c)

    def process(self, img, median_filtering=True, blur_kn_size=3, 
                artif_suppression=True, low_int_threshold=.05, kernel_size=15,
                pect_removal=False, high_int_threshold=.8, **pect_kwargs):
        img_proc = img.copy()
        if median_filtering:
            img_proc = cv2.medianBlur(img_proc, blur_kn_size)

        if artif_suppression:
            img_proc, mask_ = self.suppress_artifacts(img_proc, global_threshold=low_int_threshold, kernel_size=kernel_size)
        else:
            _, mask_ = self.suppress_artifacts(img_proc)

        if pect_removal:
            img_proc, img_col = self.remove_pectoral(img_proc, mask_, high_int_threshold=high_int_threshold, **pect_kwargs)
        else:
            img_col = None

        return (img_proc, img_col)




'''
# ------------------------------------------------------
# Assume DMImagePreprocessor class is already defined/imported
# ------------------------------------------------------
preprocessor = DMImagePreprocessor()

# Take a raw image (uint16)
raw_img = BreastCancerProcessor.raw_images[0]

# Convert to 8-bit for OpenCV
raw_img_8bit = ((raw_img.astype(np.float32) / raw_img.max()) * 255).astype(np.uint8)

# ------------------------------------------------------
# Stage 1: Artifact suppression
# ------------------------------------------------------
img_suppr, breast_mask = preprocessor.suppress_artifacts(
    raw_img_8bit,
    global_threshold=0.05,
    fill_holes=True,
    smooth_boundary=True,
    kernel_size=15
)

# ------------------------------------------------------
# Stage 2: Breast segmentation (no cropping to keep mask alignment)
# ------------------------------------------------------
img_breast_only, _ = preprocessor.segment_breast(
    img_suppr,
    low_int_threshold=0.05,
    crop=False  # Keep original size
)

# ------------------------------------------------------
# Stage 3: Pectoral muscle removal
# ------------------------------------------------------
img_clean, img_with_boundary = preprocessor.remove_pectoral(
    img_breast_only,
    breast_mask,
    high_int_threshold=0.8,
    morph_kn_size=3,
    n_morph_op=7,
    sm_kn_size=25
)

# ------------------------------------------------------
# Stage 4: Optional median filtering
# ------------------------------------------------------
img_final = cv2.medianBlur(img_clean, 3)

# ------------------------------------------------------
# Stage 5: Resize & normalize for model input
# ------------------------------------------------------
img_resized = cv2.resize(img_final, (128, 128)).astype(np.float32)
img_resized = (img_resized - np.mean(img_resized)) / (np.std(img_resized) + 1e-8)
img_resized = np.expand_dims(img_resized, axis=-1)  # add channel dimension
img_tensor = torch.tensor(img_resized.transpose(2, 0, 1)[None, ...], dtype=torch.float32)

# ------------------------------------------------------
# Stage 6: Display results
# ------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(raw_img_8bit, cmap='gray')
plt.title("Raw Image (8-bit)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_clean, cmap='gray')
plt.title("Preprocessed Breast")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_with_boundary)
plt.title("Pectoral Muscle Boundary")
plt.axis('off')

plt.show()
'''
