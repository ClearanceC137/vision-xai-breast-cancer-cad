# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:55:56 2025

@author: clear

This script provides a framework for:
1. Loading and processing DICOM medical images (e.g., mammograms).
2. Extracting raw pixel data and mapping it to pathology labels.
3. Encoding labels into integers for ML tasks.
4. Preprocessing images (resizing, normalization, augmentation).
5. Defining a lightweight U-Net (tiny UNet) for autoencoder-based feature extraction.
6. Extracting bottleneck features for downstream classification.
"""

import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class Preprocessing:
    """
    A class to handle DICOM data preprocessing including:
    - Data loading from CSV mappings
    - Feature extraction (pixel data + labels)
    - Label encoding
    - Image preprocessing hooks (resize, normalize, augment)
    """

    def __init__(self, parent_dir):
        # Storage containers
        self.data_set = []                    # List of dicts: {"DICOM": ds,"PatientID": patient_id,"ImagePath": file_path, "Class": pathology}
        self.raw_images = []                  # Raw pixel arrays from DICOM files
        self.raw_classes_str = []             # Original string labels (e.g., "benign", "malignant")
        self.raw_classes = []                 # Encoded labels (integers)
        self.processed_images = None          # Images after preprocessing (resize, normalization, augmentation)
        self.feature_data = None              # Transformed/feature representation for model input
        self.augmentation_params = None       # Placeholder for augmentation settings
        self.normalization_params = None      # Placeholder for normalization parameters
        self.resize_shape = None              # Target resize shape (e.g., (128,128))
        self.image_modality = None            # Imaging type (optional: e.g., "mammogram")
        self.data_set_size = 0                # Number of DICOMs loaded
        self.label_encoder = None             # Custom label encoder dictionary

        # Load dataset and extract features on initialization
        self.load_data(parent_dir)
        self.extract_features()

    # --------------------------------------------------------------------------
    # Placeholder preprocessing methods
    # --------------------------------------------------------------------------
    def resize_images(self, images, target_shape): 
        """Resize images to a target shape (not implemented)."""
        pass

    def normalize_images(self, images): 
        """Normalize images (e.g., scale to [0,1] or standardize)."""
        pass

    def augment_images(self, images, params): 
        """Apply augmentations like rotation, flipping (not implemented)."""
        pass

    # --------------------------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------------------------
    def fit_label_encoder(self):
        """
        Convert string class labels into integer labels.
        Creates a dictionary mapping class_name → integer index.
        """
        unique_labels = sorted(set(self.raw_classes_str))  # Sort for consistency
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.raw_classes = [self.label_encoder[label] for label in self.raw_classes_str]
        print("Label Encoder Mapping:", self.label_encoder)

    # --------------------------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------------------------
    def extract_features(self):
        """
        Extract pixel arrays and store alongside their class labels.
        Skips any file that cannot be converted to pixel data.
        """
        self.features = []  # Each entry is (pixel_array, class)

        for item in self.data_set:
            ds = item["DICOM"]
            label = item["Class"]

            try:
                pixel_array = ds.pixel_array  # Extract raw pixel data
                self.features.append((pixel_array, label))
                self.raw_images.append(pixel_array)
                self.raw_classes_str.append(label)
            except Exception as e:
                print(f"⚠️ Skipping file due to pixel extraction error: {e}")
                continue

        print(f"✅ Extracted features from {len(self.features)} DICOM files.")

    # --------------------------------------------------------------------------
    # Additional preprocessing hooks (not implemented yet)
    # --------------------------------------------------------------------------
    def prepare_for_segmentation(self, images): pass
    def prepare_for_classification(self, images): pass
    def split_train_test(self, images, labels, test_size): pass

    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------
    def view_DICOM_image(self, instance):
        """
        Display a single DICOM image using matplotlib.
        """
        plt.imshow(instance["DICOM"].pixel_array)   # Add cmap='gray' for grayscale
        plt.title(f"DICOM Image : {instance['PatientID']}")
        plt.show()

    # --------------------------------------------------------------------------
    # Load Data
    # --------------------------------------------------------------------------
    def load_data(self, mapping_csv_path):
        """
        Load DICOM file paths and labels from a mapping CSV.
        CSV must contain columns: ['dicom_file_path', 'pathology'].
        """
        self.data_set = []
        self.raw_images = []

        try:
            mapping_df = pd.read_csv(mapping_csv_path, dtype=str)
        except Exception as e:
            print(f"❌ Failed to load mapping CSV: {e}")
            return

        for _, row in mapping_df.iterrows():
            file_path = row['dicom_file_path']
            pathology = row['pathology']

            try:
                ds = pydicom.dcmread(file_path)

                # Extract patient ID (DICOM tag 0010,0020) → fallback "Unknown"
                patient_id = ds.get((0x0010, 0x0020), "Unknown")
                if hasattr(patient_id, 'value'):
                    patient_id = patient_id.value
                if isinstance(patient_id, str):
                    patient_id = patient_id.replace('.dcm', '')

                # Append dataset entry
                self.data_set.append({
                    "DICOM": ds,
                    "PatientID": patient_id,
                    "ImagePath": file_path,
                    "Class": pathology
                })

            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")

        self.data_set_size = len(self.data_set)
        print(f"✅ Loaded {self.data_set_size} labeled DICOM files.")


# ------------------------------------------------------------------------------
# Define a Tiny U-Net Autoencoder for Feature Extraction
# ------------------------------------------------------------------------------
def tiny_unet(input_shape):
    """
    Build a minimal U-Net architecture for autoencoder training.
    Useful for feature compression via bottleneck extraction.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck (feature representation layer)
    bn = layers.Conv2D(64, 3, activation='relu', padding='same', name='bottleneck')(p2)

    # Decoder
    u1 = layers.UpSampling2D()(bn)
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    # Output layer → reconstructs image
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(c4)

    model = models.Model(inputs, outputs)
    return model


# ------------------------------------------------------------------------------
# Example Workflow (commented out)
# ------------------------------------------------------------------------------
# parent_dir = r"C:\...\dicom_class_mapping.csv"
# BreastCancerProcessor = Preprocessing(parent_dir)
# BreastCancerProcessor.fit_label_encoder()

# # Step 1: Preprocess all images (resize + normalize)
# processed_images = []
# for i, img in enumerate(BreastCancerProcessor.raw_images):
#     try:
#         # Resize to 128x128
#         resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

#         # Normalize to [0,1]
#         max_val = np.max(resized)
#         if max_val > 0:
#             resized = resized.astype(np.float32) / max_val
#         else:
#             resized = resized.astype(np.float32)

#         # Ensure channel dimension exists
#         if resized.ndim == 2:
#             resized = np.expand_dims(resized, axis=-1)

#         processed_images.append(resized)
#     except Exception as e:
#         print(f"⚠️ Failed processing image {i}: {e}")

# # Convert to numpy array
# processed_images_np = np.array(processed_images)
# labels = BreastCancerProcessor.raw_classes
# print(f"✅ Processed all images: {processed_images_np.shape}")

# # Step 2: Train Tiny U-Net (autoencoder style)
# model = tiny_unet((128, 128, 1))
# model.compile(optimizer='adam', loss='mse')
# model.fit(processed_images_np, processed_images_np, epochs=5, batch_size=8)

# # Step 3: Extract bottleneck features
# bottleneck_model = models.Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
# bottleneck_features = bottleneck_model.predict(processed_images_np)
# print(f"✅ Bottleneck feature shape: {bottleneck_features.shape}")

# # Step 4: Pair features with class labels
# final_dataset = list(zip(bottleneck_features, labels))
# print(f"✅ Final feature-class dataset size: {len(final_dataset)}")
