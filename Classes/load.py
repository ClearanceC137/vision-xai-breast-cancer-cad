# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:11:28 2025

@author: clear
"""
import pydicom
import matplotlib.pyplot as plt

# Load DICOM file
ds = pydicom.dcmread(r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\DICOMSET\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\Calc-Test_P_00038_RIGHT_CC_1\08-29-2017-DDSM-NA-83105\1.000000-ROI mask images-37851\1-1.dcm")

# Display image
plt.imshow(ds.pixel_array, cmap='gray')
plt.title("DICOM Image")
plt.show()

# Print metadata
print(ds)
