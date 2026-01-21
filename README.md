# Vision-Based Explainable Diagnosis Framework for Breast Cancer

**Author:** Tshepiso Mahoko  
**Institution:** University of Johannesburg, South Africa  
**Email:** 220015607@student.uj.ac.za  

---

## Abstract

Medical image analysis has significantly advanced the detection of cancer abnormalities, largely due to the success of deep learning. Modern computer-aided detection and diagnosis (CADx) systems rely on medical imaging modalities to assist clinicians in identifying and interpreting anomalies. However, traditional systems often lack transparency and remain susceptible to human fatigue, bias, and interpretability challenges.

This project presents a **vision-based explainable computer-aided diagnosis (CADx) framework for breast cancer detection**, integrating Convolutional Neural Networks (CNNs), U-Net image segmentation, and Explainable AI (XAI) techniques. The framework enhances diagnostic accuracy while providing visual explanations of model predictions using Gradient-weighted Class Activation Mapping (Grad-CAM), improving clinical trust and transparency.

---

## Keywords

Computer-Aided Detection (CADe), Computer-Aided Diagnosis (CADx),  
Convolutional Neural Networks (CNN), Explainable AI (XAI),  
Medical Imaging, Breast Cancer

---

## 1. Introduction

Breast cancer remains one of the most prevalent and deadly cancers affecting women worldwide. Early and accurate detection is critical to improving survival rates and treatment outcomes. However, the interpretation of complex medical images is highly dependent on human expertise and is prone to fatigue, workload pressure, and diagnostic bias.

Computer-aided diagnosis (CADx) systems have emerged as valuable tools to support radiologists by enhancing detection accuracy across imaging modalities such as mammography, MRI, ultrasound, and microscopic imaging. Despite their advantages, many traditional CAD systems lack explainability and visual intelligence, limiting clinical adoption.

This research addresses these limitations by proposing a **vision-based CADx system** that combines deep learning and explainable AI. The framework leverages CNNs for classification, U-Net for segmentation, and Grad-CAM for interpretability, forming a transparent and clinically relevant diagnostic pipeline.

---

## 2. Societal Relevance

Breast cancer is the most common cancer among women globally and represents a significant public health challenge. In South Africa, approximately 1 in 27 women will be diagnosed with breast cancer during their lifetime. Limited access to radiologists and delayed diagnosis further exacerbate mortality rates.

This project contributes to societal impact by enabling early-stage detection through automated image analysis, reducing diagnostic delays, supporting clinicians in high-workload environments, and improving access to reliable screening tools. The integration of Explainable AI ensures transparency, ethical AI usage, and trust in automated medical decision-making.

---

## 3. Literature Review

### 3.1 Supervised Learning in Radiology

Deep learning methods, particularly CNNs, have demonstrated superior performance in radiology by learning hierarchical features directly from raw medical images. Compared to traditional machine learning models, CNNs significantly reduce false positives and false negatives in diagnostic tasks.

### 3.2 Mammography and Training Data

This project utilises publicly available datasets from **The Cancer Imaging Archive (TCIA)**, which provide annotated mammography images essential for supervised learning. While annotation quality may vary due to expert subjectivity, TCIA remains a critical resource for developing generalisable CAD systems.

### 3.3 Convolutional Neural Networks in Medical Diagnosis

CNNs automatically extract spatial features using convolutional, pooling, and dense layers, enabling detection of subtle pathological patterns such as calcifications and asymmetries. Multiple studies show CNNs achieving expert-level diagnostic accuracy in breast cancer classification tasks.

### 3.4 Image Preprocessing and Feature Extraction

A U-Net–based preprocessing pipeline was implemented to enhance image quality through artifact suppression, breast tissue isolation, and pectoral muscle removal. This ensures standardized, high-quality inputs for model training.

### 3.5 Image Segmentation with U-Net

U-Net segmentation isolates diagnostically relevant regions of interest (ROIs), allowing the CNN to focus on tumour-specific features. The encoder–decoder architecture with skip connections preserves spatial detail and improves segmentation accuracy.

### 3.6 Explainable AI (XAI) with Grad-CAM

Explainable AI techniques address the black-box nature of deep learning models. Grad-CAM generates heatmaps highlighting regions that influence predictions, enabling clinicians to validate whether the model focuses on medically relevant structures.

---

## 4. Methodology

### 4.1 Framework Overview

The proposed CADx pipeline integrates:
- Image preprocessing  
- U-Net segmentation  
- CNN-based classification  
- Grad-CAM explainability  

### 4.2 Model Training and Validation

The dataset was split into training and testing subsets. The model was trained iteratively using batch learning, with performance evaluated using accuracy, precision, recall, and F1-score.

### 4.3 CNN Architecture

The CNN consists of:
- Convolutional layers with ReLU activation  
- Max-pooling layers  
- Batch normalization  
- Dropout for regularization  
- Fully connected layers  
- Sigmoid output layer for binary classification  

---

## 5. Model Parameters and Performance

### 5.1 Dataset

- Total Samples: 245  
- Classes: Benign, Malignant  
- Train/Test Split: 220 / 25  
- Input Shape: 64 × 256 × 256  

### 5.2 Training Configuration

- Epochs: 60  
- Batch Size: 32  
- Learning Rate: 0.001  
- Device: CPU  
- Training Time: ~16 minutes  

### 5.3 Evaluation Metrics

The model achieved a **weighted F1-score of 0.76**, with balanced performance across both classes. Validation accuracy stabilized at approximately **76%**, demonstrating effective learning despite computational constraints.

---

## 6. Analysis

Model performance was constrained by the absence of GPU acceleration, requiring image downscaling and limiting experimentation with deeper architectures. Despite these limitations, the preprocessing and segmentation pipeline enabled stable convergence and robust feature extraction.

The achieved performance is comparable to reported radiologist sensitivity levels when assisted by AI-based CAD systems, highlighting the framework’s clinical relevance.

---

## 7. Conclusion

This project presents a **vision-based explainable CADx framework for breast cancer detection**, integrating CNNs, U-Net segmentation, and Grad-CAM explainability. The system demonstrates reliable diagnostic performance, improved transparency, and strong potential for clinical decision support.

Future work will focus on GPU acceleration, higher-resolution imaging, advanced XAI techniques, and larger datasets to further improve generalization and accuracy.

---

## License

This project is intended for **academic and research purposes**.

---

## Acknowledgements

- University of Johannesburg  
- The Cancer Imaging Archive (TCIA)
