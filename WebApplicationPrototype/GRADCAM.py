import os
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------
# Load pretrained ResNet50
# ------------------------------------------------------
model = resnet50(pretrained=True)
model.to(device)
model.eval()

# ------------------------------------------------------
# Prepare transforms
# ------------------------------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def generate_dual_class_gradcam_overlays_pytorch(img, classes_to_test=[0, 1], save_folder='explainability'):
    """
    Generates Grad-CAM overlays for two classes using PyTorch + pytorch-grad-cam.
    Args:
        img (np.ndarray): Input grayscale image, shape (H, W), scaled 0-255
        classes_to_test (list): List of target class indices
        save_folder (str): Folder to save overlays and heatmaps

    Returns:
        dict: {class_idx: (overlay, heatmap)}
    """
    os.makedirs(save_folder, exist_ok=True)
    overlays = {}

    # Convert grayscale to RGB normalized [0,1]
    img_rgb = np.stack([img / 255.0]*3, axis=-1)  # [H,W,3]

    # Apply transforms and add batch dimension
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)  # [1,3,H,W]

    # Last conv layer for Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Get predicted class
    with torch.no_grad():
        pred = model(input_tensor).argmax(dim=1).item()

    # If no specific classes are given, just use predicted
    if classes_to_test is None:
        classes_to_test = [pred]

    for class_idx in classes_to_test:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]

        # Overlay heatmap on image
        overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Convert heatmap to uint8 for saving
        heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)

        # Save files
        overlay_path = os.path.join(save_folder, f'gradcam_overlay_class_{class_idx}.png')
        heatmap_path = os.path.join(save_folder, f'gradcam_heatmap_class_{class_idx}.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(heatmap_path, heatmap_uint8)

        overlays[class_idx] = (overlay, heatmap_uint8)
        print(f"Saved Grad-CAM overlay and heatmap for class {class_idx} in {save_folder}")

    return overlays
