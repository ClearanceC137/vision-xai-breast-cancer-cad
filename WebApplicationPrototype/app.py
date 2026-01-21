from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import uuid
from werkzeug.utils import secure_filename
from ImageSegmentation import ImageSegmentation
import matplotlib.pyplot as plt
import shutil
import threading
from shutil import copyfile
import logging
import os
import zipfile
from werkzeug.datastructures import FileStorage
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(filename='debug.log', level=logging.INFO)

from customImageCleaner import DMImagePreprocessor   # Custom preprocessing class
import torch
import segmentation_models_pytorch as smp


# ------------------------------------------------------
# Load pretrained ImageNet model
# ------------------------------------------------------
import torchvision.transforms as T
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)


# Global variable for mode choosen

pipeline_global = 'basic'  # default

UPLOAD_FOLDER_SINGLE = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\Uploads\raw_image"
CLEAN_IMAGE_FOLDER = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\Uploads\clean_image_roi_highlight"
PREPROCESSED_FOLDER = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\Uploads\preprocessed_image"
SEGMENTATION_FOLDER = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\Uploads\segmentation_image"
CSV_PATH = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\Uploads\prediction_data.csv"
BULK_IMAGE_FOLDER = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\Bulk_images"
os.makedirs(UPLOAD_FOLDER_SINGLE, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
os.makedirs(SEGMENTATION_FOLDER, exist_ok=True)

# === Build CNNModel with real bottleneck features ===\
    

    
# ------------------------------------------------------
# Step 1: Convert to tensor
# ------------------------------------------------------
def get_image_tensor(resized):
    """
    Convert a resized image (NumPy array) into a normalized PyTorch tensor.

    - Scales pixel values to [0, 1] range.
    - Adds batch and channel dimensions so the final shape is [B, C, H, W].

    Args:
        resized (numpy.ndarray): Resized image array, typically shape [H, W] for grayscale.

    Returns:
        torch.Tensor: Preprocessed tensor with shape [1, 1, H, W].
    """
    image_tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
    return image_tensor

# ------------------------------------------------------
# Step 2: Create SMP UNet with pretrained ResNet34 encoder
# ------------------------------------------------------
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,   # grayscale
    classes=1        # binary mask
)
unet_model.eval()

# ------------------------------------------------------
# Step 3: Extract encoder features only from pretrained model
# ------------------------------------------------------
def extract_encoder_features(model, x):
    features = []
    for name, module in model.encoder.named_children():
        x = module(x)
        features.append(x)
    return features

def unet_process_single_image(img):
    try:
        preprocessor = DMImagePreprocessor()

        # Convert to 8-bit for OpenCV
        raw_img_8bit = ((img.astype(np.float32) / img.max()) * 255).astype(np.uint8)

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
            crop=False
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

        # Resize image to 512x512
        resized = cv2.resize(img_with_boundary, (512, 512), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32)

        # Ensure grayscale (single channel)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Convert to tensor and extract encoder features (model used from global scope)
        image_tensor = get_image_tensor(resized)
        encoder_features = extract_encoder_features(unet_model, image_tensor)

        # Take first layer features
        first_layer_features = encoder_features[0].squeeze().detach().numpy()

        return first_layer_features , img_with_boundary

    except Exception as e:
        print(f"⚠️ Failed processing image: {e}")
        return None



def preprocess_image(img, target_size=(128, 128)):
    """
    Preprocess a single image:
    1. Resize to target_size
    2. Convert to float32
    3. Add channel dimension if grayscale
    4. Normalize (zero mean, unit variance, per image)

    Returns:
        np.ndarray: preprocessed image with shape (H, W, C)
    """
    # Step 1: Resize
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Step 2: Convert type
    resized = resized.astype(np.float32)

    # Step 3: Ensure channel dimension
    if resized.ndim == 2:  # grayscale -> (H, W, 1)
        resized = np.expand_dims(resized, axis=-1)

    # Step 4: Per-image normalization
    mean = np.mean(resized, axis=(0, 1), keepdims=True)
    std  = np.std(resized, axis=(0, 1), keepdims=True) + 1e-8
    resized = (resized - mean) / std

    return resized


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def copy_image_to_static():
    # Load your CSV data
    df = pd.read_csv(CSV_PATH)
    # Get the matching row
    row = df.iloc[0]

    # Extract details
    image_path = row['dicom_file_path']
    image_name = os.path.basename(image_path)

    # Copy/move image to static folder if not already there
    static_img_path = os.path.join('static', 'images', image_name)
    if not os.path.exists(static_img_path):
        os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
        copyfile(image_path, static_img_path)

    return image_name

## Saves images musk by running the function on a thread
def save_masks(image_masks, filename, SEGMENTATION_FOLDER):
    logging.info("Save Image Masks...")
    num_channels = image_masks.shape[0]

    for i in range(num_channels):
        plt.figure(figsize=(2, 2))
        plt.imshow(image_masks[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title(f'Ch {i + 1}')
        plt.tight_layout()

        mask_filename = f"{os.path.splitext(filename)[0]}_mask_{i+1}.png"
        mask_path = os.path.join(SEGMENTATION_FOLDER, mask_filename)
        plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()
@app.route('/upload-single', methods=['POST'])
def upload_single():
    logging.info("Uploading started...")
    image = request.files.get('image1')
    breast = request.form.get('body_part1', '')
    modality = request.form.get('modality1', '') 
    if image:
        clear_folder(UPLOAD_FOLDER_SINGLE)
        clear_folder(PREPROCESSED_FOLDER)
        clear_folder(SEGMENTATION_FOLDER)
        clear_folder(CLEAN_IMAGE_FOLDER)

        filename = secure_filename(image.filename)
        raw_path = os.path.join(UPLOAD_FOLDER_SINGLE, filename)
        image.save(raw_path)

        img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error: Could not read image")
            return redirect(url_for('diagnosis'))

        try:
            

            patient_id = f"{str(uuid.uuid4())[:8]}_{secure_filename(image.filename)}"
            processed_filename = f"{os.path.splitext(filename)[0]}_processed.npy"
            processed_path = os.path.join(PREPROCESSED_FOLDER, processed_filename)
            np.save(processed_path, img)

            #processed_images_np = resized
            #segmenter = ImageSegmentation()
            #segmenter.load_image(processed_images_np)
            #segmenter.unet()

            image_masks , clean_image = unet_process_single_image(img)  # shape: (64, 256, 256)
            print(f"Segmentation shape {image_masks.shape}")
            logging.info(f"Segmentation shape clean {clean_image.shape}")
            
            logging.info(f"Clean Image shape {clean_image.shape}")
            ## save the clean image with rio highlighted
            clean_image_path = os.path.join(CLEAN_IMAGE_FOLDER , filename)
            cv2.imwrite( clean_image_path , clean_image)

            # Run the saving function on a separate thread
            thread = threading.Thread(target=save_masks, args=(image_masks, filename, SEGMENTATION_FOLDER))
            thread.start()
             

            # Save all masks stacked as a single 3D array
            raw_array_path = os.path.join(SEGMENTATION_FOLDER, f"{os.path.splitext(filename)[0]}_all_masks.npy")
            np.save(raw_array_path, image_masks)

          

            csv_headers = [
                "dicom_file_path", "preprocessed_file_path",
                "segmented_images_file_path",
                "patient_id", "breast", "image_view", "pathology", "modality","image_name","clean_image_path"
            ]

            new_entry = {
                "dicom_file_path": raw_path,
                "preprocessed_file_path": processed_path,
                "segmented_images_file_path": raw_array_path,
                "patient_id": patient_id,
                "breast": breast,
                "image_view": "",
                "pathology": "",
                "modality": modality,
                "image_name":filename,
                "clean_image_path":clean_image_path
                
            }

            df = pd.DataFrame([new_entry], columns=csv_headers)
            df.to_csv(CSV_PATH, index=False)

            return redirect(url_for('diagnosis'))

        except Exception as e:
            print("Processing error:", e)
            return redirect(url_for('diagnosis'))

    print("No image uploaded")
    return redirect(url_for('diagnosis'))

@app.route('/upload-bulk-image', methods=['POST'])
def upload_bulk_image():
    # Get the image filename from hidden input
    image_name = request.form.get('bulk_image_name')
    breast = request.form.get('body_part1', '')
    modality = request.form.get('modality1', '')

    if not image_name:
        return redirect(url_for('bulk_select_parameters'))

    image_path = os.path.join(BULK_IMAGE_FOLDER, image_name)

    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist")
        return redirect(url_for('bulk_select_parameters'))

    # Create a mock FileStorage object from the file on disk
    with open(image_path, 'rb') as f:
        file_storage = FileStorage(stream=f, filename=image_name, content_type='image/jpeg')

        # Create a temporary request context to reuse upload_single logic
        from flask import request as flask_request
        # Monkey-patch request.files for upload_single
        flask_request.files = {'image1': file_storage}
        flask_request.form = {'body_part1': breast, 'modality1': modality}

        # Call existing upload_single
        return upload_single()

@app.route('/')
def landing():
    return render_template('Analysis/Default/DefaultLanding.html')

@app.route('/home')
def home():
    global pipeline_global
    pipeline = request.args.get('pipeline', 'basic')  # default to basic if not provided
    pipeline_global = request.args.get('pipeline', 'basic')
    return render_template('Analysis/Default/DefaultMain.html')



@app.route('/diagnosis')
def diagnosis():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Replace missing values with 'N/A'
    df.fillna('N/A', inplace=True)
    logging.info(f"CSv Path {CSV_PATH}")
    # Convert DataFrame to dictionary list for easy template iteration
    data = df.to_dict(orient='records')

    return render_template('Analysis/Default/Diagnosis.html', cases=data)





@app.route('/view/<patient_id>')
def view_image(patient_id):
    # Load your CSV data
    df = pd.read_csv(CSV_PATH)
    # Get the matching row
    row = df[df['patient_id'] == patient_id].iloc[0]

    # Extract details for raw image
    image_path = row['dicom_file_path']
    image_name = os.path.basename(image_path)
    breast = row['breast']
    modality = row['modality']

    # Copy/move image to static folder if not already there
    static_img_path = os.path.join('static', 'original_image', "raw_image.png")
    clear_folder( os.path.join('static', 'original_image'))
    
    if not os.path.exists(static_img_path):
        os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
        # You may want to resize or reprocess here
        copyfile(image_path, static_img_path)

    return render_template('Analysis/Default/ViewImage.html',
                           image_filename=image_name,
                           image_name=image_name,
                           breast=breast,
                           modality=modality,
                           patient_id=patient_id)

@app.route('/view_segmentation')
def view_image_segmentation():
    df = pd.read_csv(CSV_PATH)

    # Get the path from the 'segmented_images_file_path' column (e.g., index 0 for now)
    segmented_path = df['segmented_images_file_path'].iloc[0]  # Update index as needed

    if not segmented_path:
        return "Segmented path not provided", 400

    # Extract the base file name, e.g., 'sample_4' from 'sample_4_all_masks.npy'
    base_filename = os.path.splitext(os.path.basename(segmented_path))[0].replace("_all_masks", "")

    # File prefix like 'sample_4_mask_'
    mask_prefix = f"{base_filename}_mask_"

    static_seg_folder = os.path.join(app.static_folder, 'segmentation_image')
    
    # Clear static/segmentation_image/ folder
    static_seg_folder = os.path.join(app.static_folder, 'segmentation_image')
    for filename in os.listdir(static_seg_folder):
        file_path = os.path.join(static_seg_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Collect matching mask PNG files from SEGMENTATION_FOLDER
    mask_images = []
    for file in os.listdir(SEGMENTATION_FOLDER):
        if file.startswith(mask_prefix) and file.endswith(".png"):
            src_path = os.path.join(SEGMENTATION_FOLDER, file)
            dst_path = os.path.join(static_seg_folder, file)
            shutil.copy2(src_path, dst_path)  # copy to static folder
            mask_images.append(file)

    mask_images.sort()

    if not mask_images:
        return f"No segmentation masks found for {base_filename}", 404

    metadata = {
        "image_name": base_filename + ".png",
        "modality": "Mammogram",
        "body_part": "Breast",
    }
    
    
    # raw clean image 
    clean_image_path = df['clean_image_path'].iloc[0]  # Update index as needed

    # Copy/move image to static folder if not already there
    static_img_path = os.path.join('static', 'clean_image', "clean_image.png")
    clear_folder( os.path.join('static', 'clean_image'))
    
    if not os.path.exists(static_img_path):
        os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
        # You may want to resize or reprocess here
        copyfile(clean_image_path, static_img_path)

    return render_template("Analysis/Default/ViewSegmentation.html", masks=mask_images, metadata=metadata)



def process_bottleneck_features(feat, resize_shape=(32, 32)):
    """
    Process and resize a single bottleneck feature map.

    Args:
        feat (torch.Tensor or np.ndarray): Input feature map [C,H,W] or similar.
        resize_shape (tuple): Output size (H, W), default (32, 32)

    Returns:
        np.ndarray: Resized feature map [H_resized, W_resized, C]
    """
    if isinstance(feat, torch.Tensor):
        feat_np = feat.detach().cpu().numpy()   # [C,H,W]
        feat_np = np.transpose(feat_np, (1, 2, 0))  # → [H,W,C]
    else:
        if feat.shape[0] < feat.shape[2]:  
            feat_np = np.transpose(feat, (1, 2, 0))
        else:
            feat_np = feat

    # Resize to target shape
    feat_resized = cv2.resize(feat_np, resize_shape, interpolation=cv2.INTER_LINEAR)

    return feat_resized


@app.route('/classify', methods=['GET'])
def classify():
    fileName = copy_image_to_static()
    npy_files = [f for f in os.listdir(SEGMENTATION_FOLDER) if f.endswith('.npy')]
    if len(npy_files) == 0:
        return "No .npy file found in segmentation folder", 404
    elif len(npy_files) > 1:
        return "More than one .npy file found in segmentation folder", 400
    
    npy_file_path = os.path.join(SEGMENTATION_FOLDER, npy_files[0])
    #bottleneck_features = np.load(npy_file_path)

    try:
        from GRADCAM import generate_dual_class_gradcam_overlays_pytorch
        # Load .npy bottleneck features from the given filepath
        features = np.load(npy_file_path)
        logging.info(f"[INFO] loaded shape Input {features.shape}")  
        
        
        # Add a new axis to create batch dimension (e.g., from (features_shape) to (1, features_shape))
        bottleneck_features = np.expand_dims(features, axis=0)

        num_samples = bottleneck_features.shape[0]
        input_shape = bottleneck_features.shape[1:]
        num_classes = 2  # fixed number of classes for now
        
        results = []    
        class_map = {
            0: "Benign",
            1: "Malignant",
            2: "Normal"
        }
        # ************  Basic pipeline selected ******************
        if pipeline_global.lower() == "basic":
            from CNNM import Model
            # Initialize your model (replace this with actual model loading)
            #model = CNNModel()
            #model.add_input_layer(input_shape)
            #model.num_classes = num_classes
            #model.add_convolutional_block(filters=2, kernel_size=2)
            #model.add_pooling_layer()
            #model.add_dense_layers(units=4)
            #model.add_output_layer()
    
            #predict_fn = model.get_model()


            for idx in range(num_samples):
                # Load raw sample (image)
                raw_img = bottleneck_features[idx]
                logging.info(f"[INFO] shape Input {raw_img.shape}")
                # Preprocess (resize, normalize, add channel if grayscale)
                preprocessed = process_bottleneck_features(raw_img)
                logging.info(f"[INFO] sImage Processed")            
                
            
                # Add batch dimension (since model usually expects shape (N, H, W, C))
                #preprocessed = np.expand_dims(preprocessed, axis=0)
            
                # Predict
                predicted_class, prediction = Model.predict(preprocessed)
                logging.info(f"[INFO] Prediction {predicted_class}{prediction}:")            
    
                results.append({
                    "sample": idx + 1,
                    "prediction_probabilities": prediction.tolist(),
                    "predicted_class": class_map[predicted_class],
                    "accuracy":round(np.max(prediction) * 100, 2),
                    "confidence":76,
                    "diagnosis":class_map[predicted_class],
                    "explainability":0.5,
                    "roiCoords": { "top": 0.20, "left": 0.30, "width": 0.40, "height": 0.35 }
                })
            
        # ************  advanced pipeline selected ******************            
        elif pipeline_global.lower() == "advanced":
            from ADCNNM import load_trained_model
            logging.info("[INFO] Running ADVANCED pipeline")
        
            json_path = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\trained_model\training_summary_advanced.json"
            weight_path = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\trained_model\best_model.pth"
        
            # Load the trained model
            model = load_trained_model(json_path, weight_path)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            for idx in range(num_samples):
                preprocessed = bottleneck_features[idx]
                logging.info(f"[INFO] Pre prediction Input shape: {preprocessed.shape}")
                
                # Convert to tensor and add batch dimension
                inputs = torch.tensor(preprocessed, dtype=torch.float32).unsqueeze(0).to(device)  # [1,C,H,W]
                
                # Run inference
                with torch.no_grad():
                    outputs = model(inputs)  # raw logits [1,num_classes]
                    _, predicted_class = torch.max(outputs, 1)  # predicted class index
                    predicted_class = predicted_class.item()
                    
                    # Optional: get probabilities if needed
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
                logging.info(f"[INFO] Prediction -> Class: {predicted_class}, Probabilities: {probabilities}")
            
                results.append({
                    "sample": idx + 1,
                    "prediction_probabilities": probabilities.tolist(),
                    "predicted_class": class_map[predicted_class],
                    "accuracy": round(float(np.max(probabilities) * 100),2),
                    "confidence": 76,
                    "diagnosis": class_map[predicted_class],
                    "explainability": 0.5,
                    "roiCoords": { "top": 0.20, "left": 0.30, "width": 0.40, "height": 0.35 }
                })


            
            
            
        # ------------------------------------------------------
        # Get list of all files in the folder
        # ------------------------------------------------------
        image_files = [f for f in os.listdir(CLEAN_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            raise ValueError("No image files found in the folder!")
        
        # Pick the first image (or loop through them if needed)
        image_path = os.path.join(CLEAN_IMAGE_FOLDER, image_files[0])
        
        # ------------------------------------------------------
        # Load image using OpenCV
        # ------------------------------------------------------
        img = cv2.imread(image_path)  # shape: (H, W, 3), BGR by default
        
        # Convert to RGB and float32
        img_rgb = ((img.astype(np.float32) / img.max()) * 255).astype(np.uint8)
        
        # ------------------------------------------------------
        # Resize to (512, 512)
        # ------------------------------------------------------
        img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        
        # ------------------------------------------------------
        # Convert to grayscale for Grad-CAM
        # ------------------------------------------------------
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # shape: (512, 512), 0–255
        img_gray = img_gray.astype(np.float32)
        # ------------------------------------------------------
        # Define output folder for Grad-CAM overlays
        # ------------------------------------------------------
        save_path = r'C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\explainability'
        
        # ------------------------------------------------------
        # Run Grad-CAM in a separate thread
        # ------------------------------------------------------
        thread = threading.Thread(
            target=generate_dual_class_gradcam_overlays_pytorch,
            args=(
                img_gray,   # (512, 512) grayscale image
                [0, 1],     # classes to visualize
                save_path   # where to save overlays
            )
        )
        thread.start()


        # Pass results to your HTML template
        return render_template('Analysis/Default/Classification.html', classificationData =results,image_filename = fileName)

    except Exception as e:
        return f"Error during classification: {str(e)}", 500

@app.route('/roi', methods=['GET'])
def roi():
    import logging
    import os
    import numpy as np
    import torch
    from flask import render_template

    fileName = copy_image_to_static()
    
    # Load .npy file from segmentation folder
    npy_files = [f for f in os.listdir(SEGMENTATION_FOLDER) if f.endswith('.npy')]
    if len(npy_files) == 0:
        return "No .npy file found in segmentation folder", 404
    elif len(npy_files) > 1:
        return "More than one .npy file found in segmentation folder", 400

    npy_file_path = os.path.join(SEGMENTATION_FOLDER, npy_files[0])

    try:
        # Load bottleneck features
        features = np.load(npy_file_path)
        bottleneck_features = np.expand_dims(features, axis=0)  # add batch dimension
        num_samples = bottleneck_features.shape[0]

        # Determine pipeline (basic or advanced)
        results = []
        class_map = {0: "Benign", 1: "Malignant"}

        if pipeline_global.lower() == "basic":
            from CNNM import Model
            logging.info("[INFO] Running BASIC pipeline")

            # Process only the first sample
            raw_img = bottleneck_features[0]
            preprocessed = process_bottleneck_features(raw_img)
            predicted_class, prediction = Model.predict(preprocessed)

            for class_idx in range(2):
                results.append({
                    "class_idx": class_idx,
                    "class_name": class_map[class_idx],
                    "prediction_probabilities": prediction.tolist(),
                    "predicted_class": class_map[predicted_class],
                    "accuracy": float(np.max(prediction) * 100),
                    "confidence": float(prediction[class_idx] * 100),
                    "diagnosis": class_map[predicted_class],
                    "explainability": 0.5,
                    "roiCoords": {"top": 0.20, "left": 0.30, "width": 0.20, "height": 0.175},
                    "overlay_path": f"explainability/gradcam_overlay_class_{class_idx}.png"
                })

        elif pipeline_global.lower() == "advanced":
            from ADCNNM import load_trained_model
            logging.info("[INFO] Running ADVANCED pipeline")

            json_path = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\trained_model\training_summary_advanced.json"
            weight_path = r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\trained_model\best_model.pth"

            # Load trained model
            model = load_trained_model(json_path, weight_path)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            raw_img = bottleneck_features[0]
            preprocessed = torch.tensor(raw_img, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(preprocessed)
                _, predicted_class = torch.max(outputs, 1)
                predicted_class = predicted_class.item()
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            for class_idx in range(2):
                results.append({
                    "class_idx": class_idx,
                    "class_name": class_map[class_idx],
                    "prediction_probabilities": probabilities.tolist(),
                    "predicted_class": class_map[predicted_class],
                    "accuracy": float(np.max(probabilities) * 100),
                    "confidence": float(probabilities[class_idx] * 100),
                    "diagnosis": class_map[predicted_class],
                    "explainability": 0.5,
                    "roiCoords": {"top": 0.20, "left": 0.30, "width": 0.20, "height": 0.175},
                    "overlay_path": f"explainability/gradcam_overlay_class_{class_idx}.png"
                })

        # Render template with results
        return render_template(
            'Analysis/Default/RegionOfInterest.html',
            classificationData=results,
            image_filename=fileName,
            class_0_image_path=results[0]['overlay_path'],
            class_1_image_path=results[1]['overlay_path']
        )

    except Exception as e:
        logging.error(f"[ERROR] {str(e)}")
        return f"Error during classification: {str(e)}", 500


# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- Route to handle bulk ZIP upload -----------------
@app.route('/upload-bulk', methods=['POST'])
def upload_bulk():
    # Clear previous images
    clear_folder(BULK_IMAGE_FOLDER)

    # Get uploaded ZIP file
    zip_file = request.files.get('bulk_images_zip')
    if zip_file and zip_file.filename.endswith('.zip'):
        zip_path = os.path.join(BULK_IMAGE_FOLDER, "temp.zip")
        zip_file.save(zip_path)

        # Extract ZIP contents directly into BULK_IMAGE_FOLDER
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                filename = os.path.basename(member.filename)  # ignore folders
                if not filename:  # skip directories
                    continue
                source = zip_ref.open(member)
                target_path = os.path.join(BULK_IMAGE_FOLDER, filename)
                with open(target_path, "wb") as target:
                    with source:
                        target.write(source.read())

        # Remove the ZIP file
        os.remove(zip_path)

    # Redirect to parameter selection page
    return redirect(url_for('bulk_select_parameters'))  # function name, not template path

# ----------------- Route to display extracted images for parameter selection -----------------
@app.route('/bulk-select-parameters', methods=['GET'])
def bulk_select_parameters():
    # Get all valid images in the folder
    images = [f for f in os.listdir(BULK_IMAGE_FOLDER) if allowed_file(f)]
    # Render the template with the relative path
    return render_template('Analysis/Default/bulk_select_parameters.html', images=images)

@app.route('/sample')
def sample_page():
    # Just render the static HTML file in templates/Analysis/Default/Sample.html
    return render_template('Analysis/Default/Sample.html')

# ---------- MAIN ----------

if __name__ == '__main__':
    logging.info("App started...")
    app.run(debug=False)
