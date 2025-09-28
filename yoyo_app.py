import streamlit as st
import os
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import uuid
import numpy as np
import cv2

# Configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("🔬 Breast Cancer Detection")
    st.markdown("---")
    
    st.subheader("Model Selection")
    model_option = st.radio(
        "Choose detection model:",
        ["Custom AI YOLOv5", "Pre-trained YOLOv5", "EfficientNet (Classification)"],
        help="Select the model architecture for detection"
    )
    
    st.markdown("---")
    st.subheader("Detection Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=0.99, 
        value=0.25,
        step=0.01,
        help="Lower values will detect more potential tumors but may increase false positives"
    )
    
    iou_threshold = st.slider(
        "IOU Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjust how much overlapping detections are merged"
    )
    
    st.markdown("---")
    st.subheader("Image Preprocessing")
    
    enhance_contrast = st.checkbox("Enhance Contrast", value=True)
    convert_grayscale = st.checkbox("Convert to Grayscale", value=False)
    apply_clahe = st.checkbox("Apply CLAHE", value=True)
    sharpen_image = st.checkbox("Sharpen Image", value=False)
    
    st.markdown("---")
    st.subheader("App Settings")
    
    dark_mode = st.checkbox("🌙 Enable Dark Mode", value=False)
    enable_webcam = st.checkbox("📷 Enable Webcam Capture", value=True)
    
    st.markdown("---")
    st.warning("""
    **Disclaimer**: This tool is for research purposes only. 
    Always consult a medical professional for diagnosis.
    """)

# CSS definitions as in original code
light_css = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-info {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
"""

dark_css = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stMarkdown {
        color: #fafafa;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4dabf7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-info {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #fafafa;
    }
    .warning-box {
        background-color: #2d1b00;
        border-left: 4px solid #ff9500;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
        color: #fafafa;
    }
    .success-box {
        background-color: #1a3c1a;
        border-left: 4px solid #28a745;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
        color: #fafafa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2a2d3e;
        border-radius: 4px 4px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #fafafa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4dabf7;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        color: #fafafa;
    }
    .stMetric > label {
        color: #fafafa;
    }
    .stMetric > div > div {
        color: #fafafa;
    }
    .stButton > button {
        background-color: #4dabf7;
        color: white;
    }
    .stSlider > div > div > div > div {
        background-color: #4dabf7;
    }
</style>
"""

if dark_mode:
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "Custom AI YOLOv5":
            weights_path = 'best.pt'
            if os.path.exists(weights_path):
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
                st.sidebar.success("Custom AI YOLOv5 model loaded successfully!")
                return model
            else:
                st.sidebar.error("Custom model weights not found. Using pre-trained model.")
                model_type = "Pre-trained YOLOv5"
        
        if model_type == "Pre-trained YOLOv5":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            st.sidebar.success("Pre-trained YOLOv5 model loaded!")
            return model
            
        if model_type == "EfficientNet (Classification)":
            try:
                model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
                model.eval()
                st.sidebar.success("EfficientNet model loaded!")
                return model
            except:
                st.sidebar.warning("Could not load EfficientNet. Using YOLOv5 instead.")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
                
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {str(e)}")
        return None

model = load_model(model_option)

def preprocess_image(image, enhance_contrast, convert_grayscale, apply_clahe, sharpen_image):
    img = image.copy()
    if convert_grayscale:
        img = img.convert('L')
        img = img.convert('RGB')
    else:
        img = img.convert('RGB')
    img_array = np.array(img)
    if apply_clahe and len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    if enhance_contrast:
        pil_img = Image.fromarray(img_array)
        enhancer = ImageEnhance.Contrast(pil_img)
        img_array = np.array(enhancer.enhance(1.5))
    if sharpen_image:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(img_array)

def run_detection_yolo(img, conf_thres, iou_thres):
    if model is None:
        st.error("Model not available. Cannot perform detection.")
        return None
    try:
        model.conf = conf_thres
        model.iou = iou_thres
        results = model(img, size=640)
        detections = results.pandas().xyxy[0]
        return detections
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        return None

def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
        label = f"{detection['name']} {detection['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1, x1 + text_width + 4, y1 + text_height + 4], fill="red")
        draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)
    return image

def process_and_display_results(image, confidence_threshold, iou_threshold, 
                               enhance_contrast, convert_grayscale, apply_clahe, sharpen_image):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Input Image", use_container_width=True)
    with col2:
        st.subheader("Detection Results")
        if model is None:
            st.error("Model failed to load - cannot perform detection")
        else:
            with st.spinner("Processing image..."):
                processed_img = preprocess_image(
                    image, enhance_contrast, convert_grayscale, apply_clahe, sharpen_image
                )
                detections = run_detection_yolo(processed_img, confidence_threshold, iou_threshold)
                if detections is not None:
                    if len(detections) > 0:
                        result_img = draw_detections(processed_img.copy(), detections)
                        st.image(result_img, caption="Detection Result", use_container_width=True)
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"Found {len(detections)} potential tumor(s)")
                        for i, (_, detection) in enumerate(detections.iterrows(), 1):
                            confidence = detection['confidence']
                            class_name = detection['name']
                            st.write(f"**Detection {i}**: {class_name} (confidence: {confidence:.3f})")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.image(processed_img, caption="Processed Image (No Detections)", use_container_width=True)
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.warning("""
                        No tumors detected with the current settings.
                        
                        **Suggestions:**
                        - Try lowering the confidence threshold
                        - Enable different preprocessing options
                        - Try a different model type
                        - Ensure the image is clear and properly focused
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Detection failed. Please try again.")

st.markdown('<p class="main-header">🔬 Advanced Breast Cancer Detection</p>', unsafe_allow_html=True)
st.markdown("Upload a medical image or use your webcam for tumor detection")

if model is None:
    st.error("Model status: Not loaded - detection unavailable")
else:
    st.success(f"Model status: {model_option} loaded successfully")

if enable_webcam:
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])
else:
    tab1 = st.tabs(["📁 Upload Image"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        help="Supported formats: JPG, PNG, TIFF, BMP",
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            # Save the uploaded image to 'uploads' directory
            save_dir = "uploads"
            os.makedirs(save_dir, exist_ok=True)
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(save_dir, unique_filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open(uploaded_file).convert("RGB")

            st.success(f"Image saved as {unique_filename}")

            if st.button("🔍 Detect Tumor", key="detect_file"):
                process_and_display_results(
                    image, confidence_threshold, iou_threshold,
                    enhance_contrast, convert_grayscale, apply_clahe, sharpen_image
                )
        except Exception as e:
            st.error(f"Error loading or saving image: {str(e)}")

if enable_webcam:
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p>Capture an image using your webcam for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        picture = st.camera_input("Take a picture", key="webcam")
        
        if picture is not None:
            try:
                image = Image.open(picture).convert("RGB")
                if st.button("🔍 Detect Tumor", key="detect_webcam"):
                    process_and_display_results(
                        image, confidence_threshold, iou_threshold,
                        enhance_contrast, convert_grayscale, apply_clahe, sharpen_image
                    )
            except Exception as e:
                st.error(f"Error processing webcam image: {str(e)}")

st.markdown("---")
st.subheader("Troubleshooting")

with st.expander("Click here if you're not getting any detections"):
    st.markdown("""
    If the model isn't detecting tumors in images that should contain them, try these steps:
    
    1. **Lower the confidence threshold** in the sidebar (try 0.1 or lower)
    2. **Try different preprocessing options** - sometimes CLAHE or contrast enhancement helps
    3. **Switch between model types** - your custom model might need specific preprocessing
    4. **Check your model file** - ensure 'best.pt' is in the correct location and is a valid model
    5. **Try different images** - some images might work better than others
    
    If you continue to have issues, the model might not be compatible with your images or might need retraining.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Breast Cancer Detection App | For research purposes only</p>
</div>
""", unsafe_allow_html=True)
