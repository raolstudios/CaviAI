import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="CaviAI v2 - Dental Cavity Detection",
    page_icon="🦷",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦷 CaviAI v2 Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Dental Radiograph Analysis powered by ResNet18 & CLAHE Contrast Enhancement</div>', unsafe_allow_html=True)

# ==========================================
# 2. MODEL SETUP & LOADING
# ==========================================
MODEL_PATH = "CaviAI_v2.pth"
CLASS_NAMES = ['Colored / Healthy', 'X-Ray / Cavity']  # Update mapping if customized

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"❌ Model file `{MODEL_PATH}` not found in root directory! Please ensure it is uploaded.")
        return None, device

model, device = load_model()

# ==========================================
# 3. CLAHE PREPROCESSING PIPELINE
# ==========================================
def apply_clahe_preprocessing(pil_image):
    """Converts PIL image to OpenCV format, applies CLAHE in LAB space, and returns PIL image."""
    img_np = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(clahe_rgb)

# PyTorch Image Transformation Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. SIDEBAR & FILE UPLOADER
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    show_clahe_view = st.checkbox("Show CLAHE Enhanced Image Comparison", value=True)
    
    st.divider()
    st.markdown("### About CaviAI v2")
    st.info(
        "CaviAI v2 leverages a fine-tuned ResNet18 architecture integrated with "
        "Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost feature visibility in dental radiographs."
    )

uploaded_file = st.file_uploader(
    "Choose a Dental Radiograph (PNG, JPG, JPEG)...", 
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# 5. INFERENCE & RESULTS DISPLAY
# ==========================================
if uploaded_file is not None and model is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    clahe_image = apply_clahe_preprocessing(raw_image)
    
    # Visual Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Original Upload")
        st.image(raw_image, use_column_width=True)
        
    with col2:
        if show_clahe_view:
            st.subheader("🔬 CLAHE Contrast Enhanced (Model Input)")
            st.image(clahe_image, use_column_width=True)
        else:
            st.subheader("📊 Diagnostic Analysis")
            st.write("Enable 'Show CLAHE Enhanced Image' in the sidebar to inspect feature enhancement.")

    st.divider()

    # Preprocess & Inference
    input_tensor = transform_pipeline(clahe_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)

    predicted_label = CLASS_NAMES[predicted_class_idx.item()]
    confidence_score = confidence.item() * 100

    # Output Diagnostic Banner
    st.subheader("🎯 Diagnostic Prediction")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Predicted Classification", value=predicted_label)
        st.metric(label="Confidence Level", value=f"{confidence_score:.2f}%")

    with res_col2:
        st.markdown("#### Probability Distribution")
        for idx, name in enumerate(CLASS_NAMES):
            prob = probabilities[idx].item() * 100
            st.write(f"**{name}** ({prob:.1f}%)")
            st.progress(int(prob))

    # Diagnostic Disclaimer
    st.caption("⚠️ Note: CaviAI v2 is designed as a proof-of-concept decision support system and should not replace professional clinical evaluation.")