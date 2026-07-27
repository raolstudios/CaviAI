import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="CaviAI — Dental Cavity Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for modern UI styling
st.markdown("""
    <style>
    /* Global background and font styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom card container */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    /* Result Badges */
    .badge-positive {
        background-color: #fff0f0;
        color: #d9534f;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #d9534f;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .badge-negative {
        background-color: #f0fff4;
        color: #2b8a3e;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #2b8a3e;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Sidebar aesthetic */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING & PREPROCESSING
# ==========================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Path to model weights
    model_path = "caviAI_v1.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.title("🦷 CaviAI Controls")
    st.caption("AI Diagnostic Assistant v1.0")
    st.markdown("---")
    
    st.subheader("⚙️ Detection Sensitivity")
    sensitivity_threshold = st.slider(
        "Cavity Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Lower threshold increases sensitivity to detect subtle lesions (fewer false negatives)."
    )
    
    st.markdown("---")
    st.markdown("### 📌 Instructions")
    st.markdown("""
    1. Upload a **cropped tooth patch** from a dental radiograph.
    2. Ensure image format is `PNG` or `JPG`.
    3. Review confidence probabilities and diagnostic alerts.
    """)
    
    st.markdown("---")
    if model_loaded:
        st.success("🟢 Model Engine Online")
    else:
        st.error("🔴 Model Weights Not Found (`caviAI_v1.pth`)")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🦷 CaviAI")
st.markdown("**Deep Learning Dental Cavity Detection Assistant**")
st.caption("Upload localized radiograph tooth patches for instant caries risk assessment.")
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "Choose a cropped dental X-ray patch...", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Grid Layout: Left column for image preview, Right column for diagnosis
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🔍 Uploaded Radiograph")
        st.image(image, use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Diagnostic Analysis")
        
        if not model_loaded:
            st.error("Unable to run inference. Please verify model weights file.")
        else:
            with st.spinner("Analyzing radiograph density & features..."):
                # Inference
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    
                non_cavity_prob = probabilities[0].item()
                cavity_prob = probabilities[1].item()
                
                is_cavity = cavity_prob >= sensitivity_threshold

            # Results Display
            if is_cavity:
                st.markdown(
                    f"""<div class="badge-positive">
                    ⚠️ <b>Caries Detected</b><br>
                    Probability exceeds sensitivity threshold ({sensitivity_threshold * 100:.0f}%).
                    </div>""", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="badge-negative">
                    ✅ <b>No Caries Detected</b><br>
                    Radiograph patch appears normal within threshold limits.
                    </div>""", 
                    unsafe_allow_html=True
                )
                
            st.write("") # Spacing
            
            # Confidence Breakdown Metrics
            st.markdown("#### Confidence Breakdown")
            st.metric(label="Cavity Probability", value=f"{cavity_prob * 100:.1f}%")
            st.progress(cavity_prob)
            
            st.metric(label="Healthy Tissue Probability", value=f"{non_cavity_prob * 100:.1f}%")
            st.progress(non_cavity_prob)

else:
    # Empty State Dashboard Preview
    st.info("👆 Please upload a cropped radiograph patch above to begin evaluation.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** CaviAI is an AI proof-of-concept for demonstration purposes only. It is not intended for clinical use or professional medical decision-making.")