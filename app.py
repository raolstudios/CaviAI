import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dental Cavity Detector",
    page_icon="🦷",
    layout="centered"
)

st.title("🦷 Dental Radiograph Cavity Detector")
st.write("Upload a cropped dental radiograph patch to analyze for caries/cavities.")

# -----------------------------------------------------------------------------
# 2. Load Model Architecture & Weights
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Initialize ResNet18 structure (matching training set-up)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load trained weights
    model_path = "caviAI_v1.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model file `caviAI_v1.pth` not found in project directory. Please place it in the same folder as `app.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Image Preprocessing Pipelines
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# 4. File Upload & Prediction Interface
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Dental X-Ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Safe image opening to catch corrupted or empty files
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Radiograph Patch", use_column_width=True)
    except Exception:
        st.error("⚠️ Could not read the image file. Please upload a valid PNG, JPG, or JPEG file.")
        st.stop()

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Class Index Mapping: 0 -> Cavity, 1 -> No Cavity
    cavity_prob = probabilities[0].item() * 100
    no_cavity_prob = probabilities[1].item() * 100

    st.markdown("---")
    st.subheader("Analysis Results")

    # -------------------------------------------------------------------------
    # Sensitivity Thresholding (Prevents False Negatives)
    # Even if the model is only 35%+ confident of a cavity, flag it for review!
    # -------------------------------------------------------------------------
    CAVITY_SENSITIVITY_THRESHOLD = 35.0

    if cavity_prob >= CAVITY_SENSITIVITY_THRESHOLD:
        st.error(f"### ⚠️ Cavity Detected")
        st.write(f"**Confidence Score:** {cavity_prob:.2f}%")
        st.progress(min(int(cavity_prob), 100))
    else:
        st.success(f"### ✅ No Cavity Detected")
        st.write(f"**Confidence Score:** {no_cavity_prob:.2f}%")
        st.progress(min(int(no_cavity_prob), 100))

    # Detailed breakdown expander
    with st.expander("See Raw Probability Breakdown"):
        st.write(f"- **Cavity Probability:** {cavity_prob:.2f}%")
        st.write(f"- **Healthy Probability:** {no_cavity_prob:.2f}%")
        st.write(f"- **Decision Threshold Used:** {CAVITY_SENSITIVITY_THRESHOLD}%")