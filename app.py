import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.cm as cm

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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦷 CaviAI v2 Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Dental Radiograph Analysis with Grad-CAM Visualizations</div>', unsafe_allow_html=True)

# ==========================================
# 2. MODEL SETUP & LOADING
# ==========================================
MODEL_PATH = "CaviAI_v2.pth"
CLASS_NAMES = ['Healthy / Normal', 'Cavity / Decay Detected']

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
        st.error(f"❌ Model file `{MODEL_PATH}` not found in root directory!")
        return None, device

model, device = load_model()

# ==========================================
# 3. GRAD-CAM HEATMAP GENERATOR
# ==========================================
def generate_gradcam(model, input_tensor, target_class):
    """Generates Grad-CAM heatmap for ResNet18 layer4 natively in PyTorch."""
    model.eval()
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Hook into last convolutional layer of ResNet18
    target_layer = model.layer4[1].conv2
    hook = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward pass for target class
    score = output[0, target_class]
    score.backward()

    # Calculate Grad-CAM
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    hook.remove()
    return cam

def overlay_heatmap(original_pil, cam_map):
    """Overlays heatmap on top of original PIL radiograph."""
    img_resized = original_pil.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    # Resize CAM to image size
    cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    cam_array = np.array(cam_pil) / 255.0

    # Apply Jet Color Map
    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam_array)[:, :, :3]

    # Blend original and heatmap
    blended = 0.5 * img_array + 0.5 * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# Transformations
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. SIDEBAR & FILE UPLOADER
# ==========================================
with st.sidebar:
    st.header("⚙️ Diagnostics Controls")
    show_heatmap = st.checkbox("Show Grad-CAM Heatmap Focus", value=True)
    show_enhanced = st.checkbox("Apply Contrast Equalization", value=True)

uploaded_file = st.file_uploader(
    "Upload Dental Radiograph (PNG, JPG, JPEG)...", 
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# 5. INFERENCE & HEATMAP DISPLAY
# ==========================================
if uploaded_file is not None and model is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    processed_image = ImageOps.equalize(ImageOps.grayscale(raw_image)).convert('RGB') if show_enhanced else raw_image
    
    input_tensor = transform_pipeline(processed_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # Forward Pass
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_class_idx = torch.max(probabilities, 0)

    target_class = predicted_class_idx.item()
    predicted_label = CLASS_NAMES[target_class]
    confidence_score = confidence.item() * 100

    # Generate Heatmap
    cam_map = generate_gradcam(model, input_tensor, target_class)
    heatmap_overlay = overlay_heatmap(raw_image, cam_map)

    # Layout Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Uploaded Radiograph")
        st.image(raw_image, use_container_width=True)
        
    with col2:
        if show_heatmap:
            st.subheader("🔥 Grad-CAM Attention Heatmap")
            st.image(heatmap_overlay, use_container_width=True)
        else:
            st.subheader("🔬 Equalized Preprocessing")
            st.image(processed_image, use_container_width=True)

    st.divider()

    # Results Section
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

    st.caption("⚠️ Note: Heatmaps highlight region intensity associated with model outputs.")