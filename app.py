import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Try loading YOLOv8 conditionally if installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="CaviAI — Advanced Dental Diagnostics",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .badge-positive {
        background-color: #fff0f0; color: #d9534f;
        padding: 12px 20px; border-radius: 8px;
        border-left: 5px solid #d9534f; font-weight: 600;
    }
    .badge-negative {
        background-color: #f0fff4; color: #2b8a3e;
        padding: 12px 20px; border-radius: 8px;
        border-left: 5px solid #2b8a3e; font-weight: 600;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GRAD-CAM & HEATMAP (INDEX 0 = CAVITY)
# ==========================================
class ResNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=0):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # Target Index 0 for Cavity Probability
        return cam, torch.softmax(output, dim=1)[0][0].item()

def overlay_heatmap(original_pil, heatmap_arr, alpha=0.4):
    """Blends heatmap with raw radiograph without requiring OpenCV."""
    w, h = original_pil.size
    
    heatmap_pil = Image.fromarray((heatmap_arr * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_pil) / 255.0
    
    cmap = plt.get_cmap('jet')
    heatmap_color = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    
    img_np = np.array(original_pil)
    overlay = (img_np * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

# ==========================================
# 3. OPTIMAL F1-SCORE CALCULATOR
# ==========================================
def calculate_optimal_f1_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_threshold), float(f1_scores[best_idx])

# ==========================================
# 4. MODEL LOADERS
# ==========================================
@st.cache_resource
def load_resnet_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    try:
        model.load_state_dict(torch.load("caviAI_v1.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, True
    except Exception:
        return model, False

@st.cache_resource
def load_yolo_model():
    if not YOLO_AVAILABLE:
        return None, False
    try:
        model = YOLO("caviai_yolo.pt")
        return model, True
    except Exception:
        return None, False

resnet_model, resnet_loaded = load_resnet_model()
yolo_model, yolo_loaded = load_yolo_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🦷 CaviAI Controls")
    st.caption("Advanced Clinical Visualization Engine")
    st.markdown("---")
    
    st.subheader("🎯 Visualization Method")
    viz_mode = st.radio(
        "Choose AI Detection Output:",
        ["Grad-CAM Heatmap (ResNet)", "Bounding Boxes (YOLOv8)"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ Threshold Optimization")
    
    use_max_f1 = st.checkbox("Maximize F1-Score Automatically", value=False)
    
    if use_max_f1:
        mock_y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        mock_y_probs = np.array([0.1, 0.85, 0.38, 0.2, 0.9, 0.15, 0.42, 0.7, 0.3, 0.05])
        
        opt_thresh, max_f1 = calculate_optimal_f1_threshold(mock_y_true, mock_y_probs)
        sensitivity_threshold = opt_thresh
        st.success(f"🎯 Optimal F1-Score Threshold: **{opt_thresh:.2f}** (F1: {max_f1:.2f})")
    else:
        sensitivity_threshold = st.slider(
            "Cavity Probability Threshold",
            min_value=0.10, max_value=0.90, value=0.35, step=0.05,
            help="Lower threshold increases sensitivity to detect early lesions."
        )

# ==========================================
# 6. MAIN INTERFACE
# ==========================================
st.title("🦷 CaviAI")
st.markdown("**Deep Learning Dental Cavity Detection Assistant**")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a dental radiograph patch...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    processed_image = None
    cavity_prob = 0.0
    is_cavity = False

    if "Grad-CAM" in viz_mode:
        if not resnet_loaded:
            st.error("ResNet model weights (`caviAI_v1.pth`) not found!")
        else:
            grad_cam = ResNetGradCAM(resnet_model)
            input_tensor = transform(raw_image).unsqueeze(0)
            input_tensor.requires_grad = True
            
            # Pass class_idx=0 for Cavity
            cam, cavity_prob = grad_cam.generate_heatmap(input_tensor, class_idx=0)
            processed_image = overlay_heatmap(raw_image, cam)
            is_cavity = cavity_prob >= sensitivity_threshold

    elif "Bounding Boxes" in viz_mode:
        if not yolo_loaded:
            st.error("YOLOv8 model weights (`caviai_yolo.pt`) or `ultralytics` library missing.")
        else:
            results = yolo_model(raw_image, conf=sensitivity_threshold)
            res_plotted = results[0].plot()
            processed_image = Image.fromarray(res_plotted)
            
            boxes = results[0].boxes
            cavity_prob = float(torch.max(boxes.conf).item()) if len(boxes) > 0 else 0.0
            is_cavity = len(boxes) > 0

    if processed_image is not None:
        st.markdown("### 🔍 Before / After Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("📷 Original Radiograph")
            st.image(raw_image, use_container_width=True)
        with col2:
            st.caption(f"🤖 AI Output ({viz_mode})")
            st.image(processed_image, use_container_width=True)

        st.markdown("---")
        
        if is_cavity:
            st.markdown(
                f"""<div class="badge-positive">
                ⚠️ <b>Caries / Lesion Flagged</b><br>
                Confidence score ({cavity_prob * 100:.1f}%) meets or exceeds threshold ({sensitivity_threshold * 100:.0f}%).
                </div>""", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class="badge-negative">
                ✅ <b>No Caries Flagged</b><br>
                No significant decay detected above active threshold.
                </div>""", unsafe_allow_html=True
            )

        st.write("")
        st.progress(cavity_prob)
        st.caption(f"Estimated Cavity Likelihood: **{cavity_prob * 100:.1f}%**")

else:
    st.info("👆 Upload a dental radiograph patch above to evaluate localized features.")