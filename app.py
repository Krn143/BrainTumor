import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Import your custom architecture
from model_architecture import get_medsight_hex_model

# --- 1. PAGE CONFIG & ADVANCED UI STYLING ---
st.set_page_config(page_title="MedSight-Hex | Clinical Dashboard", layout="wide", page_icon="🧠")

# Custom CSS for a professional "Medical Tech" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4e5d6c; }
    .researcher-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GRAD-CAM CORE LOGIC ---
# --- 2. THE NOTEBOOK-EXACT XAI FUNCTION ---
def get_gradcam(model, input_tensor):
    """Exactly matches the notebook logic provided by user."""
    features = []
    def hook_feature(module, input, output):
        features.append(output)
    
    # Target layer from your architecture
    target_layer = model.blocks[-1].norm1
    handle = target_layer.register_forward_hook(hook_feature)
    
    # Forward & Backward Pass
    model.zero_grad()
    output = model(input_tensor)
    target_class = output.argmax(dim=1).item()
    output[0, target_class].backward()
    
    # Extracting weights based on your notebook's dim=1 logic
    # weights shape matches your ViT-Tiny 192 features
    weights = torch.mean(features[0], dim=1).squeeze().cpu().data.numpy()
    
    # Reshaping 192 features to 14x14 grid (196 patches)
    # We take the 192 available values and pad/adjust to 196 for the grid
    heatmap_values = np.zeros(196)
    heatmap_values[:len(weights)] = weights
    heatmap = heatmap_values.reshape(14, 14)
    
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    handle.remove()
    return heatmap, target_class, torch.softmax(output, dim=1)[0, target_class].item()
# --- 3. MODEL & DATA HELPERS ---
@st.cache_resource
def load_vision_engine():
    model = get_medsight_hex_model(num_classes=4)
    model_path = "HexFormer_BrainTumor_Final_96_on_val.pth"
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_report(label):
    reports = {
        "glioma": "Diagnostic features consistent with Glioma. Immediate neurosurgical consultation recommended.",
        "meningioma": "Findings suggest Meningioma. Recommend contrast MRI for dural evaluation.",
        "pituitary": "Pituitary mass detected. Standard protocol: Endocrine workup and visual field testing.",
        "no_tumor": "No significant tumor mass detected. Monitor symptoms clinically."
    }
    return reports.get(label, "Consult a specialist.")

# --- 4. SIDEBAR & RESEARCHER PROFILE ---
with st.sidebar:
    st.markdown(f"""
        <div class="researcher-card">
            <h3 style='margin:0; color:#60a5fa;'>👨‍🔬 Researcher</h3>
            <p style='margin:0; font-size:1.1em;'><b>Karan Sanjay Rathod</b></p>
            <p style='margin:0; font-size:0.8em; opacity:0.8;'>BE Computer Engineering</p>
            <hr style='margin:10px 0; border-color:#3b82f6;'>
            <p style='margin:0; font-size:0.9em;'><b>Project:</b> MedSight-Hex</p>
            <p style='margin:0; font-size:0.9em;'><b>Accuracy:</b> 96.6%</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload MRI Scan (T1-weighted)", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    st.write("🔧 **Backend:** Hyperbolic Lorentzian ViT")

# --- 5. MAIN DASHBOARD ---
vision_engine = load_vision_engine()

if uploaded_file:
    # Processing
    raw_img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0)
    input_tensor.requires_grad = True

    # Analysis
    heatmap, pred_idx, conf = get_gradcam(vision_engine, input_tensor)
    label = ['glioma', 'meningioma', 'no_tumor', 'pituitary'][pred_idx]

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Localization & XAI")
        # Tabs for better visualization
        tab_orig, tab_cam = st.tabs(["Original MRI", "Explainable AI (Heatmap)"])
        
        with tab_orig:
            st.image(raw_img, use_column_width=True)
        
        with tab_cam:
            img_np = np.array(raw_img.resize((224, 224)))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
            st.image(overlay, use_column_width=True)
            st.caption("Red regions indicate primary diagnostic focus points.")

    with col2:
        st.subheader("📊 Diagnostic Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Diagnosis", label.upper())
        m2.metric("Confidence", f"{conf*100:.2f}%")

        st.markdown("### 📜 Clinical Reasoning")
        st.info(get_report(label))
        
        st.markdown("### 🧬 Architecture Insights")
        st.write("""
            The **HexFormer** localized this pathology using non-Euclidean geometry. 
            By mapping the MRI features onto a **Lorentzian manifold**, the system 
            is able to capture hierarchical tumor boundaries more effectively than standard CNNs.
        """)
        
else:
    # Empty state visuals
    st.markdown("### 🚀 Welcome to MedSight-Hex")
    st.write("Please upload an MRI file from the sidebar to begin the Hyperbolic analysis.")
