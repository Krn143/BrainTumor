import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Import your custom architecture from your notebook code
from model_architecture import get_medsight_hex_model

# --- 1. PAGE CONFIG & ENHANCED UI STYLING ---
st.set_page_config(page_title="MedSight-Hex | Clinical Dashboard", layout="wide", page_icon="🧠")

# Custom CSS for the Advanced Profile and Dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4e5d6c; }
    
    /* Advanced Profile Card */
    .profile-container {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    .profile-pic {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #3b82f6;
        margin-bottom: 15px;
    }
    .social-links {
        display: flex;
        justify-content: space-around;
        margin-top: 15px;
    }
    .social-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 1.2em;
        transition: color 0.3s;
    }
    .social-links a:hover { color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERIFIED GRAD-CAM LOGIC ---
def get_gradcam(model, input_tensor):
    """Calculates heatmaps aligned with ViT-Tiny (192-dim) architecture."""
    activations = []
    gradients = []

    def save_activation(module, input, output): activations.append(output)
    def save_gradient(module, grad_input, grad_output): gradients.append(grad_output[0])

    # Targeting the final block's norm layer as per ViT standards
    target_layer = model.blocks[-1].norm1
    h_a = target_layer.register_forward_hook(save_activation)
    h_g = target_layer.register_full_backward_hook(save_gradient)

    model.zero_grad()
    output = model(input_tensor)
    _, pred_idx = torch.max(output, 1)
    output[:, pred_idx].backward()

    # Extracting tokens (1 Batch, 197 Tokens, 192 Features)
    grads = gradients[0].cpu().data.numpy()[0] 
    acts = activations[0].cpu().data.numpy()[0]
    
    h_a.remove()
    h_g.remove()

    # Average gradients to find feature importance
    weights = np.mean(grads, axis=0) 
    # Focus only on spatial tokens (1 to 196), ignoring the Class Token (0)
    spatial_acts = acts[1:, :] 
    
    # Generate heatmap through dot product
    cam = np.dot(spatial_acts, weights)
    cam = cam.reshape(14, 14)

    # Post-processing for visualization
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    return cam, pred_idx.item(), torch.softmax(output, dim=1)[0, pred_idx].item()

# --- 3. MODEL LOADERS ---
@st.cache_resource
def load_vision_engine():
    model = get_medsight_hex_model(num_classes=4)
    # Ensure this file exists in your GitHub repo!
    model_path = "HexFormer_BrainTumor_Final_96_on_val.pth"
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 4. SIDEBAR & ADVANCED RESEARCHER PROFILE ---
with st.sidebar:
    # 1. Profile Photo (Ensure 'profile.jpg' is in your GitHub repo)
    try:
        st.markdown(f'<div class="profile-container">', unsafe_allow_html=True)
        st.image("profile.jpg", width=120, use_column_width=False)
        st.markdown(f"""
            <h3 style='margin:10px 0 5px 0; color:#ffffff;'>Karan Sanjay Rathod</h3>
            <p style='color:#3b82f6; font-size:0.9em; margin-bottom:15px;'>BE Computer Engineering</p>
            <div class="social-links">
                <a href="https://www.github.com/Krn143/BrainTumor" target="_blank">🔗 GitHub</a>
                <a href="https://www.linkedin.com/in/karan-sanjay-rathod" target="_blank">🔗 LinkedIn</a>
                <a href="https://www.kaggle.com/karansrathod1432" target="_blank">🔗 Kaggle</a>
            </div>
            </div>
        """, unsafe_allow_html=True)
    except:
        st.warning("Upload 'profile.jpg' to GitHub to see your photo!")

    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload MRI Scan (T1-weighted)", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    st.write("🔧 **Geometry:** Lorentzian Hyperbolic")

# --- 5. MAIN DASHBOARD ---
vision_engine = load_vision_engine()

if uploaded_file:
    # Pre-processing
    raw_img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0)
    input_tensor.requires_grad = True

    # Run Analysis
    heatmap, pred_idx, conf = get_gradcam(vision_engine, input_tensor)
    label = ['glioma', 'meningioma', 'no_tumor', 'pituitary'][pred_idx]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Localization (XAI)")
        tab1, tab2 = st.tabs(["Diagnostic View", "Heatmap Overlay"])
        with tab1:
            st.image(raw_img, use_column_width=True)
        with tab2:
            img_np = np.array(raw_img.resize((224, 224)))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
            st.image(overlay, use_column_width=True)
            st.caption("Red regions indicate where the Lorentzian ViT focused its diagnostic attention.")

    with col2:
        st.subheader("📊 Performance Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Diagnosis", label.upper())
        m2.metric("Confidence", f"{conf*100:.2f}%")
        
        st.markdown("### 🏛️ Architecture Note")
        st.write("The model utilizes a **Hyperbolic (Lorentzian) manifold** to better represent the hierarchical structure of tumor growth compared to standard Euclidean ViTs.")

else:
    st.markdown("### 🚀 Ready for Diagnostic Analysis")
    st.info("Upload a brain MRI to visualize the 96.6% accuracy classification results.")
