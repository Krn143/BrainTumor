import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time

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
def get_gradcam(model, input_tensor):
    """Calculates heatmaps specifically for the 192-dim HexFormer architecture."""
    activations = []
    gradients = []

    def save_activation(module, input, output): activations.append(output)
    def save_gradient(module, grad_input, grad_output): gradients.append(grad_output[0])

    # Target the last transformer block's normalization layer
    target_layer = model.blocks[-1].norm1
    h_a = target_layer.register_forward_hook(save_activation)
    h_g = target_layer.register_full_backward_hook(save_gradient)

    model.zero_grad()
    output = model(input_tensor)
    _, pred_idx = torch.max(output, 1)
    output[:, pred_idx].backward()

    # Get data and remove hooks immediately
    grads = gradients[0].cpu().data.numpy() # (1, 197, 192)
    acts = activations[0].cpu().data.numpy()  # (1, 197, 192)
    h_a.remove()
    h_g.remove()

    # 1. Average the gradients across the 197 tokens to get 192 weights
    # This matches the 192 feature dimension of your ViT-Tiny
    weights = np.mean(grads[0], axis=0) 

    # 2. Ignore the class token (index 0) to get the 196 spatial patches
    spatial_acts = acts[0, 1:, :] # Shape: (196, 192)

    # 3. Multiply (196, 192) by (192,) to get (196,)
    cam = np.dot(spatial_acts, weights)

    # 4. Reshape the 196 patches into a 14x14 grid
    cam = cam.reshape(14, 14)

    # 5. Normalize for visualization
    cam = np.maximum(cam, 0) # ReLU to keep only positive influence
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    
    return cam, pred_idx.item(), torch.softmax(output, dim=1)[0, pred_idx].item()

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


def call_medgemma_api(pred_label: str, conf_score: float, mode_simulated: bool = True):
    """Return a MedGemma-style expert summary and a patient-friendly summary.

    This function is currently a simulated/local fallback that returns prewritten
    outputs for each class. Replace the implementation with an actual Vertex AI
    or Hugging Face `generate` call in production. Example pseudocode is shown
    in the docstring for easy replacement.

    Example replacement (pseudocode):
        # build prompt from pred_label and conf_score
        response = vertex_ai.generate_text(prompt)
        return response['expert'], response['patient_friendly']
    """

    # Simulate network latency for realism
    if mode_simulated:
        time.sleep(0.6)

    expert_templates = {
        "glioma": (
            "Expert Summary: The model predicts Glioma with "
            f"{conf_score*100:.2f}% confidence. Imaging features localize to "
            "the cerebral hemisphere with heterogeneous signal and patchy enhancement. "
            "Recommend expedited neurosurgical evaluation and contrast MRI for surgical planning."
        ),
        "meningioma": (
            "Expert Summary: The model predicts Meningioma with "
            f"{conf_score*100:.2f}% confidence. Appearance is extra-axial and dural-based, "
            "often showing homogeneous enhancement. Recommend contrast-enhanced MRI and "
            "neurosurgical consultation for resection planning."
        ),
        "pituitary": (
            "Expert Summary: The model predicts a Pituitary lesion with "
            f"{conf_score*100:.2f}% confidence. Lesion is centered in the sellar region; "
            "suggest endocrine panel and dedicated pituitary MRI with contrast."
        ),
        "no_tumor": (
            "Expert Summary: No tumor detected (model confidence "
            f"{conf_score*100:.2f}%). Findings are within expected normal limits for this sequence. "
            "Correlate clinically and consider follow-up imaging if symptoms persist."
        ),
    }

    patient_templates = {
        "glioma": (
            "Patient Summary: The scan appears consistent with a type of brain tumor called a glioma. "
            "This means there is an area the model highlights that should be reviewed by a neurosurgeon. "
            "Next steps typically include further MRI with contrast and referral to a specialist."
        ),
        "meningioma": (
            "Patient Summary: The image likely shows a meningioma, a usually slow-growing tumor "
            "on the brain's surface. Doctors often confirm this with a contrast scan and discuss "
            "treatment options such as monitoring or surgery."
        ),
        "pituitary": (
            "Patient Summary: The scan suggests a small growth in the pituitary gland. "
            "Your care team will typically check hormone levels and order focused imaging to guide next steps."
        ),
        "no_tumor": (
            "Patient Summary: No clear tumor was identified on this image. If you have symptoms, "
            "your doctor may recommend monitoring or additional tests, but there are no urgent findings here."
        ),
    }

    expert = expert_templates.get(pred_label, "Expert Summary: Consult a specialist for further review.")
    patient = patient_templates.get(pred_label, "Patient Summary: Consult your clinician for interpretation.")

    # Annotate that this is a simulated/local response when mode_simulated is True
    if mode_simulated:
        expert = "[SIMULATED MedGemma] " + expert
        patient = "[SIMULATED MedGemma] " + patient

    return expert, patient

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
    st.subheader("MedGemma Integration")
    simulated_mode = st.checkbox("Use simulated MedGemma (local templates)", value=True, help="When checked the app returns prewritten MedGemma-style outputs locally. Replace with Vertex AI call for production.")
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
        # Call the MedGemma API (simulated or true integration)
        expert_report, patient_report = call_medgemma_api(label, conf, mode_simulated=simulated_mode)

        st.subheader("Expert Summary (MedGemma)")
        st.info(expert_report)

        st.subheader("Patient-friendly Summary")
        st.info(patient_report)
        
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
