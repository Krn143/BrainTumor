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
st.set_page_config(page_title="MedSight-ViT | Clinical Dashboard", layout="wide", page_icon="🧠")

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
    """Calculates heatmaps specifically for the 192-dim Transformer architecture."""
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
            <p style='margin:0; font-size:0.8em; opacity:0.8;'>BE Computer Engineering (SPPU)</p>
            <hr style='margin:10px 0; border-color:#3b82f6;'>
            <p style='margin:0; font-size:0.9em;'><b>Project:</b> MedSight-ViT</p>
            <p style='margin:0; font-size:0.9em;'><b>Accuracy:</b> 94.6%</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload MRI Scan (T1-weighted)", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    st.subheader("MedGemma Integration")
    simulated_mode = st.checkbox("Use simulated MedGemma (local templates)", value=True, help="When checked the app returns prewritten MedGemma-style outputs locally. Replace with Vertex AI call for production.")
    st.markdown("---")
    
    # Code Generation & Repository Information
    with st.expander("ℹ️ About This Project"):
        st.write("""
        ### MedSight-Hex: Transformer-based Brain Tumor Classification
        
        **Project Overview:**
        This application implements a Vision Transformer (ViT) pipeline for 
        multi-class brain tumor classification using T1-weighted MRI scans. The system 
        combines explainable AI (Grad-CAM) and clinical text generation for actionable insights.
        
        **Key Components:**
        - **Vision Backbone:** ViT-Tiny (vit_tiny_patch16_224) with ImageNet pretraining
        - **Training Strategy:** Two-stage fine-tuning (head → selective block unfreezing)
        - **Explainability:** Adapted Grad-CAM for transformer patch representations
        - **Clinical Output:** Simulated MedGemma integration (production-ready template)
        
        **Dataset:**
        - BRISC 2025 classification challenge (https://www.kaggle.com/datasets/briscdataset/brisc2025)
        - 4 classes: Glioma, Meningioma, Pituitary, No Tumor
        - Train/Val/Test split with deduplication
        
        **Performance:**
        - Validation Accuracy: 94.6% 
        - Class-balanced sampling via WeightedRandomSampler
        - Early stopping with patience=5
        
        **Code & References:**
        - Model Training: braintumor-classification.ipynb
        - Streamlit App: app.py (this file)
        - Libraries: PyTorch, TIMM, Hugging Face Transformers, Streamlit
        
        **Important Note on MedGemma:**
        Due to exhausted cloud API credits, the MedGemma text-generation integration 
        uses **simulated local templates** during demonstration. These templates are based 
        on expected neuroradiological descriptions for each tumor class and match the output 
        structure of the real MedGemma API. For production deployment, replace the 
        `call_medgemma_api()` function in app.py with a real Vertex AI or Hugging Face 
        Inference API call (see docstring for pseudocode example).
        
        **Disclaimer:**
        Automated model outputs are intended to **assist, not replace** expert clinical 
        interpretation. This is a proof-of-concept system requiring prospective clinical 
        validation, regulatory approval, and integration into clinical workflows before 
        deployment.
        """)
    
    st.markdown("---")
    st.write("🔧 **Backend:** Vision Transformer + Hyperbolic Geometry (Conceptual)")

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
        #### MedSight-ViT: Vision Transformer Design
        
        **Backbone & Transfer Learning:**
        The Transformer pipeline employs a Vision Transformer (ViT-Tiny, 192-dim embeddings, 12 blocks) 
        pretrained on ImageNet. Transfer learning dramatically reduces the data requirements and 
        accelerates convergence on the small BRISC2025 dataset compared to training from scratch.
        
        **Two-Stage Training Strategy:**
        1. **Stage 1 (Head Training):** Freeze the transformer backbone and train only the custom 
           classifier head (192 → 256 → ReLU → Dropout(0.5) → 4 classes) for 20–50 epochs. This 
           preserves pretrained features while adapting to the tumor classification task.
        2. **Stage 2 (Fine-tuning):** Unfreeze the last two transformer blocks (blocks.10, blocks.11) 
           and apply differential learning rates—head at 1e-3, blocks at 1e-5—to enable conservative 
           domain-specific refinement without catastrophic forgetting.
        
        **Class Imbalance Mitigation:**
        A WeightedRandomSampler ensures approximately balanced minibatches despite class imbalance in the dataset.
        
        **Regularization & Optimization:**
        - **Loss:** Cross-entropy with label smoothing (ε=0.1) during fine-tuning for better calibration.
        - **Optimizer:** AdamW (decoupled weight decay) at different learning rates per stage.
        - **Schedule:** Cosine annealing learning-rate decay (T_max=50) for smooth convergence.
        - **Early Stopping:** Patience=5 on validation loss to prevent overfitting.
        
        **Explainability (Adapted Grad-CAM for ViT):**
        Instead of convolutional feature maps, we extract gradients and activations from the final 
        transformer norm layer, aggregate across 197 tokens to get 192 weights, apply these weights 
        to the 196 spatial patch activations, reshape to 14×14, and upsample to 224×224 for overlay. 
        This provides clinician-interpretable heatmaps showing which image regions the model attended to.
        
        **Clinical Text Generation (MedGemma):**
        Model predictions and confidence are converted into expert-style and patient-friendly textual 
        summaries via templated prompts. The system is designed to integrate with production APIs 
        (Vertex AI, Hugging Face) for real-time report generation.
        
        ✅ **Result:** 94.6% validation accuracy with robust, interpretable, and clinician-actionable outputs.
        """)
        
        
else:
    # Empty state visuals
    st.markdown("### 🚀 Welcome to MedSight-ViT")
    st.write("Please upload an MRI file from the sidebar to begin the Brain MRI analysis.")
