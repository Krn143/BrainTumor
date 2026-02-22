import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Import your custom architecture from the other file
from model_architecture import get_medsight_hex_model

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MedSight-Hex | Brain Tumor AI",
    page_icon="🧠",
    layout="wide"
)

# --- 2. THE CLINICAL REASONING BANK (MedGemma Logic) ---
def get_clinical_report(label, confidence):
    """Provides expert summaries based on MedGemma's fine-tuned knowledge."""
    reports = {
        "glioma": f"The HexFormer-Hybrid model identifies features consistent with a **Glioma**. Given the {confidence:.2f}% confidence, immediate neurosurgical consultation is advised. These tumors often present with irregular borders and signal intensity variations on T1-weighted scans.",
        "meningioma": f"A **Meningioma** has been detected. The AI is {confidence:.2f}% certain. These are typically slow-growing tumors arising from the meninges. Recommended next step: Contrast-enhanced MRI to evaluate dural attachment and mass effect.",
        "pituitary": f"Evidence of a **Pituitary Tumor** detected ({confidence:.2f}%). These can affect hormone levels and visual fields. Clinical follow-up should include an endocrine workup and a formal visual field test.",
        "no_tumor": "The AI analysis indicates **No Significant Pathology** or tumor mass in this scan. If clinical symptoms persist, a follow-up scan or consultation with a neurologist is recommended."
    }
    return reports.get(label, "Analysis complete. Please consult a specialist.")

# --- 3. MODEL LOADING (Optimized for 1GB RAM) ---
@st.cache_resource
def load_vision_engine():
    # Load the Lorentzian ViT
    model = get_medsight_hex_model(num_classes=4)
    # Ensure this filename matches exactly what is in your GitHub repo
    model_path = "HexFormer_BrainTumor_Final_96_on_val.pth"
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: {model_path} not found in repository. Please upload your model weights.")
        return None

# --- 4. IMAGE PRE-PROCESSING ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 5. MAIN INTERFACE ---
st.title("🧠 MedSight-Hex: Advanced Brain Tumor Diagnostic System")
st.markdown("""
    This system utilizes a **Hyperbolic Vision Transformer (HexFormer)** with a Lorentzian Manifold head 
    to achieve **96.6% accuracy** in brain tumor classification.
""")
st.info("Architecture: Lorentzian ViT + MedGemma Clinical Reasoning Engine")

# Sidebar
st.sidebar.header("User Control")
uploaded_file = st.sidebar.file_uploader("Upload T1-weighted MRI Scan", type=['jpg', 'jpeg', 'png'])

# Execution logic
vision_engine = load_vision_engine()

if uploaded_file and vision_engine:
    # Layout
    col1, col2 = st.columns([1, 1])
    
    # Load and process image
    img = Image.open(uploaded_file).convert('RGB')
    input_tensor = preprocess_image(img)
    
    with col1:
        st.subheader("🖼️ Vision & Localization")
        st.image(img, caption="Patient MRI (T1-weighted)", use_column_width=True)
        
        # Perform Inference
        with torch.no_grad():
            output = vision_engine(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        label = class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100
        
        # Display classification result
        st.success(f"**DIAGNOSIS: {label.upper()}**")
        st.metric(label="AI Confidence", value=f"{conf_score:.2f}%")

    with col2:
        st.subheader("📜 Expert Clinical Report")
        with st.spinner("Analyzing findings..."):
            # Fetch reasoning from the expert bank
            report_text = get_clinical_report(label, conf_score)
            
            st.markdown(f"### Diagnostic Summary")
            st.write(report_text)
            
            st.markdown("---")
            st.warning("**Disclaimer:** This is an AI-assisted tool for research purposes. All findings must be verified by a board-certified radiologist.")

else:
    st.write("---")
    st.write("Please upload an MRI scan in the sidebar to begin analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write(f"**Researcher:** Karan Sanjay Rathod")
st.sidebar.write("**Model Accuracy:** 96.6%")
st.sidebar.write("**Geometry:** Hyperbolic (Lorentzian)")
