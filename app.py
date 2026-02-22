import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model_architecture import get_medsight_hex_model
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="MedSight-Hex", layout="wide", page_icon="🧠")
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Access the secret securely
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    # Load Vision Model
    vision_model = get_medsight_hex_model()
    # Loading the weights you saved from Kaggle
    state_dict = torch.load("HexFormer_BrainTumor_Final_96_on_val.pth", map_location='cpu')
    vision_model.load_state_dict(state_dict)
    vision_model.eval()
    
    # Load MedGemma (Using 2b for Streamlit Cloud memory limits)
    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return vision_model, tokenizer, llm_model

# Initialize models
try:
    vision_engine, m_tokenizer, medgemma = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure you have accepted the Gemma license on HuggingFace.")

# --- 3. TRANSFORMS ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. USER INTERFACE ---
st.title("🧠 MedSight-Hex: Advanced Brain Tumor Diagnostic System")
st.markdown("---")

uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    # Process Image
    img = Image.open(uploaded_file).convert('RGB')
    input_tensor = test_transform(img).unsqueeze(0)
    
    with col1:
        st.subheader("🖼️ Vision Analysis")
        st.image(img, caption="Uploaded MRI", use_column_width=True)
        
        # Run Vision Prediction
        with torch.no_grad():
            output = vision_engine(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(prob, 1)
            
        label = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item() * 100
        
        st.success(f"Prediction: {label.upper()} ({confidence:.2f}%)")

    with col2:
        st.subheader("📜 Patient-Friendly Report")
        with st.spinner("Generating clinical summary..."):
            # MedGemma Prompt for Laypeople
            prompt = f"""
            Explain a brain MRI result to a patient. 
            Result: {label} with {confidence:.2f}% certainty.
            Explain what this is simply, what the AI saw, and what they should do next.
            Do not use medical jargon.
            """
            
            inputs = m_tokenizer(prompt, return_tensors="pt")
            outputs = medgemma.generate(**inputs, max_new_tokens=250)
            report = m_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.write(report.replace(prompt, "").strip())

st.sidebar.markdown("---")
st.sidebar.write("Developed by: Karan Sanjay Rathod")
st.sidebar.write("Architecture: HexFormer-Hybrid")
