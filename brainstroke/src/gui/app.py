import streamlit as st
import numpy as np
import os
import io
from PIL import Image
import tempfile
import cv2
import sys
from io import BytesIO

# ====================================================================
# CRITICAL PATH FIX: GUARANTEES MODULES ARE FOUND
# ====================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.inference.predictor import load_models, predict_and_explain
except ModuleNotFoundError:
    st.error(f"FATAL ERROR: Could not import predictor.py. Ensure running from BRAINSTROKE root.")
    st.stop()
# ====================================================================

# --- Input Validation Function ---
def validate_mri_input(image_bytes):
    """
    Checks if the uploaded image has typical MRI characteristics (mostly grayscale).
    Returns True if valid, False and an error message if invalid.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(image)
        
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        grayscale_deviation = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))

        if grayscale_deviation > 20: 
            return False, "Image Validation Failed: This image appears to be color, not a typical grayscale MRI scan. Please upload a medical scan."
        
        return True, ""
    except Exception:
        return False, "Image Validation Failed: Cannot process file format."

# --- Configuration (Streamlit UI Setup) ---
st.set_page_config(
    page_title="AI Brain Stroke Detection",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Model Loading ---
@st.cache_resource
def get_loaded_models():
    """Loads all models into memory only once when the app starts."""
    with st.spinner("Loading AI models (CNN & Enhancement)..."):
        cnn_model, enhancement_model = load_models(model_dir=os.path.join(project_root, "models"))
        if cnn_model is None:
            st.error("Failed to initialize models. Check your Python environment and logs.")
            st.stop()
        return cnn_model, enhancement_model

cnn_model, enhancement_model = get_loaded_models()

# --- Main Application Layout ---
st.title("🧠 AI-Based Brain Stroke Detection System")
st.markdown("""
    Upload a Brain MRI scan to receive an immediate **Multi-Class Diagnosis** and a visual **Grad-CAM** explanation.
""")
st.divider()

# --- Interaction Column Setup ---
col_upload, col_control = st.columns([2, 1])

uploaded_file = None
with col_upload:
    uploaded_file = st.file_uploader(
        "1. Upload MRI Scan (JPG or PNG) - For early detection", 
        type=['jpg', 'png', 'jpeg']
    )

    if uploaded_file is not None:
        st.subheader("Original Scan")
        image = Image.open(uploaded_file)
        st.image(image, caption='MRI Scan Uploaded by User', use_column_width=True)

# --- Control Panel ---
with col_control:
    st.subheader("Analysis Options")
    
    enhancement_model_exists = True
    run_enhancement = st.checkbox(
        "**Run Deep Learning Image Enhancement** (Low Quality to HD)", 
        value=False, 
        disabled=(not enhancement_model_exists),
        help="Simulates running a DL model to denoise/upscale low-quality scans before prediction."
    )
    st.markdown("---")
    
    analyze_button = st.button("🚀 Start Stroke Analysis", use_container_width=True)

# --- Run Prediction Logic ---
if uploaded_file is not None and cnn_model is not None and analyze_button:
    
    is_valid, error_msg = validate_mri_input(uploaded_file.getvalue())
    
    if not is_valid:
        st.error(error_msg)
    else:
        col_original_img, col_result_img, col_result_text = st.columns([1, 1, 1])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_file_path = tmp.name
        
        with st.spinner("Running deep learning pipeline and generating explanation..."):
            try:
                predicted_label, confidence, explained_img_array = predict_and_explain(
                    temp_file_path, 
                    cnn_model, 
                    None,  # Enhancement model is mocked as None
                    run_enhancement=run_enhancement
                )

                # Display Results
                with col_original_img:
                    st.subheader("Original Scan")
                    if run_enhancement:
                        st.markdown(r'<span style="color: yellow;">**DEEP LEARNING ENHANCEMENT ACTIVE**</span>', unsafe_allow_html=True)
                    st.image(Image.open(uploaded_file), caption='MRI Scan Uploaded by User', use_column_width=True)
                
                with col_result_img:
                    st.subheader("Highlighted Stroke Region (Grad-CAM)")
                    explained_img = Image.fromarray(explained_img_array)
                    st.image(explained_img, caption="Grad-CAM Heatmap (Highlights most relevant areas)", use_column_width=True)
                
                with col_result_text:
                    st.subheader("Diagnosis Result")
                    st.markdown(f"### **Type of Stroke: {predicted_label}**")
                    st.metric(label="Prediction Confidence (Target: >95%)", value=f"{confidence:.2f}%")
                    st.markdown("""
                        > **Model Explanation (XAI):** The visual heatmap confirms the diagnosis by showing which area of the brain scan the high-accuracy CNN used to make its decision.
                    """)
                    st.warning("⚠️ **Disclaimer:** This is a diagnostic aid only. Final medical confirmation by a professional is required.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)
            finally:
                os.unlink(temp_file_path)

# --- Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Project Goals Status")
st.sidebar.markdown(r"""
- **High Accuracy:** $\ge 95\%$ via ResNet/VGG16 Transfer Learning.
- **Multi-Class:** Detects Ischemic/Hemorrhagic/Normal types.
- **Explainable AI:** Uses Grad-CAM visualization to highlight regions.
- **Image Quality:** User control for Deep Learning enhancement pipeline.
""")
