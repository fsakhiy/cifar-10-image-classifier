import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="CIFAR-10 CNN Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .pixelated-img {
        image-rendering: pixelated;
        image-rendering: crisp-edges;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Constants & Class Definitions
# ==============================================================================
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_EMOJIS = {
    'airplane': '✈️ Airplane',
    'automobile': '🚗 Automobile',
    'bird': '🐦 Bird',
    'cat': '🐱 Cat',
    'deer': '🦌 Deer',
    'dog': '🐶 Dog',
    'frog': '🐸 Frog',
    'horse': '🐴 Horse',
    'ship': '🚢 Ship',
    'truck': '🚚 Truck'
}

DEFAULT_MODEL_PATH = Path("model/cnn_cifar10.keras")
DOWNLOADS_MODEL_PATH = Path.home() / "Downloads" / "cnn_cifar10.keras"
SAMPLES_DIR = Path("samples")

# ==============================================================================
# Model Loading Helper (Cached)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path_str: str):
    """Loads and caches the Keras CNN model."""
    try:
        model = tf.keras.models.load_model(model_path_str)
        return model, None
    except Exception as e:
        return None, str(e)


def find_model_file():
    """Detects available model file from default paths or user directory."""
    if DEFAULT_MODEL_PATH.exists():
        return str(DEFAULT_MODEL_PATH), "Found in project 'model/' folder"
    elif DOWNLOADS_MODEL_PATH.exists():
        return str(DOWNLOADS_MODEL_PATH), "Found in Downloads folder"
    
    # Check if any .keras file is in model/
    if Path("model").exists():
        keras_files = list(Path("model").glob("*.keras"))
        if keras_files:
            return str(keras_files[0]), f"Found {keras_files[0].name} in 'model/' folder"
            
    return None, "Model file not found"


def preprocess_image(image: Image.Image, target_size=(32, 32)):
    """Preprocess PIL image for CIFAR-10 model input."""
    # Ensure 3-channel RGB (removes alpha or grayscale dimensions)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to model input shape (32x32)
    resized_img = image.resize(target_size, Image.Resampling.BILINEAR)
    
    # Scale pixel values [0, 255] -> [0.0, 1.0]
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    
    # Add batch dimension -> (1, 32, 32, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    
    return batch_array, resized_img


# ==============================================================================
# Sidebar: Model Status & Configuration
# ==============================================================================
st.sidebar.title("⚙️ Model Configuration")

# Model source selection
discovered_path, discovery_status = find_model_file()

uploaded_model = st.sidebar.file_uploader(
    "Upload custom `.keras` file (optional)",
    type=["keras", "h5"],
    help="Upload your trained model directly from your computer"
)

active_model_path = None
active_model_source = ""

if uploaded_model is not None:
    # Save uploaded model to temp location in model/
    os.makedirs("model", exist_ok=True)
    temp_path = Path("model") / uploaded_model.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    active_model_path = str(temp_path)
    active_model_source = f"Uploaded via Web UI ({uploaded_model.name})"
elif discovered_path:
    active_model_path = discovered_path
    active_model_source = discovery_status

# Model loading status indicator
model = None
if active_model_path:
    with st.sidebar:
        with st.spinner("Loading model..."):
            model, load_err = load_keras_model(active_model_path)

    if model:
        st.sidebar.success(f"**Model Loaded**\n\n`{Path(active_model_path).name}`\n\n*{active_model_source}*")
        with st.sidebar.expander("🔍 Architecture Details"):
            st.write(f"**Input shape:** `{model.input_shape}`")
            st.write(f"**Output shape:** `{model.output_shape}`")
            st.write(f"**Total parameters:** `{model.count_params():,}`")
    else:
        st.sidebar.error(f"❌ Failed to load model: {load_err}")
else:
    st.sidebar.warning(
        "⚠️ **No `.keras` model found.**\n\n"
        "Place your `cnn_cifar10.keras` in the `model/` folder or upload it above."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("About CIFAR-10")
st.sidebar.caption(
    "CIFAR-10 is a benchmark dataset of 60,000 32x32 color images across 10 classes: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck."
)

# ==============================================================================
# Main UI: Image Selection & Prediction
# ==============================================================================
st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Test your Convolutional Neural Network on images outside the training dataset. "
    "Upload your own photo or choose from sample test images below."
)

# Input method toggle
input_mode = st.radio(
    "Select Input Method:",
    ["Upload an Image", "Choose Sample Image"],
    horizontal=True
)

selected_image = None
image_caption = ""

if input_mode == "Upload an Image":
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, WEBP)...",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
            image_caption = f"Uploaded: {uploaded_file.name} ({selected_image.width}x{selected_image.height})"
        except Exception as e:
            st.error(f"Error opening image: {e}")
else:
    # Sample images selector
    sample_files = list(SAMPLES_DIR.glob("*.*")) if SAMPLES_DIR.exists() else []
    sample_options = [f.name for f in sample_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    
    if sample_options:
        chosen_sample = st.selectbox("Select a sample image:", sample_options)
        if chosen_sample:
            sample_path = SAMPLES_DIR / chosen_sample
            selected_image = Image.open(sample_path)
            image_caption = f"Sample: {chosen_sample} ({selected_image.width}x{selected_image.height})"
    else:
        st.info("No sample images found in `samples/` directory yet. Switch to 'Upload an Image' above.")

# ==============================================================================
# Inference and Visualization
# ==============================================================================
if selected_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Image Preview")
        st.image(selected_image, caption=image_caption, use_container_width=True)
        
        # Preprocess image
        input_tensor, resized_preview = preprocess_image(selected_image, target_size=(32, 32))
        
        with st.expander("🔬 View what the model actually sees (32x32 pixels)"):
            st.image(
                resized_preview,
                caption="Resized to 32x32 (CIFAR-10 resolution)",
                width=160
            )
            st.caption(
                "Note: Real-world photos get compressed to 32x32 pixels before entering the CNN. "
                "Notice how fine details appear at this resolution!"
            )

    with col2:
        st.subheader("🎯 Classification Results")
        
        if model is None:
            st.warning("Please provide a `.keras` model in the sidebar to run predictions.")
        else:
            with st.spinner("Analyzing image with CNN..."):
                predictions = model.predict(input_tensor, verbose=0)[0]
                
            top_idx = int(np.argmax(predictions))
            top_class = CLASS_NAMES[top_idx]
            top_emoji_class = CLASS_EMOJIS.get(top_class, top_class)
            top_confidence = float(predictions[top_idx])

            # Top Prediction Card
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1f77b4 0%, #115582 100%);
                            padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Top Prediction</div>
                    <div style="font-size: 32px; font-weight: bold; margin: 8px 0;">{top_emoji_class}</div>
                    <div style="font-size: 20px; font-weight: 500;">Confidence: {top_confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probabilities breakdown table / chart
            st.markdown("#### Class Probabilities Breakdown")
            
            df_probs = pd.DataFrame({
                "Class": [CLASS_EMOJIS.get(c, c) for c in CLASS_NAMES],
                "Probability": predictions
            }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

            # Format for percentage display
            df_probs["Confidence"] = df_probs["Probability"].apply(lambda p: f"{p * 100:.2f}%")
            
            # Interactive Bar Chart
            st.bar_chart(
                df_probs.set_index("Class")["Probability"],
                color="#1f77b4",
                height=300
            )
            
            # Detailed breakdown table
            with st.expander("📋 View Exact Probabilities Table"):
                st.dataframe(
                    df_probs[["Class", "Confidence"]],
                    use_container_width=True,
                    hide_index=True
                )
else:
    st.info("👆 Please upload an image or choose a sample image to classify.")
