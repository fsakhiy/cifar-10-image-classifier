from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10")

st.title("Klasifikasi Gambar CIFAR-10")
st.write("Unggah gambar untuk menguji model CNN.")

# Label kelas dalam bahasa Indonesia
CLASSES = [
    'Pesawat', 'Mobil', 'Burung', 'Kucing', 'Rusa',
    'Anjing', 'Katak', 'Kuda', 'Kapal', 'Truk'
]

# Load model
@st.cache_resource
def get_model():
    model_paths = [
        Path("model/cnn_cifar10.keras"),
        Path("cnn_cifar10.keras"),
        Path.home() / "Downloads" / "cnn_cifar10.keras"
    ]
    for p in model_paths:
        if p.exists():
            return tf.keras.models.load_model(str(p)), p.name
    return None, None

model, model_name = get_model()

if model is None:
    st.error("Model cnn_cifar10.keras tidak ditemukan. Silakan letakkan file model di dalam folder model/.")
    st.stop()

# Input gambar
option = st.radio("Metode input gambar:", ("Unggah gambar", "Pilih sampel gambar"), horizontal=True)

image = None
if option == "Unggah gambar":
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    sample_dir = Path("samples")
    sample_files = list(sample_dir.glob("*.*")) if sample_dir.exists() else []
    if sample_files:
        selected_sample = st.selectbox("Pilih sampel:", [f.name for f in sample_files])
        image = Image.open(sample_dir / selected_sample)
    else:
        st.write("Tidak ada gambar sampel di folder samples/.")

# Prediksi
if image is not None:
    st.image(image, caption="Gambar yang Diuji", width=300)

    # Preprocessing
    img_rgb = image.convert("RGB").resize((32, 32))
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)

    # Prediksi model
    preds = model.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = CLASSES[top_idx]
    confidence = preds[top_idx] * 100

    st.subheader(f"Hasil Prediksi: {label}")
    st.write(f"Tingkat Keyakinan: {confidence:.2f}%")

    # Grafik probabilitas
    st.write("Distribusi Probabilitas Tiap Kelas:")
    df = pd.DataFrame({"Probabilitas": preds}, index=CLASSES)
    st.bar_chart(df)
