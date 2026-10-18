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

# Cache loading model agar cepat saat berganti opsi
@st.cache_resource
def load_model(path_str):
    return tf.keras.models.load_model(path_str)

# Deteksi model yang tersedia di folder model/
model_options = {}
opt_path = Path("model/cnn_cifar10_optimized.keras")
base_path = Path("model/cnn_cifar10.keras")

if opt_path.exists():
    model_options["Model Sesudah Optimasi"] = str(opt_path)
if base_path.exists():
    model_options["Model Sebelum Optimasi (Baseline)"] = str(base_path)

# Tambahan jika ada file .keras lain
for f in Path("model").glob("*.keras"):
    if f not in [opt_path, base_path]:
        model_options[f"Model ({f.name})"] = str(f)

if not model_options:
    st.error("Tidak ditemukan file model (.keras) di folder model/.")
    st.stop()

# Dropdown pemilihan model
col_m1, col_m2 = st.columns([2, 1])
with col_m1:
    selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
with col_m2:
    st.write("")
    st.write("")
    compare_both = st.checkbox("Bandingkan kedua model", value=False) if len(model_options) >= 2 else False

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

# Fungsi helper prediksi
def predict_image(model_obj, img_input):
    img_rgb = img_input.convert("RGB").resize((32, 32))
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)
    preds = model_obj.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return CLASSES[top_idx], preds[top_idx] * 100, preds

# Tampilkan gambar dan hasil prediksi
if image is not None:
    st.image(image, caption="Gambar yang Diuji", width=250)

    if compare_both and len(model_options) >= 2:
        st.markdown("---")
        st.subheader("Perbandingan Hasil Model")
        
        # Load kedua model
        m_base = load_model(model_options["Model Sebelum Optimasi (Baseline)"])
        m_opt = load_model(model_options["Model Sesudah Optimasi"])

        pred_base_label, conf_base, probs_base = predict_image(m_base, image)
        pred_opt_label, conf_opt, probs_opt = predict_image(m_opt, image)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📌 Sebelum Optimasi")
            st.write(f"**Prediksi:** {pred_base_label}")
            st.write(f"**Keyakinan:** {conf_base:.2f}%")
            df_base = pd.DataFrame({"Probabilitas": probs_base}, index=CLASSES)
            st.bar_chart(df_base)

        with c2:
            st.markdown("### 🚀 Sesudah Optimasi")
            st.write(f"**Prediksi:** {pred_opt_label}")
            st.write(f"**Keyakinan:** {conf_opt:.2f}%")
            df_opt = pd.DataFrame({"Probabilitas": probs_opt}, index=CLASSES)
            st.bar_chart(df_opt)
    else:
        # Prediksi dengan model yang dipilih pada dropdown
        current_model = load_model(model_options[selected_model_name])
        label, confidence, probs = predict_image(current_model, image)

        st.markdown("---")
        st.subheader(f"Hasil Prediksi ({selected_model_name})")
        st.write(f"**Prediksi:** {label}")
        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")

        st.write("Distribusi Probabilitas Tiap Kelas:")
        df = pd.DataFrame({"Probabilitas": probs}, index=CLASSES)
        st.bar_chart(df)
