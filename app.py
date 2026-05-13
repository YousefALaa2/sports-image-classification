import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

# ===== Model URL (Google Drive Direct Link) =====
MODEL_URL = "https://drive.google.com/uc?export=download&id=1hFaD_R2SgKxn9fHW-3jOLVv8KHenkkn7"

MODEL_PATH = "sports_mobilenet.keras"

# ===== Download model if not exists =====
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... please wait ⏳")
    response = requests.get(MODEL_URL)
    open(MODEL_PATH, "wb").write(response.content)
    st.success("Model downloaded successfully!")

# ===== Load Model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Class Mapping (صح 100%) =====
class_names = {
    0: 'Badminton',
    1: 'Cricket',
    2: 'Karate',
    3: 'Soccer',
    4: 'Swimming',
    5: 'Tennis',
    6: 'Wrestling'
}

# ===== UI =====
st.title("🏆 Sports Image Classification")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index]

    st.markdown("## 🔍 Result")
    st.success(f"{class_names[pred_index]}")
    st.info(f"Confidence: {confidence:.2f}")

    st.markdown("### Top 3")

    top3 = np.argsort(prediction[0])[::-1][:3]
    for i in top3:
        st.write(f"{class_names[i]} → {prediction[0][i]:.2f}")