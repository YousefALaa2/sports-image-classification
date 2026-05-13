import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===== Load Model =====
model = tf.keras.models.load_model("sports_mobilenet.keras")

# ===== Correct Class Mapping (مهم جدًا) =====
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
st.set_page_config(page_title="Sports Classifier", layout="centered")

st.title("🏆 Sports Image Classification")
st.write("Upload an image and the model will predict the sport")

# ===== Upload =====
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===== Preprocessing =====
    img = image.resize((224, 224))
    img = np.array(img)

    # Handle RGBA
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ===== Prediction =====
    prediction = model.predict(img)
    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index]

    # ===== Result =====
    st.markdown("## 🔍 Prediction Result")
    st.success(f"🏷️ {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")

    # ===== Top 3 Predictions =====
    st.markdown("### 📊 Top 3 Predictions")

    top_3 = np.argsort(prediction[0])[::-1][:3]

    for i in top_3:
        st.write(f"{class_names[i]} → {prediction[0][i]:.2f}")