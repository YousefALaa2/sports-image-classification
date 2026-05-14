import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# ===== Load Model (local file inside Space) =====
model = tf.keras.models.load_model("sports_mobilenet.keras")

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

# ===== Prediction Function =====
def predict(image):
    img = image.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return f"{class_names[idx]} ({confidence:.2f})"

# ===== UI =====
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🏆 Sports Image Classifier"
)

demo.launch()
