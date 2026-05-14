import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

# ===== Model Download from Google Drive =====
MODEL_URL = "https://drive.google.com/uc?export=download&id=1hFaD_R2SgKxn9fHW-3jOLVv8KHenkkn7"
MODEL_PATH = "sports_mobilenet.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# ===== Load Model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Class Mapping =====
class_names = {
    0: 'Badminton',
    1: 'Cricket',
    2: 'Karate',
    3: 'Soccer',
    4: 'Swimming',
    5: 'Tennis',
    6: 'Wrestling'
}

class_icons = {
    'Badminton': '🏸',
    'Cricket':   '🏏',
    'Karate':    '🥋',
    'Soccer':    '⚽',
    'Swimming':  '🏊',
    'Tennis':    '🎾',
    'Wrestling': '🤼'
}

# ===== Prediction Function =====
def predict(image):
    if image is None:
        return "<p style='text-align:center; color:#888; padding: 2rem;'>Upload an image to get predictions.</p>"

    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    top3_idx = np.argsort(prediction[0])[::-1][:3]

    top_label = class_names[idx]
    top_icon  = class_icons.get(top_label, '🏆')
    conf_pct  = round(confidence * 100, 1)

    # Confidence colour
    if confidence >= 0.85:
        conf_color = "#22c55e"
        conf_label = "High confidence"
    elif confidence >= 0.60:
        conf_color = "#f59e0b"
        conf_label = "Moderate confidence"
    else:
        conf_color = "#ef4444"
        conf_label = "Low confidence"

    # Build top-3 bars
    bars_html = ""
    for rank, i in enumerate(top3_idx):
        name  = class_names[i]
        icon  = class_icons.get(name, '🏅')
        pct   = round(float(prediction[0][i]) * 100, 1)
        width = max(pct, 3)
        name_color  = "#ffffff" if rank == 0 else "#b0b8c8"
        pct_color   = "#818cf8" if rank == 0 else "#6b7280"
        bar_bg      = "linear-gradient(90deg,#6366f1,#818cf8)" if rank == 0 else "#374151"
        bars_html += f"""
        <div style="margin-bottom:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:15px; font-weight:{'700' if rank==0 else '400'}; color:{name_color};">
              {icon} {name}
            </span>
            <span style="font-size:14px; font-weight:600; color:{pct_color};">{pct}%</span>
          </div>
          <div style="background:#1f2937; border-radius:999px; height:10px; overflow:hidden;">
            <div style="height:10px; border-radius:999px; width:{width}%;
                        background:{bar_bg};"></div>
          </div>
        </div>"""

    html = f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif; padding:8px 4px;">

      <!-- Hero Result -->
      <div style="background:linear-gradient(135deg,#0f0f1a 0%,#1a1040 60%,#2d1f6e 100%);
                  border:1px solid #3730a3; border-radius:20px; padding:32px 28px;
                  text-align:center; margin-bottom:20px;">
        <div style="font-size:64px; margin-bottom:10px;">
          {top_icon}
        </div>
        <div style="font-size:11px; letter-spacing:3px; text-transform:uppercase;
                    color:#818cf8; font-weight:600; margin-bottom:8px;">Sport Detected</div>
        <div style="font-size:38px; font-weight:800; color:#ffffff; margin-bottom:18px;
                    letter-spacing:-0.5px;">
          {top_label}
        </div>
        <div style="display:inline-flex; align-items:center; gap:8px;
                    background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15);
                    border-radius:999px; padding:8px 20px;">
          <div style="width:10px; height:10px; border-radius:50%; background:{conf_color};"></div>
          <span style="color:#e2e8f0; font-size:14px; font-weight:600;">{conf_pct}% — {conf_label}</span>
        </div>
      </div>

      <!-- Top 3 Breakdown -->
      <div style="background:#111827; border:1px solid #1f2937; border-radius:16px; padding:24px 22px;">
        <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
                    color:#6b7280; font-weight:600; margin-bottom:18px;">Top Predictions</div>
        {bars_html}
      </div>

    </div>
    """
    return html


# ===== Custom CSS =====
custom_css = """
/* Dark background */
body, .gradio-container, .gradio-container > div {
    background: #0a0a0f !important;
    min-height: 100vh;
}

/* All text white by default */
.gradio-container, .gradio-container * {
    color: #e2e8f0;
}

/* Labels */
label span, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Input panels dark */
.block, .form, .panel {
    background: #111827 !important;
    border-color: #1f2937 !important;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #3730a3 !important;
    border-radius: 18px !important;
    background: #0d0d1a !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.upload-zone:hover {
    border-color: #6366f1 !important;
    background: #111827 !important;
}
.upload-zone .wrap {
    color: #6b7280 !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
button.primary:hover {
    opacity: 0.9;
}

/* Remove white output panel bg */
.output-html {
    background: transparent !important;
    border: none !important;
}

footer { display: none !important; }
"""

# ===== Build UI =====
with gr.Blocks(css=custom_css, theme=gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
)) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 28px 0 12px; font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="font-size:48px; margin-bottom:12px;">🏆</div>
      <h1 style="font-size:36px; font-weight:800; color:#ffffff; margin:0 0 10px;
                 letter-spacing:-1px;">
        Sports Classifier <span style="color:#818cf8;">AI</span>
      </h1>
      <p style="font-size:15px; color:#94a3b8; margin:0 auto; max-width:480px;">
        Upload any sports image and our AI model will identify the sport with confidence scores.
      </p>
      <div style="display:flex; justify-content:center; gap:8px; margin-top:18px; flex-wrap:wrap;">
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🏸 Badminton</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🏏 Cricket</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🥋 Karate</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">⚽ Soccer</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🏊 Swimming</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🎾 Tennis</span>
        <span style="background:#1e1b4b; color:#a5b4fc; font-size:12px; font-weight:600;
                     padding:5px 14px; border-radius:999px; border:1px solid #3730a3;">🤼 Wrestling</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Sports Image",
                elem_classes=["upload-zone"],
                height=320,
            )
            classify_btn = gr.Button(
                "✨  Classify Sport",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            result_output = gr.HTML(
                value="""
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                            height:320px; font-family:'Segoe UI',system-ui,sans-serif;">
                  <div style="font-size:52px; margin-bottom:12px; opacity:0.5;">📸</div>
                  <p style="font-size:15px; font-weight:500; margin:0; color:#94a3b8;">Upload an image to see results</p>
                  <p style="font-size:13px; margin:6px 0 0; color:#4b5563;">
                    Supports JPG, PNG, WEBP
                  </p>
                </div>
                """,
                label="Prediction Results",
            )

    classify_btn.click(fn=predict, inputs=image_input, outputs=result_output)
    image_input.change(fn=predict, inputs=image_input, outputs=result_output)

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 8px; font-family:'Segoe UI',system-ui,sans-serif;">
      <p style="font-size:12px; color:#4b5563; margin:0;">
        Powered by MobileNet · 7 Sport Categories · Deep Learning
      </p>
    </div>
    """)

demo.launch()
