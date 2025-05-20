import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError

# App title
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-Ray")

# Load the model
model = load_model("pneumonia_densenet_model.h5")

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        result = "🫁 Pneumonia" if prediction > 0.5 else "✅ Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Prediction: **{result}**")
        st.markdown(f"Confidence: `{confidence:.2f}`")

    except UnidentifiedImageError:
        st.error("⚠️ Unable to process the image. Please upload a valid image file.")
else:
    st.info("👆 Upload a chest X-ray image to get started.")
