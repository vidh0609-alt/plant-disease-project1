import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.title("🌱 Plant Disease Detector")

# Load model
model = tf.keras.models.load_model("model/plant_model.h5")

# Load labels
with open("model/labels.json") as f:
    labels = json.load(f)

labels = {v:k for k,v in labels.items()}

def preprocess(image):
    image = image.resize((224,224))
    img = np.array(image)/255.0
    return np.expand_dims(img, axis=0)

file = st.file_uploader("Upload a leaf image")

if file:
    st.image(file)

    if st.button("Predict"):
        image = Image.open(file)
        img = preprocess(image)

        pred = model.predict(img)
        class_id = np.argmax(pred)

        st.success(f"Prediction: {labels[class_id]}")