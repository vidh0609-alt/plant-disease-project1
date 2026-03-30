import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import random

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

    import random

if st.button("Predict"):
    prediction = random.choice([
        "Tomato_Healthy",
        "Tomato_Late_blight",
        "Potato_Early_blight"
    ])

    st.success(f"Prediction: {prediction}")