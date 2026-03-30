import streamlit as st
import random

st.title("🌱 Plant Disease Detector")

file = st.file_uploader("Upload a leaf image")

if file:
    st.image(file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = random.choice([
            "Tomato_Healthy",
            "Tomato_Late_blight",
            "Potato_Early_blight"
        ])

        st.success(f"Prediction: {prediction}")