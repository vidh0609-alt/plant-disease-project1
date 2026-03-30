import streamlit as st
import requests

st.title("🌱 Plant Disease Detector")

file = st.file_uploader("Upload a leaf image")

if file:
    st.image(file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": file}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
        else:
            st.error("Error in prediction")