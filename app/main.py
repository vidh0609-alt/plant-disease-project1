from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = FastAPI()

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

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    img = preprocess(image)

    pred = model.predict(img)
    class_id = np.argmax(pred)

    return {"prediction": labels[class_id]}