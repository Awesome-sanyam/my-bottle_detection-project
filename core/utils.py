# core/utils.py
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the model once when the server starts to save memory
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml_model/bottle_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

def predict_bottle(image_path):
    # 1. Load and prepare the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # 2. Run prediction
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]
    
    # 3. Return label and confidence
    label = "DEFECTIVE" if score > 0.5 else "OK"
    confidence = score if label == "DEFECTIVE" else (1 - score)
    
    return label, round(float(confidence) * 100, 2)