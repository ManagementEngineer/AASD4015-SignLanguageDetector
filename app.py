import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.write("""
Sign Language Classification Model
""")

# Load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Function to preprocess the image and make predictions
def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((228, 228))  # Resize the image to match the input size of the model
    img = np.array(img) / 255.0  # Normalize the image pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions using the model
    predictions = model.predict(img)
    predicted_class = tf.argmax(predictions, axis=1)[0].numpy()
    confidence = tf.reduce_max(predictions, axis=1)[0]

    return predicted_class, confidence.numpy()

def file_selector():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    return uploaded_file

uploaded_image = file_selector()

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Your Image", use_column_width=True)
    st.write("Classifying...")
    label_class, label_confidence = predict(uploaded_image)
    st.write('The image is %d with %.2f%% probability' % (label_class, label_confidence * 100))

