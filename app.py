import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.write("""
# Simple VGG16 Model Classifier
""")

# Load the pre-trained model
def load_model():
    # model = tf.keras.models.load_model('best_model.h5')
    model = tf.keras.models.load_model('./results_vgg/vgg_digits_best_model.h5')
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

def file_selector(folders):
    filenames = []
    for folder_path in folders:
        folder_files = os.listdir(folder_path)
        filenames.extend([os.path.join(folder_path, file) for file in folder_files])
    
    selected_filename = st.selectbox('Select your Image', filenames)
    return selected_filename

folders = ['./ASL Digits/test/0', './ASL Digits/test/1', './ASL Digits/test/2', './ASL Digits/test/3', './ASL Digits/test/4', './ASL Digits/test/5', './ASL Digits/test/6','./ASL Digits/test/7','./ASL Digits/test/8','./ASL Digits/test/9',]  # Add your folder paths here
filename = file_selector(folders)
st.write('You selected `%s`' % filename)

if filename:
    img = Image.open(filename)
    st.image(img, caption="Your Image", use_column_width=True)
    st.write("Classifying...")
    label_class, label_confidence = predict(filename)
    st.write('The image is %d with %.2f%% probability' % (label_class, label_confidence * 100))