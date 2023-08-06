import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Custom CSS to style the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #feca57;
        color: #ffffff;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom header
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhWi5-x1EvY3iq6DTWLttZ7-Ym0fLpXEuSR5Ctu_jPxHyI8PL7axqyt6iCajI7QuKgVqX15gaiLP_RJr-A6GNINxsRv3noZkyDBaT2pZsoW2BNdxppejifHQRZwdCExYlZLIzIPYyJ_PDJ9z_-HqGse1T_gW8f4sFW4NF5T2mI_UOjT7HCrMXMr7Vor/w1200-h630-p-k-no-nu/GDS_ASL_FingerspellingCompetition_Banners_Metacard.png", use_column_width=True)
st.title(":wave: Sign Language Classification Model :muscle:")

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
    st.markdown("### :rocket: Classifying Image...")
    label_class, label_confidence = predict(uploaded_image)

    # Emojis representing the predicted class
    emojis = [":zero:", ":one:", ":two:", ":three:", ":four:", ":five:", ":six:", ":seven:", ":eight:", ":nine:"]
    predicted_emoji = emojis[label_class]

    # Display the prediction with a colorful banner
    st.markdown(
        """
        <div style="background-color: #feca57; padding: 10px; border-radius: 5px;">
            <h3 style="color: #ffffff;">Prediction Result:</h3>
            <p style="font-size: 24px; color: #ffffff;">The image is {} with {:.2f}% probability</p>
        </div>
        """.format(predicted_emoji, label_confidence * 100),
        unsafe_allow_html=True
    )

