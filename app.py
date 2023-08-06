
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import cv2

# Load the pre-trained VGG model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained=True)
vgg = vgg.to(device)
vgg.eval()

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image.unsqueeze(0)

# Perform inference using the VGG model
def predict(image):
    with torch.no_grad():
        image = image.to(device)
        output = vgg(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities

# Streamlit app
def main():
    st.title("VGG Model Image Classification")
    st.write("Use the camera to capture an image and let the VGG model predict its class.")

    video_capture = cv2.VideoCapture(0)

    if st.button("Capture"):
        if video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                # Convert the OpenCV frame to PIL format
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption="Captured Image.", use_column_width=True)

                # Perform prediction
                preprocessed_image = preprocess_image(image)
                probabilities = predict(preprocessed_image)

                # Get class labels (e.g., from ImageNet)
                # Replace the labels below with the appropriate class labels for your model
                class_labels = ["class1", "class2", "class3", ...]

                # Get the top predicted class
                top_class_idx = torch.argmax(probabilities).item()
                top_class_prob = probabilities[top_class_idx].item()

                st.write(f"Prediction: {class_labels[top_class_idx]}")
                st.write(f"Probability: {top_class_prob:.2f}")

    # Release the camera when the app is closed
    st.experimental_rerun()

if __name__ == "__main__":
    main()