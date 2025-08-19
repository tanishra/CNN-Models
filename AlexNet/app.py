import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/alexnet.keras")
    return model

model = load_model()

# App title
st.title("CIFAR-10 Image Classifier (AlexNet)")

# Upload image
uploaded_file = st.file_uploader("Choose a CIFAR-10 image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((227, 227))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 227, 227, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown(f"### Prediction: **{predicted_class}**")
