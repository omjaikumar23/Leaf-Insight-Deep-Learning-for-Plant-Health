import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained leaf classifier model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('leaf_disease_classifier.h5')  # Replace with your model path
    return model

model = load_model()

# List of class names (44 leaf classes)
class_names = ['Apple leaf with disease apple scab', 'Apple leaf with disease black rot', 'Apple leaf with disease cedar apple rust', 'Cassava leaf with disease Bacterial Blight (CBB)', 'Cassava leaf with disease Brown Streak Disease (CBSD)', 'Cassava leaf with disease Green Mottle (CGM)', 'Cassava leaf with disease Mosaic Disease (CMD)', 'Cherry (including sour) leaf with disease powdery mildew', 'Corn (maize) leaf with disease cercospora leaf spot gray leaf spot', 'Corn (maize) leaf with disease common rust', 'Corn (maize) leaf with disease northern leaf blight', 'Grape leaf with disease black rot', 'Grape leaf with disease esca (black measles)', 'Grape leaf with disease leaf blight (isariopsis leaf spot)', 'Healthy Apple leaf', 'Healthy Cassava leaf', 'Healthy Cherry (including sour) leaf', 'Healthy Corn (maize) leaf', 'Healthy Grape leaf', 'Healthy Peach leaf', 'Healthy Pepper, bell leaf', 'Healthy Potato leaf', 'Healthy Rice leaf', 'Healthy Strawberry leaf', 'Healthy Tomato leaf', 'Orange leaf with disease haunglongbing (citrus greening)', 'Peach leaf with disease bacterial spot', 'Pepper, bell leaf with disease bacterial spot', 'Potato leaf with disease early blight', 'Potato leaf with disease late blight', 'Rice leaf with disease BrownSpot', 'Rice leaf with disease Hispa', 'Rice leaf with disease LeafBlast', 'Squash leaf with disease powdery mildew', 'Strawberry leaf with disease leaf scorch', 'Tomato leaf with disease bacterial spot', 'Tomato leaf with disease early blight', 'Tomato leaf with disease late blight', 'Tomato leaf with disease leaf mold', 'Tomato leaf with disease septoria leaf spot', 'Tomato leaf with disease spider mites two-spotted spider mite', 'Tomato leaf with disease target spot', 'Tomato leaf with disease tomato mosaic virus', 'Tomato leaf with disease tomato yellow leaf curl virus']

st.title("Leaf Insight: Deep Learning For Plant Health")

option = st.radio("Select Input Type:", ('Upload Image', 'Capture Live Image'))

def preprocess_image(image):
    img = image.resize((256, 256))  # Match model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        pred_label = np.argmax(predictions, axis=1)[0]
        st.write(f"### Prediction: **{class_names[pred_label]}**")
elif option == 'Capture Live Image':
    img_file_buffer = st.camera_input("Capture a leaf image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        st.image(image, caption='Captured Leaf Image', use_container_width=True)
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        pred_label = np.argmax(predictions, axis=1)[0]
        st.write(f"### Prediction: **{class_names[pred_label]}**")
