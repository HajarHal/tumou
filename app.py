import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model_path = 'braintumor.h5'
model = tf.keras.models.load_model(model_path)

st.markdown("<h1 style='text-align: center; color: #256c70;'>Brain Tumor Classification</h1>", unsafe_allow_html=True)

st.sidebar.image('H (3).png', width=250, use_column_width=True)
st.sidebar.info('This app classifies brain tumor images into four categories.')
st.sidebar.markdown("**Contact:** [hajarhalmaoui.1@gmail.com](mailto:hajarhalmaoui.1@gmail.com)")
st.sidebar.markdown("[Learn More](https://github.com/HajarHal/brain_tumour)")


st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #fbf6f1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def predict(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image.reshape(1, 150, 150, 3)
    prediction = model.predict(image)
    return prediction

# Main app code
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prediction = predict(image)
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    predicted_label = labels[np.argmax(prediction)]

    if predicted_label == 'no_tumor':
        st.write('**Prediction:** ', f'<span style="color:green; font-weight:bold">Normal</span>', unsafe_allow_html=True)
    elif predicted_label == 'meningioma_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Meningioma Tumor</span>', unsafe_allow_html=True)
    elif predicted_label == 'glioma_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Glioma Tumor</span>', unsafe_allow_html=True)
    elif predicted_label == 'pituitary_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Pituitary Tumor</span>', unsafe_allow_html=True)
