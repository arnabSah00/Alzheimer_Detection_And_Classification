import streamlit as st
import os
import sys
import tempfile

# import path
sys.path.append(os.path.abspath("."))

from src.config import MODEL_PATH
from src.dataset import split_dataset
from src.feature_extraction import extract_features
from src.pca_transform import apply_pca
from src.train_svm import train_model

# UI config
st.set_page_config(page_title="Alzheimer Detection", layout="centered")

st.title("Alzheimer MRI Detection System")
st.write("Upload an MRI image to classify Alzheimer stage.")

# SETUP PIPELINE
with st.spinner(" Setting up system..."):

    if not os.path.exists(MODEL_PATH):
        st.write("First time setup: running full pipeline...")

        split_dataset()
        extract_features()
        apply_pca()
        train_model()

        st.success("Model trained successfully")
    else:
        st.write("Model already available")

# Import func to classify
from src.classify import classify_image

st.success("System Ready!")

# Upload Image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Show image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Predict
    with st.spinner("Classifying..."):
        result = classify_image(temp_path)

    st.success(f"Prediction: {result}")