import streamlit as st
import os
import sys
import tempfile

# Fix import path
sys.path.append(os.path.abspath("."))


# IMPORT YOUR MODULES
from src.config import MODEL_PATH
from src.dataset import split_dataset
from src.feature_extraction import extract_features
from src.pca_transform import apply_pca
from src.train_svm import train_model
from src.classify import classify_image

from rag.pipeline import build_rag, ask_rag


# UI CONFIG
st.set_page_config(page_title="Alzheimer Detection", layout="centered")

st.title("Alzheimer MRI Detection System")
st.write("Upload an MRI image to classify Alzheimer stage and get treatment guidance.")


# SETUP PIPELINE
with st.spinner("Setting up system..."):

    # ML MODEL
    if not os.path.exists(MODEL_PATH):
        st.write("First time setup: running ML pipeline...")

        split_dataset()
        extract_features()
        apply_pca()
        train_model()

        st.success("ML Model trained successfully")
    else:
        st.write("ML Model already available")

    # RAG SYSTEM
    if not os.path.exists("rag/vector_db"):
        st.write("Building RAG database (first time)...")
        build_rag()
        st.success("RAG system ready")
    else:
        st.write("RAG already available")

st.success("System Ready!")


# IMAGE UPLOAD
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Show image
    st.image(uploaded_file, caption="Uploaded MRI", width=400)

    
    # CLASSIFICATION
    with st.spinner("Classifying MRI..."):
        result = classify_image(temp_path)

    st.success(f"Prediction: {result}")

    
    # RAG SECTION
    if result.lower() != "non":

        st.divider()
        st.subheader("Ask about treatment")

        user_query = st.text_input("Enter your question")

        if st.button("Get Answer"):

            if user_query.strip() == "":
                st.warning("Please enter a question")
            else:
                with st.spinner("Generating answer..."):

                    response = ask_rag(result, user_query)

                    st.subheader("Answer")
                    st.write(response["answer"])

                    # sources
                    with st.expander("Sources"):
                        for s in response["sources"]:
                            st.write(s)

    else:
        st.info("You are fine. No treatment needed")