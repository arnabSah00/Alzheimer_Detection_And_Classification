# Alzheimer Detection using MRI Images

This project is a complete end-to-end AI system that detects Alzheimer’s disease stages from MRI images and provides treatment guidance using RAG (Retrieval-Augmented Generation).

## Features

Feature extraction using pretrained CNN (**VGG16**)
PCA for dimensionality reduction
SVM model for classification
AG-based treatment recommendation system
Streamlit web app with interactive Q&A


### 2. Create virtual environment

```bash
conda create -n p_env python=3.10
conda activate p_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
python main.py
```

### What it does:

Splits dataset (train/val/test)
Extracts features using VGG16
Applies PCA
Trains SVM model
Saves model files


## Run Web App

```bash
streamlit run app.py
```

## Features in App

Upload MRI image
Get Alzheimer stage prediction
Ask treatment-related questions
Get **AI-generated answers from medical documents (RAG)**

---

# RAG Workflow (NEW)

The system uses Retrieval-Augmented Generation (RAG) to provide **accurate, document-based treatment guidance**.

### Flow:


Medical PDFs
   ->
Document Loader (PyPDF)
   ->
Chunking (Recursive Splitter)
   ->
Embeddings (MiniLM)
   ->
Vector Database (Chroma)
   ->
User Query + Stage
   ->
Similarity Retrieval (Top-K)
   ->
Context Building (with metadata)
   ->
LLM (TinyLlama)
   ->
Final Answer


## RAG Stages

**Document Loading** – Load treatment PDFs
**Chunking** – Split into meaningful sections
**Metadata Extraction** - extract metadata of each chunk->source id,chunk id etc
**Embedding** – Convert text → vectors
**Vector Storage** – Store in Chroma DB
**Query Processing** – Combine stage + question
**Retrieval** – Fetch relevant chunks
**Prompt Engineering** – Structured input
**LLM Generation** – Context-based answer


## Model Pipeline

MRI Image
->
Resize (224x224)
->
VGG16 (Feature Extraction)
->
Flatten
->
PCA (300 features)
->
StandardScaler
->
SVM (RBF Kernel)
->
Prediction
->
RAG System → Treatment Answer

## Output Classes

MildDemented
/nModerateDemented
/nNonDemented
/nVeryMildDemented

## Performance

~98% accuracy
/nFaster training using PCA
/nContext-aware treatment generation via RAG


## Author

Arnab Sahoo
