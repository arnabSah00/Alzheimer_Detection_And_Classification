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

### **The Pipeline Flow:**
1. **Document Loading** – Ingesting medical treatment PDFs using `PyPDF`.
2. **Chunking** – Splitting dense text into smaller, meaningful sections for better retrieval accuracy.
3. **Metadata Extraction** – Extracting specific metadata for each chunk (e.g., `source_id`, `chunk_id`) to maintain traceability.
4. **Embedding** – Converting text chunks into high-dimensional vectors using **MiniLM**.
5. **Vector Storage** – Storing and indexing vectors in **Chroma DB** for efficient similarity searching.
6. **Query Processing** – Merging the predicted Alzheimer’s stage with the user's specific question.
7. **Retrieval** – Fetching the top-K most relevant document chunks based on vector similarity.
8. **Prompt Engineering** – Constructing a structured prompt that feeds the retrieved context into the LLM.
9. **LLM Generation** – Generating a natural language answer using **TinyLlama** based strictly on the provided context.

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
`NonDemented`  `VeryMildDemented`  `MildDemented`  `ModerateDemented`

## Performance

* **High Accuracy:** Achieved **~98%** accuracy on the test dataset.
* **Optimized Efficiency:** Faster training and inference times implemented using **PCA** for dimensionality reduction.
* **Context-Aware Guidance:** Intelligent treatment generation powered by **RAG**, ensuring evidence-based medical responses.


## Author

Arnab Sahoo
