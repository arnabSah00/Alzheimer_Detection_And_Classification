# Alzheimer Detection using MRI Images

This project is a complete end-to-end Machine Learning pipeline that detects Alzheimer’s disease stages from MRI images using:

- VGG16 (Feature Extraction)
- PCA (Dimensionality Reduction)
- SVM (Classification)

---

## Features

- Full automated pipeline (no notebook required)
- Feature extraction using pretrained CNN (VGG16)
- PCA for dimensionality reduction
- SVM model for classification
- Streamlit web app for real-time prediction
- Modular and production-ready structure

---

## Project Structure

Alzheimer_Project/
│
├── data/
│ └── raw_mri/
│
├── models/
│
|___notebook/
|   |
|   |---Alzheimer_Full_Pipeline.ipynb
|
├── src/
│ |--- config.py
│ |--- dataset.py
│ |--- feature_extraction.py
│ |--- pca_transform.py
│ |--- train_svm.py
│ |--- classify.py
│
|---main.py
|--- app.py
|--- requirements.txt
|--- README.md



---

##  Installation

### 1. Clone the repository


### 2. Create virtual environment


### 3. Install dependencies


---

## Dataset
raw_mri/
├── MildDemented/
├── ModerateDemented/
├── NonDemented/
├── VeryMildDemented/

---

## Run Training Pipeline


### What it does:
python main.py
- Splits dataset (train/val/test)
- Extracts features using VGG16
- Applies PCA
- Trains SVM model
- Saves model files

## Run Web App
streamlit run app.py


### Features:

- Upload MRI image
- Get instant prediction
- Simple UI

---

## Model Pipeline
MRI Image
↓
Resize (224x224)
↓
VGG16 (Feature Extraction)
↓
Flatten
↓
PCA (300 features)
↓
StandardScaler
↓
SVM (RBF Kernel)
↓
Prediction



---

## Output Classes

- MildDemented
- ModerateDemented
- NonDemented
- VeryMildDemented

---

## Performance

- ~98% accuracy (depends on dataset)
- Optimized using PCA for faster training

---

## Future Improvements

- Add confidence score
- Deploy using FastAPI
- Docker containerization
- Improve UI design

---

## Author

- Arnab Sahoo
