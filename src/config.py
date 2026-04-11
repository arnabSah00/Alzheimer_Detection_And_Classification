import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw_mri")

TRAIN_PATH = os.path.join(DATA_PATH, "train")
VAL_PATH = os.path.join(DATA_PATH, "val")
TEST_PATH = os.path.join(DATA_PATH, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_PATH = os.path.join(MODEL_DIR, "features")
PCA_FEATURE_PATH = os.path.join(MODEL_DIR, "pca_features")

PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")