import os
import sys

# import path
sys.path.append(os.path.abspath("."))

from src.dataset import split_dataset
from src.feature_extraction import extract_features
from src.pca_transform import apply_pca
from src.train_svm import train_model
from src.config import MODEL_PATH


def run_pipeline():
    print("\nRunning full training pipeline...\n")

    split_dataset()
    extract_features()
    apply_pca()
    train_model()

    print("\nTraining pipeline completed\n")


def predict_flow():
    from src.classify import classify_image

    img_path = input("Enter MRI image path: ").strip()

    if not os.path.exists(img_path):
        print("Invalid path. Try again.")
        return

    result = classify_image(img_path)

    print(f"\nPrediction: {result}\n")


if __name__ == "__main__":

    print("Alzheimer Detection System")

    # Check model
    if os.path.exists(MODEL_PATH):
        print("Model found → Skipping training")
    else:
        run_pipeline()

    # Always predict
    predict_flow()