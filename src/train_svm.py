import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PCA_FEATURE_PATH, FEATURE_PATH, MODEL_PATH, SCALER_PATH


def train_model():

    X_train = np.load(f"{PCA_FEATURE_PATH}/X_train_pca.npy")
    y_train = np.load(f"{FEATURE_PATH}/y_train.npy")

    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    joblib.dump(scaler, SCALER_PATH)

    svm = SVC(kernel='rbf', C=5, gamma='scale', max_iter=2000)
    svm.fit(X_train, y_train)

    joblib.dump(svm, MODEL_PATH)

    print("Model trained")