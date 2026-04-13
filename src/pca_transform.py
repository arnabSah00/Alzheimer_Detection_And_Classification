import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from src.config import FEATURE_PATH, PCA_PATH, PCA_FEATURE_PATH

os.makedirs(PCA_FEATURE_PATH, exist_ok=True)


def apply_pca():

    if os.path.exists(os.path.join(PCA_FEATURE_PATH, "X_train_pca.npy")):
        print("PCA already done")
        return

    print("Applying PCA...")

    X_train = np.load(os.path.join(FEATURE_PATH, "X_train.npy"))
    X_val = np.load(os.path.join(FEATURE_PATH, "X_val.npy"))
    X_test = np.load(os.path.join(FEATURE_PATH, "X_test.npy"))

    pca = PCA(n_components=300)

    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    joblib.dump(pca, PCA_PATH)

    np.save(os.path.join(PCA_FEATURE_PATH, "X_train_pca.npy"), X_train)
    np.save(os.path.join(PCA_FEATURE_PATH, "X_val_pca.npy"), X_val)
    np.save(os.path.join(PCA_FEATURE_PATH, "X_test_pca.npy"), X_test)

    print("PCA complete")