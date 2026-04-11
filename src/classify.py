import joblib
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from src.config import PCA_PATH, MODEL_PATH, SCALER_PATH

CLASS_NAMES = [
    'MildDemented',
    'ModerateDemented',
    'NonDemented',
    'VeryMildDemented'
]

# Load models
pca = joblib.load(PCA_PATH)
svm = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load VGG16 
base_model = VGG16(weights='imagenet', include_top=False)
vgg = Model(inputs=base_model.input, outputs=base_model.output)


def classify_image(img_path):

    # Load image
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Preprocess
    img = preprocess_input(img * 255.0)

    # Extract features
    feat = vgg.predict(img, verbose=0)
    feat = feat.reshape(1, -1)

    # PCA + Scale
    feat = pca.transform(feat)
    feat = scaler.transform(feat)

    # Predict
    pred = svm.predict(feat)[0]

    return CLASS_NAMES[pred]