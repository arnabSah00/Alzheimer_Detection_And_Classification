import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.config import TRAIN_PATH, VAL_PATH, TEST_PATH, FEATURE_PATH

os.makedirs(FEATURE_PATH, exist_ok=True)


def extract_features():

    if os.path.exists(os.path.join(FEATURE_PATH, "X_train.npy")):
        print("Features already exist")
        return

    datagen = ImageDataGenerator(rescale=1./255)

    generators = {
        "train": datagen.flow_from_directory(TRAIN_PATH, target_size=(224,224), batch_size=32),
        "val": datagen.flow_from_directory(VAL_PATH, target_size=(224,224), batch_size=32),
        "test": datagen.flow_from_directory(TEST_PATH, target_size=(224,224), batch_size=32)
    }

    base_model = VGG16(weights='imagenet', include_top=False)
    vgg = Model(inputs=base_model.input, outputs=base_model.output)

    for name, gen in generators.items():

        features, labels = [], []

        for i in range(len(gen)):
            X, y = gen[i]
            X = preprocess_input(X*255.0)

            feat = vgg.predict(X, verbose=0)
            feat = feat.reshape(feat.shape[0], -1)

            features.append(feat)
            labels.append(y)

            print(f"{name} batch {i+1}/{len(gen)}")

        np.save(os.path.join(FEATURE_PATH, f"X_{name}.npy"), np.vstack(features))
        np.save(os.path.join(FEATURE_PATH, f"y_{name}.npy"), np.vstack(labels))

    print("Feature extraction complete")