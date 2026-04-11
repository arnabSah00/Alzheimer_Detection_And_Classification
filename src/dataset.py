import os
import shutil
import random
from src.config import RAW_PATH, DATA_PATH, TRAIN_PATH


def is_dataset_valid():
    if not os.path.exists(TRAIN_PATH):
        return False
    return len(os.listdir(TRAIN_PATH)) > 0


def clear_old_split():
    for d in ["train", "val", "test"]:
        path = os.path.join(DATA_PATH, d)
        if os.path.exists(path):
            shutil.rmtree(path)


def split_dataset():
    if is_dataset_valid():
        print("Dataset already valid")
        return

    print("Splitting dataset...")

    clear_old_split()

    for split in ["train", "val", "test"]:
        for cls in os.listdir(RAW_PATH):
            cls_path = os.path.join(RAW_PATH, cls)
            if not os.path.isdir(cls_path):
                continue
            os.makedirs(os.path.join(DATA_PATH, split, cls), exist_ok=True)

    for cls in os.listdir(RAW_PATH):
        cls_path = os.path.join(RAW_PATH, cls)
        if not os.path.isdir(cls_path):
            continue

        imgs = os.listdir(cls_path)
        random.shuffle(imgs)

        n = len(imgs)
        train_end = int(0.7*n)
        val_end = int(0.85*n)

        splits = {
            "train": imgs[:train_end],
            "val": imgs[train_end:val_end],
            "test": imgs[val_end:]
        }

        for s in splits:
            for img in splits[s]:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(DATA_PATH, s, cls, img)
                )

    print("Dataset split complete")