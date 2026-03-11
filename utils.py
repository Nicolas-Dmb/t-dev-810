import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

datasets = ("Test", "Train", "Val")


def display_distribution(
    list_test_normal_imgs: List[Path],
    list_test_pneumonia_imgs: List[Path],
    list_train_normal_imgs: List[Path],
    list_train_pneumonia_imgs: List[Path],
    list_val_normal_imgs: List[Path],
    list_val_pneumonia_imgs: List[Path],
):
    groups = {
        "normal": (
            len(list_test_normal_imgs),
            len(list_train_normal_imgs),
            len(list_val_normal_imgs),
        ),
        "pneumonia": (
            len(list_test_pneumonia_imgs),
            len(list_train_pneumonia_imgs),
            len(list_val_pneumonia_imgs),
        ),
    }

    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0

    _, ax = plt.subplots(layout="constrained")

    for attribute, measurement in groups.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("count")
    ax.set_title("datasets")
    ax.set_xticks(x + width, datasets)
    ax.legend(loc="upper left", ncols=3)

    plt.show()


def dataset_checker(X: List[Path], y: List[int]):
    assert len(X) == len(y), "The number of samples in X and y must be the same."
    for img_path, label in zip(X, y):
        if img_path.parent.name == "NORMAL":
            assert label == 0, f"Expected label 0 for NORMAL image, got {label}."
        elif img_path.parent.name == "PNEUMONIA":
            assert label == 1, f"Expected label 1 for PNEUMONIA image, got {label}."
        else:
            raise ValueError(f"Unexpected directory name: {img_path.parent.name}")


def crop_images(imgs: list[Path], subFolder: str) -> list[Path]:
    """Crop the images and save them in a new folder."""
    cropped_images: list[Path] = []
    dataset_link = "dataset/" + subFolder + "/"
    dataset_path = Path(dataset_link)
    # Clean the dataset folder if it already exists
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=True)
    # Crop the images and save them in the dataset folder
    for img_path in imgs:
        with Image.open(img_path) as image:
            cropped_image = cropping(image)

            folder_path = dataset_path / img_path.parent.name
            folder_path.mkdir(parents=True, exist_ok=True)

            output_path = folder_path / img_path.name
            cropped_image.save(output_path)
            cropped_images.append(output_path)
    return cropped_images


def cropping(img: Image.Image) -> Image.Image:
    """Crop the image by removing 10% of the width and height from each side."""
    width, height = img.size

    margin_w = width * 0.1
    margin_h = height * 0.1

    return img.crop((margin_w, margin_h, width - margin_w, height - margin_h))
