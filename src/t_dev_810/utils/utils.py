import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

from t_dev_810.config import PROCESS_TYPE, Config
from t_dev_810.utils.model import ImageFile

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


def init_dataset():
    """Initialize the dataset by creating a new folder and cleaning it if it already exists."""
    dataset_path = Path("dataset/")
    # Clean the dataset folder if it already exists
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)


def cropping(img_file: ImageFile) -> ImageFile:
    """Crop the image by removing 10% of the width and height from each side."""
    width, height = img_file.img.size

    margin_w = width * 0.1
    margin_h = height * 0.1

    return ImageFile(
        img_file.img.crop((margin_w, margin_h, width - margin_w, height - margin_h)),
        img_file.path,
    )


def enhance_contrast(imgs: List[ImageFile], factor: float) -> List[ImageFile]:
    """Enhance the contrast of the images by a given factor."""

    return [
        ImageFile(ImageEnhance.Contrast(img.img).enhance(factor), img.path)
        for img in imgs
    ]


def register_processed_images(X: List[ImageFile], subfolder: str):
    """Save the processed images in the appropriate folders based on their class."""

    if Path("dataset/" + subfolder).exists():
        shutil.rmtree("dataset/" + subfolder)
    for imageFile in X:
        new_path = (
            "dataset/"
            + imageFile.path.parents[1].name
            + "/"
            + imageFile.path.parents[0].name
            + "/"
        )

        if not Path(new_path).exists():
            Path(new_path).mkdir(parents=True, exist_ok=True)

        imageFile.img.save(new_path + imageFile.path.name)


def flatten_images(
    images: List[Image.Image],
) -> np.ndarray:
    """Flatten images into 1D arrays."""
    flattened_images: List[np.ndarray] = []

    for image in images:
        assert isinstance(image, Image.Image)
        img_array = np.array(image)
        img_flat = img_array.flatten()
        flattened_images.append(img_flat)

    return np.array(flattened_images)


def normalise_pixel(
    config: Config, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize pixel values to the range [0, 1]."""
    config.PROCESSED.append(PROCESS_TYPE.normalize_pixel)
    X_train_normalized = X_train / 255.0
    X_val_normalized = X_val / 255.0
    X_test_normalized = X_test / 255.0

    return X_train_normalized, X_val_normalized, X_test_normalized
