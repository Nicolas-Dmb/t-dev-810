from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageEnhance

from config import PROCESS_TYPE, Config


def normalize_img(
    config: Config,
    train_paths: List[Path],
    val_paths: List[Path],
    test_paths: List[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize images by resizing them to a common size and flattening them into 1D arrays."""
    config.PROCESSED.append(PROCESS_TYPE.normalize_img)
    if PROCESS_TYPE.enhance_color in config.PROCESS:
        config.PROCESSED.append(PROCESS_TYPE.enhance_color)
    images_train: List[np.ndarray] = []
    images_val: List[np.ndarray] = []
    images_test: List[np.ndarray] = []

    for list_imgs, images in zip(
        [train_paths, val_paths, test_paths], [images_train, images_val, images_test]
    ):
        for img_path in list_imgs:
            assert isinstance(img_path, Path)
            img = Image.open(img_path).convert("L")
            img = img.resize(config.IMG_SIZE)
            if PROCESS_TYPE.enhance_color in config.PROCESS:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(config.ENHANCE_COLOR_FACTOR)
            img_array = np.array(img)
            img_flat = img_array.flatten()
            images.append(img_flat)

    return np.array(images_train), np.array(images_val), np.array(images_test)


def normalise_pixel(
    config: Config, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize pixel values to the range [0, 1]."""
    config.PROCESSED.append(PROCESS_TYPE.normalize_pixel)
    X_train_normalized = X_train / 255.0
    X_val_normalized = X_val / 255.0
    X_test_normalized = X_test / 255.0

    return X_train_normalized, X_val_normalized, X_test_normalized
