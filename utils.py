from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

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
