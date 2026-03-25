from typing import List

import numpy as np

from t_dev_810.data.schema import DatasetData, DatasetImg, ImageData


def to_numpy(dataset: List[ImageData]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([img.data for img in dataset])
    y = np.array([img.label for img in dataset])
    return X, y


def flatten_image(dataset: DatasetImg) -> DatasetData:
    """Flatten the image to a 1D array."""
    return DatasetData(
        test=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.test
        ],
        train=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.train
        ],
        val=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.val
        ],
    )
