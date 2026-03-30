import enum
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


class DatasetTypes(enum.Enum):
    TEST = "test"
    TRAIN = "train"
    VAL = "val"


@dataclass
class ImagePath:
    path: str
    label: int


@dataclass
class ImageData:
    data: np.ndarray
    label: int


@dataclass
class ImageFile:
    data: Image.Image
    label: int


@dataclass
class DatasetFile:
    test: List[ImagePath]
    train: List[ImagePath]
    val: List[ImagePath]


@dataclass
class DatasetImg:
    test: List[ImageFile]
    train: List[ImageFile]
    val: List[ImageFile]


@dataclass
class DatasetData:
    test: List[ImageData]
    train: List[ImageData]
    val: List[ImageData]
