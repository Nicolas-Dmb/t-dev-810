import enum
from dataclasses import dataclass
from pathlib import Path


class PROCESS_TYPE(enum.Enum):
    data_spliting_category = "data_spliting_category"
    data_splitting_test_val = "data_splitting_test_val"
    normalize_img = "normalize_img"
    normalize_pixel = "normalize_pixel"
    enhance_color = "enhance_color"
    data_splitting_label = "data_splitting_label"


class MODEL_TYPE(enum.Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"  # Not implemented yet
    svm = "svm"  # Not implemented yet


@dataclass
class Config:
    PROCESS: list[PROCESS_TYPE]
    PROCESSED: list[PROCESS_TYPE]
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    IMG_SIZE: tuple[int, int] = (255, 255)
    MAX_ITER: int = 2000
    ENHANCE_COLOR_FACTOR: float = 0.5
    MODEL: MODEL_TYPE = MODEL_TYPE.logistic_regression
    MESSAGE: str = ""

    @staticmethod
    def default() -> "Config":
        return Config(
            PROCESSED=[],
            PROCESS=[
                PROCESS_TYPE.data_splitting_test_val,
                PROCESS_TYPE.normalize_img,
                PROCESS_TYPE.data_splitting_label,
            ],
        )


@dataclass
class DatasetPaths:
    test_normal_paths: list[Path]
    test_pneumonia_paths: list[Path]
    train_normal_paths: list[Path]
    train_pneumonia_paths: list[Path]
    val_normal_paths: list[Path]
    val_pneumonia_paths: list[Path]

    @staticmethod
    def load() -> "DatasetPaths":
        test_normal_imgs = Path("test/NORMAL")
        test_pneumonia_imgs = Path("test/PNEUMONIA")

        train_normal_imgs = Path("train/NORMAL")
        train_pneumonia_imgs = Path("train/PNEUMONIA")

        val_normal_imgs = Path("val/NORMAL")
        val_pneumonia_imgs = Path("val/PNEUMONIA")
        return DatasetPaths(
            test_normal_paths=list(test_normal_imgs.glob("*.jpeg")),
            test_pneumonia_paths=list(test_pneumonia_imgs.glob("*.jpeg")),
            train_normal_paths=list(train_normal_imgs.glob("*.jpeg")),
            train_pneumonia_paths=list(train_pneumonia_imgs.glob("*.jpeg")),
            val_normal_paths=list(val_normal_imgs.glob("*.jpeg")),
            val_pneumonia_paths=list(val_pneumonia_imgs.glob("*.jpeg")),
        )
