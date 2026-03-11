from pathlib import Path

from sklearn.model_selection import train_test_split

from t_dev_810.config import PROCESS_TYPE, Config
from t_dev_810.utils.model import ImageFile


def data_splitting(
    config: Config,
    ImageFile_train: list[ImageFile],
    ImageFile_val: list[ImageFile],
) -> tuple[list[ImageFile], list[ImageFile]]:
    config.PROCESSED.append(PROCESS_TYPE.data_distribution)

    X = ImageFile_train + ImageFile_val
    y: list[int] = []
    for image in X:
        if image.path.parent.name == "NORMAL":
            y.append(0)
        elif image.path.parent.name == "PNEUMONIA":
            y.append(1)
        else:
            raise ValueError(
                f"Unexpected folder name {image.path.parent.name} for image {image.path}"
            )

    ImageFile_train, ImageFile_val = train_test_split(
        X,
        test_size=config.VAL_SIZE,
        random_state=42,
        stratify=y,
    )
    X_train: list[ImageFile] = []
    X_val: list[ImageFile] = []

    for image in ImageFile_train:
        assert isinstance(image, ImageFile)
        if image.path.parents[1].name == "val":
            imgFile = ImageFile(
                path=Path("train/" + image.path.parent.name + "/" + image.path.name),
                img=image.img,
            )
            X_train.append(imgFile)
        else:
            X_train.append(image)

    for image in ImageFile_val:
        assert isinstance(image, ImageFile)
        if image.path.parents[1].name == "train":
            imgFile = ImageFile(
                path=Path("val/" + image.path.parent.name + "/" + image.path.name),
                img=image.img,
            )
            X_val.append(imgFile)
        else:
            X_val.append(image)

    return X_train, X_val
