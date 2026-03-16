from pathlib import Path

from PIL import Image

from t_dev_810.data.schema import DatasetFile, DatasetImg, ImageFile, ImagePath

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = PROJECT_ROOT / "data"


def load() -> DatasetFile:
    print("loading dataset ...")
    test_path = str(DATA_RAW / "test")
    train_path = str(DATA_RAW / "train")
    val_path = str(DATA_RAW / "val")
    return DatasetFile(
        [
            ImagePath(path=str(p), label=int(p.parent.name == "pneumonia"))
            for p in Path(test_path).glob("**/*.jpeg")
        ],
        [
            ImagePath(path=str(p), label=int(p.parent.name == "pneumonia"))
            for p in Path(train_path).glob("**/*.jpeg")
        ],
        [
            ImagePath(path=str(p), label=int(p.parent.name == "pneumonia"))
            for p in Path(val_path).glob("**/*.jpeg")
        ],
    )


def load_image(dataset_file: DatasetFile) -> DatasetImg:
    print("loading image in dataset ...")
    return DatasetImg(
        [ImageFile(data=Image.open(str(p)), label=p.label) for p in dataset_file.test],
        [ImageFile(data=Image.open(str(p)), label=p.label) for p in dataset_file.train],
        [ImageFile(data=Image.open(str(p)), label=p.label) for p in dataset_file.val],
    )
