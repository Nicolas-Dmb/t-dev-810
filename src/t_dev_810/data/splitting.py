from typing import List

from sklearn.model_selection import train_test_split  # type: ignore

from .schema import DatasetFile, ImagePath


def data_splitting(dataset: DatasetFile) -> DatasetFile:
    "fix distribution between train and test as 80/20"

    dataset_train = dataset.train + dataset.val
    label = [value.label for value in dataset_train]

    dataset_train, dataset_val = train_test_split(
        dataset_train,
        test_size=0.2,
        random_state=42,
        stratify=label,
    )
    assert isinstance(dataset_train, List) and isinstance(dataset_train[0], ImagePath)
    assert isinstance(dataset_val, List) and isinstance(dataset_val[0], ImagePath)
    return DatasetFile(dataset.test, dataset_train, dataset_val)  # type: ignore
