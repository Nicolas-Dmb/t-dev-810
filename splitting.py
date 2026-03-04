from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

from config import PROCESS_TYPE, Config


def data_splitting_all(
    config: Config,
    train_normal_paths: List[Path],
    train_pneumonia_paths: List[Path],
    val_normal_paths: List[Path],
    val_pneumonia_paths: List[Path],
    test_normal_paths: List[Path],
    test_pneumonia_paths: List[Path],
) -> tuple[List[Path], List[Path], List[Path], List[int], List[int], List[int]]:
    config.PROCESSED.append(PROCESS_TYPE.data_spliting_category)
    all_paths = (
        train_normal_paths
        + train_pneumonia_paths
        + val_normal_paths
        + val_pneumonia_paths
        + test_normal_paths
        + test_pneumonia_paths
    )
    all_labels = (
        [0] * len(train_normal_paths)
        + [1] * len(train_pneumonia_paths)
        + [0] * len(val_normal_paths)
        + [1] * len(val_pneumonia_paths)
        + [0] * len(test_normal_paths)
        + [1] * len(test_pneumonia_paths)
    )

    train_paths, val_paths, y_train, y_val = train_test_split(
        all_paths,
        all_labels,
        test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=42,
        stratify=all_labels,
    )

    val_paths, test_paths, y_val, y_test = train_test_split(
        val_paths,
        y_val,
        test_size=config.VAL_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state=42,
        stratify=y_val,
    )

    return train_paths, val_paths, test_paths, y_train, y_val, y_test


def data_splitting_val(
    config: Config,
    train_normal_paths: List[Path],
    train_pneumonia_paths: List[Path],
    val_normal_paths: List[Path],
    val_pneumonia_paths: List[Path],
) -> tuple[List[Path], List[Path], List[int], List[int]]:
    config.PROCESSED.append(PROCESS_TYPE.data_splitting_test_val)
    train_paths, val_paths, y_train, y_val = train_test_split(
        train_normal_paths + train_pneumonia_paths,
        [0] * len(train_normal_paths) + [1] * len(train_pneumonia_paths),
        test_size=config.VAL_SIZE,
        random_state=42,
        stratify=[0] * len(train_normal_paths) + [1] * len(train_pneumonia_paths),
    )

    val_paths += val_normal_paths + val_pneumonia_paths
    y_val += [0] * len(val_normal_paths) + [1] * len(val_pneumonia_paths)

    return train_paths, val_paths, y_train, y_val
