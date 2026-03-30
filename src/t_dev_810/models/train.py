from typing import Any

import numpy as np

from t_dev_810.data.schema import DatasetData


def train_model(model: Any, dataset: DatasetData) -> Any:

    X_train = np.array([train.data for train in dataset.train])
    y_train = [train.label for train in dataset.train]

    X_val = np.array([val.data for val in dataset.val])
    y_val = [val.label for val in dataset.val]

    X_train_final = np.concatenate((X_train, X_val), axis=0)
    model.fit(X_train_final, y_train + y_val)
    return model
