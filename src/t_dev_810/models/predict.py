from typing import Any

import numpy as np

from t_dev_810.data.schema import DatasetData


def predict_model(model: Any, dataset: DatasetData) -> tuple[list[int], list[int]]:
    X_test = np.array([test.data for test in dataset.test])
    y_test = [test.label for test in dataset.test]
    y_pred: list[int] = model.predict(X_test)
    return y_test, y_pred
