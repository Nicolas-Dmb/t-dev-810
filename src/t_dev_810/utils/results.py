import json
from datetime import datetime
from typing import Any, Dict, List

from numpy import ndarray
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

from t_dev_810.config import Config


def evaluate_model(
    config: Config,
    model: Any,
    X_train: ndarray,
    y_train: List[int],
    X_test: ndarray,
    y_test: List[int],
    y_pred: ndarray,
) -> None:
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    )

    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]

    recall = tp / (fn + tp)

    precision = tp / (fp + tp)

    f1 = 2 * (precision * recall) / (precision + recall)

    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    envs_dict = {
        "TEST_SIZE": config.TEST_SIZE,
        "VAL_SIZE": config.VAL_SIZE,
        "IMG_SIZE": config.IMG_SIZE,
        "MAX_ITER": config.MAX_ITER,
        "PROCESS": [process.value for process in config.PROCESSED],
        "ENHANCE_COLOR_FACTOR": config.ENHANCE_COLOR_FACTOR,
        "MESSAGE": config.MESSAGE,
    }

    result_dict = {
        "accuracy": accuracy,
        "confusion_matrix": {
            "TP": int(cm[1, 1]),
            "FP": int(cm[0, 1]),
            "TN": int(cm[0, 0]),
            "FN": int(cm[1, 0]),
        },
        "cv_auc": (scores.mean(), scores.std()),
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "test_auc": test_auc,
    }
    _register_result(envs_dict, result_dict)


def _register_result(envs_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
    with open("./version.json", "r+") as f:
        file = f.read()
        if file.strip():
            version_data = json.loads(file)
        else:
            version_data = {"versions": {}}
        version_data["versions"][datetime.now().isoformat()] = {
            "envs": envs_dict,
            "results": result_dict,
            "model": Config.MODEL.value,
        }
        f.seek(0)
        f.truncate()
        json.dump(version_data, f, indent=4)
