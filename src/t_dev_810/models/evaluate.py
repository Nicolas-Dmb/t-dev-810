from typing import Any

from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(
    model: Any,
    X_train: ndarray,
    y_train: list[int],
    X_test: ndarray,
    y_test: list[int],
    y_pred: list[int],
) -> dict[str, Any]:
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError(
            "Model must implement predict_proba or decision_function to compute ROC AUC."
        )

    test_auc = roc_auc_score(y_test, y_score)

    return {
        "accuracy": accuracy,
        "confusion_matrix": {
            "TP": int(cm[1, 1]),
            "FP": int(cm[0, 1]),
            "TN": int(cm[0, 0]),
            "FN": int(cm[1, 0]),
        },
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "test_auc": test_auc,
    }
