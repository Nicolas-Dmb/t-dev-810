from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from PIL import Image

from t_dev_810.data import DatasetFile

from .io import load_version_json

datasets = ("Test", "Train", "Val")


def display_distribution(
    dataset: DatasetFile,
):

    groups = _count_labels(dataset)

    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0

    _, ax = plt.subplots(layout="constrained")

    for attribute, measurement in groups.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("count")
    ax.set_title("datasets")
    ax.set_xticks(x + width, datasets)
    ax.legend(loc="upper left", ncols=3)

    plt.show()


def _count_labels(dataset: DatasetFile) -> dict[str, tuple[int, int, int]]:
    list_test_normal_imgs = [img for img in dataset.test if img.label == 0]
    list_train_normal_imgs = [img for img in dataset.train if img.label == 0]
    list_val_normal_imgs = [img for img in dataset.val if img.label == 0]

    list_test_pneumonia_imgs = len(dataset.test) - len(list_test_normal_imgs)
    list_train_pneumonia_imgs = len(dataset.train) - len(list_train_normal_imgs)
    list_val_pneumonia_imgs = len(dataset.val) - len(list_val_normal_imgs)

    return {
        "normal": (
            len(list_test_normal_imgs),
            len(list_train_normal_imgs),
            len(list_val_normal_imgs),
        ),
        "pneumonia": (
            list_test_pneumonia_imgs,
            list_train_pneumonia_imgs,
            list_val_pneumonia_imgs,
        ),
    }


def show_image_from_path(path: str, label: int | None = None):
    img = Image.open(path)

    plt.imshow(img, cmap="gray")
    if label is not None:
        title = "PNEUMONIA" if label == 1 else "NORMAL"
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_image_size(dataset: DatasetFile):
    img_width: List[int] = []
    img_height: List[int] = []

    for img_lists in [dataset.train, dataset.val, dataset.test]:
        for img in img_lists:
            assert isinstance(img.path, str)
            img = Image.open(img.path)
            width, height = img.size
            img_width.append(width)
            img_height.append(height)

    plt.scatter(img_width, img_height)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Scatter Plot of Image Dimensions")
    plt.show()


def _format_value(value: Any) -> str:
    """Format values for markdown display."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_experiment_markdown(experiment_id: str, experiment: Dict[str, Any]) -> str:
    """Build a neutral markdown summary for one experiment."""
    envs = experiment["envs"]
    results = experiment["results"]
    confusion_matrix = results["confusion_matrix"]

    return f"""
## 🧪 Experiment — {experiment_id}

### Hypothesis
{envs["hypothesis"]}

### Parameters
- **Image size**: {envs["image_size"]}x{envs["image_size"]}
- **Normalize**: {_format_value(envs["normalize"])}
- **PCA components**: {_format_value(envs["pca_components"])}
- **Crop factor**: {_format_value(envs["crop_factor"])}
- **Enhance factor**: {_format_value(envs["enhance_factor"])}
- **Model**: {_format_value(envs["model"])}
- **Penalty**: {_format_value(envs["penalty"])}
- **Solver**: {_format_value(envs["solver"])}
- **L1 ratio**: {_format_value(envs["l1_ratio"])}
- **Regularization C**: {_format_value(envs["regularization_c"])}
- **Class weight**: {_format_value(envs["class_weight"])}
- **Max iter**: {_format_value(envs["max_iter"])}

### Results
"""


def display_experiment_markdown(experiment_id: str, experiment: Dict[str, Any]) -> None:
    """Display experiment summary as markdown."""
    display(Markdown(_build_experiment_markdown(experiment_id, experiment)))


def plot_experiment_metrics(experiment: Dict[str, Any]) -> None:
    """Plot the main evaluation metrics."""
    results = experiment["results"]

    metric_names = ["Accuracy", "Recall", "Precision", "F1-score", "Test AUC"]
    metric_values = [
        results["accuracy"],
        results["recall"],
        results["precision"],
        results["f1_score"],
        results["test_auc"],
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1)
    plt.title("Main Evaluation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_auc_comparison(experiment: Dict[str, Any]) -> None:
    """Plot CV AUC mean vs Test AUC."""
    results = experiment["results"]

    auc_names = ["CV AUC mean", "Test AUC"]
    auc_values = [results["cv_auc_mean"], results["test_auc"]]

    plt.figure(figsize=(6, 4))
    plt.bar(auc_names, auc_values)
    plt.ylim(0, 1)
    plt.title("CV AUC vs Test AUC")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(experiment: Dict[str, Any]) -> None:
    """Display confusion matrix as a heatmap."""
    cm = experiment["results"]["confusion_matrix"]

    matrix = np.array(
        [
            [cm["TN"], cm["FP"]],
            [cm["FN"], cm["TP"]],
        ]
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred Normal", "Pred Pneumonia"])
    plt.yticks([0, 1], ["Actual Normal", "Actual Pneumonia"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def _display_experiment(experiment_id: str, experiment: Dict[str, Any]) -> None:
    """Display a full experiment summary with markdown and plots."""
    display_experiment_markdown(experiment_id, experiment)
    plot_experiment_metrics(experiment)
    plot_auc_comparison(experiment)
    plot_confusion_matrix(experiment)


def get_experiment_by_id(
    version_data: Dict[str, Any], experiment_id: str
) -> Dict[str, Any]:
    """Get one experiment by its id."""
    if experiment_id not in version_data["versions"].keys():
        raise KeyError(f"Experiment ID '{experiment_id}' not found.")
    return version_data["versions"][experiment_id]


def display_experiment_from_json(experiment_id: str) -> None:
    """Load one experiment from JSON and display it."""
    version_data = load_version_json()
    experiment = get_experiment_by_id(version_data, experiment_id)
    _display_experiment(experiment_id, experiment)
