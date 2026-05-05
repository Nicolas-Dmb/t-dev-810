from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from PIL import Image

from t_dev_810.data.schema import DatasetFile

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

    base = f"""
## Experiment — {experiment_id}

### Hypothesis
{envs["hypothesis"]}

### Parameters
- **Image size**: {envs["image_size"]}x{envs["image_size"]}
- **Normalize**: {_format_value(envs["normalize"])}
- **PCA components**: {_format_value(envs["pca_components"])}
- **Crop factor**: {_format_value(envs["crop_factor"])}
- **Enhance factor**: {_format_value(envs["enhance_factor"])}
- **Model**: {_format_value(envs["model"])}
"""

    if envs.get("model") == "random_forest":
        base += f"""- **N estimators**: {_format_value(envs.get("n_estimators"))}
- **Max depth**: {_format_value(envs.get("max_depth"))}
- **Min samples split**: {_format_value(envs.get("min_samples_split"))}
- **Min samples leaf**: {_format_value(envs.get("min_samples_leaf"))}
- **Class weight**: {_format_value(envs.get("class_weight"))}
"""
    else:
        base += f"""- **Penalty**: {_format_value(envs.get("penalty"))}
- **Solver**: {_format_value(envs.get("solver"))}
- **L1 ratio**: {_format_value(envs.get("l1_ratio"))}
- **Regularization C**: {_format_value(envs.get("regularization_c"))}
- **Class weight**: {_format_value(envs.get("class_weight"))}
- **Max iter**: {_format_value(envs.get("max_iter"))}
"""

    base += "\n### Results\n"
    return base


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


def _old_display_experiment(experiment_id: str, experiment: Dict[str, Any]) -> None:
    """Display a full experiment summary with markdown and plots."""
    display_experiment_markdown(experiment_id, experiment)
    plot_experiment_metrics(experiment)
    plot_auc_comparison(experiment)
    plot_confusion_matrix(experiment)


def _find_varying_params(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Find config parameters that differ across experiments."""
    configs = [r["config"] for r in results]
    all_keys: set[str] = set()
    for config in configs:
        all_keys.update(config.keys())

    varying: Dict[str, List[Any]] = {}
    for key in sorted(all_keys):
        values = [config.get(key) for config in configs]
        if len(set(str(v) for v in values)) > 1:
            varying[key] = values
    return varying


def _sweep_experiment_labels(
    results: List[Dict[str, Any]], varying: Dict[str, List[Any]]
) -> List[str]:
    """Build short labels from varying params for each experiment."""
    labels: List[str] = []
    for i, r in enumerate(results):
        parts = [f"{k}={_format_value(r['config'].get(k))}" for k in varying]
        labels.append(", ".join(parts) if parts else f"Exp {i + 1}")
    return labels


def _build_sweep_markdown(experiment_id: str, experiment: Dict[str, Any]) -> str:
    """Build a markdown comparison table for a sweep experiment."""
    results = experiment["results"]
    hypothesis = experiment.get("hypothesis", "N/A")
    n = len(results)
    varying = _find_varying_params(results)
    labels = _sweep_experiment_labels(results, varying)

    md = f"\n## Sweep — {experiment_id}\n\n"
    md += f"### Hypothesis\n{hypothesis}\n\n"

    # Config differences table
    if varying:
        header = "| Parameter | " + " | ".join(labels) + " |"
        sep = "|---|" + "|".join("---" for _ in range(n)) + "|"
        md += "### Configuration differences\n"
        md += header + "\n" + sep + "\n"
        for param, values in varying.items():
            row = (
                f"| **{param}** | "
                + " | ".join(_format_value(v) for v in values)
                + " |"
            )
            md += row + "\n"
        md += "\n"

    # Metrics comparison table
    metric_keys = [
        "accuracy",
        "recall",
        "precision",
        "f1_score",
        "test_auc",
        "cv_auc_mean",
    ]
    metric_labels = [
        "Accuracy",
        "Recall",
        "Precision",
        "F1-score",
        "Test AUC",
        "CV AUC mean",
    ]

    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join("---" for _ in range(n)) + "|"
    md += "### Metrics comparison\n"
    md += header + "\n" + sep + "\n"
    for label, key in zip(metric_labels, metric_keys):
        values = [r["metrics"].get(key) for r in results]
        row = f"| **{label}** | " + " | ".join(_format_value(v) for v in values) + " |"
        md += row + "\n"

    # Best experiment by F1-score
    f1_scores = [r["metrics"].get("f1_score", 0) for r in results]
    best_idx = int(np.argmax(f1_scores))
    md += (
        f"\n**Best F1-score**: {labels[best_idx]}"
        f" ({_format_value(f1_scores[best_idx])})\n"
    )
    return md


def plot_sweep_metrics(experiment: Dict[str, Any]) -> None:
    """Grouped bar chart comparing metrics across sweep experiments."""
    results = experiment["results"]
    n = len(results)
    varying = _find_varying_params(results)
    labels = _sweep_experiment_labels(results, varying)

    metric_keys = ["accuracy", "recall", "precision", "f1_score", "test_auc"]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1-score", "Test AUC"]

    x = np.arange(len(metric_labels))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (result, label) in enumerate(zip(results, labels)):
        values = [result["metrics"].get(m, 0) for m in metric_keys]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label)

    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sweep_roc_curves(experiment: Dict[str, Any]) -> None:
    """Overlay ROC curves for all sweep experiments on a single plot."""
    results = experiment["results"]
    varying = _find_varying_params(results)
    labels = _sweep_experiment_labels(results, varying)

    plt.figure(figsize=(7, 6))
    for result, label in zip(results, labels):
        roc = result["metrics"].get("roc_curve")
        if roc is None:
            continue
        auc = result["metrics"]["test_auc"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_sweep_confusion_matrices(experiment: Dict[str, Any]) -> None:
    """Side-by-side confusion matrices for each sweep experiment."""
    results = experiment["results"]
    n = len(results)
    varying = _find_varying_params(results)
    labels = _sweep_experiment_labels(results, varying)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 2:
        axes = list(axes)
    elif n == 1:
        axes = [axes]

    for ax, result, label in zip(axes, results, labels):
        cm = result["metrics"]["confusion_matrix"]
        matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])

        ax.imshow(matrix)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Pneumonia"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Pneumonia"], fontsize=8)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, matrix[r, c], ha="center", va="center")

    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    plt.show()


def _display_sweep(experiment_id: str, experiment: Dict[str, Any]) -> None:
    """Display a sweep experiment with comparison tables and plots."""
    display(Markdown(_build_sweep_markdown(experiment_id, experiment)))
    plot_sweep_metrics(experiment)
    plot_sweep_roc_curves(experiment)
    plot_sweep_confusion_matrices(experiment)


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
    results = experiment.get("results")

    if isinstance(results, list):
        if len(results) == 1:
            # New format, single experiment → adapt to old format display
            single = results[0]
            compat = {
                "envs": {
                    **single["config"],
                    "hypothesis": experiment.get("hypothesis", ""),
                },
                "results": single["metrics"],
                "model": single["model"],
            }
            _old_display_experiment(experiment_id, compat)
        else:
            # New format, multiple experiments → comparative display
            _display_sweep(experiment_id, experiment)
    else:
        # Old format (results is a dict or missing)
        _old_display_experiment(experiment_id, experiment)
