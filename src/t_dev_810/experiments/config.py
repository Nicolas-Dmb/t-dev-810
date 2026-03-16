from typing import Optional

from .schema import ExperimentConf, GridSearchConf, Model, Penalty, Solver


def ask_bool(question: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    answer = input(f"{question} [y/n] (default {default_str}): ").strip().lower()

    if answer == "":
        return default
    return answer in ["y", "yes"]


def ask_int(question: str, default: int) -> int:
    answer = input(f"{question} (default {default}): ").strip()
    return int(answer) if answer else default


def ask_opt_int(question: str, default: Optional[int]) -> Optional[int]:
    answer = input(f"{question} (default {default}): ").strip()
    return int(answer) if answer else default


def ask_float(question: str, default: float) -> float:
    answer = input(f"{question} (default {default}): ").strip()
    return float(answer) if answer else default


def build_config() -> ExperimentConf | GridSearchConf:

    print("\n===== Experiment configuration =====\n")

    image_size = ask_int("Image size", 64)

    normalize = ask_bool("Normalize pixels", True)

    pca_components = ask_opt_int("PCA components", None)

    crop_factor = ask_int("Crop factor", 0)

    experiment_type = ask_bool("Is it GridSearch experiment ?", False)
    if experiment_type:
        hypothesis = input("Experiment hypothesis: ").strip()
        return GridSearchConf.from_quizz(
            image_size,
            normalize,
            pca_components,
            crop_factor=crop_factor,
            hypothesis=hypothesis,
        )

    penalty = (
        input("Penalty [l1/l2/elasticnet] (default l2): ").strip() or "l2"
    )  # TODO enumerat all available values

    solver = input("Solver (default liblinear): ").strip() or "liblinear"

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = ask_float("l1_ratio", 0.5)

    C = ask_float("Regularization C", 1.0)

    class_weight = ask_bool("Class weight", False)

    hypothesis = input("Experiment hypothesis: ").strip()

    return ExperimentConf(
        image_size=image_size,
        normalize=normalize,
        pca_components=pca_components,
        crop_factor=crop_factor,
        model=Model.logistic_regression,
        penalty=Penalty.elasticnet,  # TODO : find correct value
        solver=Solver.liblinear,  # TODO
        l1_ratio=l1_ratio,
        regularization_c=C,
        class_weight=class_weight,
        hypothesis=hypothesis,
    )
