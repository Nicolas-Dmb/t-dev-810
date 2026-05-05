from dataclasses import fields, replace
from typing import Any, Optional

from .schema import (
    ExperimentConf,
    Experiments,
    GridSearchConf,
    Model,
    Penalty,
    RandomForestExperimentConf,
    Solver,
)

# Parameters available for sweep, with their parser.
# "opt_int" means Optional[int] (accepts "None").
SWEEP_PARAMS: dict[str, type | str] = {
    "image_size": int,
    "pca_components": "opt_int",
    "crop_factor": int,
    "enhance_factor": int,
    "regularization_c": float,
    "max_iter": int,
    "n_estimators": int,
    "max_depth": "opt_int",
    "min_samples_split": int,
    "min_samples_leaf": int,
}


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


# ── Main entry point ─────────────────────────────────────────────────


def build_config() -> Experiments:
    hypothesis = input("Experiment hypothesis: ").strip()

    mode = (
        input("Mode [single/sweep] (default single): ").strip().lower()
        or "single"
    )

    if mode == "sweep":
        return _build_sweep_config(hypothesis)

    # single / manual multi mode
    experiments: list[ExperimentConf | RandomForestExperimentConf | GridSearchConf] = []
    while True:
        config = build_single_config()
        experiments.append(config)
        if isinstance(config, GridSearchConf):
            break
        if not ask_bool("Do you want to add another experiment?", False):
            break
    return Experiments(experiments=experiments, hypothesis=hypothesis)


# ── Sweep config builder ─────────────────────────────────────────────


def _parse_sweep_value(raw: str, type_hint: type | str) -> Any:
    """Parse a single user-entered value according to the param type."""
    raw = raw.strip()
    if type_hint == "opt_int":
        return None if raw.lower() == "none" else int(raw)
    if isinstance(type_hint, type):
        return type_hint(raw)
    return raw


def _get_sweepable_params(
    config: ExperimentConf | RandomForestExperimentConf,
) -> dict[str, type | str]:
    """Return sweep-eligible params that exist on this config type."""
    field_names = {f.name for f in fields(config)}
    return {k: v for k, v in SWEEP_PARAMS.items() if k in field_names}


def _build_sweep_config(hypothesis: str) -> Experiments:
    """Configure a base experiment, then generate variants by sweeping one param."""
    print("\n── Configure the base experiment ──\n")
    base = build_single_config()

    if isinstance(base, GridSearchConf):
        raise ValueError("Cannot sweep over a GridSearch experiment.")

    sweepable = _get_sweepable_params(base)
    params_list = list(sweepable.keys())

    print("\nSweepable parameters:")
    for i, name in enumerate(params_list, 1):
        current = getattr(base, name)
        print(f"  {i}. {name} (current: {current})")

    choice = ask_int("Parameter to sweep (number)", 1) - 1
    param_name = params_list[choice]
    type_hint = sweepable[param_name]

    raw_values = input(f"Values for {param_name} (comma-separated): ").strip()
    values = [_parse_sweep_value(v, type_hint) for v in raw_values.split(",")]

    configs: list[ExperimentConf | RandomForestExperimentConf | GridSearchConf] = [
        replace(base, **{param_name: val}) for val in values
    ]

    print(f"\n→ {len(configs)} experiments generated (sweep on {param_name})")
    return Experiments(experiments=configs, hypothesis=hypothesis)


# ── Single config builders ────────────────────────────────────────────


def build_single_config() -> (
    ExperimentConf | RandomForestExperimentConf | GridSearchConf
):

    print("\n===== Experiment configuration =====\n")

    model_choice = (
        input(
            "Model [logistic_regression/random_forest] (default logistic_regression): "
        ).strip()
        or "logistic_regression"
    )
    model = Model(model_choice)

    image_size = ask_int("Image size", 64)

    normalize = ask_bool("Normalize pixels", True)

    pca_components = ask_opt_int("PCA components", None)

    crop_factor = ask_int("Crop factor", 0)

    enhance_factor = ask_int("Enhance factor", 0)

    experiment_type = ask_bool("Is it GridSearch experiment ?", False)
    if experiment_type:
        return GridSearchConf.from_quizz(
            image_size=image_size,
            normalize_pixel=normalize,
            pca=pca_components,
            crop_factor=crop_factor,
            enhance_factor=enhance_factor,
            model=model,
        )

    if model == Model.random_forest:
        return _build_rf_config(
            image_size=image_size,
            normalize=normalize,
            pca_components=pca_components,
            crop_factor=crop_factor,
            enhance_factor=enhance_factor,
        )

    penalty = input("Penalty [l1/l2/elasticnet] (default l2): ").strip() or "l2"

    solver = input("Solver (default liblinear): ").strip() or "liblinear"

    max_iter = ask_int("Max iterations", 2000)

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = ask_float("l1_ratio", 0.5)

    C = ask_float("Regularization C", 1.0)

    class_weight = ask_bool("Class weight", False)

    return ExperimentConf(
        image_size=image_size,
        normalize=normalize,
        pca_components=pca_components,
        enhance_factor=enhance_factor,
        crop_factor=crop_factor,
        model=Model.logistic_regression,
        penalty=Penalty(penalty),
        solver=Solver(solver),
        l1_ratio=l1_ratio,
        regularization_c=C,
        class_weight=class_weight,
        max_iter=max_iter,
    )


def _build_rf_config(
    image_size: int,
    normalize: bool,
    pca_components: Optional[int],
    crop_factor: int,
    enhance_factor: int,
) -> RandomForestExperimentConf:
    n_estimators = ask_int("Number of trees (n_estimators)", 100)

    max_depth = ask_opt_int("Max depth (None = unlimited)", None)

    min_samples_split = ask_int("Min samples split", 2)

    min_samples_leaf = ask_int("Min samples leaf", 1)

    class_weight = ask_bool("Class weight balanced", False)

    return RandomForestExperimentConf(
        image_size=image_size,
        normalize=normalize,
        pca_components=pca_components,
        crop_factor=crop_factor,
        enhance_factor=enhance_factor,
        model=Model.random_forest,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
    )