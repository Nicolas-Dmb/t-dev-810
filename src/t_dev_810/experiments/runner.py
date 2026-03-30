from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from t_dev_810.data import (
    DatasetData,
    DatasetFile,
    crop_dataset,
    data_splitting,
    enhance_constrast,
    load,
    load_image,
    normalize_pixel,
    pca,
    resize_img,
)
from t_dev_810.features.transforms import flatten_image
from t_dev_810.models import evaluate_model, predict_model, train_model
from t_dev_810.utils import save_result

from .schema import ExperimentConf, GridSearchConf, Model

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = PROJECT_ROOT / "experiments.csv"

PreprocessStep = Callable[[DatasetFile], Any]


def runner(experiment_conf: ExperimentConf | GridSearchConf):
    print("Starting experiment, loading dataset...")
    dataset_file = load()
    print("Dataset loaded, starting preprocessing...")
    processed_dataset = _preprocess(experiment_conf, dataset_file)
    print("Dataset preprocessed, starting training...")
    if isinstance(experiment_conf, GridSearchConf):
        config_dict, result_dict = gridsearch_runner(experiment_conf, processed_dataset)
    else:
        config_dict, result_dict = experiment_runner(experiment_conf, processed_dataset)

    print("Saving results...")
    save_result(config_dict, result_dict, experiment_conf.model.value)
    print("Experiment completed and results saved.")
    print(f"Results: {result_dict}")


def _preprocess(
    experiment_conf: ExperimentConf | GridSearchConf,
    dataset_file: DatasetFile,
) -> DatasetData:
    pipeline: list[Callable[[DatasetFile], Any]] = build_preprocess_pipeline(
        experiment_conf
    )

    for step in pipeline:
        dataset_file = step(dataset_file)

    return dataset_file


def build_preprocess_pipeline(
    experiment_conf: ExperimentConf | GridSearchConf,
) -> list[PreprocessStep]:
    pipeline: list[PreprocessStep] = []

    pipeline.append(lambda dataset: data_splitting(dataset))

    # Load dataset as images
    pipeline.append(lambda dataset: load_image(dataset))

    pipeline.append(
        lambda dataset: resize_img(
            dataset,
            image_size=experiment_conf.image_size,
        )
    )

    if experiment_conf.crop_factor > 0:
        pipeline.append(
            lambda dataset: crop_dataset(
                dataset,
                crop_factor=experiment_conf.crop_factor,
            )
        )

    if experiment_conf.enhance_factor > 0:
        pipeline.append(
            lambda dataset: enhance_constrast(
                dataset,
                enhance_factor=experiment_conf.enhance_factor,
            )
        )

    pipeline.append(lambda dataset: flatten_image(dataset))

    if experiment_conf.pca_components is not None:
        pipeline.append(
            lambda dataset: pca(
                dataset,
                n_components=experiment_conf.pca_components,
            )
        )

    if experiment_conf.normalize:
        pipeline.append(lambda dataset: normalize_pixel(dataset=dataset))

    return pipeline


def gridsearch_runner(
    experiment_conf: GridSearchConf, dataset: DatasetData
) -> Tuple[dict[str, Any], dict[str, Any]]:
    match experiment_conf.model:
        case Model.logistic_regression:
            model = LogisticRegression()
        case _:
            raise NotImplementedError("model not implemented yet")

    grid_search = GridSearchCV(model, experiment_conf.conf, cv=5, n_jobs=-1, verbose=3)

    model = train_model(grid_search, dataset)
    y_test, y_pred = predict_model(model, dataset)

    best_params: dict[str, Any] = model.best_params_
    config_dict: dict[str, Any] = experiment_conf.to_json(best_params)

    result_dict = evaluate_model(
        model=model,
        X_train=np.array([train.data for train in dataset.train]),
        y_train=[train.label for train in dataset.train],
        X_test=np.array([test.data for test in dataset.test]),
        y_test=y_test,
        y_pred=y_pred,
    )

    return config_dict, result_dict


def experiment_runner(
    experiment_conf: ExperimentConf, dataset: DatasetData
) -> Tuple[dict[str, Any], dict[str, Any]]:
    match experiment_conf.model:
        case Model.logistic_regression:
            model = LogisticRegression(max_iter=experiment_conf.max_iter)
        case _:
            raise NotImplementedError("model not implemented yet")

    config_dict = experiment_conf.to_json()

    model = train_model(model, dataset)
    y_test, y_pred = predict_model(model, dataset)

    result_dict = evaluate_model(
        model=model,
        X_train=np.array([train.data for train in dataset.train]),
        y_train=[train.label for train in dataset.train],
        X_test=np.array([test.data for test in dataset.test]),
        y_test=y_test,
        y_pred=y_pred,
    )

    return config_dict, result_dict
