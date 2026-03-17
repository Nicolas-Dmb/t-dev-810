from pathlib import Path
from typing import Any, Callable

from t_dev_810.data import (
    DatasetFile,
    crop_dataset,
    data_splitting,
    enhance_constrast,
    flatten_image,
    load,
    load_image,
    normalize_pixel,
    pca,
    resize_img,
)

from .schema import ExperimentConf, GridSearchConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = PROJECT_ROOT / "experiments.csv"

PreprocessStep = Callable[[DatasetFile], Any]


def runner(experiment_conf: ExperimentConf | GridSearchConf):
    dataset_file = load()
    if isinstance(experiment_conf, GridSearchConf):
        return


def _preprocess(
    experiment_conf: ExperimentConf | GridSearchConf,
    dataset_file: DatasetFile,
) -> DatasetFile | DatasetImg:
    pipeline = build_preprocess_pipeline(experiment_conf)

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

    # Load flatten Image
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


def test(dataset_file: DatasetFile) -> DatasetFile:
    return dataset_file


def gridsearch_runner(experiment_conf: GridSearchConf):
    pass


def experiment_runner(experiment_conf: ExperimentConf) -> None:
    pass
