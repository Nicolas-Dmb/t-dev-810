from pathlib import Path
from typing import Callable

from t_dev_810.data import (
    DatasetFile,
    cropping,
    data_splitting,
    load,
    load_image,
    resize_img,
)

from .schema import ExperimentConf, GridSearchConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = PROJECT_ROOT / "experiments.csv"

PreprocessStep = Callable[[DatasetFile], DatasetFile]


def runner(experiment_conf: ExperimentConf | GridSearchConf):
    dataset_file = load()
    if isinstance(experiment_conf, GridSearchConf):
        return


def _preprocess(
    experiment_conf: ExperimentConf | GridSearchConf,
    dataset_file: DatasetFile,
) -> DatasetFile:
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

    if experiment_conf.crop_factor > 0:
        pipeline.append(
            lambda dataset: cropping(
                dataset,
                crop_factor=experiment_conf.crop_factor,
            )
        )

    if experiment_conf.image_size is not None:
        pipeline.append(
            lambda dataset: resize_img(
                dataset,
                image_size=experiment_conf.image_size,
            )
        )

    return pipeline


def test(dataset_file: DatasetFile) -> DatasetFile:
    return dataset_file


def gridsearch_runner(experiment_conf: GridSearchConf):
    pass


def experiment_runner(experiment_conf: ExperimentConf) -> None:
    pass
