from pathlib import Path
from typing import Callable, Optional, Dict

from t_dev_810.data import DatasetFile, load, resize_img, data_splitting, cropping

from .schema import ExperimentConf, GridSearchConf 

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = PROJECT_ROOT / "experiments.csv"


def runner(experiment_conf: ExperimentConf | GridSearchConf):
    dataset_file = load()
    if isinstance(experiment_conf, GridSearchConf):
        return


def _preprocess(
    experiment_conf: ExperimentConf | GridSearchConf, dataset_file: DatasetFile
):
    

def processing_callable(function:Callable, dataset_file: DatasetFile):


def mapper(experiment_conf: ExperimentConf | GridSearchConf, dataset_file: DatasetFile) -> Dict[str, Optional[Callable]]:
    return {
        'data_splitting': data_splitting,
        'image_size':resize_img,
        'cropping': 
        'pca_components': 
        

    }
    

def test(dataset_file: DatasetFile)->DatasetFile:
    return dataset_file


def gridsearch_runner(experiment_conf: GridSearchConf):
    pass


def experiment_runner(experiment_conf: ExperimentConf) -> None:
    pass
