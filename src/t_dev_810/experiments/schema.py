import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class Penalty(enum.Enum):
    l1 = "l1"
    l2 = "l2"
    elasticnet = "elasticnet"


class Solver(enum.Enum):
    lbfgs = "lbfgs"
    liblinear = "liblinear"
    saga = "saga"


class Model(enum.Enum):
    logistic_regression = "logistic_regression"


@dataclass
class ExperimentConf:
    image_size: int
    normalize: bool
    pca_components: Optional[int]
    crop_factor: int
    enhance_factor: int
    model: Model
    penalty: Penalty
    solver: Solver
    l1_ratio: Optional[float]  # between 0 and 1
    regularization_c: float  # between 0 and 100
    class_weight: bool
    max_iter: int
    hypothesis: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "normalize": self.normalize,
            "pca_components": self.pca_components,
            "crop_factor": self.crop_factor,
            "enhance_factor": self.enhance_factor,
            "model": self.model.value,
            "penalty": self.penalty.value,
            "solver": self.solver.value,
            "l1_ratio": self.l1_ratio,
            "regularization_c": self.regularization_c,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "hypothesis": self.hypothesis,
        }


@dataclass
class GridSearchConf:
    model: Model
    image_size: int
    normalize: bool
    pca_components: Optional[int]
    enhance_factor: int
    crop_factor: int
    hypothesis: str
    conf: List[Dict[str, Any]]

    @staticmethod
    def from_quizz(
        image_size: int,
        normalize_pixel: bool,
        enhance_factor: int,
        pca: Optional[int],
        crop_factor: int,
        hypothesis: str,
        model: Model = Model.logistic_regression,
    ) -> "GridSearchConf":
        gridsearch_conf: List[Dict[str, Any]]

        match model.name:
            case Model.logistic_regression.name:
                gridsearch_conf = LOGISTIC_REG_GRIDSEARCH_CONF
            case _:
                raise NotImplementedError("model not implemented yet")

        return GridSearchConf(
            model=model,
            image_size=image_size,
            normalize=normalize_pixel,
            enhance_factor=enhance_factor,
            pca_components=pca,
            crop_factor=crop_factor,
            hypothesis=hypothesis,
            conf=gridsearch_conf,
        )

    def to_json(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "normalize": self.normalize,
            "pca_components": self.pca_components,
            "crop_factor": self.crop_factor,
            "enhance_factor": self.enhance_factor,
            "model": self.model.value,
            "penalty": best_params.get("penalty"),
            "solver": best_params.get("solver"),
            "l1_ratio": best_params.get("l1_ratio"),
            "regularization_c": best_params.get("C"),
            "class_weight": best_params.get("class_weight"),
            "max_iter": best_params.get("max_iter"),
            "hypothesis": self.hypothesis,
        }


LOGISTIC_REG_GRIDSEARCH_CONF = [
    {
        "solver": ["lbfgs"],
        "max_iter": [2000],
        "l1_ratio": [0],
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"],
    },
    {
        "solver": ["liblinear"],
        "max_iter": [2000],
        "l1_ratio": [0, 1],
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"],
    },
    # {
    #     "solver": ["saga"],
    #     "max_iter": [6000],
    #     "l1_ratio": [0, 0.5, 1],
    #     "C": [0.01, 0.1, 1],
    #     "class_weight": [None, "balanced"],
    # },
]
