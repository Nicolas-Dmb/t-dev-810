import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de utils/ à t_dev_810/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VERSION_FILE = BASE_DIR / "version.json"
MODEL_DIR = PROJECT_ROOT / "models"


def save_result(
    results: list[tuple[dict[str, Any], dict[str, Any], str]], hypothesis: str
) -> None:
    """Save experiment results to version.json."""
    with open(VERSION_FILE, "r+") as f:
        file = f.read()
        if file.strip():
            version_data = json.loads(file)
        else:
            version_data = {"versions": {}}
        version_data["versions"][datetime.now().isoformat()] = {
            "hypothesis": hypothesis,
            "results": [
                {
                    "config": config_dict,
                    "metrics": result_dict,
                    "model": model,
                }
                for config_dict, result_dict, model in results
            ],
        }
        f.seek(0)
        f.truncate()
        json.dump(version_data, f, indent=4)


def load_version_json() -> Dict[str, Any]:
    """Load version.json."""
    with open(VERSION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model: Any, filename: str) -> Path:
    """Save a trained model to the models/ directory."""
    MODEL_DIR.mkdir(exist_ok=True)
    filepath = MODEL_DIR / filename
    joblib.dump(model, filepath)
    return filepath
