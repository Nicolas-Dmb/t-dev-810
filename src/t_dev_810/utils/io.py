import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de utils/ à t_dev_810/

VERSION_FILE = BASE_DIR / "version.json"


def save_result(
    config_dict: dict[str, Any],
    result_dict: dict[str, Any],
    model_name: str,
) -> None:
    with open(VERSION_FILE, "r+") as f:
        file = f.read()
        if file.strip():
            version_data = json.loads(file)
        else:
            version_data = {"versions": {}}
        version_data["versions"][datetime.now().isoformat()] = {
            "envs": config_dict,
            "results": result_dict,
            "model": model_name,
        }
        f.seek(0)
        f.truncate()
        json.dump(version_data, f, indent=4)


def load_version_json() -> Dict[str, Any]:
    """Load version.json."""
    with open(VERSION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)
