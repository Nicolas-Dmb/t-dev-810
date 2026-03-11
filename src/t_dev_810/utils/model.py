from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class ImageFile:
    img: Image.Image
    path: Path
