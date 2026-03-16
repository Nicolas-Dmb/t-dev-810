from dataclasses import dataclass

from PIL import Image


@dataclass
class ImageFile:
    img: Image.Image
    is_pneumonia: bool
