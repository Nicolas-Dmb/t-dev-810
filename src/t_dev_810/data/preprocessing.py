from ast import List

from .schema import DatasetImg, ImageFile


def resize_img(size: int):
    # TODO
    pass


# TODO ; restart hir
def cropping(datasets: DatasetImg, crop_factor: int) -> DatasetImg:
    """Crop the image by removing 10% of the width and height from each side."""
    img_datasets = [List[ImageFile], List[ImageFile], List[ImageFile]]
    for index, dataset in enumerate([datasets.test, datasets.train, datasets.val]):
        
        for 
        width, height = img_file.img.size

        margin_w = width * crop_factor
        margin_h = height * crop_factor



    return DatasetImg(
        img_file.img.crop((margin_w, margin_h, width - margin_w, height - margin_h)),
        img_file.path,
    )
