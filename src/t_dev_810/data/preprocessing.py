from ast import List

from .schema import DatasetImg, ImageFile


def resize_img(dataset: DatasetImg, image_size: int) -> DatasetImg:
    """Resize the image to the given size."""
    
    return DatasetImg(
        test=[ImageFile(data=img_file.data.resize((image_size, image_size)), label=img_file.label) for img_file in dataset.test],
        train=[ImageFile(data=img_file.data.resize((image_size, image_size)), label=img_file.label) for img_file in dataset.train],
        val=[ImageFile(data=img_file.data.resize((image_size, image_size)), label=img_file.label) for img_file in dataset.val],
    )


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
