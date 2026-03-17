import numpy as np
from PIL import ImageEnhance
from sklearn.decomposition import PCA

from .schema import DatasetData, DatasetImg, ImageData, ImageFile


def resize_img(dataset: DatasetImg, image_size: int) -> DatasetImg:
    """Resize the image to the given size."""

    return DatasetImg(
        test=[
            ImageFile(
                data=img_file.data.resize((image_size, image_size)),
                label=img_file.label,
            )
            for img_file in dataset.test
        ],
        train=[
            ImageFile(
                data=img_file.data.resize((image_size, image_size)),
                label=img_file.label,
            )
            for img_file in dataset.train
        ],
        val=[
            ImageFile(
                data=img_file.data.resize((image_size, image_size)),
                label=img_file.label,
            )
            for img_file in dataset.val
        ],
    )


def crop_dataset(datasets: DatasetImg, crop_factor: int) -> DatasetImg:
    """Crop the image by removing 10% of the width and height from each side."""
    return DatasetImg(
        test=[_cropping(img_file, crop_factor) for img_file in datasets.test],
        train=[_cropping(img_file, crop_factor) for img_file in datasets.train],
        val=[_cropping(img_file, crop_factor) for img_file in datasets.val],
    )


def _cropping(img: ImageFile, crop_factor: int) -> ImageFile:
    return ImageFile(
        data=img.data.crop(
            (
                crop_factor,
                crop_factor,
                img.data.width - crop_factor,
                img.data.height - crop_factor,
            )
        ),
        label=img.label,
    )


def flatten_image(dataset: DatasetImg) -> DatasetData:
    """Flatten the image to a 1D array."""
    return DatasetData(
        test=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.test
        ],
        train=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.train
        ],
        val=[
            ImageData(
                data=np.array(img_file.data).flatten(),
                label=img_file.label,
            )
            for img_file in dataset.val
        ],
    )


def pca(dataset: DatasetData, n_components: int) -> DatasetData:
    """Apply PCA to the dataset."""
    pca = PCA(n_components=n_components)

    return DatasetData(
        test=[
            ImageData(
                data=pca.fit_transform(data.data),
                label=data.label,
            )
            for data in dataset.test
        ],
        train=[
            ImageData(
                data=pca.fit_transform(data.data),
                label=data.label,
            )
            for data in dataset.train
        ],
        val=[
            ImageData(
                data=pca.fit_transform(data.data),
                label=data.label,
            )
            for data in dataset.val
        ],
    )


def normalize_pixel(dataset: DatasetData) -> DatasetData:
    """Normalize the pixel values to be between 0 and 1."""
    return DatasetData(
        test=[
            ImageData(
                data=data.data / 255.0,
                label=data.label,
            )
            for data in dataset.test
        ],
        train=[
            ImageData(
                data=data.data / 255.0,
                label=data.label,
            )
            for data in dataset.train
        ],
        val=[
            ImageData(
                data=data.data / 255.0,
                label=data.label,
            )
            for data in dataset.val
        ],
    )


def enhance_constrast(dataset: DatasetImg, enhance_factor: int) -> DatasetImg:
    """Enhance the contrast of the image."""
    return DatasetImg(
        test=[_enhance_contrast(img_file, enhance_factor) for img_file in dataset.test],
        train=[
            _enhance_contrast(img_file, enhance_factor) for img_file in dataset.train
        ],
        val=[_enhance_contrast(img_file, enhance_factor) for img_file in dataset.val],
    )


def _enhance_contrast(img: ImageFile, enhance_factor: int) -> ImageFile:
    enhancer = ImageEnhance.Contrast(img.data)
    enhanced_img = enhancer.enhance(enhance_factor)

    return ImageFile(
        data=enhanced_img,
        label=img.label,
    )


def data_augmentation(dataset: DatasetImg) -> DatasetImg:
    """Apply data augmentation to the dataset."""
    # TODO: Implement data augmentation logic
