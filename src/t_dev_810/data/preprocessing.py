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


def pca(dataset: DatasetData, n_components: int) -> DatasetData:
    """Apply PCA on the whole dataset using train split only."""

    X_train = np.array([img.data for img in dataset.train])
    y_train = [img.label for img in dataset.train]

    X_val = np.array([img.data for img in dataset.val])
    y_val = [img.label for img in dataset.val]

    X_test = np.array([img.data for img in dataset.test])
    y_test = [img.label for img in dataset.test]

    pca_model = PCA(n_components=n_components)

    X_train_pca = pca_model.fit_transform(X_train)
    X_val_pca = pca_model.transform(X_val)
    X_test_pca = pca_model.transform(X_test)

    return DatasetData(
        train=[
            ImageData(data=X_train_pca[i], label=y_train[i])
            for i in range(len(y_train))
        ],
        val=[ImageData(data=X_val_pca[i], label=y_val[i]) for i in range(len(y_val))],
        test=[
            ImageData(data=X_test_pca[i], label=y_test[i]) for i in range(len(y_test))
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
