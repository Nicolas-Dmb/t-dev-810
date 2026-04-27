import sys

from sklearn.linear_model import LogisticRegression

from t_dev_810.data import data_splitting, load, load_image, normalize_pixel, resize_img
from t_dev_810.features.transforms import flatten_image
from t_dev_810.models import train_model
from t_dev_810.utils import save_model


def main():
    try:
        print("Loading dataset...")
        dataset_file = load()

        print("Preprocessing...")
        dataset_file = data_splitting(dataset_file)
        dataset_img = load_image(dataset_file)
        dataset_img = resize_img(dataset_img, image_size=64)
        dataset_data = flatten_image(dataset_img)
        dataset_data = normalize_pixel(dataset=dataset_data)

        print("Training model (exp 4: class_weight=balanced)...")
        model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
        )
        model = train_model(model, dataset_data)

        print("Exporting model...")
        path = save_model(model, "logistic_regression_balanced.joblib")
        print(f"Model saved to {path}")

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
