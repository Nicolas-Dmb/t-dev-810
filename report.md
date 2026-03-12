# Dataset Visualization

## Dataset Overview

This dataset is divided into three subsets:

- **train**
- **validation**
- **test**

Each subset contains two classes of JPEG chest X-ray images:

- **Normal**
- **Pneumonia**

## Data Distribution

The class distribution is unbalanced. The training set contains many more images than the validation set, and the pneumonia class is more represented than the normal class, especially in the training set.

The validation set is particularly small, with only a few images per class, which makes it difficult to use reliably for model selection.

## Image Sizes

The images do not all have the same dimensions, so they cannot be used directly by a standard machine learning model.

## Example Image

Each image is a grayscale chest radiograph.

---

# Preprocessing

Before training the model, several preprocessing steps are required.

## 1. Rebalancing the Train/Validation Split

The original validation set is too small to be representative. To obtain a more reliable validation dataset, part of the training set is moved into the validation set.

The test set must remain untouched, since it is supposed to represent unseen data used only for final evaluation.

## 2. Resizing Images

Machine learning models require all samples to have the same number of features. Since the original images have different sizes, they must be resized to a common resolution.

Changing the image size also changes the number of features:

- a **smaller size** reduces the number of features and may help reduce overfitting
- a **larger size** preserves more visual information but may increase the risk of overfitting

## 3. Flattening Images into 1D Vectors

A logistic regression model cannot directly process 2D images. Each image must therefore be transformed into a 1D vector.

For a grayscale image, each pixel is represented by an intensity value between **0 and 255**. After flattening, the image becomes a sequence of numerical features that can be used as input for the model.

## 4. Pixel Normalization

Pixel values can also be normalized from **[0, 255]** to **[0, 1]**. This can improve numerical stability and make training easier for some algorithms.

## 5. Cropping Borders

Some images contain large dark borders or irrelevant information near the edges. Cropping can reduce useless background noise and force the model to focus more on the lungs.

---

# Model Choice

## Logistic Regression

I chose **logistic regression** as my first classification algorithm because it is one of the first supervised learning models I studied.

It is simple, interpretable, and easy to use as a baseline model. Even though it is not specifically designed for image data, it is a good starting point to understand the impact of preprocessing and dataset preparation.

---

# Evaluation Metrics

To evaluate the model, I used several metrics:

- **Accuracy**: proportion of correct predictions among all predictions.
- **Recall**: proportion of actual pneumonia cases correctly detected.
- **Precision**: proportion of predicted pneumonia cases that are actually pneumonia.
- **F1-score**: harmonic mean of precision and recall.
- **AUC**: measures the model’s ability to rank positive samples above negative ones.

Recall is particularly important in this project because missing a pneumonia case is more serious than incorrectly flagging a normal image as pneumonia.

However, recall alone is not enough. Since the dataset is imbalanced, a model could obtain a high recall by predicting pneumonia too often. That is why precision and F1-score are also useful.

---

# Experiments

## Experiment 1

### Hypothesis

First, I tried a basic pipeline using only image resizing, without respecting the original separation between train, validation, and test sets.

### Parameters and Preprocessing

- image size: **256 × 256**
- dataset split: mixed **train, validation, and test**, then redistributed into **60/20/20**

### Results

- Accuracy: **0.914**
- AUC: **0.955**
- Recall: **0.945**
- Train AUC: **0.973**

### Analysis

The results were good, but this experiment was methodologically incorrect because I mixed the original test set with the other subsets. The test set should remain isolated until the final evaluation.

---

## Experiment 2

### Hypothesis

I expected that keeping the original test set separate would produce worse but more realistic results.

### Parameters and Preprocessing

- image size: **256 × 256**
- split: original **test set kept separate**
- train/validation split: **80/20**

### Results

- Accuracy: **0.741**
- AUC: **0.754**
- Recall: **0.982**
- Train AUC: **0.986**

### Analysis

The test performance dropped significantly, while the training score remained very high. This suggests clear **overfitting**.

---

## Experiment 3

### Hypothesis

Some images contain more dark background than others. I expected pixel normalization to reduce this variation and improve the results.

### Parameters and Preprocessing

- image size: **256 × 256**
- test set kept separate
- train/validation split: **80/20**
- pixel normalization from **0–255** to **0–1**

### Results

- Accuracy: **0.746**
- AUC: **0.895**
- Recall: **0.987**
- Train AUC: **0.989**

### Analysis

Normalization did not significantly improve the results. The model still seemed to **overfit**.

---

## Experiment 4

### Hypothesis

I tested different image sizes to reduce overfitting. I also tried contrast enhancement to emphasize useful grayscale details in the lungs.

### Parameters and Preprocessing

- image size: **64 × 64** to **500 × 500**
- test set kept separate
- train/validation split: **80/20**
- pixel normalization
- contrast enhancement factor: **0.5 to 2**

### Results

No configuration performed better than the previous experiment.

### Analysis

These preprocessing changes were not sufficient to improve generalization. I needed another way to reduce overfitting.

---

## Experiment 5

### Hypothesis

Cropping the borders might reduce useless pixels, remove noise such as annotations or background, and help the model focus on the lungs.

### Parameters and Preprocessing

- test set kept separate
- train/validation split: **80/20**
- border cropping: **10% of the image**

### Results

- Accuracy: **0.746**
- AUC: **0.828**
- Recall: **0.982**
- Precision: **0.717**
- F1-score: **0.829**
- Train AUC: **0.979**

### Analysis

I added precision and F1-score because the dataset is imbalanced. A high recall alone would not be enough to prove that the model is good.

Cropping did not improve the overall performance. Overfitting was still present.

---

## Experiment 6

### Hypothesis

I tried to reduce overfitting by tuning the logistic regression hyperparameters:

- regularization to limit coefficient magnitude
- class weights to compensate for the class imbalance

### Parameters

- penalty: **L1, L2, elastic net**
- class weight: **None, balanced**

### Best Configuration

- `C = 1`
- `class_weight = balanced`
- `l1_ratio = 0`
- `solver = liblinear`

### Results

- Accuracy: **0.750**
- AUC: **0.893**
- Recall: **0.719**
- Precision: **0.717**
- F1-score: **0.831**
- Train AUC: **0.987**

### Analysis

This experiment slightly reduced overfitting, but the difference between training and test performance remained large. In addition, recall dropped compared with previous experiments, which is a problem for pneumonia detection.



---
## Experiment 7

### Hypothesis

To reduce overfitting, I applied **Principal Component Analysis (PCA)** to reduce the number of features.  
By compressing the original pixel space into a smaller set of principal components, the model may generalize better and rely less on noisy or redundant information.

### Parameters

- PCA components: **100**
- penalty: **L1**
- class_weight: **balanced**
- C: **1**

### Results

- Accuracy: **0.743**
- AUC: **0.888**
- Recall: **0.982**
- Precision: **0.714**
- F1-score: **0.827**
- Train AUC: **0.985**

### Analysis

Applying PCA slightly reduced overfitting, but the difference between training and test performance remained significant.  
While the model still detects most pneumonia cases (high recall), the overall performance did not significantly improve compared to previous experiments.

This suggests that dimensionality reduction alone is not sufficient to solve the overfitting issue in this dataset.

---

# Conclusion

This project showed that a simple **logistic regression** model can achieve acceptable results on chest X-ray classification, but it struggles to generalize well.

The main issue is **overfitting**:

- training performance remains very high
- test performance is significantly lower

Several preprocessing strategies were tested:

- resizing
- normalization
- contrast enhancement
- cropping
- regularization
- class weighting

These changes slightly affected performance but did not solve the generalization problem.

This suggests that logistic regression is probably too limited for this image classification task. A more suitable model for image data, such as a **convolutional neural network (CNN)**, would likely perform better.

## Experiments Summary

| Exp | Image Size | Preprocessing | Regularization | Accuracy | AUC | Recall | Precision | F1 | Train AUC |
|----|----|----|----|----|----|----|----|----|----|
| 1 | 256x256 | Mixed dataset | None | 0.914 | 0.955 | 0.945 | - | - | 0.973 |
| 2 | 256x256 | Proper split | None | 0.741 | 0.754 | 0.982 | - | - | 0.986 |
| 3 | 256x256 | Pixel normalization | None | 0.746 | 0.895 | 0.987 | - | - | 0.989 |
| 4 | 64–500 | Resize + contrast | None | - | - | - | - | - | - |
| 5 | 256x256 | Crop 10% | None | 0.746 | 0.828 | 0.982 | 0.717 | 0.829 | 0.979 |
| 6 | 256x256 | Normalization | L2 + balanced | 0.750 | 0.893 | 0.719 | 0.717 | 0.831 | 0.987 |
| 7 | 128x128 | Normalization & PCA | L2 + balanced | 0.74 | 0.88 | 0.98 | 0.71 | 0.82 | 0.98 | 