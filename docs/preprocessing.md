# Step 1: Dataset Preprocessing

This document details the preprocessing pipeline implemented for the Higgs Boson Detection dataset. Preprocessing is a critical step to ensure data quality and prepare the features for the subsequent learning paradigms (Evolutionary, Active, and Ensemble learning).

## Overview

The preprocessing pipeline is orchestrated by [`main_preprocessing.py`](../main_preprocessing.py) and utilizes modular functions defined in the `src/` directory.

### Key Steps

1. **Data Loading**
2. **Missing Value Handling**
3. **Label Encoding**
4. **Data Splitting**
5. **Feature Scaling (Normalization)**
6. **Dimensionality Reduction (PCA)**

---

## 1. Data Loading and Cleaning

The raw data is loaded from `data/atlas-higgs-challenge-2014-v2.csv` using [`src/data_loader.py`](../src/data_loader.py).

### Handling Missing Values
The Higgs Boson dataset uses `-999.0` as a placeholder for missing values. These are addressed in the `clean_missing_values()` function:
- **Detection**: All instances of `-999.0` are replaced with `NaN`.
- **Imputation**: Missing values in numeric feature columns are imputed using the **median** of the respective column. Median imputation was chosen over mean to provide robustness against potential outliers in the physical measurements.

## 2. Feature Engineering & Preparation

Implemented in [`src/preprocessing.py`](../src/preprocessing.py).

### Label Encoding
The target variable `Label` contains two classes:
- `s` (Signal)
- `b` (Background)

The `encode_labels()` function converts these categorical labels into numeric values: `1` for signal and `0` for background, making them compatible with Scikit-learn classifiers.

### Data Splitting
The dataset is split into training and testing sets using an **80/20 ratio**.
- **Random Seed**: `42` is used to ensure reproducibility across different runs.
- **Weights**: The dataset includes a `Weight` column which is preserved and split alongside the features and labels, as it is often required for calculating the Approximate Median Significance (AMS) metric.

---

## 3. Normalization and Dimensionality Reduction

### Feature Scaling
To ensure uniform scaling across all physical measurements (which vary significantly in magnitude), we apply `StandardScaler`. This transforms features to have a mean of 0 and a standard deviation of 1.

### Dimensionality Reduction (PCA)
To improve computational efficiency and reduce noise, Principal Component Analysis (PCA) is applied:
- **Variance Retained**: 95% of the total variance is preserved.
- **Result**: In the initial run, the feature space was reduced from **30 to 20 dimensions**.

---

## 4. Output Artifacts

After running the pipeline, the following artifacts are generated in the `processed_data/` directory:

| Artifact | Description |
| --- | --- |
| `train_data.npz` | Compressed NumPy archive containing `X_train`, `y_train`, and `w_train`. |
| `test_data.npz` | Compressed NumPy archive containing `X_test`, `y_test`, and `w_test`. |
| `scaler.joblib` | The fitted `StandardScaler` object for future inference. |
| `pca.joblib` | The fitted `PCA` object for future inference. |
| `label_encoder.joblib` | The fitted `LabelEncoder` object. |

## How to Run

To execute the preprocessing pipeline:

```bash
python main_preprocessing.py
```
