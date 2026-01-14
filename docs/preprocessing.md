# Step 1: Dataset Preprocessing

This document details the preprocessing pipeline implemented for the Higgs Boson Detection dataset. Preprocessing is a critical step to ensure data quality and prepare the features for the subsequent learning paradigms (Evolutionary, Active, and Ensemble learning).

## Overview

The preprocessing pipeline is orchestrated by [`main_preprocessing.py`](../main_preprocessing.py) and utilizes modular functions defined in the `src/` directory.

### Key Steps

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Missing Value Handling (SimpleImputer)**
4. **Label Encoding**
5. **Stratified Data Splitting**
6. **Feature Scaling (PowerTransformer)**
7. **Dimensionality Reduction (PCA)**

---

## 1. Data Loading and Cleaning

The raw data is loaded from `data/atlas-higgs-challenge-2014-v2.csv` using [`src/data_loader.py`](../src/data_loader.py).

### Initial Cleanup
Before processing, superfluous columns `EventId`, `Weight`, `KaggleSet`, and `KaggleWeight` are removed to focus on the feature set.

---

## 2. Exploratory Data Analysis (EDA) & Decisions

An EDA was performed to understand the data characteristics and inform preprocessing decisions.

### Class Balance
*   **Background (b):** 65.83%
*   **Signal (s):** 34.17%
*   **Imbalance Ratio:** 1.93

**Decision:** The dataset is moderately imbalanced. We implemented **Stratified Splitting** to ensure this ratio is maintained in both training and testing sets.

### Outlier Detection (IQR Method)
Significant outliers were detected in several features, often related to the imputation of missing physical quantities (originally -999.0).

| Feature | Outlier Count | Percentage |
| :--- | :--- | :--- |
| `DER_mass_jet_jet` | 237,984 | 29.08% |
| `PRI_jet_subleading_pt` | 237,980 | 29.08% |
| `PRI_jet_leading_eta` | 184,023 | 22.49% |
| ... | ... | ... |

**Decision:**
1.  **PowerTransformer (Yeo-Johnson):** Initially, standard scaling and robust scaling led to unbalanced variance in PCA due to extreme heavy tails. We switched to **`PowerTransformer`**, which applies a non-linear transformation to stabilize variance and make features more Gaussian. This prevents extreme outliers from dominating the PCA components.
2.  **PCA Variance:** To ensure the maximum amount of information is preserved for the models, the PCA variance retention was set to **95%**.

---

## 3. Handling Missing Values

The Higgs Boson dataset uses `-999.0` as a placeholder for missing values.
*   **Implementation:** We utilize Scikit-learn's `SimpleImputer` with the **median** strategy.
*   **Justification:** Median imputation is robust to the skewed distributions observed in the physical variables.

---

## 4. Feature Engineering & Preparation

Implemented in [`src/preprocessing.py`](../src/preprocessing.py).

### Label Encoding
The target variable `Label` contains two classes:
- `s` (Signal) -> `1`
- `b` (Background) -> `0`

### Stratified Data Splitting
The dataset is split into training and testing sets using an **70/30 ratio**.
- **Stratification:** Enabled to preserve the 1.93 imbalance ratio.
- **Random Seed:** `42` for reproducibility.

---

## 5. Normalization and Dimensionality Reduction

### Feature Scaling (PowerTransformer)
We apply `PowerTransformer` (using the Yeo-Johnson method) to transform features. This method stabilizes variance and minimizes the impact of heavy tails (outliers) by mapping the data to a more Gaussian distribution. This is essential for PCA, ensuring that components are not dominated by a few extreme values.

### Dimensionality Reduction (PCA)
To improve computational efficiency and reduce noise:
- **Variance Retained:** **95%**.
- **Benefit:** Reduces the feature space while preserving almost all original information, filtering out only the most insignificant noise components.

---

## 6. Output Artifacts

After running the pipeline, the following artifacts are generated in the `processed_data/` directory:

| Artifact | Description |
| --- | --- |
| `train_data.npz` | Compressed NumPy archive containing `X_train`, `y_train`, and `w_train`. |
| `test_data.npz` | Compressed NumPy archive containing `X_test`, `y_test`, and `w_test`. |
| `scaler.joblib` | The fitted `PowerTransformer` object. |
| `pca.joblib` | The fitted `PCA` object. |
| `label_encoder.joblib` | The fitted `LabelEncoder` object. |

## How to Run

To execute the preprocessing pipeline:

```bash
python main_preprocessing.py
```
