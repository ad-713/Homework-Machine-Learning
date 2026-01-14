# Comparative Analysis of Evolutionary, Active, and Ensemble Learning for Higgs Boson Detection

## 1. Introduction

The detection of the Higgs Boson particle represents a fundamental challenge in High Energy Physics, relying heavily on the analysis of collision events produced by particle accelerators. This project addresses the classification problem of distinguishing between the Higgs Boson signal ($s$) and background noise ($b$) using the **ATLAS Higgs Boson Challenge 2014 dataset**.

The primary objective is to evaluate and synthesize three distinct machine learning paradigms:
1.  **Evolutionary Learning:** Utilizing Genetic Programming (GP) to evolve mathematical expressions for classification.
2.  **Active Learning (AL):** Integrating uncertainty sampling to optimize the training process and reduce data dependency.
3.  **Ensemble Learning (EL):** Aggregating predictions from multiple evolved individuals to improve robustness and precision.

This report details the implementation methodology, presents quantitative results regarding performance and computational efficiency, and discusses the trade-offs inherent in each approach.

## 2. Methodology

The implementation leverages a modular Python architecture, utilizing `DEAP` for evolutionary algorithms, `modAL` for active learning, and `scikit-learn` for preprocessing and evaluation.

### 2.1. Dataset Preprocessing
The raw dataset, containing physical measurements from simulated collision events, underwent a rigorous cleaning and transformation pipeline:
- **Missing Value Imputation:** Values marked as `-999.0` were identified and imputed using the median of their respective columns to preserve statistical robustness against outliers.
- **Label Encoding:** Categorical targets were mapped to binary values: Signal ($1$) and Background ($0$).
- **Normalization:** Feature scaling was performed using `StandardScaler` to standardize features to zero mean and unit variance.
- **Dimensionality Reduction:** Principal Component Analysis (PCA) reduced the feature space from 30 to 20 dimensions, retaining 95% of the variance to improve computational efficiency.

### 2.2. Evolutionary Learning (Genetic Programming)
We implemented a Genetic Programming classifier where individuals represent mathematical formulas evolved to discriminate between signal and background.
- **Primitive Set:** Basic arithmetic operations: $\{+, -, *\}$.
- **Terminal Set:** The 20 principal components derived from the dataset features.
- **Fitness Function:** Classification accuracy on the training set.
- **Parameters:** Population size of 50, evolved over 10 generations.

### 2.3. Active Learning Integration
To address the computational cost of evaluating fitness on large datasets, we integrated an Active Learning loop using **Uncertainty Sampling**.
- **Strategy:** The algorithm iteratively queries samples from an unlabeled pool where the current model is least confident.
- **Probability Mapping:** GP outputs ($f(x)$) were mapped to probabilities using the logistic sigmoid function:
  $$P(y=1|x) = \frac{1}{1 + e^{-f(x)}}$$
- **Workflow:** The training set was updated every 3 generations with the 20 most "uncertain" samples, allowing the GP to focus on the decision boundary.

### 2.4. Ensemble Learning
The Ensemble Learning component utilizes the diversity of the final GP population to refine predictions.
- **Selection:** The top 10 performing individuals from the final generation were selected.
- **Voting Mechanism:** We compared Hard Voting (majority rule), Soft Voting (averaged probabilities), and Weighted Voting (fitness-weighted). The final results reported utilize a voting mechanism to balance precision and recall.

## 3. Results

The performance of the Standard GA, Active Learning augmented GA (GA+AL), and the Ensemble method (GA+AL+EL) was evaluated on a held-out test set. A Random Sampling baseline was included for comparison.

### 3.1. Quantitative Metrics

The following table summarizes the key performance indicators. Notably, the Active Learning approach achieved the highest Accuracy and F1-Score while requiring significantly less training time.

| Method | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GA (Baseline)** | 0.6907 | 0.5463 | 0.2319 | 0.3256 | 5.30 |
| **Random Baseline** | 0.6453 | 0.4222 | 0.2754 | 0.3333 | 0.59 |
| **GA + AL** | **0.7120** | 0.5389 | **0.7308** | **0.6204** | **0.59** |
| **GA + AL + EL** | 0.7107 | **0.5675** | 0.4265 | 0.4870 | 0.68 |

### 3.2. Visual Analysis

**Performance Comparison:**
As illustrated in Figure 1, the GA+AL approach (green) significantly outperforms the baseline GA (blue) in terms of Recall and F1-Score, demonstrating the effectiveness of targeted sampling.

![Metrics Comparison](experiment/plots/metrics_comparison.png)
*Figure 1: Comparative bar chart of Accuracy, Precision, Recall, and F1-Score across all methods.*

**Computational Efficiency:**
Figure 2 highlights the dramatic reduction in computational cost. The Active Learning approach yields a speedup of approximately 9x compared to the full-batch GA, making it highly efficient for iterative experimentation.

![Time Comparison](experiment/plots/time_comparison.png)
*Figure 2: Training time comparison (in seconds).*

**Ensemble Performance:**
The ensemble method's ability to distinguish classes is further detailed in the Confusion Matrix (Figure 3) and ROC Curve (Figure 4). While the ensemble sacrificed some recall compared to the single best AL individual, it achieved the highest precision, reducing false positives.

![Confusion Matrix](experiment/plots/confusion_matrix.png)
*Figure 3: Confusion Matrix for the Ensemble Classifier.*

![ROC Curve](experiment/plots/roc_curve.png)
*Figure 4: Receiver Operating Characteristic (ROC) Curve.*

## 4. Discussion

### 4.1. Analysis of Approaches
The results demonstrate a clear hierarchy in efficiency and efficacy:
- **Standard GA:** While establishing a baseline accuracy of ~69%, the standard GA suffered from poor recall (0.23) and high computational cost (5.3s). It likely converged to a local optimum favoring the majority background class.
- **Active Learning (GA+AL):** This was the most successful approach, achieving the highest **Accuracy (71.2%)** and **F1-Score (0.62)**. By focusing training on uncertain samples near the decision boundary, the model learned to identify the signal class much more effectively (Recall increased to 0.73). Crucially, this was achieved using less than 10% of the training data, validating the hypothesis that data *quality* matters more than *quantity* in this domain.
- **Ensemble Learning (GA+AL+EL):** The ensemble method offered a trade-off. It achieved the highest **Precision (0.5675)**, making it the preferred choice if the cost of False Positives is high. However, the voting mechanism dampened the high recall achieved by the single best AL individual, resulting in a lower overall F1-score.

### 4.2. Limitations and Future Work
- **Population Size:** The experiments were constrained to a population of 50 and 10 generations. Larger populations could potentially evolve more complex features.
- **Primitive Set:** The set was limited to basic arithmetic. Including trigonometric or logarithmic functions could allow the GP to model more complex physical laws governing particle physics.
- **Stability:** Genetic Programming is stochastic. While Active Learning improved average performance, the variance between runs remains a factor to consider in deployment.

In conclusion, combining Evolutionary Algorithms with Active Learning provides a powerful framework for High Energy Physics classification, delivering superior performance at a fraction of the computational cost of traditional methods.
