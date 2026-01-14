# Higgs Boson Detection: A Comparative Study of Evolutionary, Active, and Ensemble Learning Paradigms

## 1. Introduction

The discovery of the Higgs Boson particle is a landmark achievement in High Energy Physics. Detecting the signal of the Higgs Boson against a vast background of other physical processes is a complex classification challenge characterized by noise and class imbalance. This study aims to evaluate and analyze the performance of three machine learning paradigms—**Evolutionary Learning**, **Active Learning**, and **Ensemble Learning**—applied to the [Higgs Boson Detection dataset](https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz).

The primary objectives of this research are:
1.  To implement a Genetic Programming (GP) classifier using the DEAP library to evolve mathematical models for classification.
2.  To integrate Active Learning strategies to optimize the training process by dynamically sampling informative data points.
3.  To construct an Ensemble classifier from the evolved population to improve generalization and robustness.
4.  To perform a comparative analysis of these approaches regarding classification performance (Accuracy, F1-Score) and computational efficiency.

## 2. Methodology

### 2.1 Dataset Preprocessing
The raw dataset undergoes a rigorous preprocessing pipeline detailed in [`docs/preprocessing.md`](docs/preprocessing.md) and implemented in [`src/preprocessing.py`](src/preprocessing.py). This ensures data quality and compatibility with the learning algorithms.

*   **Missing Values**: Occurrences of `-999.0` are identified as missing data and are imputed using the column median. This method was chosen to ensure robustness against outliers in the physical measurements.
*   **Normalization**: Features are scaled to zero mean and unit variance using `StandardScaler`, ensuring that features with larger magnitudes do not dominate the evolutionary process.
*   **Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to reduce the feature space from 30 to 20 dimensions, preserving 95% of the total variance to improve computational efficiency.
*   **Data Splitting**: The data is partitioned into an 80/20 train-test split with a fixed random seed for reproducibility.

### 2.2 Evolutionary Learning (Genetic Programming)
We employ Genetic Programming (GP) to evolve mathematical expressions that classify events as signal ($s$) or background ($b$). The implementation utilizes the [`src/genetic_programming.py`](src/genetic_programming.py) module.

*   **Primitives**: The function set is restricted to basic arithmetic operations to maintain interpretability:
    $$F = \{+, -, \times\}$$
*   **Terminals**: The scaled physical features (represented as `ARG0` through `ARG29`) serve as input leaves for the expression trees.
*   **Fitness Function**: Individuals are evaluated based on their classification accuracy on the training set. The raw output of the expression tree, $f(x)$, is thresholded at 0 to determine the class:
    $$ \hat{y} = \begin{cases} 1 (Signal) & \text{if } f(x) > 0 \\ 0 (Background) & \text{if } f(x) \leq 0 \end{cases} $$

### 2.3 Active Learning Integration
To address the high computational cost of evaluating individuals on large datasets, we integrate Active Learning using **Uncertainty Sampling**.

*   **Uncertainty Metric**: We map the raw GP output to a probability $P(y=1|x)$ using a logistic sigmoid function:
    $$P(y=1|x) = \frac{1}{1 + e^{-f(x)}}$$
*   **Sampling Strategy**: The training loop is modified to intervene every $k=3$ generations. The algorithm queries the unlabeled pool for the $n=20$ samples where the model is least confident (i.e., where $P(y=1|x) \approx 0.5$) and adds them to the training set. This focuses the evolutionary pressure on the decision boundary, allowing the model to learn efficiently from fewer samples.

### 2.4 Ensemble Learning
The final classification model aggregates the "knowledge" of the top $N$ individuals from the final generation of the Genetic Algorithm.

*   **Mechanisms**: We explored Hard Voting (majority rule), Soft Voting (averaged probabilities), and Weighted Voting (fitness-weighted).
*   **Goal**: The ensemble aims to reduce prediction variance and improve precision by mitigating the idiosyncrasies of single evolved individuals.

## 3. Results

The experimental results demonstrate the trade-offs between the different approaches. Data is derived from the project's experimental logs found in [`experiment/metrics.json`](experiment/metrics.json).

### 3.1 Performance Metrics

The following table summarizes the performance on the held-out test set. The **GA+AL** method achieves the highest accuracy and F1-score, while the **Ensemble** method maximizes precision.

| Method | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Data Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GA (Full Data)** | 0.6907 | 0.5463 | 0.2319 | 0.3256 | 5.30 | 3,500 |
| **Random Baseline** | 0.6453 | 0.4222 | 0.2754 | 0.3333 | 0.59 | 260 |
| **GA+AL** | **0.7120** | 0.5389 | **0.7308** | **0.6204** | **0.59** | 260 |
| **GA+AL+EL** | 0.7107 | **0.5675** | 0.4265 | 0.4870 | 0.68 | 260 |

### 3.2 Visual Analysis

Figure 1 illustrates the comparative performance across four key metrics. The Active Learning approach (GA+AL) significantly boosts Recall compared to the baseline GA, which struggled with the minority signal class.

![Metrics Comparison](experiment/plots/metrics_comparison.png)
*Figure 1: Bar chart comparing Accuracy, Precision, Recall, and F1-Score across approaches.*

The efficiency gains are visualized in Figure 2. The GA+AL approach yields a ~9x speedup by training on a fraction of the data (starting with 200 samples) compared to the full dataset (3500 samples).

![Time Comparison](experiment/plots/time_comparison.png)
*Figure 2: Training time comparison demonstrating the efficiency of Active Learning.*

Detailed performance for the Ensemble model is shown in the ROC curve below.

![ROC Curve](experiment/plots/roc_curve.png)
*Figure 3: ROC Curve for the Ensemble Classifier.*

## 4. Discussion

The comparative analysis reveals that the integration of **Active Learning** was the most effective strategy in this study.

**Performance Analysis**:
By dynamically selecting informative samples, the GA+AL model achieved the highest Accuracy (**71.2%**) and F1-Score (**0.620**), significantly outperforming the baseline GA trained on the full dataset. The baseline GA exhibited poor Recall (0.23), likely getting stuck in a local optimum that favored the majority background class. In contrast, the uncertainty sampling in GA+AL forced the model to confront ambiguous cases, leading to a Recall of **0.73**.

**Efficiency**:
The computational cost was drastically reduced from 5.30 seconds (Full GA) to 0.59 seconds (GA+AL). This result confirms that Genetic Programming, which is typically computationally expensive due to population-based evaluation, can be made feasible for larger problems through intelligent data sampling.

**Ensemble Learning**:
While the Ensemble approach (GA+AL+EL) did not surpass the single best GA+AL model in F1-score, it achieved the highest **Precision** (**0.5675**). In the context of high-energy physics, where false positives (claiming a discovery when there is none) can be scientifically costly, the ensemble method offers a valuable trade-off.

**Limitations**:
*   **Recall Variance**: The ensemble model showed a drop in Recall (0.42) compared to the single best GA+AL individual. This suggests that not all top individuals in the population shared the high-recall characteristic, and averaging them diluted this specific gain.
*   **Complexity**: The resulting mathematical expressions can become complex. While bloat control (height limits) was implemented, further measures like parsimony pressure could enhance interpretability.

In conclusion, combining Evolutionary Learning with Active Learning offers a powerful synergy, delivering superior classification performance with a fraction of the computational cost.
