# Step 4: Ensemble Learning Implementation (GA+AL+EL)

This document describes the integration of **Ensemble Learning (EL)** into the Genetic Programming and Active Learning pipeline.

## Overview

After the Genetic Algorithm (GA) finishes its evolutionary process (refined by Active Learning), we are left with a final population of individuals. Instead of selecting only the single best individual, we combine the "knowledge" of multiple top-performing individuals to improve generalization and robustness.

---

## 1. Ensemble Construction

The ensemble is created by selecting the **Top $N$ individuals** from the final population based on their fitness (accuracy on the training set). 

- **Default Selection**: Top 10 individuals.
- **Diversity**: Because GA naturally maintains some diversity in the population, these top individuals often capture different aspects of the feature space.

---

## 2. Voting Mechanisms

We implemented three types of voting mechanisms in the [`GPEnsembleClassifier`](../src/ensemble_learning.py):

### Hard Voting
In Hard Voting, each individual in the ensemble predicts a class label (0 or 1). The final prediction is the majority vote.
- **Formula**: $Y = \text{mode}(h_1(x), h_2(x), ..., h_n(x))$

### Soft Voting
In Soft Voting, we average the predicted probabilities from each individual. The final class is determined by thresholding the average probability at 0.5.
- **Probability Mapping**: Probabilities are calculated using a sigmoid function applied to the GP tree output: $P(y=1|x) = \sigma(f(x))$.
- **Formula**: $Y = 1 \text{ if } \frac{1}{n} \sum \sigma(f_i(x)) > 0.5 \text{ else } 0$

### Weighted Voting (Alternative Solution)
We propose **Weighted Voting** as an alternative to treat high-performing individuals with more "authority". Each individual's vote is multiplied by its training fitness score.
- **Formula**: $Y = 1 \text{ if } \sum w_i \cdot h_i(x) > 0.5 \text{ else } 0$
  where $w_i$ is the normalized fitness of individual $i$.

---

## 3. Results Comparison

In the final evaluation step, we compare the performance of:
1.  **Best Single Individual**: The standard output of the GA.
2.  **Ensemble (Hard Voting)**: Simple majority.
3.  **Ensemble (Soft Voting)**: Averaged probabilities (usually more robust).
4.  **Ensemble (Weighted Voting)**: Fitness-aware majority.

Ensembles typically outperform single individuals by reducing the variance of the predictions and mitigating the risk of selecting an individual that overfits a specific subset of the training data.

---

## 4. How to Run

The ensemble evaluation is now part of the main genetic programming script:

```bash
python main_genetic_programming.py
```

The script will output classification reports for each voting method, allowing for direct comparison on the test set.

To see how the ensemble performs relative to the standard GA and GA+AL models, refer to the [**Comparative Analysis**](comparative_analysis.md).
