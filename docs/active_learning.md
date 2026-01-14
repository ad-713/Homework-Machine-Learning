# Step 3: Integrating Active Learning (GA+AL)

This document details the enhancement of the Genetic Programming (GP) loop with **Active Learning (AL)** techniques using the [`modAL`](https://modal-python.readthedocs.io/) library.

## Overview

In traditional supervised learning, the model is trained on a static dataset. In **Active Learning**, the algorithm interacts with a pool of unlabeled data and selectively queries the most informative samples to be labeled and added to the training set.

In this project, we integrated Active Learning into the Genetic Programming evolutionary loop to dynamically refine the training set based on the model's uncertainty.

---

## 1. Implementation Details

### Sampling Technique: Uncertainty Sampling
We use **Uncertainty Sampling** as our core active learning strategy. The algorithm identifies instances in the unlabeled pool where the current best GP individual is least confident about the classification.

- **Metric**: For binary classification, uncertainty is measured by how close the predicted probability is to 0.5.
- **Probability Mapping**: Since GP trees output raw floats, we apply a **Logistic Sigmoid** function to map these outputs to a probability range $[0, 1]$:
  $$P(y=1|x) = \frac{1}{1 + e^{-f(x)}}$$
  where $f(x)$ is the output of the GP tree.

### GP Classifier Wrapper
The [`GPClassifierWrapper`](../src/genetic_programming.py) class was implemented to bridge the gap between the DEAP library and `modAL`. It wraps a GP individual and provides:
- `predict(X)`: Returns binary labels.
- `predict_proba(X)`: Returns class probabilities using the sigmoid function.

---

## 2. Dynamic Training Loop

The standard `eaSimple` algorithm from DEAP was replaced with a custom iterative loop in [`run_active_learning_ga`](../src/genetic_programming.py) to allow interventions between generations.

### The GA+AL Workflow:
1. **Initialize**: Start with a small labeled training set (e.g., 200 samples) and a large unlabeled pool.
2. **Evolve**: Run the Genetic Algorithm for $k$ generations.
3. **Query (AL Step)**: Every $k$ generations:
   - Identify the best individual in the current population.
   - Use the `uncertainty_sampling` strategy to select the $n$ most uncertain instances from the pool.
   - "Label" these instances (revealing their true labels from the dataset) and move them from the **Pool** to the **Training Set**.
4. **Re-evaluate**: Reset and re-calculate the fitness of the entire population based on the expanded training set.
5. **Repeat**: Continue evolution until the maximum number of generations is reached.

---

## 3. Configuration and Results

### Parameters
- **Query Interval ($k$)**: 3 generations.
- **Batch Size ($n$)**: 20 samples per query.
- **Population Size**: 50.
- **Generations**: 10.

### Observed Impact
As the training set grows, the fitness landscape changes. The re-evaluation step ensures that individuals are tested against increasingly difficult or informative samples.

**Sample Output:**
```text
--- Active Learning Step at Generation 3 ---
Added 20 samples. New training set size: 220
gen     nevals  avg             max
3       50      0.597545        0.677273

--- Active Learning Step at Generation 6 ---
Added 20 samples. New training set size: 240
6       50      0.60875         0.683333
```

- **Improved Generalization**: By focusing on uncertain samples, the GP is forced to resolve ambiguities near the decision boundary, often leading to better performance on the test set with fewer total training samples.

---

## 4. How to Run

1. Ensure `modAL-python` is installed:
   ```bash
   pip install modAL-python
   ```
2. Run the main script:
   ```bash
   python main_genetic_programming.py
   ```

The script will output the progress of the evolution and explicitly log when an Active Learning step occurs and how the training set size changes.
