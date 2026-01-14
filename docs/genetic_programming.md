# Step 2: Evolutionary Learning with Genetic Programming

This document describes the implementation of a Genetic Programming (GP) classifier for the Higgs Boson detection task using the **DEAP (Distributed Evolutionary Algorithms in Python)** library.

## Overview

Genetic Programming is an evolutionary technique where the individuals in the population are computer programs (in this case, mathematical expressions). The goal is to evolve an expression that takes dataset features as input and outputs a value that can be used to distinguish between signal (`s`) and background (`b`).

The implementation is split between [`src/genetic_programming.py`](../src/genetic_programming.py) (logic) and [`main_genetic_programming.py`](../main_genetic_programming.py) (execution).

---

## 1. GP Configuration

### Primitive Set
The primitive set defines the building blocks for the evolved programs. As per the requirements, we included:
- **Addition** (`+`)
- **Subtraction** (`-`)
- **Multiplication** (`*`)

These basic operations allow the GP to discover non-linear relationships between physical measurements.

### Terminal Set
The terminal set consists of the input variables for the GP trees.
- **Dataset Features**: All 30 features from the Higgs Boson dataset are used as terminals (represented as `ARG0` to `ARG29`).
- **Interpretability**: By using the original features (after scaling) as terminals, the resulting best individual can be interpreted as a physical formula.

---

## 2. Fitness and Selection

### Fitness Function
The fitness of an individual is evaluated based on its **Classification Accuracy** on the training set:
1. The GP tree is compiled into a callable function.
2. The function is applied to the training samples.
3. The output (a float) is thresholded:
   - **Output > 0**: Classified as `1` (Signal)
   - **Output <= 0**: Classified as `0` (Background)
4. Accuracy is calculated using `sklearn.metrics.accuracy_score`.

### Evolution Loop
The Genetic Algorithm (GA) loop follows these parameters:
- **Population Size**: 50 individuals.
- **Generations**: 10.
- **Selection**: Tournament selection (size 3).
- **Crossover**: One-point crossover (`gp.cxOnePoint`).
- **Mutation**: Uniform mutation (`gp.mutUniform`).
- **Bloat Control**: Tree height is limited to 17 to prevent overly complex, uninterpretable trees.

---

## 3. Results and Evaluation

The model is evaluated on a held-out test set (30% of the data) after the evolution completes.

### Typical Output
```text
Best Individual found: add(ARG13, ARG12)
Best Fitness (Accuracy): 0.6921

Classification Report (Test Set):
              precision    recall  f1-score   support
           b       0.77      0.75      0.76       405
           s       0.51      0.53      0.52       195
    accuracy                           0.68       600
```

- **Observation**: Even with a simple primitive set and few generations, the GP finds meaningful combinations of features (e.g., combining specific momentum or mass measurements) that perform significantly better than random guessing.

---

## 4. How to Run

1. Ensure the dependencies are installed:
   ```bash
   pip install deap
   ```
2. Run the GP script:
   ```bash
   python main_genetic_programming.py
   ```

*Note: The script currently loads a subset (2000 rows) of the main dataset to ensure efficient execution and to avoid memory/token limits in the environment.*
