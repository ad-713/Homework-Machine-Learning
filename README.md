# Higgs Boson Detection - Machine Learning Homework

This project implements and combines three learning paradigms for the Higgs Boson classification problem: **Evolutionary Learning**, **Active Learning**, and **Ensemble Learning**.

The goal is to evaluate and analyze the performance of these approaches on the [Higgs Boson Detection dataset](https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz).

> **Note**: The raw dataset should be downloaded and extracted into the `data/` directory.

## Project Structure

- `data/`: Contains the raw dataset.
  - `atlas-higgs-challenge-2014-v2.csv`: The main CSV file.
- `src/`: Modular Python source code.
  - [`data_loader.py`](src/data_loader.py): Loading and initial cleaning.
  - [`preprocessing.py`](src/preprocessing.py): Scaling, encoding, and dimensionality reduction.
  - [`genetic_programming.py`](src/genetic_programming.py): Evolutionary learning implementation.
- `docs/`: Detailed documentation for each step.
  - [**Step 1: Dataset Preprocessing**](docs/preprocessing.md)
  - [**Step 2: Evolutionary Learning**](docs/genetic_programming.md)
  - [**Step 3: Active Learning**](docs/active_learning.md)
  - [**Step 4: Ensemble Learning**](docs/ensemble_learning.md)
  - [**Step 5: Comparative Analysis**](docs/comparative_analysis.md)
- `experiment/`: Results from the comparative analysis.
  - `report.md`: Summary table of metrics.
  - `plots/`: Performance and efficiency visualizations.
- `processed_data/`: Created by the preprocessing pipeline. Contains split and scaled NumPy archives.
- `main_preprocessing.py`: Orchestration script for data preparation.
- `main_genetic_programming.py`: Orchestration script for evolutionary learning.
- `comparative_analysis.py`: Final script for benchmarking all methods.
- [`Project_Report.md`](Project_Report.md): Comprehensive project report summarizing methodology and findings.

## Implementation Steps

### 1. Dataset Preprocessing
Detailed documentation: [Step 1: Preprocessing](docs/preprocessing.md)

- Handled missing values (`-999.0`) via median imputation.
- Normalized features using `StandardScaler`.
- Performed 80/20 train-test split.
- Reduced dimensionality from 30 to 20 features using PCA (95% variance).

### 2. Evolutionary Learning
Detailed documentation: [Step 2: Evolutionary Learning](docs/genetic_programming.md)

- Implemented Genetic Programming (GP) using the DEAP library.
- Evolved mathematical expressions using addition, subtraction, and multiplication.
- Optimized for classification accuracy on raw physical features.

### 3. Active Learning
Detailed documentation: [Step 3: Active Learning](docs/active_learning.md)

- Integrated the `modAL` library for active sampling.
- Implemented **Uncertainty Sampling** to dynamically query informative samples from an unlabeled pool.
- Created a `GPClassifierWrapper` to map GP outputs to probabilities via a sigmoid function.
- Enhanced the evolutionary loop to update training data every $k$ generations.

### 4. Ensemble Learning
Detailed documentation: [Step 4: Ensemble Learning](docs/ensemble_learning.md)

- Implemented a `GPEnsembleClassifier` supporting Hard, Soft, and Weighted voting.
- Leverages the final population of the GA to improve generalization.
- Reduces prediction variance by combining multiple top-performing individuals.

### 5. Comparative Analysis and Deliverables
Detailed documentation: [Step 5: Comparative Analysis](docs/comparative_analysis.md)

- Evaluated GA, GA+AL, and GA+AL+EL on common metrics (Accuracy, Precision, Recall, F1).
- Analyzed computational efficiency and training time trade-offs.
- Generated performance plots and a comparative summary table.
- Results are stored in the `experiment/` directory, including a [comprehensive report](experiment/report.md).

## Performance Summary

| Approach | Accuracy | F1-Score | Time Speedup |
| :--- | :--- | :--- | :--- |
| **GA (Baseline)** | 69.1% | 0.326 | 1x |
| **GA + AL** | **71.2%** | **0.620** | **~9x faster** |
| **GA + AL + EL** | 71.1% | 0.487 | ~8x faster |

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Preprocessing:
   ```bash
   python main_preprocessing.py
   ```

3. Run Evolutionary Learning:
   ```bash
   python main_genetic_programming.py
   ```

4. Run Comparative Analysis:
   ```bash
   python comparative_analysis.py
   ```
