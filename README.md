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
- `processed_data/`: Created by the preprocessing pipeline. Contains split and scaled NumPy archives.
- `main_preprocessing.py`: Orchestration script for data preparation.
- `main_genetic_programming.py`: Orchestration script for evolutionary learning.

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
*Documentation pending.*

### 4. Ensemble Learning
*Documentation pending.*

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
