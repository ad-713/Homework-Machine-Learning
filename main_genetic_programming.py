import pandas as pd
import numpy as np
import os
from src.data_loader import clean_missing_values
from src.preprocessing import encode_labels, split_data, scale_features
from src.genetic_programming import setup_gp, run_ga, eval_classifier
from sklearn.metrics import classification_report

def main():
    # Load a subset of data to avoid token flooding and for speed
    raw_data_path = 'data/atlas-higgs-challenge-2014-v2.csv'
    print(f"Loading data subset from {raw_data_path}...")
    
    # Read first 2000 rows as a sample
    try:
        df = pd.read_csv(raw_data_path, nrows=2000)
    except FileNotFoundError:
        print(f"Error: {raw_data_path} not found. Please ensure the dataset is in the data/ directory.")
        return

    print(f"Loaded {len(df)} rows.")
    
    # Preprocessing
    print("Cleaning and preprocessing data...")
    df = clean_missing_values(df)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, test_size=0.3)
    
    # Scaling - important for GP to avoid huge numbers
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Genetic Programming Setup
    n_features = X_train_scaled.shape[1]
    print(f"Setting up GP with {n_features} features...")
    pset, toolbox = setup_gp(n_features)
    
    # Run GA
    print("Starting Evolutionary Loop...")
    # Using small pop and gen for demonstration
    pop, log, hof = run_ga(X_train_scaled, y_train.values, pset, toolbox, n_gen=10, pop_size=50)
    
    best_ind = hof[0]
    print(f"\nBest Individual found: {best_ind}")
    print(f"Best Fitness (Accuracy): {best_ind.fitness.values[0]:.4f}")
    
    # Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    # We need to manually evaluate the best individual on the test set
    func = toolbox.compile(expr=best_ind)
    test_preds = [1 if func(*p) > 0 else 0 for p in X_test_scaled]
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=le.classes_))

if __name__ == "__main__":
    main()
