import pandas as pd
import numpy as np
import os
from src.data_loader import clean_missing_values
from src.preprocessing import encode_labels, split_data, scale_features
from src.genetic_programming import setup_gp, run_active_learning_ga
from src.ensemble_learning import GPEnsembleClassifier
from sklearn.metrics import classification_report

def main():
    # Load a subset of data to avoid token flooding and for speed
    raw_data_path = 'data/atlas-higgs-challenge-2014-v2.csv'
    print(f"Loading data subset from {raw_data_path}...")
    
    # Read first 5000 rows for a better AL demonstration
    try:
        df = pd.read_csv(raw_data_path, nrows=5000)
    except FileNotFoundError:
        print(f"Error: {raw_data_path} not found. Please ensure the dataset is in the data/ directory.")
        return

    print(f"Loaded {len(df)} rows.")
    
    # Preprocessing
    print("Cleaning and preprocessing data...")
    df = clean_missing_values(df)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, test_size=0.3)
    
    # Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Active Learning Split: Initial Train vs Pool
    # We take 200 samples for initial training
    n_initial = 200
    X_train_initial = X_train_scaled[:n_initial]
    y_train_initial = y_train.values[:n_initial]
    
    X_pool = X_train_scaled[n_initial:]
    y_pool = y_train.values[n_initial:]
    
    print(f"Initial Training Set size: {len(X_train_initial)}")
    print(f"Unlabeled Pool size: {len(X_pool)}")
    
    # Genetic Programming Setup
    n_features = X_train_scaled.shape[1]
    print(f"Setting up GP with {n_features} features...")
    pset, toolbox = setup_gp(n_features)
    
    # Run GA with Active Learning
    print("Starting Evolutionary Loop with Active Learning (Uncertainty Sampling)...")
    # Step every 3 generations, add 20 samples
    pop, log, hof = run_active_learning_ga(
        X_train_initial, y_train_initial, 
        X_pool, y_pool, 
        pset, toolbox, 
        n_gen=10, pop_size=50, 
        k=3, n_instances=20
    )
    
    best_ind = hof[0]
    print(f"\nBest Individual found: {best_ind}")
    print(f"Best Fitness (Accuracy): {best_ind.fitness.values[0]:.4f}")
    
    # Selection of Top N for Ensemble
    # Sort population by fitness and take top 10
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
    top_n = 10
    ensemble_pop = sorted_pop[:top_n]
    print(f"\nCreating Ensemble from Top {len(ensemble_pop)} individuals...")
    
    ensemble = GPEnsembleClassifier(ensemble_pop, toolbox)
    
    # Evaluation on Test Set
    print("\n--- Evaluating Models on Test Set ---")
    
    # 1. Best Individual
    func = toolbox.compile(expr=best_ind)
    best_ind_preds = [1 if func(*p) > 0 else 0 for p in X_test_scaled]
    print("\n[Best Individual Classifier]")
    print(classification_report(y_test, best_ind_preds, target_names=le.classes_))
    
    # 2. Ensemble - Hard Voting
    hard_preds = ensemble.predict(X_test_scaled, voting='hard')
    print("\n[Ensemble - Hard Voting]")
    print(classification_report(y_test, hard_preds, target_names=le.classes_))
    
    # 3. Ensemble - Soft Voting
    soft_preds = ensemble.predict(X_test_scaled, voting='soft')
    print("\n[Ensemble - Soft Voting]")
    print(classification_report(y_test, soft_preds, target_names=le.classes_))
    
    # 4. Ensemble - Weighted Voting (Alternative Solution)
    weighted_preds = ensemble.predict(X_test_scaled, voting='weighted')
    print("\n[Ensemble - Weighted Voting (Alternative)]")
    print(classification_report(y_test, weighted_preds, target_names=le.classes_))

if __name__ == "__main__":
    main()
