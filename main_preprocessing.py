import os
import joblib
import numpy as np
from src.data_loader import load_higgs_data, clean_missing_values
from src.preprocessing import (
    drop_columns, check_class_balance, check_outliers,
    encode_labels, split_data, scale_features, apply_pca
)

def main():
    # Paths
    raw_data_path = 'data/atlas-higgs-challenge-2014-v2.csv'
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_higgs_data(raw_data_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    # 1. Delete superfluous columns
    superfluous_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight']
    print(f"Deleting superfluous columns: {superfluous_cols}...")
    df = drop_columns(df, superfluous_cols)
    
    # 2. Clean missing values
    print("Cleaning missing values (-999.0)...")
    df = clean_missing_values(df)
    
    # 3. Perform EDA
    print("\nPerforming Exploratory Data Analysis...")
    check_class_balance(df)
    check_outliers(df)
    print("\nEDA complete.\n")
    
    # 4. Encoding labels
    print("Encoding labels...")
    df, label_encoder = encode_labels(df)
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("Scaling features using RobustScaler...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    print("Applying PCA (preserving 90% variance)...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, n_components=0.90)
    joblib.dump(pca, os.path.join(output_dir, 'pca.joblib'))
    print(f"PCA reduced dimensions from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")
    
    print("Saving processed data...")
    np.savez(os.path.join(output_dir, 'train_data.npz'), X=X_train_pca, y=y_train, w=w_train)
    np.savez(os.path.join(output_dir, 'test_data.npz'), X=X_test_pca, y=y_test, w=w_test)
    
    print("Preprocessing complete. Files saved in 'processed_data/'")

if __name__ == "__main__":
    main()
