import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_higgs_data(csv_path):
    """
    Load the Higgs Boson dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    return df

def analyze_missing_values(df):
    """
    EDA: Calculate and print the percentage of missing values for each column.
    """
    missing_pct = df.isnull().mean() * 100
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    print("\nMissing Values Report (Percentage):")
    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report)
    print("-" * 30)
    return missing_pct

def clean_missing_values(df, missing_val=-999.0, threshold=70.0):
    """
    Handle missing values:
    1. Replaces missing_val with NaN.
    2. Drops columns with missing values > threshold (%).
    3. Imputes remaining missing values using SimpleImputer with median strategy.
    """
    # Replace the specific missing value marker with NaN
    df_cleaned = df.replace(missing_val, np.nan)
    
    # Perform EDA to check % of missing values
    missing_pct = analyze_missing_values(df_cleaned)
    
    # Identify metadata columns to exclude from dropping/imputation if needed
    metadata_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label']
    
    # 1. Drop features with high missingness (> 70%)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    # Ensure we don't drop metadata columns even if they have missing values (though they shouldn't)
    cols_to_drop = [col for col in cols_to_drop if col not in metadata_cols]
    
    if cols_to_drop:
        print(f"Dropping columns with >{threshold}% missing values: {cols_to_drop}")
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    # Identify remaining feature columns for imputation
    feature_cols = [col for col in df_cleaned.columns if col not in metadata_cols]
    
    # 2. Handle missing values with SimpleImputer class (median strategy)
    imputer = SimpleImputer(strategy='median')
    
    if df_cleaned[feature_cols].isnull().any().any():
        print("Imputing remaining missing values using SimpleImputer (median)...")
        df_cleaned[feature_cols] = imputer.fit_transform(df_cleaned[feature_cols])
            
    return df_cleaned
