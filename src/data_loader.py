import pandas as pd
import numpy as np

def load_higgs_data(csv_path):
    """
    Load the Higgs Boson dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    return df

def clean_missing_values(df, missing_val=-999.0):
    """
    Handle missing values represented by -999.0.
    Replaces them with NaN for easier handling with sklearn.
    """
    df_cleaned = df.replace(missing_val, np.nan)
    
    # Identify numeric columns (excluding metadata and label)
    # Based on dataset_sample.json, metadata columns are: EventId, Weight, KaggleSet, KaggleWeight, Label
    metadata_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label']
    feature_cols = [col for col in df_cleaned.columns if col not in metadata_cols]
    
    # Simple imputation: replace NaN with the median of the column
    # We use median because some distributions might be skewed
    for col in feature_cols:
        if df_cleaned[col].isnull().any():
            median = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median)
            
    return df_cleaned
