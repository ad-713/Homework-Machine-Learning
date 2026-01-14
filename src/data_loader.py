import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_higgs_data(csv_path):
    """
    Load the Higgs Boson dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    return df

def clean_missing_values(df, missing_val=-999.0):
    """
    Handle missing values represented by -999.0.
    Uses SimpleImputer with median strategy and adds a 'HasMissing' indicator.
    """
    # Identify numeric columns (excluding metadata and label)
    metadata_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Create missing indicator: True if ANY feature column is missing
    df['HasMissing'] = df[feature_cols].eq(missing_val).any(axis=1)

    df_cleaned = df.replace(missing_val, np.nan)
    
    # Initialize SimpleImputer with median strategy
    imputer = SimpleImputer(strategy='median')
    
    # Impute missing values only for feature columns
    if feature_cols:
        df_cleaned[feature_cols] = imputer.fit_transform(df_cleaned[feature_cols])
            
    return df_cleaned
