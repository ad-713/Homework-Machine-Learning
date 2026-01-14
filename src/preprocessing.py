from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def drop_columns(df, columns):
    """
    Drop specified columns from the dataframe.
    """
    df_dropped = df.drop(columns=columns, errors='ignore')
    return df_dropped

def check_class_balance(df, target_col='Label'):
    """
    Check and print the distribution of classes in the target column.
    """
    if target_col not in df.columns:
        print(f"Column '{target_col}' not found for balance check.")
        return
    
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    print("\n--- Class Balance ---")
    for cls, count in counts.items():
        print(f"Class {cls}: {count} ({percentages[cls]:.2f}%)")
    
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    print(f"Imbalance Ratio (Majority/Minority): {imbalance_ratio:.2f}")

def check_outliers(df, threshold=1.5):
    """
    Identify outliers using the IQR method and print a summary.
    Excludes non-numeric columns and 'Label'.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filter out columns that might be metadata if they still exist
    exclude = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label']
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    print("\n--- Outlier Detection (IQR Method) ---")
    total_outliers = 0
    outlier_summary = []
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = len(outliers)
        
        if num_outliers > 0:
            outlier_summary.append({
                'column': col,
                'count': num_outliers,
                'percentage': (num_outliers / len(df)) * 100
            })
            total_outliers += num_outliers
            
    if not outlier_summary:
        print("No outliers detected.")
    else:
        # Sort by percentage descending
        outlier_summary.sort(key=lambda x: x['percentage'], reverse=True)
        print(f"{'Feature':<30} | {'Count':<10} | {'Percentage':<10}")
        print("-" * 55)
        for item in outlier_summary:
            print(f"{item['column']:<30} | {item['count']:<10} | {item['percentage']:>8.2f}%")
            
    print(f"\nTotal potential outlier detections across all features: {total_outliers}")

def encode_labels(df):
    """
    Encode the 'Label' column ('s' for signal, 'b' for background) to 1 and 0.
    """
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label']) # 'b' -> 0, 's' -> 1 (usually alphabetically)
    return df, le

def scale_features(X_train, X_test):
    """
    Scale features using RobustScaler to handle outliers.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(X_train, X_test, n_components=0.90):
    """
    Apply PCA to reduce dimensionality while preserving a certain percentage of variance.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def split_data(df, test_size=0.3, random_state=42):
    """
    Split the dataframe into training and testing sets.
    Returns X_train, X_test, y_train, y_test, weights, and missing_mask.
    """
    metadata_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label', 'HasMissing']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols]
    y = df['Label']
    
    # Handle optional metadata
    weights = df['Weight'] if 'Weight' in df.columns else pd.Series([1.0] * len(df))
    mask = df['HasMissing'] if 'HasMissing' in df.columns else pd.Series([False] * len(df))
    
    # Bundle metadata to split it consistently
    meta = pd.concat([weights.rename('w'), mask.rename('m')], axis=1)
    
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return (X_train, X_test, y_train, y_test,
            meta_train['w'].values, meta_test['w'].values,
            meta_train['m'].values, meta_test['m'].values)
