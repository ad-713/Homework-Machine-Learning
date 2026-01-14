from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def encode_labels(df):
    """
    Encode the 'Label' column ('s' for signal, 'b' for background) to 1 and 0.
    """
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label']) # 'b' -> 0, 's' -> 1 (usually alphabetically)
    return df, le

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(X_train, X_test, n_components=0.95):
    """
    Apply PCA to reduce dimensionality while preserving a certain percentage of variance.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets.
    Returns X_train, X_test, y_train, y_test, and weights if available.
    """
    metadata_cols = ['EventId', 'Weight', 'KaggleSet', 'KaggleWeight', 'Label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols]
    y = df['Label']
    weights = df['Weight'] if 'Weight' in df.columns else None
    
    if weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, w_train, w_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, None, None
