import pandas as pd
import numpy as np


def load_and_preprocess_data(filename, feature_columns, target_column):
    data = pd.read_csv(filename)
    X = data[feature_columns].values
    y = data[target_column].values

    # Standardizing features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std

    return X_standardized, y, X_mean, X_std
