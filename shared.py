import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def standardize(X_train, X_other):
    """
    Standardize data using only the training data's mean and std.
    Apply those same values to both datasets.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    # Prevent division by zero
    std = std.replace(0, 1)

    X_train_scaled = (X_train - mean) / std
    X_other_scaled = (X_other - mean) / std

    return X_train_scaled, X_other_scaled
