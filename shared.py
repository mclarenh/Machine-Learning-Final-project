import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def standardize(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    std = std.replace(0, 1)

    X_train_scaled = (X_train - mean) / std
    X_other_scaled = (X_other - mean) / std

    return X_train_scaled, X_other_scaled

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)
