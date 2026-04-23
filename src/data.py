"""
data.py — Dataset loading and SSL-style train/val/test splits.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """
    Load Wisconsin Breast Cancer dataset.
    Returns X (features), y (labels: 1=malignant, 0=benign).
    """
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target  # Flip labels so malignant is class 1

    print("Dataset Overview:")
    print(f"  Features : {X.shape[1]}")
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Classes  : malignant={np.sum(y == 1)}, benign={np.sum(y == 0)}")

    return X, y


def make_ssl_split(X, y, labeled_ratio=0.10, test_ratio=0.50, seed=42):
    """
    50% test / labeled_ratio labeled / rest unlabeled, all stratified.

    Returns
    -------
    X_lab, X_unlab, X_test, y_lab, y_unlab, y_test
    """
    # Hold out the test set first
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )

    # Compute labeled share within the remaining pool
    lab_share = labeled_ratio / (1.0 - test_ratio)

    # Split remaining data into labeled and unlabeled sets
    X_lab, X_unlab, y_lab, y_unlab = train_test_split(
        X_rest, y_rest, train_size=lab_share, stratify=y_rest, random_state=seed
    )

    return X_lab, X_unlab, X_test, y_lab, y_unlab, y_test


def preprocess(X_lab, X_unlab, X_test):
    """
    Fit StandardScaler on labeled set only (no leakage),
    then transform all splits.
    """
    scaler = StandardScaler()

    # Fit on labeled data only, then apply to unlabeled and test sets
    X_lab_sc = scaler.fit_transform(X_lab)
    X_unlab_sc = scaler.transform(X_unlab)
    X_test_sc = scaler.transform(X_test)

    return X_lab_sc, X_unlab_sc, X_test_sc, scaler