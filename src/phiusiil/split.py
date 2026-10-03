"""
Dataset splitting utilities for the PhiUSIIL dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data into training, validation, and test sets.

    Final proportions:
        64% training
        16% validation
        20% test
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_relative_size = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
