"""
Feature analysis utilities for the PhiUSIIL dataset.

Analysis is performed on the training split only so that
validation and test data remain unseen during preprocessing decisions.
"""

from __future__ import annotations

import pandas as pd


def summarize_features(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Create a statistical summary of the training features.

    The function does not modify the input DataFrame.
    """

    summary = pd.DataFrame(index=X_train.columns)

    summary["dtype"] = X_train.dtypes.astype(str)
    summary["missing"] = X_train.isna().sum()
    summary["unique"] = X_train.nunique()
    summary["mean"] = X_train.mean()
    summary["std"] = X_train.std()
    summary["min"] = X_train.min()
    summary["median"] = X_train.median()
    summary["max"] = X_train.max()
    summary["skew"] = X_train.skew()

    return summary


def find_constant_features(X_train: pd.DataFrame) -> list[str]:
    """Return features containing only one unique value."""

    return [
        column
        for column in X_train.columns
        if X_train[column].nunique() <= 1
    ]
