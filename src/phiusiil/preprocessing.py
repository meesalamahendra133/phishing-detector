"""
Preprocessing pipeline for PhiUSIIL URL features.

The transformer is fitted only on the training data to prevent
data leakage into validation and test sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Highly right-skewed count features.
LOG_FEATURES = [
    "URLLength",
    "NoOfObfuscatedChar",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
]


class URLPreprocessor:
    """Preprocess numerical URL features."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "URLPreprocessor":
        """Fit preprocessing using training data only."""

        self.feature_columns = list(X.columns)

        X_transformed = self._log_transform(X)

        self.scaler.fit(X_transformed)

        self.fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform a dataset using the fitted preprocessing pipeline."""

        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()."
            )

        X = X[self.feature_columns].copy()

        X_transformed = self._log_transform(X)

        return self.scaler.transform(X_transformed)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor and transform the training data."""

        self.fit(X)
        return self.transform(X)

    @staticmethod
    def _log_transform(X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to highly skewed count features."""

        X = X.copy()

        for column in LOG_FEATURES:
            if column in X.columns:
                X[column] = np.log1p(X[column])

        return X
