"""
PhiUSIIL dataset preparation.

Loads the PhiUSIIL Phishing URL Dataset from UCI,
removes duplicate URLs, selects the initial numerical
feature set, and converts labels to the project's convention.

Project convention:
    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 967

FEATURE_COLUMNS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
]


def load_phiusiil() -> pd.DataFrame:
    """Load and prepare the PhiUSIIL dataset."""

    dataset = fetch_ucirepo(id=DATASET_ID)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = pd.concat([features, targets], axis=1)

    required_columns = FEATURE_COLUMNS + ["URL", "label"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: {missing_columns}"
        )

    # Remove duplicate URLs before any train/test split.
    df = df.drop_duplicates(subset="URL", keep="first").copy()

    # Verify that every URL has exactly one label.
    conflicting_labels = (
        df.groupby("URL")["label"].nunique().gt(1).sum()
    )

    if conflicting_labels:
        raise ValueError(
            f"Found {conflicting_labels} URLs with conflicting labels."
        )

    # UCI convention:
    #   0 = phishing
    #   1 = legitimate
    #
    # Project convention:
    #   0 = legitimate
    #   1 = phishing
    df["label"] = df["label"].map({0: 1, 1: 0})

    if df["label"].isna().any():
        raise ValueError("Unexpected label value found in the dataset.")

    return df[["URL"] + FEATURE_COLUMNS + ["label"]].reset_index(drop=True)


def get_features_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target labels."""

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    return X, y
