"""
PhiUSIIL dataset preparation.

Loads the local PhiUSIIL Phishing URL Dataset,
removes duplicate URLs, selects the initial numerical
feature set, and converts labels to the project's convention.

Project convention:
    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

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
    """Load and prepare the local PhiUSIIL dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"PhiUSIIL dataset not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURE_COLUMNS + ["URL", "label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: "
            f"{missing_columns}"
        )

    # Check whether the same URL appears with different labels.
    conflicting_labels = (
        df.groupby("URL")["label"].nunique().gt(1).sum()
    )

    if conflicting_labels:
        raise ValueError(
            f"Found {conflicting_labels} URLs with conflicting labels."
        )

    # Remove duplicate URLs before model splitting.
    df = df.drop_duplicates(
        subset="URL",
        keep="first",
    ).copy()

    # UCI convention:
    #   0 = phishing
    #   1 = legitimate
    #
    # Project convention:
    #   0 = legitimate
    #   1 = phishing
    df["label"] = df["label"].map({
        0: 1,
        1: 0,
    })

    if df["label"].isna().any():
        raise ValueError(
            "Unexpected label value found in the dataset."
        )

    return df[
        ["URL"] + FEATURE_COLUMNS + ["label"]
    ].reset_index(drop=True)


def get_features_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target labels."""

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    return X, y
