from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.phiusiil.dataset import DATASET_PATH, FEATURE_COLUMNS
from src.phiusiil.url_features import extract_phiusiil_features


OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "phiusiil_url_features.csv"
)


def main():
    print("=" * 60)
    print("BUILDING EXTRACTED URL FEATURE DATASET")
    print("=" * 60)

    print("\n[1] Loading raw dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows loaded: {len(df)}")

    # Check for conflicting labels before removing duplicates.
    conflicts = (
        df.groupby("URL")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    if conflicts:
        raise ValueError(
            f"Found {conflicts} URLs with conflicting labels."
        )

    # Remove duplicate URLs.
    df = df.drop_duplicates(
        subset="URL",
        keep="first",
    ).copy()

    print(f"Rows after duplicate removal: {len(df)}")

    # Convert labels:
    # Original UCI:
    #   0 = phishing
    #   1 = legitimate
    #
    # Project:
    #   0 = legitimate
    #   1 = phishing
    df["label"] = df["label"].map({
        0: 1,
        1: 0,
    })

    print("\n[2] Extracting URL features...")

    records = []

    total = len(df)

    for index, row in enumerate(
        df[["URL", "label"]].itertuples(index=False),
        start=1,
    ):
        features = extract_phiusiil_features(row.URL)

        record = {
            "URL": row.URL,
            **features,
            "label": row.label,
        }

        records.append(record)

        if index % 10000 == 0 or index == total:
            print(
                f"Processed {index}/{total} URLs"
            )

    feature_df = pd.DataFrame(records)

    expected_columns = [
        "URL",
        *FEATURE_COLUMNS,
        "label",
    ]

    missing = [
        column
        for column in expected_columns
        if column not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}"
        )

    feature_df = feature_df[expected_columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE DATASET CREATED")
    print("=" * 60)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows  : {len(feature_df)}")
    print(f"Columns: {len(feature_df.columns)}")

    print("\nLabel distribution:")
    print(feature_df["label"].value_counts())

    print("\nFeature columns:")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")


if __name__ == "__main__":
    main()
