from __future__ import annotations

import pandas as pd

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels


def main():
    print("=" * 60)
    print("FEATURE / TARGET LEAKAGE AUDIT")
    print("=" * 60)

    print("\n[1] Loading dataset...")

    df = load_phiusiil()
    X, y = get_features_and_labels(df)

    print(f"Rows     : {len(df)}")
    print(f"Features : {X.shape[1]}")

    print("\n[2] Feature correlation with label...")

    correlation = X.copy()
    correlation["label"] = y

    correlations = (
        correlation.corr(numeric_only=True)["label"]
        .drop("label")
        .abs()
        .sort_values(ascending=False)
    )

    print("\nAbsolute correlation with label:")
    print(correlations.to_string())

    print("\n" + "=" * 60)
    print("TOP 5 FEATURES")
    print("=" * 60)

    for feature in correlations.head(5).index:
        print(f"\nFeature: {feature}")

        legitimate = df.loc[df["label"] == 0, feature]
        phishing = df.loc[df["label"] == 1, feature]

        print(
            f"Legitimate -> "
            f"min={legitimate.min():.4f}, "
            f"mean={legitimate.mean():.4f}, "
            f"median={legitimate.median():.4f}, "
            f"max={legitimate.max():.4f}"
        )

        print(
            f"Phishing   -> "
            f"min={phishing.min():.4f}, "
            f"mean={phishing.mean():.4f}, "
            f"median={phishing.median():.4f}, "
            f"max={phishing.max():.4f}"
        )

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

