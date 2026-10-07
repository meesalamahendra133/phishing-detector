from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.phiusiil.dataset import FEATURE_COLUMNS



MODEL_PATH = Path("models/xgboost_extracted_features.joblib")

def predict_url(features: dict) -> float:
    """
    Predict phishing probability for one URL.

    Returns:
        Probability between 0 and 1.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = pd.DataFrame(
        [[features[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    probability = model.predict_proba(X)[0][1]

    return float(probability)


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python -m src.predictor.predict_url <url>"
        )
        sys.exit(1)

    url = sys.argv[1]

    print("=" * 60)
    print("PHISHING URL PREDICTION")
    print("=" * 60)

    print(f"\nURL: {url}")

    # Temporary test feature vector.
    #
    # These are NOT extracted from the URL yet.
    # Feature extraction will be connected in the next step.

    features = {
        "URLLength": len(url),
        "DomainLength": 0,
        "IsDomainIP": 0,
        "TLDLength": 0,
        "NoOfSubDomain": 0,
        "HasObfuscation": 0,
        "NoOfObfuscatedChar": 0,
        "ObfuscationRatio": 0,
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "LetterRatioInURL": 0,
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "DegitRatioInURL": 0,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": 0,
        "SpacialCharRatioInURL": 0,
        "IsHTTPS": int(url.lower().startswith("https://")),
        "CharContinuationRate": 0,
        "TLDLegitimateProb": 0,
        "URLCharProb": 0,
    }

    probability = predict_url(features)

    print(
        f"\nPhishing probability: "
        f"{probability:.4f}"
    )

    print(
        f"Phishing percentage: "
        f"{probability * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
