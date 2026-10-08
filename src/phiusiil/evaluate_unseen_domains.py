from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.phiusiil.dataset import FEATURE_COLUMNS
from src.phiusiil.url_features import extract_phiusiil_features


RANDOM_STATE = 42

DATASET_PATH = (
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

MODEL_FEATURE_COLUMNS = [
    feature
    for feature in FEATURE_COLUMNS
    if feature != "IsHTTPS"
]


def extract_domain(url: str) -> str:
    """Extract hostname/domain used for domain-level splitting."""

    url = str(url).strip().lower()

    if not re.match(r"^[a-z][a-z0-9+.-]*://", url):
        url = "http://" + url

    from urllib.parse import urlparse

    parsed = urlparse(url)

    return parsed.hostname or ""


def extract_features(url: str) -> dict[str, float]:
    """Extract the same features used during model inference."""

    return extract_phiusiil_features(url)


def main():

    print("=" * 60)
    print("UNSEEN-DOMAIN EVALUATION")
    print("17 FEATURES - WITHOUT HTTPS")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------

    print("\n[1] Loading raw dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows loaded: {len(df)}")

    # --------------------------------------------------
    # 2. Remove duplicate URLs
    # --------------------------------------------------

    print("\n[2] Removing duplicate URLs...")

    conflicting_labels = (
        df.groupby("URL")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    if conflicting_labels:
        raise ValueError(
            f"Found {conflicting_labels} URLs with conflicting labels."
        )

    df = df.drop_duplicates(
        subset="URL",
        keep="first",
    ).copy()

    print(f"Rows after duplicate removal: {len(df)}")

    # --------------------------------------------------
    # 3. Convert labels
    # --------------------------------------------------

    # Original PhiUSIIL:
    # 0 = phishing
    # 1 = legitimate
    #
    # Project:
    # 0 = legitimate
    # 1 = phishing

    df["label"] = df["label"].map({
        0: 1,
        1: 0,
    })

    # --------------------------------------------------
    # 4. Extract domains
    # --------------------------------------------------

    print("\n[3] Extracting domains...")

    df["domain"] = df["URL"].apply(
        extract_domain
    )

    empty_domains = (
        df["domain"].eq("")
        .sum()
    )

    if empty_domains:
        raise ValueError(
            f"Could not extract domains from "
            f"{empty_domains} URLs."
        )

    print(
        f"Unique domains: "
        f"{df['domain'].nunique()}"
    )

    # --------------------------------------------------
    # 5. Domain-level split
    # --------------------------------------------------

    print("\n[4] Creating unseen-domain split...")

    domains = (
        df["domain"]
        .drop_duplicates()
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(domains) * 0.80
    )

    train_domains = set(
        domains.iloc[:split_index]
    )

    test_domains = set(
        domains.iloc[split_index:]
    )

    train_df = df[
        df["domain"].isin(train_domains)
    ].copy()

    test_df = df[
        df["domain"].isin(test_domains)
    ].copy()

    print(
        f"Training domains: "
        f"{len(train_domains)}"
    )

    print(
        f"Testing domains : "
        f"{len(test_domains)}"
    )

    print(
        f"Training URLs   : "
        f"{len(train_df)}"
    )

    print(
        f"Testing URLs    : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------
    # 6. Extract model features
    # --------------------------------------------------

    print("\n[5] Extracting URL features...")

    X_train_records = []

    for index, url in enumerate(
        train_df["URL"],
        start=1,
    ):

        features = extract_features(url)

        X_train_records.append(
            {
                feature: features[feature]
                for feature in MODEL_FEATURE_COLUMNS
            }
        )

        if index % 10000 == 0:
            print(
                f"Training features: "
                f"{index}/{len(train_df)}"
            )

    X_test_records = []

    for index, url in enumerate(
        test_df["URL"],
        start=1,
    ):

        features = extract_features(url)

        X_test_records.append(
            {
                feature: features[feature]
                for feature in MODEL_FEATURE_COLUMNS
            }
        )

        if index % 10000 == 0:
            print(
                f"Testing features: "
                f"{index}/{len(test_df)}"
            )

    X_train = pd.DataFrame(
        X_train_records,
        columns=MODEL_FEATURE_COLUMNS,
    )

    X_test = pd.DataFrame(
        X_test_records,
        columns=MODEL_FEATURE_COLUMNS,
    )

    y_train = train_df["label"].reset_index(
        drop=True
    )

    y_test = test_df["label"].reset_index(
        drop=True
    )

    print(
        f"\nFeatures used: "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------
    # 7. Train XGBoost
    # --------------------------------------------------

    print("\n[6] Training XGBoost...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------
    # 8. Evaluate
    # --------------------------------------------------

    print("\n[7] Evaluating on unseen domains...")

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n" + "=" * 60)
    print("UNSEEN-DOMAIN XGBOOST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # --------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------

    print("\n[8] Feature importance:")

    feature_importance = sorted(
        zip(
            MODEL_FEATURE_COLUMNS,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in feature_importance:

        print(
            f"{feature:<30} "
            f"{importance:.6f}"
        )

    print("\n" + "=" * 60)
    print("UNSEEN-DOMAIN EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
