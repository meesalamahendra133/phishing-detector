from __future__ import annotations

import re

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


RANDOM_STATE = 42


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
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
]


def extract_domain(url: str) -> str:
    url = str(url).strip().lower()

    url = re.sub(r"^https?://", "", url)

    domain = url.split("/")[0]
    domain = domain.split("?")[0]
    domain = domain.split("#")[0]
    domain = domain.split(":")[0]

    return domain


def evaluate_model(name, model, X_train, y_train, X_test, y_test):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nTraining model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)

    print("\nResults:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


def main():

    print("=" * 60)
    print("UNSEEN-DOMAIN PHISHING MODEL EVALUATION")
    print("=" * 60)

    print("\n[1] Loading dataset...")

    df = pd.read_csv(
        "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
    )

    print(f"Total rows: {len(df)}")

    # Original PhiUSIIL labels:
    # 0 = phishing
    # 1 = legitimate
    #
    # Convert to project convention:
    # 0 = legitimate
    # 1 = phishing

    df["label"] = df["label"].map({
        0: 1,
        1: 0,
    })

    print("\n[2] Extracting domains...")

    df["domain"] = df["URL"].apply(extract_domain)

    print(f"Unique domains: {df['domain'].nunique()}")

    # --------------------------------------------------
    # Remove duplicate URL/domain-label conflicts
    # --------------------------------------------------

    df = df.drop_duplicates(subset=["URL"])

    conflicting = (
        df.groupby("URL")["label"]
        .nunique()
    )

    conflicting_urls = conflicting[conflicting > 1]

    if len(conflicting_urls) > 0:
        print(
            f"WARNING: {len(conflicting_urls)} URLs "
            "have conflicting labels."
        )

    # --------------------------------------------------
    # Domain-level split
    # --------------------------------------------------

    print("\n[3] Creating unseen-domain split...")

    domains = df["domain"].drop_duplicates()

    domains = domains.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    split_index = int(len(domains) * 0.80)

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

    print(f"Training domains: {len(train_domains)}")
    print(f"Testing domains : {len(test_domains)}")

    print(f"Training URLs   : {len(train_df)}")
    print(f"Testing URLs    : {len(test_df)}")

    # --------------------------------------------------
    # Features
    # --------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["label"].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["label"].copy()

    print(f"\nFeatures used: {len(FEATURE_COLUMNS)}")

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    evaluate_model(
        "RANDOM FOREST - UNSEEN DOMAINS",
        rf_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    xgb_model = XGBClassifier(
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

    evaluate_model(
        "XGBOOST - UNSEEN DOMAINS",
        xgb_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("UNSEEN-DOMAIN EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

