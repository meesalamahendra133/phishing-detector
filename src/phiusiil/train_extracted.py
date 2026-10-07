from __future__ import annotations

from pathlib import Path

import joblib
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
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "phiusiil_url_features.csv"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "xgboost_extracted_features.joblib"
)


def main():
    print("=" * 60)
    print("XGBOOST TRAINING - EXTRACTED URL FEATURES")
    print("=" * 60)

    print("\n[1] Loading extracted feature dataset...")

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    print(f"Rows     : {len(df)}")
    print(f"Features : {X.shape[1]}")

    print("\n[2] Creating train/validation/test split...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Testing    : {X_test.shape}")

    print("\n[3] Training XGBoost...")

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

    print("XGBoost training completed.")

    print("\n[4] Evaluating model...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\n[5] Feature importance:")

    importances = model.feature_importances_

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in feature_importance:
        print(
            f"{feature:<30} {importance:.6f}"
        )

    print("\n[6] Saving model...")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Model saved to:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
