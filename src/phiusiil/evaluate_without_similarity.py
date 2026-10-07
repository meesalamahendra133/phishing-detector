from __future__ import annotations

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

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42


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
    print("PHISHING MODEL AUDIT")
    print("WITHOUT URLSimilarityIndex")
    print("=" * 60)

    print("\n[1] Loading dataset...")

    df = load_phiusiil()

    X, y = get_features_and_labels(df)

    print(f"Original features: {X.shape[1]}")

    # Remove suspicious feature
    X = X.drop(columns=["URLSimilarityIndex"])

    print(f"Features after removal: {X.shape[1]}")

    print("\n[2] Splitting dataset...")

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

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    evaluate_model(
        "RANDOM FOREST",
        rf_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # XGBOOST
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
        "XGBOOST",
        xgb_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
