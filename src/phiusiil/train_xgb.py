from __future__ import annotations

from xgboost import XGBClassifier

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("XGBOOST - PHISHING URL CLASSIFICATION")
    print("=" * 60)

    # Load prepared dataset
    print("\n[1] Loading dataset...")
    df = load_phiusiil()

    X, y = get_features_and_labels(df)

    print(f"Rows: {len(df)}")
    print(f"Features: {X.shape[1]}")

    # Split dataset
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

    # Create XGBoost model
    print("\n[3] Creating XGBoost model...")

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

    # Train
    print("\n[4] Training XGBoost...")

    model.fit(X_train, y_train)

    print("XGBoost training completed.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
