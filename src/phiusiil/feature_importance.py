from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("FEATURE IMPORTANCE AUDIT")
    print("=" * 60)

    print("\n[1] Loading dataset...")
    df = load_phiusiil()

    X, y = get_features_and_labels(df)

    print(f"Rows     : {len(df)}")
    print(f"Features : {X.shape[1]}")

    print("\n[2] Splitting dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    print("\n[3] Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf_model.fit(X_train, y_train)

    rf_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf_model.feature_importances_,
    })

    rf_importance = rf_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nRandom Forest Feature Importance:")
    print(rf_importance.to_string(index=False))

    # --------------------------------------------------
    # XGBOOST
    # --------------------------------------------------

    print("\n[4] Training XGBoost...")

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

    xgb_model.fit(X_train, y_train)

    xgb_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_,
    })

    xgb_importance = xgb_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nXGBoost Feature Importance:")
    print(xgb_importance.to_string(index=False))

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
