from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("RANDOM FOREST - PHISHING URL CLASSIFICATION")
    print("=" * 60)

    # Load prepared dataset
    print("\n[1] Loading dataset...")
    df = load_phiusiil()

    print(f"Rows: {len(df)}")
    print(f"Features: {len(get_features_and_labels(df)[0].columns)}")

    # Separate features and labels
    X, y = get_features_and_labels(df)

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

    # Create Random Forest
    print("\n[3] Creating Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Train
    print("\n[4] Training Random Forest...")

    model.fit(X_train, y_train)

    print("Random Forest training completed.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
