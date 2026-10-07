from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.phiusiil.dataset import load_phiusiil, get_features_and_labels
from src.phiusiil.split import split_dataset


RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("PHISHING DETECTION - RANDOM FOREST TRAINING")
    print("=" * 60)

    # 1. Load prepared dataset
    print("\n[1] Loading dataset...")
    df = load_phiusiil()

    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns) - 2}")

    # 2. Separate features and labels
    X, y = get_features_and_labels(df)

    # 3. Split dataset
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

    # 4. Create Random Forest model
    print("\n[3] Creating Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    # 5. Train
    print("\n[4] Training model...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # 6. Validation evaluation
    print("\n[5] Validation evaluation...")

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("\nValidation Classification Report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            target_names=["Legitimate", "Phishing"],
        )
    )

    # 7. Final test evaluation
    print("\n[6] Test evaluation...")

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nTest Classification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Phishing"],
        )
    )

    print("=" * 60)
    print("RANDOM FOREST TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
