
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_v0.2.joblib"


def train_model():

    # Load the diabetes dataset
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    # Split dataset in the same random state for reproducible evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    #TODO shouldn't version 2 be with a random forest????????????????????????????????
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )

    # Train and evaluate model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)

    # Save the fitted model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model version: v0.2")
    print("Model: Ridge Regression (alpha=10.0)")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Saved model: {MODEL_PATH}")

    return rmse


if __name__ == "__main__":
    train_model()
