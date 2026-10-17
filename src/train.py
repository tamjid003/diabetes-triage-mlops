
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


def train_model():

    # Load the diabetes dataset from sklearn
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    
    # Use the same random split for reproducible evaluation for v01 and v02
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Baseline model: simple StandardScaler + LinearRegression
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    # Train and evaluate v01 model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

    # Save the fitted v01 model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model version: v0.1")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Saved model: {MODEL_PATH}")

    return rmse


if __name__ == "__main__":
    train_model()
