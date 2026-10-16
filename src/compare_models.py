

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "v0.1 Linear Regression": LinearRegression(),
        "Ridge alpha=0.1": Ridge(alpha=0.1),
        "Ridge alpha=1.0": Ridge(alpha=1.0),
        "Ridge alpha=10.0": Ridge(alpha=10.0),
    }

    results = []

    for name, regressor in models.items():
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)

        results.append((name, rmse))
        print(f"{name}: RMSE = {rmse:.4f}")

    print("\nLower RMSE is better.")
    best_name, best_rmse = min(results, key=lambda result: result[1])
    print(f"Best model: {best_name}")
    print(f"Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
