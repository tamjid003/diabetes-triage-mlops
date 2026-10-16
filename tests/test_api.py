
import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health_returns_model_version():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v0.2",
    }


def test_predict_returns_numeric_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    prediction = response.json()["prediction"]
    assert isinstance(prediction, (int, float))
    assert math.isfinite(prediction)


def test_predict_rejects_missing_feature():
    payload = VALID_PAYLOAD.copy()
    payload.pop("s6")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "s6" for error in errors)


def test_predict_rejects_non_numeric_feature():
    payload = VALID_PAYLOAD.copy()
    payload["bmi"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_extra_feature():
    payload = VALID_PAYLOAD.copy()
    payload["unexpected"] = 123

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
