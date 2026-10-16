
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model_v0.2.joblib"
MODEL_VERSION = "v0.2"

app = FastAPI(title="Virtual Diabetes Clinic Triage")


class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(allow_inf_nan=False)
    sex: float = Field(allow_inf_nan=False)
    bmi: float = Field(allow_inf_nan=False)
    bp: float = Field(allow_inf_nan=False)
    s1: float = Field(allow_inf_nan=False)
    s2: float = Field(allow_inf_nan=False)
    s3: float = Field(allow_inf_nan=False)
    s4: float = Field(allow_inf_nan=False)
    s5: float = Field(allow_inf_nan=False)
    s6: float = Field(allow_inf_nan=False)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: DiabetesFeatures):
    if not MODEL_PATH.exists():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Model file not found. Train the model first.",
        )

    model = joblib.load(MODEL_PATH)
    input_data = pd.DataFrame(
    [[
        features.age,
        features.sex,
        features.bmi,
        features.bp,
        features.s1,
        features.s2,
        features.s3,
        features.s4,
        features.s5,
        features.s6,
    ]],
    columns=[
        "age", "sex", "bmi", "bp", "s1",
        "s2", "s3", "s4", "s5", "s6",
    ],
)

    prediction = float(model.predict(input_data)[0])
    return {"prediction": prediction}
