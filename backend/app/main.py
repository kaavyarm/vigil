import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

import math  # noqa: E402

import shap  # noqa: E402

from explain_model import explain_patient, explain_vector, load_model_and_data  # noqa: E402
from process_one_patient import ICU_TYPE_MAP  # noqa: E402
from risk_trend import build_risk_trend  # noqa: E402
from timeline_events import build_patient_timeline_events  # noqa: E402


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


app = FastAPI(title="Vigil ICU Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATIENT_DIR = PROJECT_ROOT / "data/processed/patients"

model, df, X, feature_columns = load_model_and_data()
explainer = shap.TreeExplainer(model)
train_medians = joblib.load(PROJECT_ROOT / "models/train_medians.joblib")


class PatientVitals(BaseModel):
    Age: float | None = None
    HR: float | None = None
    MAP: float | None = None
    GCS: float | None = None
    RespRate: float | None = None
    Temp: float | None = None
    SaO2: float | None = None
    Creatinine: float | None = None
    Lactate: float | None = None
    BUN: float | None = None
    Urine: float | None = None
    MechVent: int | None = None


def _build_predict_vector(vitals: PatientVitals) -> pd.DataFrame:
    features = dict(train_medians)

    for param, value in vitals.model_dump(exclude_none=True).items():
        if param == "Age":
            features["Age"] = float(value)
        elif param == "MechVent":
            v = float(value)
            features.update({
                "MechVent_last": v, "MechVent_mean": v, "MechVent_min": v,
                "MechVent_max": v, "MechVent_std": 0.0, "MechVent_count": 1,
                "MechVent_trend": 0.0, "MechVent_measured": 1,
                "MechVent_flag": int(value > 0),
            })
        else:
            v = float(value)
            features.update({
                f"{param}_last": v, f"{param}_mean": v, f"{param}_min": v,
                f"{param}_max": v, f"{param}_std": 0.0, f"{param}_count": 1,
                f"{param}_trend": 0.0, f"{param}_measured": 1,
            })
            if param == "Urine":
                features["Urine_total"] = v

    return pd.DataFrame([{col: features.get(col, np.nan) for col in feature_columns}])


@app.get("/")
def root():
    return {"message": "Vigil API is running"}


@app.get("/patients")
def get_patients(limit: int = 500):
    subset = df.head(limit)
    risks = model.predict_proba(X.loc[subset.index])[:, 1]

    patients = []
    for i, (_, row) in enumerate(subset.iterrows()):
        risk = float(risks[i])
        icu_raw = None if pd.isna(row["ICUType"]) else int(row["ICUType"])

        patients.append(
            {
                "record_id": int(row["RecordID"]),
                "age": None if pd.isna(row["Age"]) else int(row["Age"]),
                "icu_type": icu_raw,
                "icu_type_label": ICU_TYPE_MAP.get(icu_raw, "ICU") if icu_raw else "ICU",
                "mortality_risk": risk,
                "mortality_risk_percent": round(risk * 100, 1),
                "status": (
                    "Critical" if risk >= 0.8
                    else "High" if risk >= 0.5
                    else "Moderate" if risk >= 0.2
                    else "Low"
                ),
            }
        )

    patients.sort(key=lambda p: p["mortality_risk"], reverse=True)
    return patients


@app.post("/predict")
def predict_custom(vitals: PatientVitals):
    vector = _build_predict_vector(vitals)
    result = explain_vector(model=model, feature_vector=vector, top_n=10, explainer=explainer)
    return make_json_safe(result)


@app.get("/patients/{record_id}")
def get_patient(record_id: int):
    patient_path = PATIENT_DIR / f"{record_id}.json"
    if not patient_path.exists():
        raise HTTPException(status_code=404, detail="Patient not found")
    with open(patient_path, "r") as f:
        patient_data = json.load(f)
    return make_json_safe(patient_data)


@app.get("/patients/{record_id}/explanation")
def get_patient_explanation(record_id: int):
    try:
        return explain_patient(model=model, df=df, X=X, record_id=record_id, top_n=10, explainer=explainer)
    except ValueError:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/patients/{record_id}/risk-trend")
def get_patient_risk_trend(record_id: int):
    try:
        return build_risk_trend(record_id, model, feature_columns)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/patients/{record_id}/timeline-events")
def get_patient_timeline_events(record_id: int):
    try:
        return build_patient_timeline_events(record_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Patient not found")
