import json
import logging
import logging.config
import math
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
from explain_model import (
    explain_patient,
    explain_vector,
    get_risk_level,
    load_model_and_data,
)
from process_one_patient import ICU_TYPE_MAP
from risk_trend import build_risk_trend
from s3_loader import sync_from_s3
from timeline_events import build_patient_timeline_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Logging ───────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Structured JSON log lines — compatible with CloudWatch Logs Insights."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        })


logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": _JsonFormatter}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "json"}},
    "root": {"handlers": ["console"], "level": "INFO"},
})

logger = logging.getLogger(__name__)


# ── Startup ───────────────────────────────────────────────────────────────────

sync_from_s3(PROJECT_ROOT)

model, df, X, feature_columns = load_model_and_data()
explainer = shap.TreeExplainer(model)
train_medians = joblib.load(PROJECT_ROOT / "models/train_medians.joblib")

# SHAP TreeExplainer is not thread-safe; serialize all calls through this lock.
_explainer_lock = threading.Lock()

db.init_pool()

PATIENT_DIR = PROJECT_ROOT / "data/processed/patients"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Vigil ICU Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


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


def _patient_summary(record_id: int, age: int | None, icu_type: int | None, risk: float) -> dict:
    return {
        "record_id": record_id,
        "age": age,
        "icu_type": icu_type,
        "icu_type_label": ICU_TYPE_MAP.get(icu_type, "ICU") if icu_type else "ICU",
        "mortality_risk": risk,
        "mortality_risk_percent": round(risk * 100, 1),
        "status": get_risk_level(risk),
    }


_monitoring_report: dict | None = None
_monitoring_lock = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {"message": "Vigil API is running"}


@app.get("/health")
def health() -> dict:
    """ECS / load-balancer health check."""
    return {
        "status": "ok",
        "db": "connected" if db.is_available() else "unavailable",
    }


@app.get("/monitoring/report")
def get_monitoring_report() -> dict:
    global _monitoring_report
    if _monitoring_report is None:
        with _monitoring_lock:
            if _monitoring_report is None:
                report_path = PROJECT_ROOT / "data/monitoring/report.json"
                if not report_path.exists():
                    raise HTTPException(
                        status_code=503,
                        detail="Monitoring report not available. Run scripts/simulate_monitoring.py first.",
                    )
                with open(report_path) as f:
                    _monitoring_report = json.load(f)
    return _monitoring_report


@app.get("/patients")
def get_patients(limit: int = 500) -> list:
    if db.is_available():
        return db.list_patients(limit)

    subset = df.head(limit)
    risks = model.predict_proba(X.loc[subset.index])[:, 1]

    patients = [
        _patient_summary(
            record_id=int(row["RecordID"]),
            age=None if pd.isna(row["Age"]) else int(row["Age"]),
            icu_type=None if pd.isna(row["ICUType"]) else int(row["ICUType"]),
            risk=float(risks[i]),
        )
        for i, (_, row) in enumerate(subset.iterrows())
    ]
    patients.sort(key=lambda p: p["mortality_risk"], reverse=True)
    return patients


@app.post("/predict")
def predict_custom(vitals: PatientVitals) -> dict:
    vector = _build_predict_vector(vitals)
    with _explainer_lock:
        result = explain_vector(model=model, feature_vector=vector, top_n=10, explainer=explainer)
    return make_json_safe(result)


@app.get("/patients/{record_id}")
def get_patient(record_id: int) -> dict:
    if db.is_available():
        data = db.get_patient(record_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Patient not found") from None
        return make_json_safe(data)

    patient_path = PATIENT_DIR / f"{record_id}.json"
    if not patient_path.exists():
        raise HTTPException(status_code=404, detail="Patient not found") from None
    with open(patient_path) as f:
        return make_json_safe(json.load(f))


@app.get("/patients/{record_id}/explanation")
def get_patient_explanation(record_id: int) -> dict:
    try:
        with _explainer_lock:
            result = explain_patient(
                model=model, df=df, X=X, record_id=record_id, top_n=10, explainer=explainer
            )
        return make_json_safe(result)
    except ValueError:
        raise HTTPException(status_code=404, detail="Patient not found") from None


@app.get("/patients/{record_id}/risk-trend")
def get_patient_risk_trend(record_id: int) -> dict:
    try:
        return make_json_safe(build_risk_trend(record_id, model, feature_columns))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Patient not found") from None


@app.get("/patients/{record_id}/timeline-events")
def get_patient_timeline_events(record_id: int) -> dict:
    try:
        return make_json_safe(build_patient_timeline_events(record_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Patient not found") from None
