import json
from pathlib import Path

import joblib
import pandas as pd

from explain_model import get_risk_level
from feature_engineering import extract_patient_features, load_patient_json

_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATIENT_DIR = _ROOT / "data/processed/patients"
MODEL_PATH = _ROOT / "models/vigil_xgboost_initial.joblib"
FEATURE_COLUMNS_PATH = _ROOT / "models/feature_columns.joblib"
OUTPUT_DIR = _ROOT / "data/predictions"


SNAPSHOT_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]


def create_patient_snapshot(patient_data: dict, max_hour: int) -> dict:
    """Keep only timeline events up to a given hour."""
    filtered_timeline = [
        event for event in patient_data["timeline"]
        if event["hour"] <= max_hour
    ]

    return {
        "static_info": patient_data["static_info"],
        "timeline": filtered_timeline,
    }


def predict_snapshot_risk(model, feature_columns: list[str], snapshot_data: dict) -> float:
    features = extract_patient_features(snapshot_data)
    feature_row = pd.DataFrame([features]).reindex(columns=feature_columns)

    risk_probability = model.predict_proba(feature_row)[0, 1]

    return float(risk_probability)


def build_risk_trend(record_id: int, model, feature_columns: list[str]) -> dict:
    patient_path = PROCESSED_PATIENT_DIR / f"{record_id}.json"
    patient_data = load_patient_json(patient_path)

    trend = []

    for hour in SNAPSHOT_HOURS:
        snapshot_data = create_patient_snapshot(patient_data, hour)

        if not snapshot_data["timeline"]:
            continue

        risk_probability = predict_snapshot_risk(
            model=model,
            feature_columns=feature_columns,
            snapshot_data=snapshot_data,
        )

        trend.append(
            {
                "hour": hour,
                "mortality_risk": risk_probability,
                "mortality_risk_percent": round(risk_probability * 100, 1),
                "risk_level": get_risk_level(risk_probability),
            }
        )

    return {
        "record_id": record_id,
        "risk_trend": trend,
    }


def save_risk_trend_plot(record_id: int, risk_trend: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    hours = [point["hour"] for point in risk_trend]
    risks = [point["mortality_risk_percent"] for point in risk_trend]

    output_path = OUTPUT_DIR / f"{record_id}_risk_trend.png"

    plt.figure(figsize=(8, 4))
    plt.plot(hours, risks, marker="o")
    plt.title(f"Mortality Risk Trend for Patient {record_id}")
    plt.xlabel("ICU Hour")
    plt.ylabel("Predicted Mortality Risk (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def main():
    record_id = int(input("Enter Record ID: "))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    result = build_risk_trend(record_id, model, feature_columns)

    json_output_path = OUTPUT_DIR / f"{record_id}_risk_trend.json"

    with open(json_output_path, "w") as f:
        json.dump(result, f, indent=2)

    plot_output_path = save_risk_trend_plot(
        record_id=record_id,
        risk_trend=result["risk_trend"],
    )

    print(json.dumps(result, indent=2))
    print(f"\nSaved risk trend JSON to {json_output_path}")
    print(f"Saved risk trend plot to {plot_output_path}")


if __name__ == "__main__":
    main()