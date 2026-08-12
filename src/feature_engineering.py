import json
from pathlib import Path

import numpy as np


def load_patient_json(file_path: str | Path) -> dict:
    with open(file_path) as f:
        return json.load(f)


def extract_static_features(patient_data: dict) -> dict:
    static = patient_data["static_info"]

    return {
        "RecordID": int(static.get("RecordID")),
        "Age": static.get("Age"),
        "Gender": static.get("Gender"),
        "Height": static.get("Height"),
        "Weight": static.get("Weight"),
        "ICUType": static.get("ICUType"),
    }


def get_measurements_for_parameter(patient_data: dict, parameter: str) -> list[dict]:
    measurements = []

    for event in patient_data["timeline"]:
        if parameter in event["measurements"]:
            measurements.append(
                {
                    "hour": event["hour"],
                    "value": event["measurements"][parameter],
                }
            )

    return measurements


def extract_numeric_summary(measurements: list[dict], prefix: str) -> dict:
    if len(measurements) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_count": 0,
            f"{prefix}_trend": np.nan,
            f"{prefix}_measured": 0,
        }

    values = np.array([m["value"] for m in measurements], dtype=float)
    hours = np.array([m["hour"] for m in measurements], dtype=float)

    trend = float(np.polyfit(hours, values, 1)[0]) if len(values) >= 2 else np.nan

    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_last": float(values[-1]),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_count": len(values),
        f"{prefix}_trend": trend if not np.isnan(trend) else np.nan,
        f"{prefix}_measured": 1,
    }


def extract_patient_features(patient_data: dict) -> dict:
    features = extract_static_features(patient_data)

    parameters = [
        "HR", "MAP", "SysABP", "DiasABP", "NIMAP", "NISysABP", "NIDiasABP",
        "RespRate", "SaO2", "FiO2", "PaO2", "PaCO2", "GCS", "Temp",
        "Creatinine", "BUN", "Urine", "Na", "K", "Mg", "HCO3", "WBC",
        "HCT", "Platelets", "Albumin", "Bilirubin", "ALT", "AST", "ALP",
        "Glucose", "Lactate", "MechVent",
    ]

    for parameter in parameters:
        measurements = get_measurements_for_parameter(patient_data, parameter)
        features.update(extract_numeric_summary(measurements, parameter))

    urine_measurements = get_measurements_for_parameter(patient_data, "Urine")
    if urine_measurements:
        urine_values = np.array([m["value"] for m in urine_measurements], dtype=float)
        features["Urine_total"] = float(np.nansum(urine_values))
    else:
        features["Urine_total"] = np.nan

    mechvent_measurements = get_measurements_for_parameter(patient_data, "MechVent")
    if mechvent_measurements:
        mechvent_values = np.array([m["value"] for m in mechvent_measurements], dtype=float)
        features["MechVent_flag"] = int(np.nanmax(mechvent_values) > 0)
    else:
        features["MechVent_flag"] = 0

    return features


def main():
    file_path = Path("data/processed/patients/132539.json")
    patient_data = load_patient_json(file_path)
    features = extract_patient_features(patient_data)

    print("\nFEATURES FOR ONE PATIENT")
    print("========================")
    for key, value in features.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
