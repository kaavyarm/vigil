from pathlib import Path
import json

from feature_engineering import load_patient_json


_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATIENT_DIR = _ROOT / "data/processed/patients"
OUTPUT_DIR = _ROOT / "data/predictions"


PARAMETER_CATEGORY = {
    # Hemodynamic vitals
    "HR": "vital",
    "Temp": "vital",
    "MAP": "vital",
    "NIMAP": "vital",
    "SysABP": "vital",
    "NISysABP": "vital",
    "DiasABP": "vital",
    "NIDiasABP": "vital",
    # Neurological
    "GCS": "neurological",
    # Respiratory
    "RespRate": "respiratory",
    "SaO2": "respiratory",
    "PaO2": "respiratory",
    "PaCO2": "respiratory",
    "FiO2": "respiratory",
    "MechVent": "respiratory",
    # Labs
    "Creatinine": "lab",
    "BUN": "lab",
    "WBC": "lab",
    "Platelets": "lab",
    "Bilirubin": "lab",
    "Glucose": "lab",
    "HCO3": "lab",
    "Lactate": "lab",
    "Na": "lab",
    "K": "lab",
    "Mg": "lab",
    "HCT": "lab",
    "Albumin": "lab",
    "ALT": "lab",
    "AST": "lab",
    "ALP": "lab",
    # Fluids
    "Urine": "fluid",
}


ALERT_RULES = {
    "HR": [
        (lambda v: v > 120, "Heart rate spike", "High"),
        (lambda v: v < 50, "Low heart rate", "Moderate"),
    ],
    "Temp": [
        (lambda v: v >= 38.5, "Fever", "Moderate"),
        (lambda v: v < 35.5, "Low temperature", "Moderate"),
    ],
    "MAP": [
        (lambda v: v < 65, "Low mean arterial pressure", "High"),
    ],
    "NIMAP": [
        (lambda v: v < 65, "Low non-invasive mean arterial pressure", "High"),
    ],
    "SysABP": [
        (lambda v: v < 90, "Low systolic blood pressure", "High"),
    ],
    "NISysABP": [
        (lambda v: v < 90, "Low non-invasive systolic blood pressure", "High"),
    ],
    "RespRate": [
        (lambda v: v > 24, "High respiratory rate", "Moderate"),
        (lambda v: v < 8, "Low respiratory rate", "High"),
    ],
    "SaO2": [
        (lambda v: v < 90, "Low oxygen saturation", "High"),
    ],
    "PaO2": [
        (lambda v: v < 60, "Low arterial oxygen", "High"),
    ],
    "GCS": [
        (lambda v: v < 8, "Severely reduced consciousness", "Critical"),
        (lambda v: v < 13, "Reduced consciousness", "High"),
    ],
    "Lactate": [
        (lambda v: v >= 4, "Severely elevated lactate", "Critical"),
        (lambda v: v > 2, "Elevated lactate", "High"),
    ],
    "Creatinine": [
        (lambda v: v > 2, "Elevated creatinine", "High"),
    ],
    "BUN": [
        (lambda v: v > 30, "Elevated BUN", "Moderate"),
    ],
    "WBC": [
        (lambda v: v > 20, "High white blood cell count", "High"),
        (lambda v: v < 4, "Low white blood cell count", "Moderate"),
    ],
    "Platelets": [
        (lambda v: v < 100, "Low platelets", "High"),
    ],
    "Bilirubin": [
        (lambda v: v > 2, "Elevated bilirubin", "High"),
    ],
    "Glucose": [
        (lambda v: v > 250, "High glucose", "Moderate"),
        (lambda v: v < 70, "Low glucose", "High"),
    ],
    "HCO3": [
        (lambda v: v < 18, "Low bicarbonate / possible acidosis", "High"),
    ],
    "MechVent": [
        (lambda v: v == 1, "Mechanical ventilation", "High"),
    ],
    "Urine": [
        (lambda v: v == 0, "No urine output recorded", "High"),
    ],
}


def classify_event(parameter: str, value: float):
    """Return a clinical event if the measurement crosses a threshold."""

    if value is None:
        return None

    if parameter not in ALERT_RULES:
        return None

    for condition, label, severity in ALERT_RULES[parameter]:
        if condition(value):
            return {
                "parameter": parameter,
                "value": value,
                "event": label,
                "severity": severity,
                "category": PARAMETER_CATEGORY.get(parameter, "other"),
            }

    return None


def extract_timeline_events(patient_data: dict) -> list[dict]:
    events = []

    for timeline_point in patient_data["timeline"]:
        time = timeline_point["time"]
        measurements = timeline_point["measurements"]

        for parameter, value in measurements.items():
            event = classify_event(parameter, value)

            if event:
                events.append({"time": time, **event})

    return events


def build_patient_timeline_events(record_id: int) -> dict:
    patient_path = PROCESSED_PATIENT_DIR / f"{record_id}.json"
    patient_data = load_patient_json(patient_path)

    events = extract_timeline_events(patient_data)

    return {
        "record_id": record_id,
        "timeline_events": events,
    }


def main():
    record_id = int(input("Enter Record ID: "))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = build_patient_timeline_events(record_id)

    output_path = OUTPUT_DIR / f"{record_id}_timeline_events.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved timeline events to {output_path}")


if __name__ == "__main__":
    main()