from pathlib import Path
import json

import joblib
import pandas as pd
import shap


_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _ROOT / "data/features/training_dataset.csv"
MODEL_PATH = _ROOT / "models/vigil_xgboost_initial.joblib"
FEATURE_COLUMNS_PATH = _ROOT / "models/feature_columns.joblib"
PREDICTIONS_DIR = _ROOT / "data/predictions"


PARAMETER_METADATA = {
    "GCS":       {"unit": "/15",      "name": "Consciousness (GCS)"},
    "HR":        {"unit": "bpm",      "name": "Heart Rate"},
    "MAP":       {"unit": "mmHg",     "name": "Mean Arterial Pressure"},
    "SysABP":    {"unit": "mmHg",     "name": "Systolic BP (arterial)"},
    "DiasABP":   {"unit": "mmHg",     "name": "Diastolic BP (arterial)"},
    "NIMAP":     {"unit": "mmHg",     "name": "Non-invasive MAP"},
    "NISysABP":  {"unit": "mmHg",     "name": "Non-invasive Systolic BP"},
    "NIDiasABP": {"unit": "mmHg",     "name": "Non-invasive Diastolic BP"},
    "Temp":      {"unit": "°C",       "name": "Temperature"},
    "RespRate":  {"unit": "br/min",   "name": "Respiratory Rate"},
    "SaO2":      {"unit": "%",        "name": "O₂ Saturation"},
    "FiO2":      {"unit": "",         "name": "Inspired O₂ Fraction"},
    "PaO2":      {"unit": "mmHg",     "name": "Arterial PO₂"},
    "PaCO2":     {"unit": "mmHg",     "name": "Arterial PCO₂"},
    "Creatinine":{"unit": "mg/dL",    "name": "Creatinine"},
    "BUN":       {"unit": "mg/dL",    "name": "BUN"},
    "WBC":       {"unit": "×10³/μL",  "name": "White Blood Cells"},
    "Platelets": {"unit": "×10³/μL",  "name": "Platelets"},
    "HCT":       {"unit": "%",        "name": "Hematocrit"},
    "Na":        {"unit": "mEq/L",    "name": "Sodium"},
    "K":         {"unit": "mEq/L",    "name": "Potassium"},
    "Mg":        {"unit": "mg/dL",    "name": "Magnesium"},
    "HCO3":      {"unit": "mEq/L",    "name": "Bicarbonate"},
    "Bilirubin": {"unit": "mg/dL",    "name": "Bilirubin"},
    "ALT":       {"unit": "U/L",      "name": "ALT"},
    "AST":       {"unit": "U/L",      "name": "AST"},
    "ALP":       {"unit": "U/L",      "name": "ALP"},
    "Albumin":   {"unit": "g/dL",     "name": "Albumin"},
    "Glucose":   {"unit": "mg/dL",    "name": "Glucose"},
    "Lactate":   {"unit": "mmol/L",   "name": "Lactate"},
    "Urine":     {"unit": "mL",       "name": "Urine Output"},
    "MechVent":  {"unit": "",         "name": "Mechanical Ventilation"},
    "Age":       {"unit": "yrs",      "name": "Age"},
    "Gender":    {"unit": "",         "name": "Sex"},
    "Height":    {"unit": "cm",       "name": "Height"},
    "Weight":    {"unit": "kg",       "name": "Weight"},
    "ICUType":   {"unit": "",         "name": "ICU Type"},
}

RECOMMENDED_ACTIONS = {
    "Critical": "Immediate physician review required. Patient shows signs of critical deterioration — consider rapid assessment and ICU escalation.",
    "High": "Prioritize for clinical review within the hour. Reassess recent vitals, active medications, and trending lab values.",
    "Moderate": "Monitor closely. Flag for nursing assessment and watch for further deterioration in key indicators.",
    "Low": "Continue routine monitoring per standard ICU protocol.",
}


def build_feature_label(feature_name: str) -> str:
    parts = feature_name.rsplit("_", 1)
    if len(parts) != 2:
        return feature_name

    param, suffix = parts
    meta = PARAMETER_METADATA.get(param)
    if not meta:
        return feature_name

    name = meta["name"]
    if suffix == "mean":
        return f"Avg. {name}"
    if suffix == "min":
        return f"Min. {name}"
    if suffix == "max":
        return f"Max. {name}"
    if suffix == "last":
        return f"Latest {name}"
    if suffix == "std":
        return f"{name} Variability"
    if suffix == "trend":
        return f"{name} Trend"
    if suffix == "count":
        return f"{name} Readings"
    if suffix in ("measured", "flag"):
        return name
    if suffix == "total":
        return f"Total {name}"
    return feature_name


def build_value_label(feature_name: str, value) -> str:
    if value is None:
        return "not measured"

    parts = feature_name.rsplit("_", 1)
    if len(parts) != 2:
        return f"{round(value, 2)}" if isinstance(value, float) else str(value)

    param, suffix = parts
    meta = PARAMETER_METADATA.get(param)

    if suffix == "trend":
        direction = "↑" if value > 0 else "↓" if value < 0 else "→"
        return f"{direction} {abs(value):.3f}/hr"
    if suffix == "std":
        return f"σ = {value:.2f}"
    if suffix in ("measured", "flag"):
        return "yes" if value == 1 else "no"
    if suffix == "count":
        return f"{int(value)} readings"

    if meta and meta["unit"]:
        return f"{round(value, 2)} {meta['unit']}"

    return f"{round(value, 2)}" if isinstance(value, float) else str(value)


def calculate_confidence(risk_probability: float) -> dict:
    risk_probability = float(risk_probability)
    distance = abs(risk_probability - 0.5)
    confidence_score = float(distance / 0.5)

    if confidence_score >= 0.8:
        label = "High"
    elif confidence_score >= 0.5:
        label = "Moderate"
    else:
        label = "Low"

    return {"score": float(round(confidence_score, 3)), "label": label}


def get_risk_level(risk_probability: float) -> str:
    risk_probability = float(risk_probability)
    if risk_probability < 0.2:
        return "Low"
    elif risk_probability < 0.5:
        return "Moderate"
    elif risk_probability < 0.8:
        return "High"
    return "Critical"


def clean_value(value):
    if pd.isna(value):
        return None
    return float(value)


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[feature_columns]
    return model, df, X, feature_columns


def _shap_explanation(model, patient_features: pd.DataFrame, top_n: int, explainer) -> dict:
    risk_probability = float(model.predict_proba(patient_features)[0, 1])

    if explainer is None:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_features)

    rows = []
    for feat, val, sv in zip(
        patient_features.columns,
        patient_features.iloc[0],
        shap_values[0],
    ):
        rows.append({
            "feature": str(feat),
            "label": build_feature_label(str(feat)),
            "value": clean_value(val),
            "value_label": build_value_label(str(feat), clean_value(val)),
            "shap_value": float(sv),
            "abs_shap": float(abs(sv)),
        })

    all_df = pd.DataFrame(rows)

    risk_df = (
        all_df[all_df["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .head(top_n)
    )
    protective_df = (
        all_df[all_df["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .head(top_n)
    )

    total_risk = risk_df["shap_value"].sum()
    total_protective = protective_df["abs_shap"].sum()

    def format_factors(factor_df, total):
        return [
            {
                "feature": row["feature"],
                "label": row["label"],
                "value": row["value"],
                "value_label": row["value_label"],
                "shap_value": row["shap_value"],
                "contribution_pct": round(row["abs_shap"] / total * 100, 1) if total > 0 else 0,
            }
            for row in factor_df.to_dict(orient="records")
        ]

    risk_level = get_risk_level(risk_probability)

    return {
        "mortality_risk": float(risk_probability),
        "mortality_risk_percent": float(round(risk_probability * 100, 1)),
        "risk_level": risk_level,
        "confidence": calculate_confidence(risk_probability),
        "recommended_action": RECOMMENDED_ACTIONS[risk_level],
        "risk_factors": format_factors(risk_df, total_risk),
        "protective_factors": format_factors(protective_df, total_protective),
    }


def explain_patient(model, df, X, record_id: int, top_n: int = 10, explainer=None) -> dict:
    patient_row = df[df["RecordID"] == record_id]
    if patient_row.empty:
        raise ValueError(f"No patient found with RecordID {record_id}")

    patient_features = X.loc[[patient_row.index[0]]]
    result = _shap_explanation(model, patient_features, top_n, explainer)
    result["record_id"] = int(record_id)
    return result


def explain_vector(model, feature_vector: pd.DataFrame, top_n: int = 10, explainer=None) -> dict:
    return _shap_explanation(model, feature_vector, top_n, explainer)


def main():
    model, df, X, _ = load_model_and_data()
    record_id = int(input("Enter Record ID: "))

    explanation = explain_patient(model=model, df=df, X=X, record_id=record_id, top_n=10)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"{record_id}_explanation.json"
    with open(output_path, "w") as f:
        json.dump(explanation, f, indent=2)

    print(json.dumps(explanation, indent=2))
    print(f"\nSaved explanation to {output_path}")


if __name__ == "__main__":
    main()
