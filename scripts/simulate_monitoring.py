"""
Simulates a monitoring run against an incoming patient cohort.

Uses the last 500 patients in the feature dataset as the "incoming" cohort
and everything before that as the reference (training) distribution.

Outputs: data/monitoring/report.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from monitoring import generate_report  # noqa: E402

DATA_PATH = ROOT / "data/features/training_dataset.csv"
MODEL_PATH = ROOT / "models/vigil_xgboost_initial.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models/feature_columns.joblib"
OUTPUT_DIR = ROOT / "data/monitoring"

TARGET_COL = "In-hospital_death"
LEAKAGE_COLS = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", TARGET_COL]

# Training-time baselines from Phase 1 evaluation (tuned XGBoost, held-out test set)
BASELINE_BRIER = 0.0892
BASELINE_ECE = 0.0300

INCOMING_COHORT_SIZE = 500


def main():
    print("Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    X = df.drop(columns=LEAKAGE_COLS)
    y = df[TARGET_COL]

    # Simulate: last N patients as incoming cohort, everything before as reference.
    train_df = X.iloc[:-INCOMING_COHORT_SIZE].copy()
    incoming_df = X.iloc[-INCOMING_COHORT_SIZE:].copy()
    y_incoming = y.iloc[-INCOMING_COHORT_SIZE:].values

    print(f"Reference cohort:  {len(train_df):,} patients")
    print(f"Incoming cohort:   {len(incoming_df):,} patients")

    train_preds = model.predict_proba(train_df.reindex(columns=feature_columns))[:, 1]
    incoming_preds = model.predict_proba(incoming_df.reindex(columns=feature_columns))[:, 1]

    print("Computing monitoring report...")
    report = generate_report(
        train_df=train_df,
        incoming_df=incoming_df,
        train_preds=train_preds,
        incoming_preds=incoming_preds,
        y_true=y_incoming,
        y_proba=incoming_preds,
        baseline_brier=BASELINE_BRIER,
        baseline_ece=BASELINE_ECE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "report.json"
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n{'=' * 40}")
    print(f"Overall health:         {report.overall_health.upper()}")
    print(f"Features drifted:       {report.n_features_drifted}/{report.n_features_checked}")
    print(f"Features warning:       {report.n_features_warning}/{report.n_features_checked}")
    print(f"Missingness drifted:    {len(report.missingness_drifted)} features")
    print(f"Prediction drift (KS):  drifted={report.prediction_drift.drifted}  "
          f"p={report.prediction_drift.ks_p_value}  "
          f"mean_shift={report.prediction_drift.mean_shift:+.4f}")
    if report.calibration_drift:
        cd = report.calibration_drift
        print(f"Calibration drift:      drifted={cd.drifted}  "
              f"brier_delta={cd.brier_delta:+.4f}  ece_delta={cd.ece_delta:+.4f}")
    if report.top_drifted_features:
        print("\nTop drifted features:")
        for r in report.top_drifted_features[:5]:
            print(f"  {r.feature:<30s} PSI={r.psi:.4f}  [{r.status}]")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
