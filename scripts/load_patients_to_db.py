"""
Bulk-loads all processed patient JSONs into PostgreSQL.

Usage:
    DATABASE_URL=postgresql://... python scripts/load_patients_to_db.py

Each patient's predicted_risk is computed from the production model so the
DB can serve pre-computed risk scores without a model call on every request.
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PATIENT_DIR = ROOT / "data/processed/patients"
FEATURE_DIR = ROOT / "data/features/training_dataset.csv"
MODEL_PATH = ROOT / "models/vigil_xgboost_initial.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models/feature_columns.joblib"
TARGET_COL = "In-hospital_death"
LEAKAGE_COLS = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", TARGET_COL]


def _load_risk_map() -> dict[int, tuple[float, int]]:
    """Return {record_id: (predicted_risk, outcome)} from the feature CSV + model."""
    df = pd.read_csv(FEATURE_DIR)
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    record_ids = df["RecordID"].values
    outcomes = df[TARGET_COL].values
    X = df.drop(columns=LEAKAGE_COLS).reindex(columns=feature_columns)
    risks = model.predict_proba(X)[:, 1]

    return {
        int(rid): (float(risk), int(outcome))
        for rid, risk, outcome in zip(record_ids, risks, outcomes, strict=False)
    }


def main() -> None:
    if not os.environ.get("DATABASE_URL"):
        logger.error("DATABASE_URL is not set. Exiting.")
        sys.exit(1)

    db.init_pool()
    if not db.is_available():
        logger.error("Could not connect to the database. Exiting.")
        sys.exit(1)

    logger.info("Computing predicted risks from model…")
    risk_map = _load_risk_map()

    patient_files = sorted(PATIENT_DIR.glob("*.json"))
    logger.info("Loading %d patient records into PostgreSQL…", len(patient_files))

    failed = 0
    for path in tqdm(patient_files, unit="patient"):
        try:
            with open(path) as f:
                data = json.load(f)

            record_id = int(data["static_info"]["RecordID"])
            risk, outcome = risk_map.get(record_id, (None, None))
            db.upsert_patient(record_id, data, outcome=outcome, predicted_risk=risk)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path.name, exc)
            failed += 1

    loaded = len(patient_files) - failed
    logger.info("Done. %d loaded, %d failed.", loaded, failed)


if __name__ == "__main__":
    main()
