import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from feature_engineering import extract_patient_features, load_patient_json

PROCESSED_PATIENT_DIR = Path("data/processed/patients")
FEATURE_OUTPUT_PATH = Path("data/features/patient_features.csv")


def build_feature_table() -> pd.DataFrame:
    """Build one feature table from all processed patient JSON files."""
    patient_files = sorted(PROCESSED_PATIENT_DIR.glob("*.json"))

    print(f"Found {len(patient_files)} processed patient files.")

    feature_rows = []

    for file_path in tqdm(patient_files, desc="Extracting features"):
        patient_data = load_patient_json(file_path)
        features = extract_patient_features(patient_data)
        feature_rows.append(features)

    feature_table = pd.DataFrame(feature_rows)

    return feature_table


def main():
    FEATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_table = build_feature_table()

    feature_table.to_csv(FEATURE_OUTPUT_PATH, index=False)

    print(f"Saved feature table to {FEATURE_OUTPUT_PATH}")
    print(f"Shape: {feature_table.shape}")
    print("\nFirst few rows:")
    print(feature_table.head())


if __name__ == "__main__":
    main()
