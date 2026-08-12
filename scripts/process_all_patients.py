import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from process_one_patient import (
    build_timeline,
    extract_static_info,
    load_patient_record,
)

RAW_PATIENT_DIR = Path("data/raw/Set A")
PROCESSED_PATIENT_DIR = Path("data/processed/patients")


def process_all_patients():
    PROCESSED_PATIENT_DIR.mkdir(parents=True, exist_ok=True)

    patient_files = sorted(RAW_PATIENT_DIR.glob("*.txt"))
    print(f"Found {len(patient_files)} patient files.")

    failed = []

    for file_path in tqdm(patient_files, desc="Processing patients"):
        try:
            df = load_patient_record(file_path)
            static_info = extract_static_info(df)
            timeline = build_timeline(df)

            record_id = int(static_info["RecordID"])

            patient_data = {
                "static_info": static_info,
                "timeline": timeline,
            }

            output_path = PROCESSED_PATIENT_DIR / f"{record_id}.json"
            with open(output_path, "w") as f:
                json.dump(patient_data, f, indent=2)

        except Exception as e:
            failed.append((file_path.name, str(e)))

    print(f"\nSaved processed patients to {PROCESSED_PATIENT_DIR}")

    if failed:
        print(f"\nFailed to process {len(failed)} file(s):")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    process_all_patients()
