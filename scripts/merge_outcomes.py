from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("data/features/patient_features.csv")
OUTCOMES_PATH = Path("data/raw/Outcomes-a.txt")
OUTPUT_PATH = Path("data/features/training_dataset.csv")


def main():
    print("Loading feature table...")
    features_df = pd.read_csv(FEATURES_PATH)

    print("Loading outcomes...")
    outcomes_df = pd.read_csv(OUTCOMES_PATH)

    print("\nFeature table shape:")
    print(features_df.shape)

    print("\nOutcomes table shape:")
    print(outcomes_df.shape)

    training_df = features_df.merge(
        outcomes_df,
        on="RecordID",
        how="inner"
    )

    print("\nMerged dataset shape:")
    print(training_df.shape)

    training_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved training dataset to {OUTPUT_PATH}")

    print("\nFirst few rows:")
    print(training_df.head())


if __name__ == "__main__":
    main()