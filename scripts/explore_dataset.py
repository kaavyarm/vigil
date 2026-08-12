from pathlib import Path

import pandas as pd

TRAINING_DATA_PATH = Path("data/features/training_dataset.csv")


def main():
    df = pd.read_csv(TRAINING_DATA_PATH)

    print("\n==============================")
    print("DATASET SHAPE")
    print("==============================")
    print(df.shape)

    print("\n==============================")
    print("TARGET DISTRIBUTION")
    print("==============================")
    print(df["In-hospital_death"].value_counts())
    print("\nPercentages:")
    print(df["In-hospital_death"].value_counts(normalize=True) * 100)

    print("\n==============================")
    print("COLUMNS")
    print("==============================")
    print(df.columns.tolist())

    print("\n==============================")
    print("MISSINGNESS: TOP 30")
    print("==============================")
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": missing_percent,
        }
    )

    print(missing_summary.head(30))

    print("\n==============================")
    print("NUMERIC SUMMARY")
    print("==============================")
    print(df.describe().T.head(40))

    print("\n==============================")
    print("POTENTIAL LEAKAGE COLUMNS")
    print("==============================")
    leakage_cols = ["SAPS-I", "SOFA", "Length_of_stay", "Survival"]
    for col in leakage_cols:
        if col in df.columns:
            print(f"{col}: present")

    print("\n==============================")
    print("DONE")
    print("==============================")

    print("\n==============================")
    print("MISSINGNESS FLAGS (measured counts)")
    print("==============================")
    measurement_cols = [c for c in df.columns if c.endswith("_measured")]
    for col in measurement_cols:
        print(f"  {col}: {int(df[col].sum())}")


if __name__ == "__main__":
    main()
