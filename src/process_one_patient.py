from pathlib import Path

import pandas as pd

STATIC_FIELDS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}

ICU_TYPE_MAP = {
    1: "Coronary Care Unit",
    2: "Cardiac Surgery Recovery Unit",
    3: "Medical ICU",
    4: "Surgical ICU",
}

GENDER_MAP = {
    0: "Female",
    1: "Male",
}


def time_to_minutes(time_str: str) -> int:
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)


def clean_value(value):
    try:
        value = float(value)
        if value == -1:
            return None
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        return value


def load_patient_record(file_path: str | Path):
    df = pd.read_csv(file_path)
    df["Value"] = df["Value"].apply(clean_value)
    df["Minutes"] = df["Time"].apply(time_to_minutes)
    df["Hours"] = (df["Minutes"] / 60).round(2)
    return df


def extract_static_info(df: pd.DataFrame) -> dict:
    static_df = df[df["Parameter"].isin(STATIC_FIELDS)]

    info = {}
    for _, row in static_df.iterrows():
        info[row["Parameter"]] = row["Value"]

    if "Gender" in info:
        info["GenderLabel"] = GENDER_MAP.get(info["Gender"], "Unknown")

    if "ICUType" in info:
        info["ICUTypeLabel"] = ICU_TYPE_MAP.get(info["ICUType"], "Unknown ICU Type")

    return info


def build_timeline(df: pd.DataFrame) -> list[dict]:
    clinical_df = df[~df["Parameter"].isin(STATIC_FIELDS)].copy()

    timeline = []

    for time, group in clinical_df.groupby("Time"):
        measurements = {}
        for _, row in group.iterrows():
            measurements[row["Parameter"]] = row["Value"]

        timeline.append(
            {
                "time": time,
                "hour": round(time_to_minutes(time) / 60, 2),
                "measurements": measurements,
            }
        )

    timeline.sort(key=lambda event: event["hour"])
    return timeline


def print_patient_summary(static_info: dict, timeline: list[dict]):
    print("\n==============================")
    print("PATIENT SUMMARY")
    print("==============================")
    print(f"Record ID: {static_info.get('RecordID')}")
    print(f"Age: {static_info.get('Age')}")
    print(f"Gender: {static_info.get('GenderLabel')}")
    print(f"Height: {static_info.get('Height')}")
    print(f"Weight: {static_info.get('Weight')}")
    print(f"ICU Type: {static_info.get('ICUTypeLabel')}")

    print("\n==============================")
    print("PATIENT TIMELINE")
    print("==============================")
    for event in timeline:
        measurement_text = ", ".join(
            f"{key}: {value}" for key, value in event["measurements"].items()
        )
        print(f"{event['time']} | Hour {event['hour']}: {measurement_text}")


def main():
    file_path = Path("data/raw/Set A/132539.txt")
    df = load_patient_record(file_path)
    static_info = extract_static_info(df)
    timeline = build_timeline(df)
    print_patient_summary(static_info, timeline)


if __name__ == "__main__":
    main()
