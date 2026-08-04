import numpy as np
import pytest
from feature_engineering import extract_numeric_summary, extract_patient_features


def test_extract_numeric_summary_empty():
    result = extract_numeric_summary([], "HR")
    assert result["HR_count"] == 0
    assert result["HR_measured"] == 0
    assert np.isnan(result["HR_mean"])
    assert np.isnan(result["HR_min"])
    assert np.isnan(result["HR_max"])


def test_extract_numeric_summary_single_point():
    measurements = [{"hour": 1.0, "value": 80.0}]
    result = extract_numeric_summary(measurements, "HR")
    assert result["HR_mean"] == 80.0
    assert result["HR_min"] == 80.0
    assert result["HR_max"] == 80.0
    assert result["HR_last"] == 80.0
    assert result["HR_count"] == 1
    assert result["HR_measured"] == 1
    assert np.isnan(result["HR_trend"])  # trend requires >= 2 points


def test_extract_numeric_summary_trend_increasing():
    measurements = [
        {"hour": 1.0, "value": 70.0},
        {"hour": 2.0, "value": 90.0},
    ]
    result = extract_numeric_summary(measurements, "HR")
    assert result["HR_trend"] > 0


def test_extract_numeric_summary_trend_decreasing():
    measurements = [
        {"hour": 1.0, "value": 90.0},
        {"hour": 2.0, "value": 70.0},
    ]
    result = extract_numeric_summary(measurements, "HR")
    assert result["HR_trend"] < 0


def test_extract_numeric_summary_last_value():
    measurements = [
        {"hour": 1.0, "value": 70.0},
        {"hour": 2.0, "value": 85.0},
        {"hour": 3.0, "value": 90.0},
    ]
    result = extract_numeric_summary(measurements, "HR")
    assert result["HR_last"] == 90.0
    assert result["HR_count"] == 3


def test_extract_patient_features_contains_expected_keys():
    patient_data = {
        "static_info": {
            "RecordID": 999, "Age": 65, "Gender": 1,
            "Height": 170.0, "Weight": 70.0, "ICUType": 1,
        },
        "timeline": [
            {"time": "00:30", "hour": 0.5, "measurements": {"HR": 80, "GCS": 15}},
        ],
    }
    features = extract_patient_features(patient_data)
    assert features["RecordID"] == 999
    assert "HR_mean" in features
    assert "HR_last" in features
    assert "HR_trend" in features
    assert "GCS_mean" in features
    assert "Urine_total" in features
    assert "MechVent_flag" in features


def test_extract_patient_features_urine_total():
    patient_data = {
        "static_info": {"RecordID": 1, "Age": 60},
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"Urine": 40}},
            {"time": "02:00", "hour": 2.0, "measurements": {"Urine": 60}},
        ],
    }
    features = extract_patient_features(patient_data)
    assert features["Urine_total"] == pytest.approx(100.0)


def test_extract_patient_features_mechvent_flag():
    patient_data = {
        "static_info": {"RecordID": 1},
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"MechVent": 1}},
        ],
    }
    features = extract_patient_features(patient_data)
    assert features["MechVent_flag"] == 1


def test_extract_patient_features_no_mechvent():
    patient_data = {
        "static_info": {"RecordID": 1},
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"HR": 80}},
        ],
    }
    features = extract_patient_features(patient_data)
    assert features["MechVent_flag"] == 0
