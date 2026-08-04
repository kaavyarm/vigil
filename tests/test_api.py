import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Vigil API is running"


def test_get_patients_returns_list():
    response = client.get("/patients?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10


def test_get_patients_sorted_by_risk_descending():
    response = client.get("/patients?limit=20")
    data = response.json()
    risks = [p["mortality_risk"] for p in data]
    assert risks == sorted(risks, reverse=True)


def test_get_patients_has_expected_fields():
    response = client.get("/patients?limit=1")
    patient = response.json()[0]
    assert "record_id" in patient
    assert "age" in patient
    assert "icu_type_label" in patient
    assert "mortality_risk" in patient
    assert "mortality_risk_percent" in patient
    assert "status" in patient


def test_predict_returns_expected_fields():
    response = client.post("/predict", json={
        "Age": 65, "HR": 80, "MAP": 85, "GCS": 15,
        "Lactate": 1.2, "Creatinine": 0.9,
    })
    assert response.status_code == 200
    data = response.json()
    assert "mortality_risk" in data
    assert "mortality_risk_percent" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "protective_factors" in data
    assert "recommended_action" in data
    assert "confidence" in data


def test_predict_risk_in_valid_range():
    response = client.post("/predict", json={"Age": 65, "HR": 80})
    data = response.json()
    assert 0.0 <= data["mortality_risk"] <= 1.0


def test_predict_critical_higher_than_stable():
    stable = client.post("/predict", json={
        "Age": 58, "HR": 76, "MAP": 88, "GCS": 15,
        "RespRate": 14, "Temp": 37.1, "SaO2": 98,
        "Creatinine": 0.9, "Lactate": 1.1, "BUN": 16, "Urine": 65, "MechVent": 0,
    }).json()

    critical = client.post("/predict", json={
        "Age": 74, "HR": 128, "MAP": 52, "GCS": 6,
        "RespRate": 30, "Temp": 38.9, "SaO2": 83,
        "Creatinine": 3.8, "Lactate": 5.5, "BUN": 48, "Urine": 8, "MechVent": 1,
    }).json()

    assert critical["mortality_risk"] > stable["mortality_risk"]


def test_predict_empty_body_uses_medians():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert "mortality_risk" in response.json()


def test_predict_risk_factors_are_positive_shap():
    response = client.post("/predict", json={"GCS": 5, "Lactate": 6.0})
    data = response.json()
    for factor in data["risk_factors"]:
        assert factor["shap_value"] > 0


def test_predict_protective_factors_are_negative_shap():
    response = client.post("/predict", json={"GCS": 15, "SaO2": 99})
    data = response.json()
    for factor in data["protective_factors"]:
        assert factor["shap_value"] < 0


def test_get_nonexistent_patient_explanation_404():
    response = client.get("/patients/999999/explanation")
    assert response.status_code == 404


def test_get_nonexistent_patient_risk_trend_404():
    response = client.get("/patients/999999/risk-trend")
    assert response.status_code == 404


def test_get_nonexistent_patient_timeline_404():
    response = client.get("/patients/999999/timeline-events")
    assert response.status_code == 404
