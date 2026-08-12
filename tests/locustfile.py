"""
Load test for the Vigil ICU Risk API.

Run with:
    locust -f tests/locustfile.py --headless -u 20 -r 5 --run-time 60s --host http://localhost:8001
"""

import random

from locust import HttpUser, between, task

# A small sample of real record IDs to use in per-patient endpoint tests.
SAMPLE_RECORD_IDS = [
    132539, 132540, 132541, 132543, 132545,
    132547, 132548, 132551, 132554, 132555,
]

STABLE_VITALS = {
    "Age": 58, "HR": 76, "MAP": 88, "GCS": 15,
    "RespRate": 14, "Temp": 37.1, "SaO2": 98,
    "Creatinine": 0.9, "Lactate": 1.1, "BUN": 16,
    "Urine": 65, "MechVent": 0,
}

CRITICAL_VITALS = {
    "Age": 74, "HR": 128, "MAP": 52, "GCS": 6,
    "RespRate": 30, "Temp": 38.9, "SaO2": 83,
    "Creatinine": 3.8, "Lactate": 5.5, "BUN": 48,
    "Urine": 8, "MechVent": 1,
}


class VIgilAPIUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def list_patients(self):
        self.client.get("/patients?limit=100", name="/patients")

    @task(2)
    def get_patient_explanation(self):
        record_id = random.choice(SAMPLE_RECORD_IDS)
        self.client.get(
            f"/patients/{record_id}/explanation",
            name="/patients/{id}/explanation",
        )

    @task(2)
    def predict_custom_stable(self):
        self.client.post(
            "/predict",
            json=STABLE_VITALS,
            name="/predict (stable)",
        )

    @task(2)
    def predict_custom_critical(self):
        self.client.post(
            "/predict",
            json=CRITICAL_VITALS,
            name="/predict (critical)",
        )

    @task(1)
    def get_risk_trend(self):
        record_id = random.choice(SAMPLE_RECORD_IDS)
        self.client.get(
            f"/patients/{record_id}/risk-trend",
            name="/patients/{id}/risk-trend",
        )
