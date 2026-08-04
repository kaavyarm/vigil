from timeline_events import classify_event, extract_timeline_events


# ── classify_event ────────────────────────────────────────────

def test_hr_spike_triggers():
    event = classify_event("HR", 130)
    assert event is not None
    assert event["severity"] == "High"
    assert event["category"] == "vital"
    assert event["parameter"] == "HR"


def test_hr_normal_no_event():
    assert classify_event("HR", 80) is None


def test_hr_low_triggers():
    event = classify_event("HR", 45)
    assert event is not None
    assert event["severity"] == "Moderate"


def test_gcs_critical_triggers():
    event = classify_event("GCS", 6)
    assert event is not None
    assert event["severity"] == "Critical"
    assert event["category"] == "neurological"


def test_gcs_reduced_triggers():
    event = classify_event("GCS", 10)
    assert event is not None
    assert event["severity"] == "High"


def test_gcs_normal_no_event():
    assert classify_event("GCS", 14) is None


def test_lactate_critical():
    event = classify_event("Lactate", 5.0)
    assert event is not None
    assert event["severity"] == "Critical"


def test_lactate_elevated():
    event = classify_event("Lactate", 2.5)
    assert event is not None
    assert event["severity"] == "High"


def test_lactate_normal_no_event():
    assert classify_event("Lactate", 1.5) is None


def test_mechvent_on_triggers():
    event = classify_event("MechVent", 1)
    assert event is not None
    assert event["category"] == "respiratory"


def test_mechvent_off_no_event():
    assert classify_event("MechVent", 0) is None


def test_none_value_no_event():
    assert classify_event("HR", None) is None


def test_unknown_parameter_no_event():
    assert classify_event("UnknownParam", 999) is None


def test_low_sao2_triggers():
    event = classify_event("SaO2", 85)
    assert event is not None
    assert event["severity"] == "High"
    assert event["category"] == "respiratory"


# ── extract_timeline_events ───────────────────────────────────

def test_extract_returns_only_alerts():
    patient_data = {
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"HR": 130, "MAP": 80}},
            {"time": "02:00", "hour": 2.0, "measurements": {"HR": 75}},
        ]
    }
    events = extract_timeline_events(patient_data)
    assert len(events) == 1
    assert events[0]["parameter"] == "HR"


def test_extract_event_has_time_not_hour():
    patient_data = {
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"GCS": 5}},
        ]
    }
    events = extract_timeline_events(patient_data)
    assert len(events) == 1
    assert "time" in events[0]
    assert "hour" not in events[0]


def test_extract_empty_timeline():
    assert extract_timeline_events({"timeline": []}) == []


def test_extract_multiple_events_same_timepoint():
    patient_data = {
        "timeline": [
            {"time": "01:00", "hour": 1.0, "measurements": {"HR": 130, "GCS": 5}},
        ]
    }
    events = extract_timeline_events(patient_data)
    assert len(events) == 2
    parameters = {e["parameter"] for e in events}
    assert parameters == {"HR", "GCS"}
