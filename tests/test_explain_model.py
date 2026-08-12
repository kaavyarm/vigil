from explain_model import (
    build_feature_label,
    build_value_label,
    calculate_confidence,
    get_risk_level,
)

# ── build_feature_label ───────────────────────────────────────

def test_label_last():
    assert build_feature_label("GCS_last") == "Latest Consciousness (GCS)"


def test_label_mean():
    assert build_feature_label("HR_mean") == "Avg. Heart Rate"


def test_label_min():
    assert build_feature_label("MAP_min") == "Min. Mean Arterial Pressure"


def test_label_max():
    assert build_feature_label("Temp_max") == "Max. Temperature"


def test_label_std():
    assert build_feature_label("Creatinine_std") == "Creatinine Variability"


def test_label_trend():
    assert build_feature_label("Lactate_trend") == "Lactate Trend"


def test_label_total():
    assert build_feature_label("Urine_total") == "Total Urine Output"


def test_label_measured():
    assert build_feature_label("HR_measured") == "Heart Rate"


def test_label_unknown_feature():
    assert build_feature_label("unknown_feature") == "unknown_feature"


def test_label_no_suffix():
    assert build_feature_label("HR") == "HR"


# ── build_value_label ─────────────────────────────────────────

def test_value_label_with_unit():
    result = build_value_label("HR_last", 80.0)
    assert "bpm" in result


def test_value_label_trend_up():
    result = build_value_label("HR_trend", 0.5)
    assert "↑" in result
    assert "/hr" in result


def test_value_label_trend_down():
    result = build_value_label("HR_trend", -0.3)
    assert "↓" in result


def test_value_label_trend_flat():
    result = build_value_label("HR_trend", 0.0)
    assert "→" in result


def test_value_label_std():
    result = build_value_label("HR_std", 5.0)
    assert "σ" in result


def test_value_label_measured_yes():
    assert build_value_label("HR_measured", 1) == "yes"


def test_value_label_measured_no():
    assert build_value_label("HR_measured", 0) == "no"


def test_value_label_count():
    result = build_value_label("HR_count", 8)
    assert "readings" in result


def test_value_label_none():
    assert build_value_label("HR_last", None) == "not measured"


# ── get_risk_level ────────────────────────────────────────────

def test_risk_level_low():
    assert get_risk_level(0.1) == "Low"


def test_risk_level_moderate():
    assert get_risk_level(0.35) == "Moderate"


def test_risk_level_high():
    assert get_risk_level(0.65) == "High"


def test_risk_level_critical():
    assert get_risk_level(0.9) == "Critical"


def test_risk_level_boundaries():
    assert get_risk_level(0.2) == "Moderate"   # exactly at Low/Moderate boundary
    assert get_risk_level(0.5) == "High"        # exactly at Moderate/High boundary
    assert get_risk_level(0.8) == "Critical"    # exactly at High/Critical boundary


# ── calculate_confidence ──────────────────────────────────────

def test_confidence_at_boundary_is_zero():
    result = calculate_confidence(0.5)
    assert result["score"] == 0.0
    assert result["label"] == "Low"


def test_confidence_high_risk():
    result = calculate_confidence(0.95)
    assert result["label"] == "High"
    assert result["score"] > 0.8


def test_confidence_score_range():
    for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
        result = calculate_confidence(p)
        assert 0.0 <= result["score"] <= 1.0
