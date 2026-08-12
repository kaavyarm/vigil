import numpy as np
import pandas as pd

from monitoring import (
    compute_calibration_drift,
    compute_feature_drift,
    compute_missingness_drift,
    compute_prediction_drift,
    compute_psi,
    generate_report,
)

# ── PSI ───────────────────────────────────────────────────────────────────────

def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 1000)
    assert compute_psi(data, data) < 0.01


def test_psi_large_for_very_different_distributions():
    rng = np.random.default_rng(0)
    exp = rng.normal(0, 1, 1000)
    act = rng.normal(5, 1, 1000)   # Mean shifted by 5σ
    assert compute_psi(exp, act) > 0.2


def test_psi_always_non_negative():
    rng = np.random.default_rng(42)
    for _ in range(20):
        exp = rng.normal(0, 1, 300)
        act = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 300)
        assert compute_psi(exp, act) >= 0.0


def test_psi_constant_column_returns_zero():
    data = np.ones(100)
    assert compute_psi(data, data) == 0.0


def test_psi_too_few_samples_returns_zero():
    assert compute_psi(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


# ── Feature drift ─────────────────────────────────────────────────────────────

def test_feature_drift_flags_heavily_shifted_column():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"hr": rng.normal(70, 10, 500), "temp": rng.normal(37.0, 0.5, 500)})
    # HR distribution shifts from 70 to 130 — should be flagged drifted
    inc = pd.DataFrame({"hr": rng.normal(130, 10, 500), "temp": rng.normal(37.0, 0.5, 500)})
    results = compute_feature_drift(train, inc)
    hr = next(r for r in results if r.feature == "hr")
    assert hr.status == "drifted"


def test_feature_drift_stable_column_not_flagged():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"hr": rng.normal(70, 10, 1000)})
    inc = pd.DataFrame({"hr": rng.normal(71, 10, 1000)})  # Tiny 1-bpm shift
    results = compute_feature_drift(train, inc)
    hr = next(r for r in results if r.feature == "hr")
    assert hr.status == "stable"


def test_feature_drift_results_sorted_by_psi_descending():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "a": rng.normal(0, 1, 500),
        "b": rng.normal(0, 1, 500),
    })
    inc = pd.DataFrame({
        "a": rng.normal(5, 1, 500),   # Large shift
        "b": rng.normal(0.1, 1, 500), # Small shift
    })
    results = compute_feature_drift(train, inc)
    psi_values = [r.psi for r in results]
    assert psi_values == sorted(psi_values, reverse=True)


# ── Missingness drift ─────────────────────────────────────────────────────────

def test_missingness_drift_detects_new_missingness():
    rng = np.random.default_rng(0)
    train_data = rng.normal(0, 1, 500)
    inc_data = rng.normal(0, 1, 500)
    inc_data[:150] = np.nan  # 30% missing in incoming vs 0% in train

    results = compute_missingness_drift(
        pd.DataFrame({"x": train_data}),
        pd.DataFrame({"x": inc_data}),
    )
    assert len(results) == 1
    assert results[0].feature == "x"
    assert results[0].drifted is True
    assert results[0].delta > 0.1


def test_missingness_drift_similar_rates_not_flagged():
    # Both ~20% missing — delta well below 0.10 threshold
    train_df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    inc_df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    assert compute_missingness_drift(train_df, inc_df) == []


def test_missingness_drift_sorted_by_delta_descending():
    train_df = pd.DataFrame({"a": [1.0] * 100, "b": [1.0] * 100})
    inc_a = [np.nan] * 50 + [1.0] * 50  # 50% missing
    inc_b = [np.nan] * 20 + [1.0] * 80  # 20% missing
    inc_df = pd.DataFrame({"a": inc_a, "b": inc_b})
    results = compute_missingness_drift(train_df, inc_df)
    assert results[0].feature == "a"
    assert results[1].feature == "b"


# ── Prediction drift ──────────────────────────────────────────────────────────

def test_prediction_drift_detects_large_distribution_shift():
    rng = np.random.default_rng(0)
    train_preds = rng.beta(2, 8, 1000)    # Low-risk distribution
    inc_preds = rng.beta(8, 2, 1000)      # High-risk distribution
    result = compute_prediction_drift(train_preds, inc_preds)
    assert result.drifted is True
    assert result.mean_shift > 0.3


def test_prediction_drift_same_array_not_drifted():
    rng = np.random.default_rng(0)
    preds = rng.beta(2, 8, 500)
    result = compute_prediction_drift(preds, preds)
    assert result.drifted is False
    assert result.mean_shift == 0.0


def test_prediction_drift_mean_shift_sign():
    train_preds = np.array([0.1, 0.1, 0.1])
    inc_preds = np.array([0.9, 0.9, 0.9])
    result = compute_prediction_drift(train_preds, inc_preds)
    assert result.mean_shift > 0


# ── Calibration drift ─────────────────────────────────────────────────────────

def test_calibration_drift_detects_ece_degradation():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.15, 500)
    # Probabilities wildly miscalibrated (all near 0.5 regardless of outcome)
    y_proba = np.full(500, 0.5)
    result = compute_calibration_drift(y_true, y_proba, baseline_brier=0.089, baseline_ece=0.030)
    assert result.drifted is True


def test_calibration_drift_no_degradation_not_flagged():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.15, 1000)
    # Near-perfect calibration
    y_proba = np.clip(y_true + rng.normal(0, 0.05, 1000), 0, 1)
    result = compute_calibration_drift(y_true, y_proba, baseline_brier=0.25, baseline_ece=0.20)
    assert result.drifted is False


# ── generate_report integration ───────────────────────────────────────────────

def test_generate_report_returns_correct_structure():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
    inc_df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)})
    train_preds = rng.beta(2, 8, 500)
    inc_preds = rng.beta(2, 8, 100)

    report = generate_report(train_df, inc_df, train_preds, inc_preds)

    assert report.n_reference == 500
    assert report.n_incoming == 100
    assert report.n_features_checked == 2
    assert report.overall_health in ("healthy", "warning", "critical")
    assert isinstance(report.prediction_drift.ks_statistic, float)
    assert report.calibration_drift is None


def test_generate_report_health_critical_on_many_drifted_features():
    rng = np.random.default_rng(0)
    n_features = 15
    train_df = pd.DataFrame({f"f{i}": rng.normal(0, 1, 500) for i in range(n_features)})
    # Every feature shifted by 5σ — all should be "drifted"
    inc_df = pd.DataFrame({f"f{i}": rng.normal(5, 1, 100) for i in range(n_features)})
    train_preds = rng.beta(2, 8, 500)
    inc_preds = rng.beta(2, 8, 100)

    report = generate_report(train_df, inc_df, train_preds, inc_preds)
    assert report.overall_health == "critical"
    assert report.n_features_drifted > 10
