from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss

PSI_WARN = 0.1        # slight drift — watch
PSI_CRITICAL = 0.2    # significant drift — act
MISSINGNESS_DELTA = 0.10   # absolute change in missing rate to flag
KS_ALPHA = 0.05            # KS test significance level
BRIER_DRIFT = 0.02         # Brier score degradation to flag
ECE_DRIFT = 0.02           # ECE degradation to flag

@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    status: str   # "stable" | "warning" | "drifted"

@dataclass
class MissingnessDriftResult:
    feature: str
    train_rate: float
    incoming_rate: float
    delta: float
    drifted: bool

@dataclass
class PredictionDriftResult:
    ks_statistic: float
    ks_p_value: float
    train_mean: float
    incoming_mean: float
    mean_shift: float
    drifted: bool

@dataclass
class CalibrationDriftResult:
    current_brier: float
    baseline_brier: float
    brier_delta: float
    current_ece: float
    baseline_ece: float
    ece_delta: float
    drifted: bool

@dataclass
class MonitoringReport:
    n_reference: int
    n_incoming: int
    n_features_checked: int
    n_features_drifted: int
    n_features_warning: int
    top_drifted_features: list
    missingness_drifted: list
    prediction_drift: PredictionDriftResult
    calibration_drift: CalibrationDriftResult | None
    overall_health: str  # "healthy" | "warning" | "critical"

    def to_dict(self) -> dict:
        return asdict(self)

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 5 or len(actual) < 5:
        return 0.0

    min_val = min(float(expected.min()), float(actual.min()))
    max_val = max(float(expected.max()), float(actual.max()))

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    # Laplace smoothing to avoid log(0)
    exp_pcts = (exp_counts + 0.5) / (len(expected) + 0.5 * n_bins)
    act_pcts = (act_counts + 0.5) / (len(actual) + 0.5 * n_bins)

    psi = float(np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts)))
    return max(psi, 0.0)

def _expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(float(y_proba[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)

def compute_feature_drift(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    n_bins: int = 10,
) -> list[FeatureDriftResult]:
    """PSI-based drift detection for every numeric feature."""
    results = []
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        exp = train_df[col].dropna().values
        act = incoming_df[col].dropna().values

        if len(exp) < 10 or len(act) < 10:
            continue
        if exp.std() == 0 and act.std() == 0:
            continue

        psi = compute_psi(exp, act, n_bins)
        status = "drifted" if psi >= PSI_CRITICAL else ("warning" if psi >= PSI_WARN else "stable")
        results.append(FeatureDriftResult(feature=col, psi=round(psi, 4), status=status))

    return sorted(results, key=lambda r: r.psi, reverse=True)

def compute_missingness_drift(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    threshold: float = MISSINGNESS_DELTA,
) -> list[MissingnessDriftResult]:
    """Flag features whose missingness rate changed more than `threshold`."""
    results = []
    for col in train_df.columns:
        train_rate = float(train_df[col].isna().mean())
        inc_rate = float(incoming_df[col].isna().mean())
        delta = abs(inc_rate - train_rate)
        if delta >= threshold:
            results.append(MissingnessDriftResult(
                feature=col,
                train_rate=round(train_rate, 4),
                incoming_rate=round(inc_rate, 4),
                delta=round(delta, 4),
                drifted=True,
            ))
    return sorted(results, key=lambda r: r.delta, reverse=True)

def compute_prediction_drift(
    train_preds: np.ndarray,
    incoming_preds: np.ndarray,
) -> PredictionDriftResult:
    """KS test on predicted probability distributions."""
    ks_stat, ks_p = stats.ks_2samp(train_preds, incoming_preds)
    return PredictionDriftResult(
        ks_statistic=round(float(ks_stat), 4),
        ks_p_value=round(float(ks_p), 4),
        train_mean=round(float(train_preds.mean()), 4),
        incoming_mean=round(float(incoming_preds.mean()), 4),
        mean_shift=round(float(incoming_preds.mean() - train_preds.mean()), 4),
        drifted=bool(ks_p < KS_ALPHA),
    )

def compute_calibration_drift(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    baseline_brier: float,
    baseline_ece: float,
) -> CalibrationDriftResult:
    """Compare current calibration against training-time baselines."""
    current_brier = float(brier_score_loss(y_true, y_proba))
    current_ece = _expected_calibration_error(y_true, y_proba)
    brier_delta = current_brier - baseline_brier
    ece_delta = current_ece - baseline_ece
    return CalibrationDriftResult(
        current_brier=round(current_brier, 4),
        baseline_brier=round(baseline_brier, 4),
        brier_delta=round(brier_delta, 4),
        current_ece=round(current_ece, 4),
        baseline_ece=round(baseline_ece, 4),
        ece_delta=round(ece_delta, 4),
        drifted=bool(brier_delta > BRIER_DRIFT or ece_delta > ECE_DRIFT),
    )

def generate_report(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    train_preds: np.ndarray,
    incoming_preds: np.ndarray,
    y_true: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    baseline_brier: float | None = None,
    baseline_ece: float | None = None,
) -> MonitoringReport:
    feature_results = compute_feature_drift(train_df, incoming_df)
    n_drifted = sum(1 for r in feature_results if r.status == "drifted")
    n_warning = sum(1 for r in feature_results if r.status == "warning")
    top = [r for r in feature_results if r.status in ("drifted", "warning")][:10]

    missingness = compute_missingness_drift(train_df, incoming_df)
    pred_drift = compute_prediction_drift(train_preds, incoming_preds)

    cal_drift = None
    if y_true is not None and y_proba is not None and baseline_brier is not None:
        cal_drift = compute_calibration_drift(y_true, y_proba, baseline_brier, baseline_ece or 0.0)

    if n_drifted > 10 or (cal_drift and cal_drift.drifted):
        health = "critical"
    elif n_drifted > 0 or n_warning > 5 or pred_drift.drifted or len(missingness) > 3:
        health = "warning"
    else:
        health = "healthy"

    return MonitoringReport(
        n_reference=len(train_df),
        n_incoming=len(incoming_df),
        n_features_checked=len(feature_results),
        n_features_drifted=n_drifted,
        n_features_warning=n_warning,
        top_drifted_features=top,
        missingness_drifted=missingness,
        prediction_drift=pred_drift,
        calibration_drift=cal_drift,
        overall_health=health,
    )
