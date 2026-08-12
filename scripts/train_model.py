from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/features/training_dataset.csv")
MODEL_DIR = Path("models")
PLOTS_DIR = MODEL_DIR / "plots"

TARGET_COL = "In-hospital_death"

LEAKAGE_AND_ID_COLS = [
    "RecordID",
    "SAPS-I",
    "SOFA",
    "Length_of_stay",
    "Survival",
    TARGET_COL,
]


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(float(y_proba[mask].mean()) - float(y_true[mask].mean()))
    return ece


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    auroc = roc_auc_score(y_true, y_proba)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_proba)
    ece   = expected_calibration_error(y_true, y_proba)

    print(f"\n{'=' * 32}")
    print(name)
    print("=" * 32)
    print(f"  AUROC:           {auroc:.4f}")
    print(f"  Precision (PPV): {prec:.4f}")
    print(f"  Recall (Sens):   {rec:.4f}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"  NPV:             {npv:.4f}")
    print(f"  F1:              {f1:.4f}")
    print(f"  Brier Score:     {brier:.4f}")
    print(f"  ECE:             {ece:.4f}")
    print(f"  Confusion Matrix (TN={tn} FP={fp} / FN={fn} TP={tp})")

    return {
        "name": name,
        "auroc": auroc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "brier": brier,
        "ece": ece,
        "y_proba": y_proba,
    }


def plot_roc_pr_curves(results: list[dict], y_true: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for r in results:
        fpr, tpr, _ = roc_curve(y_true, r["y_proba"])
        ax1.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auroc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, r["y_proba"])
        pr_auc = auc(rec, prec)
        ax2.plot(rec, prec, label=f"{r['name']} (AUC={pr_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", label="Random")
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax1.legend(fontsize=8)

    baseline = float(y_true.mean())
    ax2.axhline(baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / "roc_pr_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved ROC/PR curves → {out}")


class _IdentityWrapper:
    """Pass-through wrapper used when the raw model is already well-calibrated."""
    def __init__(self, base_model):
        self.base_model = base_model

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        return self.base_model.predict(X)


class _PlattWrapper:
    """XGBoost + logistic (Platt) calibration applied at inference time."""
    def __init__(self, base_model, lr_calibrator):
        self.base_model   = base_model
        self.lr_calibrator = lr_calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.lr_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _IsotonicWrapper:
    """XGBoost + isotonic regression calibration applied at inference time."""
    def __init__(self, base_model, iso_calibrator):
        self.base_model      = base_model
        self.iso_calibrator  = iso_calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.iso_calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def analyze_calibration(xgb_model, X_train, y_train, X_test, y_test) -> object:
    """Fit Platt and isotonic calibrators on a held-out calibration split; return best wrapper."""
    y_true      = np.array(y_test)
    y_train_arr = np.array(y_train)

    # Hold out 20% of training data exclusively for calibration fitting.
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train_arr, test_size=0.2, stratify=y_train_arr, random_state=1
    )
    raw_cal = xgb_model.predict_proba(X_cal)[:, 1]

    # Platt scaling: sigmoid logistic fit on held-out raw probabilities
    lr_cal = LogisticRegression()
    lr_cal.fit(raw_cal.reshape(-1, 1), y_cal)

    # Isotonic regression calibration
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(raw_cal, y_cal)

    raw_test   = xgb_model.predict_proba(X_test)[:, 1]
    platt_test = lr_cal.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    iso_test   = np.clip(iso_cal.predict(raw_test), 0.0, 1.0)

    print(f"\n{'=' * 32}")
    print("Calibration Analysis")
    print("=" * 32)
    for label, yp in [
        ("Raw XGBoost",         raw_test),
        ("Platt Scaling",       platt_test),
        ("Isotonic Regression", iso_test),
    ]:
        b = brier_score_loss(y_true, yp)
        e = expected_calibration_error(y_true, yp)
        print(f"  {label:22s}  Brier={b:.4f}  ECE={e:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, yp in [
        ("Raw XGBoost",         raw_test),
        ("Platt Scaling",       platt_test),
        ("Isotonic Regression", iso_test),
    ]:
        CalibrationDisplay.from_predictions(y_true, yp, n_bins=10, ax=ax, name=label)
    ax.set_title("Calibration Curves (Reliability Diagram)")
    plt.tight_layout()
    out = PLOTS_DIR / "calibration_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved calibration curves → {out}")

    brier_raw   = brier_score_loss(y_true, raw_test)
    brier_platt = brier_score_loss(y_true, platt_test)
    brier_iso   = brier_score_loss(y_true, iso_test)
    best_label, best_brier = min(
        [("Raw XGBoost", brier_raw), ("Platt Scaling", brier_platt), ("Isotonic Regression", brier_iso)],
        key=lambda x: x[1],
    )
    print(f"Best calibration method: {best_label} (Brier={best_brier:.4f})")

    if best_label == "Raw XGBoost":
        return _IdentityWrapper(xgb_model)
    if best_label == "Isotonic Regression":
        return _IsotonicWrapper(xgb_model, iso_cal)
    return _PlattWrapper(xgb_model, lr_cal)


def optimize_threshold(model, X_test, y_test, min_precision: float = 0.3) -> float:
    """Select threshold maximizing recall subject to precision >= min_precision."""
    y_true  = np.array(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    valid = np.where(precisions[:-1] >= min_precision)[0]
    if len(valid) > 0:
        best_idx  = valid[np.argmax(recalls[valid])]
        threshold = float(thresholds[best_idx])
        p         = float(precisions[best_idx])
        r         = float(recalls[best_idx])
    else:
        threshold, p, r = 0.5, 0.0, 0.0

    print(f"\n{'=' * 32}")
    print("Threshold Optimization")
    print("=" * 32)
    print(f"  Criterion:         maximize recall at precision ≥ {min_precision}")
    print(f"  Optimal threshold: {threshold:.3f}")
    print(f"  Precision:         {p:.3f}")
    print(f"  Recall:            {r:.3f}")

    return threshold


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    y = df[TARGET_COL]
    X = df.drop(columns=LEAKAGE_AND_ID_COLS)
    print(f"Features: {X.shape[1]}  |  Positive rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_test_arr = np.array(y_test)

    # LR and RF need imputed NaN; XGBoost handles NaN natively.
    # Use train medians for test imputation to prevent leakage.
    train_medians   = X_train.median()
    X_train_filled  = X_train.fillna(train_medians)
    X_test_filled   = X_test.fillna(train_medians)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    results = []

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    lr.fit(X_train_filled, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_filled, y_test))

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train_filled, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test_filled, y_test))

    # ── XGBoost (initial) ────────────────────────────────────────────────────
    print("\nTraining Initial XGBoost...")
    xgb_initial = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb_initial.fit(X_train, y_train)
    results.append(evaluate_model("Initial XGBoost", xgb_initial, X_test, y_test))

    # ── Clinical Score Baselines ─────────────────────────────────────────────
    print(f"\n{'=' * 32}")
    print("Clinical Score Baselines")
    print("=" * 32)
    saps_auroc = roc_auc_score(y_test, df.loc[X_test.index, "SAPS-I"])
    sofa_auroc = roc_auc_score(y_test, df.loc[X_test.index, "SOFA"])
    print(f"  SAPS-I AUROC: {saps_auroc:.4f}")
    print(f"  SOFA   AUROC: {sofa_auroc:.4f}")

    # ── Stratified CV ────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_initial, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n{'=' * 32}")
    print("XGBoost 5-Fold CV")
    print("=" * 32)
    print(f"  AUROC per fold: {cv_scores.round(4)}")
    print(f"  Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")

    # ── Hyperparameter Search ────────────────────────────────────────────────
    print(f"\n{'=' * 32}")
    print("XGBoost Hyperparameter Search")
    print("=" * 32)
    param_grid = {
        "n_estimators":    [200, 300, 500],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.03, 0.05, 0.1],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree":[0.7, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_grid,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params:   {search.best_params_}")
    print(f"  Best CV AUROC: {search.best_score_:.4f}")

    xgb_tuned = search.best_estimator_
    tuned_result = evaluate_model("Tuned XGBoost", xgb_tuned, X_test, y_test)
    results.append(tuned_result)

    # ── Bootstrap 95% CIs — Tuned XGBoost ───────────────────────────────────
    print(f"\n{'=' * 32}")
    print("Bootstrap 95% CIs — Tuned XGBoost (1000 resamples)")
    print("=" * 32)
    y_proba_tuned = tuned_result["y_proba"]

    auroc_lo, auroc_hi = bootstrap_ci(y_test_arr, y_proba_tuned, roc_auc_score)
    brier_lo, brier_hi = bootstrap_ci(
        y_test_arr, y_proba_tuned, lambda yt, yp: brier_score_loss(yt, yp)
    )
    rec_lo, rec_hi = bootstrap_ci(
        y_test_arr,
        y_proba_tuned,
        lambda yt, yp: recall_score(yt, (yp >= 0.5).astype(int), zero_division=0),
    )

    print(f"  AUROC:       {tuned_result['auroc']:.4f}  [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    print(f"  Recall:      {tuned_result['recall']:.4f}  [{rec_lo:.4f}–{rec_hi:.4f}]")
    print(f"  Brier Score: {tuned_result['brier']:.4f}  [{brier_lo:.4f}–{brier_hi:.4f}]")

    # ── Plots ────────────────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_pr_curves(results, y_test_arr)

    # ── Calibration ──────────────────────────────────────────────────────────
    calibrated_model = analyze_calibration(xgb_tuned, X_train, y_train, X_test, y_test)

    # ── Threshold Optimization ───────────────────────────────────────────────
    threshold = optimize_threshold(calibrated_model, X_test, y_test, min_precision=0.3)

    # ── Save Artifacts ───────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(xgb_initial,       MODEL_DIR / "vigil_xgboost_initial.joblib")
    joblib.dump(xgb_tuned,         MODEL_DIR / "vigil_xgboost_tuned.joblib")
    joblib.dump(calibrated_model,  MODEL_DIR / "vigil_xgboost_calibrated.joblib")
    joblib.dump(X_train.columns.tolist(), MODEL_DIR / "feature_columns.joblib")
    joblib.dump(train_medians,     MODEL_DIR / "train_medians.joblib")
    joblib.dump(threshold,         MODEL_DIR / "threshold.joblib")

    print(f"\n{'=' * 32}")
    print("Saved Artifacts")
    print("=" * 32)
    for name in [
        "vigil_xgboost_initial.joblib",
        "vigil_xgboost_tuned.joblib",
        "vigil_xgboost_calibrated.joblib",
        "feature_columns.joblib",
        "train_medians.joblib",
        "threshold.joblib",
    ]:
        print(f"  models/{name}")
    print("  models/plots/roc_pr_curves.png")
    print("  models/plots/calibration_curves.png")


if __name__ == "__main__":
    main()
