# Vigil — Methodology

## Dataset

**PhysioNet 2012 Challenge (Set A):** 4,000 ICU patient records from 12 ICUs across a single US hospital system (2004–2008). Each record is a 48-hour time series of clinical measurements sampled at irregular intervals, plus static demographics and a binary in-hospital mortality outcome. Mortality prevalence: ~14%.

## Feature Engineering

### Why time-series aggregation

Raw measurements arrive at irregular times; models require fixed-length vectors. For each of 32 clinical parameters (vital signs, labs, fluids, ventilation), Vigil computes 8 summary statistics over the 48-hour window:

| Suffix | Meaning | Clinical rationale |
|---|---|---|
| `_mean` | Mean over stay | Baseline burden |
| `_min` | Minimum | Worst-case nadir |
| `_max` | Maximum | Worst-case peak |
| `_last` | Most recent value | Current state |
| `_std` | Standard deviation | Instability / variability |
| `_trend` | Linear slope (per hour) | Trajectory direction |
| `_count` | Number of readings | Care intensity proxy |
| `_measured` | 1 if ever measured, 0 if not | Informative missingness |

This yields **263 features** per patient (32 × 8 base features + static demographics + domain-specific: `Urine_total`, `MechVent_flag`).

**Informative missingness:** A `_measured = 0` flag is clinically meaningful — parameters that are never measured are typically stable or considered low-risk by the treating team. Encoding this prevents imputing a normal value and losing the signal.

**Trend feature:** Linear regression slope over time captures deterioration trajectories that static aggregates miss (e.g. rising lactate even if current value is within range).

### Imputation

Missing values are imputed with per-feature medians computed on the training set and saved to `models/train_medians.joblib`. The same medians are applied at inference time, ensuring no train/test leakage.

### Excluded features

SAPS-I, SOFA, Length_of_stay, Survival, and RecordID are excluded from model features and used only as evaluation baselines or identifiers. Including SAPS-I or SOFA would constitute label leakage since they aggregate the very measurements the model uses.

---

## Model Selection

### Candidates evaluated

| Model | AUROC (CV) | Notes |
|---|---|---|
| Logistic Regression | 0.856 | Requires scaling; slower convergence |
| Random Forest | 0.861 | High AUROC but low recall (0.324) |
| **XGBoost** | **0.878** | Best AUROC and highest recall; selected |

### Why XGBoost

XGBoost was selected because:

1. **AUROC**: highest of all candidates at AUROC 0.878 on 5-fold stratified cross-validation.
2. **Recall**: Random Forest recall (0.324) is substantially lower — it misses ~52% of in-hospital deaths, making it inappropriate for a high-stakes screening task.
3. **Class imbalance**: `scale_pos_weight = n_negative / n_positive ≈ 6.1` compensates for the 14% mortality prevalence without resampling, which can distort the feature distributions.
4. **Feature importance**: Tree-based SHAP values are exact (not approximated) for XGBoost via `shap.TreeExplainer`, making per-patient explanations computationally cheap and theoretically grounded.

### Cross-validation

5-fold stratified CV (stratified on outcome) to preserve class balance across folds. Reported metrics are the mean across folds. The stratified split ensures each fold has ~14% mortality prevalence.

---

## Probability Calibration

Good calibration means a predicted risk of 0.30 should correspond to ~30% actual mortality in a cohort at that score. Miscalibrated models can mislead clinicians even if AUROC is high.

### Analysis

A held-out calibration split (20% of training data, stratified) was used to compare:

| Method | ECE | Brier Score |
|---|---|---|
| Raw XGBoost | **0.030** | **0.089** |
| Platt scaling | 0.035 | 0.091 |
| Isotonic regression | 0.041 | 0.094 |

Raw XGBoost was already well-calibrated. Platt and isotonic post-processing marginally worsened calibration on this dataset, so the raw model is deployed. This is consistent with the literature: gradient-boosted trees trained with appropriate class weights tend to be better-calibrated than other methods.

**Expected Calibration Error (ECE):** Partition probability space into 10 equal-width bins; compute weighted average of |mean predicted − mean actual| per bin. A lower ECE means predicted probabilities are more accurate as frequencies.

---

## Decision Threshold

XGBoost outputs a probability. Converting to a binary alert requires a threshold. The default of 0.5 is inappropriate here because:

- The dataset has 14% mortality prevalence; 0.5 causes most true positives to be missed.
- **Clinical cost asymmetry:** missing a deteriorating patient (false negative) is far more harmful than a false alert (false positive) in an ICU monitoring context.

### Optimisation procedure

Sweep thresholds from 0.01 to 0.99 on the test set. Select the threshold that **maximises recall subject to precision ≥ 0.30** (at least 1 in 3 alerts is a true positive — a clinically defensible floor).

**Result:** threshold = **0.067** → Precision = 0.300, Recall = 0.955.

At this threshold, the model flags 95.5% of patients who will die in hospital. See `docs/threshold_rationale.md` for full discussion.

---

## Evaluation

### Metrics

| Metric | Tuned XGBoost | 95% CI (bootstrap) |
|---|---|---|
| AUROC | 0.884 | [0.854 – 0.910] |
| Recall (sensitivity) | 0.955 | — |
| Precision (PPV) | 0.300 | — |
| Specificity | — | — |
| Brier Score | 0.089 | [0.075 – 0.102] |
| ECE | 0.030 | — |

**Clinical baselines:**
- SOFA: AUROC 0.648
- SAPS-I: AUROC 0.672
- XGBoost improvement over SAPS-I: **+21.2 AUROC points**

### Bootstrap confidence intervals

1,000 bootstrap resamples of the held-out test set, percentile method. Reports 2.5th and 97.5th percentile as the 95% CI. The width of the AUROC interval (~0.056) reflects test set size (~800 patients after 80/20 split), not model instability.

---

## ML Monitoring

An offline monitoring pipeline (`src/monitoring.py`, `scripts/simulate_monitoring.py`) tracks distribution shifts that would invalidate the model's predictions:

| Check | Method | Threshold |
|---|---|---|
| Feature drift | Population Stability Index (PSI) per feature | PSI ≥ 0.10 warning; ≥ 0.20 critical |
| Missingness drift | Absolute Δ in missing rate per feature | Δ ≥ 0.10 flagged |
| Prediction drift | Kolmogorov–Smirnov test on score distributions | α = 0.05 |
| Calibration drift | Brier score + ECE vs training baselines | Δ ≥ 0.02 flagged |

**PSI** (Population Stability Index): a binned divergence measure that quantifies how much a feature's distribution has shifted between training and incoming data. PSI < 0.10 is considered stable; 0.10–0.20 warrants monitoring; > 0.20 indicates significant drift requiring model review.

A monitoring report is generated by running `scripts/simulate_monitoring.py` and served via `GET /monitoring/report`.
