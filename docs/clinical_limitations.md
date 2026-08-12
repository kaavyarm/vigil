# Vigil — Clinical Limitations

This document describes the known limitations of the Vigil model. It is intended for anyone evaluating whether the system is appropriate for a given use case.

**Vigil is a research and portfolio project. It is not validated for clinical use and must not be used to guide patient care decisions.**

---

## 1. Retrospective, not prospective

The model was trained and evaluated on historical records. It has never been tested in a prospective trial where clinicians act on its predictions. Retrospective performance does not guarantee prospective performance — clinician behaviour changes when an alert system is present, which can affect both outcomes and the data distribution.

## 2. Single-institution, single-era data

The PhysioNet 2012 Challenge dataset contains records from 12 ICUs at a single US hospital system, collected between 2004 and 2008. The model may not generalise to:

- **Different institutions** with different care protocols, patient populations, or documentation practices.
- **Modern ICUs** where practices, equipment, and medications have changed substantially since 2008.
- **Non-US healthcare systems** where measurement conventions, normal ranges, and documentation styles differ.

## 3. Missing data is not random

Approximately 30–60% of measurements are missing for most clinical parameters. The model treats missingness as informative (a `_measured = 0` flag is a feature). This assumption — that "not measured" carries clinical meaning — is reasonable in the training data but may not hold in a different hospital where measurement protocols differ.

Imputing missing values with training-set medians, as Vigil does, introduces bias if the incoming population's baseline physiology differs from the training cohort.

## 4. Only 32 parameters

The PhysioNet dataset records 37 parameters; Vigil uses 32 of those. Many clinically important inputs are absent entirely: medication history, nursing assessments, imaging findings, surgical history, prior hospital admissions, comorbidity indices, and social determinants of health.

## 5. 48-hour window only

The model only uses data from the first 48 hours of ICU admission. It cannot account for clinical trajectories that develop after 48 hours, or for events that occurred before ICU admission.

## 6. Binary outcome

The model predicts binary in-hospital mortality. It does not predict:

- Time to death or discharge
- ICU length of stay
- Cause of death
- Quality of life outcomes
- Readmission risk

## 7. Calibration on held-out data, not prospective cohorts

The ECE of 0.030 was measured on a held-out split of the same dataset. Calibration on an external prospective cohort may differ substantially, particularly if the baseline mortality rate differs from the ~14% in this dataset.

## 8. No causal interpretation

SHAP values explain which features most influenced the model's prediction for a given patient. They do not imply that changing those features would change the patient's outcome. High lactate contributing to a high-risk prediction does not mean that treating the lactate will lower mortality — the causal relationship requires separate investigation.

## 9. Alert fatigue risk

At the deployed threshold (0.067), precision is 0.300: approximately 70% of flagged patients will not die in hospital. In a real clinical setting, a 70% false positive rate may contribute to alert fatigue, causing clinicians to ignore or dismiss alerts. The threshold was chosen to maximise recall (0.955) at a floor of precision ≥ 0.30, which was judged acceptable for a monitoring tool — but this trade-off should be re-evaluated for any real deployment.

## 10. No external validation

The model has not been validated on any dataset other than PhysioNet 2012 Set A. External validation on independent cohorts (e.g. MIMIC-IV, eICU) is a prerequisite before any clinical consideration.

---

## Summary

| Limitation | Severity | Mitigation required before clinical use |
|---|---|---|
| No prospective validation | Critical | Randomised trial or stepped-wedge study |
| Single institution / era | High | External validation on ≥ 2 independent cohorts |
| Informative missingness assumption | Moderate | Validate in target institution's documentation context |
| 32-parameter feature set | Moderate | Assess impact of additional data sources |
| Binary outcome only | Low | Extend to time-to-event if discharge planning is the goal |
| No causal interpretation | Informational | Frame as correlation signal, not treatment target |
