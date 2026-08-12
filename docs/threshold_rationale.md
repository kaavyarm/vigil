# Decision Threshold Rationale

## Selected threshold

The deployed model uses a decision threshold selected to **maximize recall (sensitivity) subject to a minimum precision constraint of 0.30**, rather than the default 0.50 threshold or the F1-maximizing point.

## Why asymmetric costs

In ICU mortality prediction, the costs of the two error types are not equal:

- **False negative (missed deterioration):** A patient at high risk is scored as low risk. Clinical intervention is not escalated. The consequence is delayed treatment for a patient who may die.
- **False positive (unnecessary alert):** A patient at low risk is scored as high risk. An additional clinical review is triggered. The consequence is a brief increase in workload.

Missed deterioration is clinically worse than an unnecessary alert. The threshold was therefore shifted toward higher recall, accepting lower precision to ensure fewer high-risk patients are missed. A precision floor of 0.30 prevents the model from flagging the majority of patients as high-risk, which would render the alerts uninformative.

## Implication

The deployed threshold is not the F1-maximizing or accuracy-maximizing point. This is an intentional design choice: Vigil is a decision-support tool whose utility depends on catching true deterioration, not on optimizing aggregate classification accuracy.
