# Vigil

[![CI](https://github.com/kaavyarm/vigil/actions/workflows/ci.yml/badge.svg)](https://github.com/kaavyarm/vigil/actions/workflows/ci.yml)

ICU mortality risk prediction and monitoring dashboard built on the [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/1.0.0/) dataset.

![Vigil ICU Dashboard](docs/screenshot.png)

## What it does

Vigil ingests 48-hour ICU patient records, engineers 263 time-series features from raw clinical measurements, and predicts in-hospital mortality risk using a calibrated XGBoost model. A FastAPI backend serves predictions to a React dashboard showing:

- **Patient list** — all patients ranked by predicted risk with colour-coded severity badges
- **SHAP explanations** — top risk factors and protective factors for each patient, with human-readable labels and contribution percentages
- **Risk trend** — how predicted mortality risk evolves hour-by-hour as new data arrives
- **Timeline events** — rule-based clinical alerts (low MAP, elevated lactate, mechanical ventilation, etc.) with severity and category
- **Monitoring report** — distribution drift diagnostics for the prediction pipeline

## Results

### Model performance

| Model | AUROC | 95% CI | Brier Score | 95% CI |
|---|---|---|---|---|
| SOFA (clinical baseline) | 0.648 | — | — | — |
| SAPS-I (clinical baseline) | 0.672 | — | — | — |
| Logistic Regression | 0.856 | — | — | — |
| Random Forest | 0.861 | — | — | — |
| **XGBoost (deployed)** | **0.878** | — | — | — |
| XGBoost (tuned) | **0.884** | [0.854–0.910] | **0.089** | [0.075–0.102] |

XGBoost clears the best clinical baseline (SAPS-I) by **+21 AUROC points**. 5-fold stratified cross-validation mean: **0.842 ± 0.013**.

### Calibration

Expected Calibration Error (ECE) on held-out test set: **0.030** — raw XGBoost is already well-calibrated against Platt scaling and isotonic regression, which marginally worsened ECE on this dataset. See [docs/methodology.md](docs/methodology.md).

### Decision threshold

Threshold optimised for maximum recall at precision ≥ 0.30 (clinical cost asymmetry: a missed deterioration is far more harmful than a false alert):

- **Threshold:** 0.067
- **Recall:** 0.955 — flags 95.5% of patients who will die in hospital
- **Precision:** 0.300

See [docs/threshold_rationale.md](docs/threshold_rationale.md).

### Load testing

Tested with Locust (20 concurrent users, 60 seconds):

| Endpoint | p50 (ms) | p95 (ms) | Req/s |
|---|---|---|---|
| `GET /patients` | 15 | 74 | 6.7 |
| `GET /patients/{id}/explanation` | 24 | 71 | 3.4 |
| `POST /predict` | 21 | 90 | 3.6 |
| `GET /patients/{id}/risk-trend` | 92 | 220 | 1.8 |
| **Aggregated** | **21** | **110** | **19.0** |

Zero failures at 20 concurrent users. See [docs/load_test_results.md](docs/load_test_results.md).

### ML monitoring (simulation)

Reference: first 3,500 patients. Incoming: last 500 patients.

- Feature drift: 0 features critical, 8 features warning (liver enzyme trends: AST, ALT, ALP PSI ≈ 0.14–0.18)
- Prediction drift: stable (KS p = 0.57)
- Calibration drift: ECE Δ = +0.053 (flagged — incoming cohort is harder to calibrate)
- Overall health: **CRITICAL** (calibration drift)

## Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, pandas, NumPy |
| ML | XGBoost, scikit-learn, SHAP (TreeExplainer) |
| Backend | FastAPI, Uvicorn |
| Database | PostgreSQL (psycopg2 ThreadedConnectionPool) |
| Frontend | React, Recharts |
| Containerisation | Docker, Docker Compose |
| Cloud | AWS ECS/Fargate, RDS PostgreSQL, S3, ECR, CloudWatch, Secrets Manager |
| CI/CD | GitHub Actions (ruff lint + pytest on every push) |
| Load testing | Locust |

## Project structure

```
vigil/
├── src/                                # Runtime library
│   ├── explain_model.py                # SHAP-based per-patient explanations
│   ├── feature_engineering.py          # 263-feature extraction per patient
│   ├── process_one_patient.py          # Parse raw PhysioNet .txt → JSON
│   ├── risk_trend.py                   # 6-hourly risk snapshots
│   ├── timeline_events.py              # Rule-based clinical alert detection
│   ├── monitoring.py                   # PSI / KS / calibration drift detection
│   ├── db.py                           # PostgreSQL connection pool + CRUD
│   └── s3_loader.py                    # Download model artifacts from S3
├── backend/
│   └── app/main.py                     # FastAPI REST API (10 endpoints)
├── frontend/src/                       # React dashboard
├── scripts/                            # Offline pipeline scripts
│   ├── process_all_patients.py         # ETL: raw → JSON
│   ├── build_feature_table.py          # Assemble feature matrix
│   ├── merge_outcomes.py               # Join mortality outcomes
│   ├── train_model.py                  # Train + CV + calibration + threshold
│   ├── simulate_monitoring.py          # Generate monitoring report
│   └── load_patients_to_db.py          # Bulk-insert 4 000 patients to PostgreSQL
├── tests/                              # 106 unit + integration tests
├── infra/
│   ├── ecs_task_definition.json        # ECS Fargate task definition template
│   └── setup_aws.sh                    # One-shot AWS provisioning script
├── docs/
│   ├── architecture.md                 # System diagram + component descriptions
│   ├── methodology.md                  # Modelling decisions + evaluation
│   ├── clinical_limitations.md         # Scope and validation caveats
│   ├── threshold_rationale.md          # Decision threshold justification
│   └── load_test_results.md            # Locust benchmark results
├── data/
│   ├── processed/patients/             # 4 000 per-patient JSON files
│   ├── features/                       # Engineered feature matrix (CSV)
│   └── monitoring/report.json          # Latest monitoring report
├── models/                             # Saved models + feature metadata
├── Dockerfile                          # python:3.12-slim production image
├── docker-compose.yml                  # API + postgres:16 local stack
└── render.yaml                         # Render.com deployment config
```

## Local setup

**Requirements:** Python 3.12+, Node 18+

### Option A — Docker Compose (recommended)

```bash
docker compose up
# API: http://localhost:8001
# PostgreSQL: localhost:5432 (user: vigil, pass: vigil, db: vigil)
```

To seed the database with patient data after the containers are running:

```bash
DATABASE_URL=postgresql://vigil:vigil@localhost:5432/vigil \
  python scripts/load_patients_to_db.py
```

### Option B — without Docker

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download PhysioNet 2012 Set A → data/raw/Set A/

# 3. Run the data pipeline
python scripts/process_all_patients.py
python scripts/build_feature_table.py
python scripts/merge_outcomes.py
python scripts/train_model.py

# 4. Start the backend (no DATABASE_URL → uses flat JSON files)
PYTHONPATH=src:backend uvicorn backend.app.main:app --reload --port 8001

# 5. Start the frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

### Tests

```bash
pytest                  # 106 tests
pytest -v --tb=short    # verbose
```

### Load testing

```bash
pip install -r requirements-dev.txt
locust -f tests/locustfile.py --host=http://localhost:8001
# Open http://localhost:8089 → set 20 users, spawn 5/s
```

### Monitoring report

```bash
python scripts/simulate_monitoring.py
# → data/monitoring/report.json
# → served at GET /monitoring/report
```

## AWS deployment

See [docs/architecture.md](docs/architecture.md) for the full AWS architecture diagram. To provision:

```bash
# Requires AWS CLI v2 configured (aws configure)
bash infra/setup_aws.sh
```

This creates: S3 model bucket, ECR repository (builds + pushes the image), RDS PostgreSQL instance, DATABASE_URL in Secrets Manager, CloudWatch log group, and registers the ECS task definition.

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System diagram, component descriptions, AWS deployment |
| [docs/methodology.md](docs/methodology.md) | Feature engineering, model selection, calibration, threshold, monitoring |
| [docs/clinical_limitations.md](docs/clinical_limitations.md) | Dataset scope, generalisation caveats, validation requirements |
| [docs/threshold_rationale.md](docs/threshold_rationale.md) | Clinical cost asymmetry and threshold selection |
| [docs/load_test_results.md](docs/load_test_results.md) | Locust benchmark results and concurrency findings |
