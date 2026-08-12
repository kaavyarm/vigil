# Vigil — System Architecture

## Overview

Vigil is a full-stack ICU mortality risk dashboard. The system has four logical layers: an offline data pipeline, a trained ML model, a REST API backend, and a React frontend. An optional PostgreSQL store and AWS cloud deployment are available for production use.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline (offline)                  │
│                                                                 │
│  PhysioNet 2012 ──► process_all_patients.py                    │
│  raw .txt files       (parse + structure)                       │
│       │                     │                                   │
│       ▼                     ▼                                   │
│  data/raw/        data/processed/patients/   ◄── 4 000 JSONs   │
│                             │                                   │
│                   build_feature_table.py                        │
│                   merge_outcomes.py                             │
│                             │                                   │
│                   data/features/training_dataset.csv            │
│                   (4 000 patients × 263 features)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Training (offline)                   │
│                                                                 │
│  scripts/train_model.py                                         │
│  ├── 5-fold stratified CV                                       │
│  ├── Hyperparameter search (scale_pos_weight, max_depth, lr)    │
│  ├── Calibration analysis (Platt / isotonic vs raw)             │
│  ├── Bootstrap 95% CIs (1 000 resamples)                        │
│  └── Threshold optimisation (max recall @ precision ≥ 0.30)     │
│                                                                 │
│  Outputs → models/  ────────────────────────────────────────►  │
│    vigil_xgboost_initial.joblib  (deployed)                     │
│    feature_columns.joblib                                       │
│    train_medians.joblib          (imputation reference)         │
│    threshold.joblib              (0.067)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴────────────────────┐
          │  S3 (optional)                         │
          │  s3://<MODEL_BUCKET>/models/            │
          │  Downloaded on startup via              │
          │  src/s3_loader.py if MODEL_BUCKET set   │
          └───────────────────┬────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                           │
│                       backend/app/main.py                       │
│                                                                 │
│  Startup                                                        │
│  ├── sync_from_s3()   — download models if MODEL_BUCKET set     │
│  ├── load_model_and_data()  — XGBoost + feature matrix          │
│  ├── shap.TreeExplainer()   — SHAP explainer (thread-locked)    │
│  └── db.init_pool()   — PostgreSQL pool if DATABASE_URL set     │
│                                                                 │
│  Endpoints                                                      │
│  ├── GET  /health              — ECS health check               │
│  ├── GET  /patients            — ranked patient list            │
│  ├── GET  /patients/{id}       — full patient record            │
│  ├── GET  /patients/{id}/explanation   — SHAP breakdown         │
│  ├── GET  /patients/{id}/risk-trend    — 6-hourly snapshots     │
│  ├── GET  /patients/{id}/timeline-events — clinical alerts      │
│  ├── POST /predict             — ad-hoc risk from vitals        │
│  └── GET  /monitoring/report   — drift report                   │
│                                                                 │
│  Data source (runtime)                                          │
│  ├── PostgreSQL  — if DATABASE_URL is set (production)          │
│  └── Flat JSON   — data/processed/patients/ (local / CI)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  REST / JSON
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       React Frontend                            │
│                       frontend/src/                             │
│                                                                 │
│  Views                                                          │
│  ├── Patient list   — sorted by risk, colour-coded badges       │
│  ├── Patient detail — SHAP waterfall, risk trend chart          │
│  └── Timeline       — flagged clinical events with severity     │
│                                                                 │
│  Deployed to Vercel; reads VITE_API_BASE env for backend URL    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ML Monitoring Pipeline                       │
│                                                                 │
│  scripts/simulate_monitoring.py  (run periodically or on-demand)│
│  ├── PSI feature drift    — 263 features, n_bins=10             │
│  ├── Missingness drift    — flags Δ > 10 pp per feature         │
│  ├── KS prediction drift  — on full predicted probability dist  │
│  └── Calibration drift    — Brier + ECE vs training baselines   │
│                                                                 │
│  Output → data/monitoring/report.json                           │
│  Served by GET /monitoring/report                               │
└─────────────────────────────────────────────────────────────────┘
```

## AWS Production Deployment

```
Internet
    │
    ▼
Application Load Balancer
    │
    ▼
ECS Fargate (vigil-api task)
├── Container image: ECR  (ACCOUNT.dkr.ecr.REGION.amazonaws.com/vigil:latest)
├── Secrets: DATABASE_URL  ←  Secrets Manager
├── Env: MODEL_BUCKET      ←  S3 (vigil-models-ACCOUNT)
└── Logs: /ecs/vigil-api   ←  CloudWatch Logs

    │ psycopg2 (SSL)
    ▼
RDS PostgreSQL (db.t3.micro)
└── patients table  (record_id PK, data JSONB, outcome, predicted_risk, predicted_at)
```

Provisioning is scripted in `infra/setup_aws.sh` (AWS CLI) and `infra/ecs_task_definition.json`.

## Local Development

```
docker compose up       # starts postgres:16 + api on :8001
                        # DATABASE_URL wired automatically
```

Without Docker: run `uvicorn backend.app.main:app --reload --port 8001` with `PYTHONPATH=src:backend`; the API falls back to flat JSON files.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

1. `ruff check .` — linting (E, F, I, W, UP, B, C4, RUF rules)
2. `pytest tests/ -v` — 106 unit and integration tests
