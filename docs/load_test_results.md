# Load Test Results

## Setup

- **Tool:** Locust 2.37.6
- **Concurrency:** 20 users, ramp rate 5/s
- **Duration:** 60 seconds
- **Host:** localhost:8001 (single uvicorn worker, no reload)
- **Task weights:** `/patients` 3×, explanation 2×, `/predict` 2× each, risk-trend 1×

## Results

| Endpoint | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | Req/s |
|---|---|---|---|---|---|---|
| `GET /patients` | 402 | 0 | 15 | 74 | 130 | 6.7 |
| `GET /patients/{id}/explanation` | 201 | 0 | 24 | 71 | 110 | 3.4 |
| `GET /patients/{id}/risk-trend` | 105 | 0 | 92 | 220 | 270 | 1.8 |
| `POST /predict (stable)` | 212 | 0 | 21 | 79 | 140 | 3.6 |
| `POST /predict (critical)` | 216 | 0 | 20 | 100 | 180 | 3.6 |
| **Aggregated** | **1136** | **0** | **21** | **110** | **180** | **19.0** |

## Observations

- **Overall throughput:** 19 req/s at 20 concurrent users with 0% error rate.
- **Fastest endpoints:** `/patients` (p50=15ms) and `/predict` (p50~20ms). Both are lightweight: patient list reads from an in-memory dataframe; predict runs a single XGBoost forward pass plus SHAP attribution.
- **Slowest endpoint:** `/risk-trend` (p50=92ms, p95=220ms). This endpoint computes 8 snapshot predictions (one per 6-hour window) per request — each requiring a feature re-extraction and model call — making it inherently more expensive.
- **SHAP serialization:** A `threading.Lock` serializes SHAP calls across concurrent requests. This prevents race conditions in the non-thread-safe `TreeExplainer` at the cost of queuing concurrent explanation requests. Under higher concurrency, SHAP throughput would become the bottleneck.

## Bugs fixed during load testing

- `GET /patients/{id}/explanation` returned HTTP 500 under any load because NaN SHAP `value` fields bypassed `make_json_safe`. Fixed by wrapping all three patient detail endpoints in `make_json_safe`.
- Added `threading.Lock` around all SHAP calls (`/explanation` and `/predict`) to prevent concurrent TreeExplainer access.
