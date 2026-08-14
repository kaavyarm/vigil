#!/bin/bash
set -e
ROOT=$(pwd)
export PYTHONPATH="$ROOT/src:$ROOT/backend"
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
