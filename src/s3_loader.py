"""
Downloads model artifacts from S3 on startup when MODEL_BUCKET is set.

Falls back silently to local files if the env var is absent, so the app
works without AWS credentials in local / CI environments.

Expected S3 layout:
    s3://<MODEL_BUCKET>/models/vigil_xgboost_initial.joblib
    s3://<MODEL_BUCKET>/models/feature_columns.joblib
    s3://<MODEL_BUCKET>/models/train_medians.joblib
    s3://<MODEL_BUCKET>/models/threshold.joblib
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_KEYS = [
    "models/vigil_xgboost_initial.joblib",
    "models/feature_columns.joblib",
    "models/train_medians.joblib",
    "models/threshold.joblib",
]


def sync_from_s3(local_root: Path) -> None:
    """Download any missing model files from S3. No-ops if MODEL_BUCKET is unset."""
    bucket = os.environ.get("MODEL_BUCKET")
    if not bucket:
        return

    try:
        import boto3

        s3 = boto3.client("s3")
    except Exception:
        logger.exception("boto3 unavailable — skipping S3 model sync")
        return

    for key in MODEL_KEYS:
        local_path = local_root / key
        if local_path.exists():
            continue  # prefer locally cached copy

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading s3://%s/%s", bucket, key)
            s3.download_file(bucket, key, str(local_path))
        except Exception:
            logger.exception("Failed to download s3://%s/%s", bucket, key)
