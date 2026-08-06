"""Standardized MLflow run logging so every experiment is comparable apples-to-apples."""

import subprocess
from contextlib import contextmanager

import mlflow

from credit_risk.config import settings


def _git_commit() -> str:
    """Return short git commit hash, or 'unknown' outside a git repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@contextmanager
def start_run(run_name: str, config: dict):
    """Open an MLflow run, auto-logging git commit and seed so every run is reproducible."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("random_seed", settings.random_seed)
        mlflow.log_params(config)
        yield run
