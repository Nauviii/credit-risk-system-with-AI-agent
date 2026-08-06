"""Centralized runtime configuration, loaded from environment / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Single source of truth for paths and service config across all phases."""

    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    interim_data_dir: Path = ROOT_DIR / "data" / "interim"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    mlflow_tracking_uri: str = f"sqlite:///{ROOT_DIR / 'mlruns.db'}"
    random_seed: int = 42

    model_config = {"env_file": ".env", "env_prefix": "CREDIT_RISK_", "extra": "ignore"}


settings = Settings()
