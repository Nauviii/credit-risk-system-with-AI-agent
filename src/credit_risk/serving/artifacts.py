"""Persist a trained champion so serving can load it, and load it back identically."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

_BOOSTER_FILE = "model.txt"
_BUNDLE_FILE = "bundle.json"
_SCHEMA_VERSION = 2


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so a 1.6 GB CSV can be hashed without loading it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def categorical_levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Category values per categorical column, in the order training used them."""
    return {
        column: [str(v) for v in frame[column].cat.categories]
        for column in frame.columns
        if isinstance(frame[column].dtype, pd.CategoricalDtype)
    }


def save_bundle(
    directory: Path,
    booster: lgb.Booster,
    features: list[str],
    levels: dict[str, list[str]],
    calibrator,
    central_tendency_shift: float = 0.0,
    scaling: dict | None = None,
    metadata: dict | None = None,
    reference: dict | None = None,
) -> Path:
    """Write booster + JSON sidecar to `directory`, returning the path written."""
    if calibrator.method != "platt":
        raise ValueError(
            f"only platt calibrators are serialisable as parameters, got '{calibrator.method}'"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / _BOOSTER_FILE), num_iteration=booster.best_iteration)

    bundle = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": list(features),
        "categorical_levels": levels,
        "calibrator": {
            "method": "platt",
            "coef": float(calibrator.model_.coef_[0][0]),
            "intercept": float(calibrator.model_.intercept_[0]),
            "floor": float(calibrator.floor),
            "cap": float(calibrator.cap),
        },
        "central_tendency_shift": float(central_tendency_shift),
        "scaling": scaling or {"pdo": 20, "base_score": 600, "base_odds": 50.0},
        "reference": reference or {},
        "metadata": metadata or {},
    }
    (directory / _BUNDLE_FILE).write_text(json.dumps(bundle, indent=2))
    return directory


class ChampionBundle:
    """A loaded champion: raw application frame in, calibrated PD and points out.

    This is the single entry point serving should call. Keeping the whole path - column
    order, categorical levels, calibration, central tendency, point scaling - behind one
    object is what stops a serving layer from reimplementing four of those five steps
    slightly differently from training.
    """

    def __init__(self, booster: lgb.Booster, bundle: dict):
        self.booster = booster
        self.features: list[str] = bundle["features"]
        self.levels: dict[str, list[str]] = bundle["categorical_levels"]
        self.calibrator: dict = bundle["calibrator"]
        self.shift: float = bundle["central_tendency_shift"]
        self.scaling: dict = bundle["scaling"]
        self.reference: dict = bundle.get("reference", {})
        self.metadata: dict = bundle.get("metadata", {})

    @classmethod
    def load(cls, directory: Path) -> "ChampionBundle":
        directory = Path(directory)
        bundle = json.loads((directory / _BUNDLE_FILE).read_text())
        if bundle["schema_version"] != _SCHEMA_VERSION:
            raise ValueError(
                f"bundle schema {bundle['schema_version']} != expected {_SCHEMA_VERSION}"
            )
        return cls(lgb.Booster(model_file=str(directory / _BOOSTER_FILE)), bundle)

    def prepare(self, df: pl.DataFrame) -> pd.DataFrame:
        """Select features in stored order and re-apply stored categorical levels.

        Missing features raise rather than default: a silently absent column becomes an
        all-null feature and shifts every score, which is far worse than a failed request.
        """
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"missing required features: {missing}")

        frame = df.select(self.features).to_pandas()
        for column, values in self.levels.items():
            frame[column] = pd.Categorical(frame[column].astype("object"), categories=values)
        return frame

    def predict_raw(self, df: pl.DataFrame) -> np.ndarray:
        """Uncalibrated model output, before any level correction."""
        return self.booster.predict(self.prepare(df))

    def predict_pd(self, df: pl.DataFrame) -> np.ndarray:
        """Calibrated, centrally-anchored probability of default within the horizon."""
        raw = np.clip(self.predict_raw(df), 1e-9, 1 - 1e-9)
        logits = np.log(raw / (1 - raw))
        calibrated = 1 / (
            1 + np.exp(-(self.calibrator["coef"] * logits + self.calibrator["intercept"]))
        )
        calibrated = np.clip(calibrated, self.calibrator["floor"], self.calibrator["cap"])
        anchored_logits = np.log(calibrated / (1 - calibrated)) + self.shift
        return 1 / (1 + np.exp(-anchored_logits))

    def predict_score(self, df: pl.DataFrame) -> np.ndarray:
        """Points on the PDO scale - what a cutoff and an override are expressed in."""
        from credit_risk.evaluation.calibration import pd_to_score

        return pd_to_score(
            self.predict_pd(df),
            pdo=self.scaling["pdo"],
            base_score=self.scaling["base_score"],
            base_odds=self.scaling["base_odds"],
        )
