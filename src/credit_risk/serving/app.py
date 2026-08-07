"""FastAPI service for the PD champion. Target: p95 under 200 ms per request."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from credit_risk.features.cleaning import clean_features
from credit_risk.serving.artifacts import ChampionBundle
from credit_risk.serving.schema import (
    IncompleteApplication,
    build_frame,
    required_raw_fields,
    validate_completeness,
)

BUNDLE_PATH = Path("artifacts/champion")
MAX_BATCH = 1_000

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion once. A failure here must stop the process, not the first request."""
    bundle = ChampionBundle.load(BUNDLE_PATH)
    _state["bundle"] = bundle
    _state["required"] = required_raw_fields(bundle.features)
    yield
    _state.clear()


app = FastAPI(
    title="Credit Risk PD Service",
    description="24-month probability of default for a loan application, plus a PDO point score.",
    version="1.0.0",
    lifespan=lifespan,
)


_EXAMPLE_APPLICATION = {
    "loan_amnt": 15000.0,
    "term": " 36 months",
    "issue_d": "Jan-2016",
    "earliest_cr_line": "Aug-2004",
    "annual_inc": 72000.0,
    "dti": 16.4,
    "fico_range_low": 705.0,
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
    "emp_length": "10+ years",
    "open_acc": 11.0,
    "total_acc": 24.0,
    "revol_bal": 9800.0,
    "revol_util": 38.5,
    "inq_last_6mths": 0.0,
    "acc_open_past_24mths": 3.0,
    "mort_acc": 1.0,
    "tot_cur_bal": 154000.0,
    "tot_hi_cred_lim": 198000.0,
    "total_bc_limit": 22000.0,
    "bc_open_to_buy": 12500.0,
    "bc_util": 43.2,
    "percent_bc_gt_75": 25.0,
    "mo_sin_old_rev_tl_op": 138.0,
    "mo_sin_rcnt_tl": 6.0,
    "mths_since_recent_bc": 14.0,
    "mths_since_recent_inq": 9.0,
    "num_tl_op_past_12m": 2.0,
    "pct_tl_nvr_dlq": 100.0,
    "delinq_2yrs": 0.0,
    "pub_rec": 0.0,
    "application_type": "Individual",
}


class ScoreRequest(BaseModel):
    """Raw application fields. Unknown keys are ignored; absent bureau fields become null.

    Core fields (loan_amnt, annual_inc, dti, fico_range_low, term, issue_d,
    earliest_cr_line) must be present and non-null, and at least half of the model's
    required fields must be supplied - see /model for the full list.
    """

    application: dict[str, Any] = Field(
        ..., description="Raw application and bureau fields, keyed by column name."
    )

    model_config = {"json_schema_extra": {"examples": [{"application": _EXAMPLE_APPLICATION}]}}


class BatchRequest(BaseModel):
    applications: list[dict[str, Any]] = Field(..., description=f"Up to {MAX_BATCH} applications.")


class ScoreResponse(BaseModel):
    probability_of_default: float = Field(..., description="Calibrated PD within 24 months.")
    score: float = Field(..., description="Points on the PDO scale; higher is safer.")
    model_version: str
    horizon_months: int = 24
    # Surfaced, not hidden: a score built from half the fields is weaker evidence than one
    # built from all of them, and the caller is the only party able to act on that.
    completeness: dict = Field(..., description="How much of the model's input was supplied.")


def _bundle() -> ChampionBundle:
    if "bundle" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _state["bundle"]


def _score_frame(frame: pl.DataFrame) -> tuple[list[float], list[float]]:
    bundle = _bundle()
    prepared = clean_features(frame)
    return (
        [float(v) for v in bundle.predict_pd(prepared)],
        [float(v) for v in bundle.predict_score(prepared)],
    )


@app.get("/health")
def health() -> dict:
    """Liveness only: the process is up. Does not assert the model is usable."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model is loaded and can serve. This is what a load balancer polls."""
    if "bundle" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    bundle = _state["bundle"]
    return {
        "status": "ready",
        "model_version": bundle.metadata.get("input_sha256", "unknown")[:12],
        "n_features": len(bundle.features),
    }


@app.get("/model")
def model_info() -> dict:
    """Full provenance: what this artefact was trained on and how it performed.

    Exposed deliberately. A production score that cannot be traced back to a dataset hash,
    a config and an out-of-time metric is not auditable.
    """
    bundle = _bundle()
    return {
        "metadata": bundle.metadata,
        "features": bundle.features,
        "required_request_fields": _state["required"],
        "scaling": bundle.scaling,
        "central_tendency_shift": bundle.shift,
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        completeness = validate_completeness(request.application, _state["required"])
    except IncompleteApplication as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    frame = build_frame(request.application, _state["required"])
    pds, scores = _score_frame(frame)
    return ScoreResponse(
        probability_of_default=pds[0],
        score=scores[0],
        model_version=_bundle().metadata.get("input_sha256", "unknown")[:12],
        completeness=completeness,
    )


@app.post("/score/batch")
def score_batch(request: BatchRequest) -> dict:
    """Scores many applications in one model call, which is where the throughput is."""
    if not request.applications:
        raise HTTPException(status_code=422, detail="applications must not be empty")
    if len(request.applications) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"batch exceeds {MAX_BATCH} applications")

    required = _state["required"]
    try:
        reports = [validate_completeness(a, required) for a in request.applications]
    except IncompleteApplication as exc:
        # All or nothing: a partially scored batch invites the caller to use the successes
        # without noticing the failures.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    frame = pl.concat([build_frame(a, required) for a in request.applications], how="vertical")
    pds, scores = _score_frame(frame)
    return {
        "results": [
            {"probability_of_default": p, "score": s, "completeness": c}
            for p, s, c in zip(pds, scores, reports, strict=True)
        ],
        "model_version": _bundle().metadata.get("input_sha256", "unknown")[:12],
        "count": len(pds),
    }


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """Server-side latency on every response, so p95 is measurable without a separate tool."""
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response
