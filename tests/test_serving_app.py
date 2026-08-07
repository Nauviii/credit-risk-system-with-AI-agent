"""Tests for the scoring service, against a real bundle built in a temp directory."""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from credit_risk.evaluation.calibration import Calibrator
from credit_risk.serving.artifacts import categorical_levels, save_bundle


def _payload(**overrides) -> dict:
    base = {
        "term": " 36 months",
        "earliest_cr_line": "Jan-2005",
        "issue_d": "Jan-2016",
        "annual_inc": 65_000.0,
        "dti": 14.2,
        "loan_amnt": 15_000.0,
        "fico_range_low": 705.0,
        "purpose": "car",
    }
    return {**base, **overrides}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Build a small champion bundle, point the app at it, and start the app."""
    import lightgbm as lgb

    from credit_risk.models.gbm import prepare_lgb_frame
    from credit_risk.serving import app as app_module

    rng = np.random.default_rng(0)
    n = 2_000
    df = pl.DataFrame(
        {
            "annual_inc": rng.normal(60_000, 15_000, n),
            "dti": rng.normal(15, 6, n),
            "purpose": rng.choice(["car", "house", "medical"], n),
            "term_months": rng.choice([36, 60], n).astype(float),
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + df["dti"].to_numpy() / 10)))).astype(int)

    features = ["annual_inc", "dti", "purpose", "term_months"]
    frame = prepare_lgb_frame(df, features)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0, "num_leaves": 8},
        lgb.Dataset(frame, label=y),
        num_boost_round=30,
    )
    calibrator = Calibrator("platt").fit(y, booster.predict(frame))

    directory = tmp_path_factory.mktemp("artifacts") / "champion"
    save_bundle(
        directory,
        booster,
        features,
        categorical_levels(frame),
        calibrator,
        metadata={"input_sha256": "abcdef0123456789", "oot_auc": 0.6964},
    )
    app_module.BUNDLE_PATH = directory
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_does_not_assert_the_model_is_loaded(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_the_loaded_model(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["n_features"] == 4
    assert body["model_version"] == "abcdef012345"


def test_score_returns_a_pd_and_a_monotone_score(client):
    safe = client.post("/score", json={"application": _payload(dti=3.0)}).json()
    risky = client.post("/score", json={"application": _payload(dti=32.0)}).json()
    assert 0.0 < safe["probability_of_default"] < 1.0
    assert risky["probability_of_default"] > safe["probability_of_default"]
    assert risky["score"] < safe["score"]  # more points means safer
    assert safe["horizon_months"] == 24


def test_missing_optional_field_is_scored_as_null_not_rejected(client):
    """A bureau attribute a lender genuinely could not pull is a real case, not an error."""
    response = client.post("/score", json={"application": _payload(purpose=None)})
    assert response.status_code == 200
    assert 0.0 < response.json()["probability_of_default"] < 1.0


def test_empty_application_is_refused_rather_than_scored(client):
    """The bug this check exists for: an empty payload used to return a plausible PD."""
    response = client.post("/score", json={"application": {}})
    assert response.status_code == 422
    assert "core fields" in response.json()["detail"]


def test_missing_core_field_names_the_field(client):
    payload = _payload()
    del payload["annual_inc"]
    response = client.post("/score", json={"application": payload})
    assert response.status_code == 422
    assert "annual_inc" in response.json()["detail"]


def test_null_core_field_is_treated_as_absent(client):
    """Present-but-null is not provided: `annual_inc: null` must fail like omitting it."""
    response = client.post("/score", json={"application": _payload(annual_inc=None)})
    assert response.status_code == 422
    assert "annual_inc" in response.json()["detail"]


def test_response_reports_how_much_input_was_supplied(client):
    body = client.post("/score", json={"application": _payload()}).json()
    assert body["completeness"]["fields_provided"] > 0
    assert 0.0 < body["completeness"]["coverage"] <= 1.0


def test_a_single_bad_application_fails_the_whole_batch(client):
    """Partial success invites a caller to use the good rows without noticing the bad ones."""
    response = client.post(
        "/score/batch", json={"applications": [_payload(), {"loan_amnt": 1000.0}]}
    )
    assert response.status_code == 422


def test_batch_matches_single_scoring_exactly(client):
    payloads = [_payload(dti=d) for d in (3.0, 15.0, 32.0)]
    batch = client.post("/score/batch", json={"applications": payloads}).json()
    singles = [client.post("/score", json={"application": p}).json() for p in payloads]
    assert batch["count"] == 3
    for got, want in zip(batch["results"], singles, strict=True):
        assert got["probability_of_default"] == pytest.approx(want["probability_of_default"])


def test_empty_and_oversized_batches_are_refused(client):
    from credit_risk.serving.app import MAX_BATCH

    assert client.post("/score/batch", json={"applications": []}).status_code == 422
    too_many = [_payload()] * (MAX_BATCH + 1)
    assert client.post("/score/batch", json={"applications": too_many}).status_code == 413


def test_model_endpoint_exposes_provenance(client):
    body = client.get("/model").json()
    assert body["metadata"]["oot_auc"] == 0.6964
    assert "annual_inc" in body["required_request_fields"]
    assert body["scaling"]["pdo"] == 20


def test_every_response_carries_server_side_latency(client):
    response = client.post("/score", json={"application": _payload()})
    assert float(response.headers["X-Response-Time-Ms"]) >= 0.0


def test_single_request_latency_is_well_inside_the_budget(client):
    """p95 target is 200 ms end to end; the model call itself must be a small part of it."""
    import time

    payload = {"application": _payload()}
    client.post("/score", json=payload)  # warm up
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        client.post("/score", json=payload)
        timings.append((time.perf_counter() - started) * 1000)
    assert sorted(timings)[int(len(timings) * 0.95)] < 200.0