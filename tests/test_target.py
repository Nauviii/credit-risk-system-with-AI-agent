"""Tests for the fixed-horizon default definition and its observability filter."""

from pathlib import Path

import polars as pl
import pytest
import yaml

from credit_risk.data.target import build_target, load_target_config

_CONFIG = Path("configs/base.yaml")


@pytest.fixture
def config(tmp_path) -> Path:
    """Config with H=24, lag=5, cutoff 2018-12 - mirrors configs/base.yaml."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "target": {
            "horizon_months": 24,
            "charge_off_lag_months": 5,
            "data_cutoff": "2018-12",
            "bad_statuses": ["Charged Off", "Default"],
        }
    }))
    return path


def _loan(issue_d: str, status: str, last_pymnt_d: str | None) -> dict:
    return {"issue_d": issue_d, "loan_status": status, "last_pymnt_d": last_pymnt_d}


def test_real_config_is_loadable_and_horizon_fits_the_cutoff():
    cfg = load_target_config(_CONFIG)
    split = yaml.safe_load(open(_CONFIG))["split"]
    oot_end_year, oot_end_month = (int(p) for p in split["oot_test_end"].split("-"))
    cutoff_year, cutoff_month = (int(p) for p in str(cfg["data_cutoff"]).split("-"))
    observable = (cutoff_year * 12 + cutoff_month) - (oot_end_year * 12 + oot_end_month)
    assert observable >= cfg["horizon_months"]


def test_healthy_loans_are_labelled_not_dropped(config):
    df = pl.DataFrame([
        _loan("Jan-2016", "Current", None),
        _loan("Jan-2016", "Late (31-120 days)", "Jun-2017"),
        _loan("Jan-2016", "Fully Paid", "Dec-2017"),
    ])
    result = build_target(df, config)
    assert result.height == 3
    assert result["default_flag"].to_list() == [0, 0, 0]


def test_default_inside_horizon_flagged_outside_horizon_not(config):
    df = pl.DataFrame([
        _loan("Jan-2015", "Charged Off", "Jan-2016"),  # mob_event 17 <= 24
        _loan("Jan-2015", "Charged Off", "Jan-2017"),  # mob_event 29 > 24
    ])
    result = build_target(df, config)
    assert result["mob_event"].to_list() == [17, 29]
    assert result["default_flag"].to_list() == [1, 0]


def test_loan_too_young_to_observe_is_excluded(config):
    df = pl.DataFrame([
        _loan("Dec-2016", "Current", None),  # 24 months observable, kept
        _loan("Jan-2017", "Current", None),  # 23 months observable, dropped
    ])
    result = build_target(df, config)
    assert result.height == 1
    assert result["mob_observable"].to_list() == [24]


def test_never_paid_default_lands_at_earliest_month(config):
    df = pl.DataFrame([_loan("Jan-2015", "Charged Off", None)])
    result = build_target(df, config)
    assert result["mob_event"][0] == 5
    assert result["default_flag"][0] == 1


def test_legacy_prefix_statuses_are_normalised_and_counted(config):
    df = pl.DataFrame([
        _loan("Jan-2015", "Does not meet the credit policy. Status:Charged Off", "Jan-2016"),
        _loan("Jan-2015", "Does not meet the credit policy. Status:Fully Paid", "Dec-2017"),
    ])
    result = build_target(df, config)
    assert result["loan_status"].to_list() == ["Charged Off", "Fully Paid"]
    assert result["default_flag"].to_list() == [1, 0]


def test_horizon_change_shifts_labels_not_row_count(config):
    """A longer horizon must relabel loans, not silently drop or add rows."""
    long_cfg = config.parent / "long.yaml"
    cfg = yaml.safe_load(open(config))
    cfg["target"]["horizon_months"] = 36
    long_cfg.write_text(yaml.safe_dump(cfg))

    df = pl.DataFrame([_loan("Jan-2015", "Charged Off", "Jan-2017")])
    assert build_target(df, config)["default_flag"][0] == 0
    assert build_target(df, long_cfg)["default_flag"][0] == 1