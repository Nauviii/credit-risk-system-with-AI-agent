"""What a scoring request must contain, derived from the bundle rather than hand-listed.

The API accepts RAW application fields, not model features, and runs the same
`clean_features` pipeline the training path runs. Asking a client to send `term_months`,
`credit_history_months` and the `has_*` flags would push feature engineering across the
network boundary, which is the textbook source of training/serving skew: two
implementations of the same derivation that drift apart silently.

The required field list is computed from the loaded bundle, so it cannot fall out of sync
with the model. Retrain with a different feature set and the contract follows.
"""

import polars as pl

from credit_risk.data.schema import STRUCTURALLY_MISSING_COLUMNS

# Model feature -> raw fields it is derived from. Everything else is passed through as-is.
DERIVED_FROM: dict[str, list[str]] = {
    "term_months": ["term"],
    "credit_history_months": ["earliest_cr_line", "issue_d"],
}

# `clean_features` reads these regardless of whether the model uses their output.
_ALWAYS_REQUIRED = ["term", "earliest_cr_line", "issue_d"]

# Fields that must be present AND non-null for a score to mean anything. Everything else
# is a bureau attribute that a lender can genuinely fail to pull, and null is a real value
# the model handles. These are the ones that exist on any application by construction -
# if they are absent, the caller has sent something malformed, not something incomplete.
#
# Without this check the service happily scores an EMPTY payload: every field becomes null,
# the model returns its "everything unknown" prediction near the base rate, and the caller
# receives a plausible-looking PD computed from nothing. In credit decisioning that is far
# worse than a rejected request.
CORE_REQUIRED_FIELDS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "term",
    "issue_d",
    "earliest_cr_line",
]

# Beyond the core fields, refuse an application whose bureau section is mostly absent. The
# model tolerates scattered nulls; it was never evaluated on a near-empty applicant.
MIN_FIELD_COVERAGE = 0.5


class IncompleteApplication(ValueError):
    """Raised when a payload cannot support a score, as opposed to merely lacking a field."""


def validate_completeness(
    payload: dict, required: list[str], min_coverage: float = MIN_FIELD_COVERAGE
) -> dict:
    """Check a payload can support a score. Returns a completeness report, or raises.

    Two distinct failures, reported separately because they mean different things:
    a missing core field is a malformed request, while low coverage is a data-quality
    problem upstream of the caller.
    """
    provided = {k for k, v in payload.items() if k in required and v is not None}
    # Intersect with `required`: demanding a core field the loaded model does not use would
    # tie the contract to one feature set rather than to the bundle.
    core = [f for f in CORE_REQUIRED_FIELDS if f in required]
    missing_core = [f for f in core if f not in provided]
    if missing_core:
        raise IncompleteApplication(f"missing or null core fields: {missing_core}")

    coverage = len(provided) / len(required) if required else 0.0
    if coverage < min_coverage:
        raise IncompleteApplication(
            f"only {len(provided)}/{len(required)} fields provided "
            f"({coverage:.0%}); minimum is {min_coverage:.0%}"
        )
    return {
        "fields_provided": len(provided),
        "fields_required": len(required),
        "coverage": round(coverage, 4),
    }


def required_raw_fields(features: list[str]) -> list[str]:
    """Raw fields a request must carry so `clean_features` can produce `features`.

    A `has_x` flag needs `x`, not itself. A derived feature needs its sources. Everything
    else is required under its own name.
    """
    required: set[str] = set(_ALWAYS_REQUIRED)
    for feature in features:
        if feature in DERIVED_FROM:
            required.update(DERIVED_FROM[feature])
        elif feature.startswith("has_") and feature[4:] in STRUCTURALLY_MISSING_COLUMNS:
            required.add(feature[4:])
        else:
            required.add(feature)
    return sorted(required)


def build_frame(payload: dict, required: list[str]) -> pl.DataFrame:
    """One-row frame with every required field present, missing ones as explicit nulls.

    Absent optional fields become null rather than an error: null is meaningful in this
    model (WOE and LightGBM both treat it as its own group), and a bureau field a lender
    genuinely could not pull is a real production case, not a malformed request.
    """
    row = {field: payload.get(field) for field in required}
    return pl.DataFrame([row], schema={field: _dtype_for(field, row[field]) for field in required})


def _dtype_for(field: str, value) -> pl.DataType:
    """String columns must stay Utf8 even when null, or downstream string ops fail."""
    if field in ("term", "earliest_cr_line", "issue_d") or isinstance(value, str):
        return pl.Utf8
    return pl.Float64