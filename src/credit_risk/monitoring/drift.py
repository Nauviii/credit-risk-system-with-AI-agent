"""Distribution drift against the reference frozen inside the model bundle.

Pairs with monitoring/performance.py and answers a different question. This module asks
whether the population still looks like the one the model was fitted on; that one asks
whether outcomes still match what it predicted. Phase 5 showed a case where the first said
yes and the second said no, so neither substitutes for the other.

What a green result here actually licenses: it says the score's inputs are in familiar
territory, so a performance alert is more likely to be genuine deterioration than a feed
problem. That is useful, and it is not the same as saying the model is fine.
"""

import polars as pl

from credit_risk.evaluation.stability import outside_reference_range, psi_against_reference

WATCH_THRESHOLD = 0.10
MATERIAL_THRESHOLD = 0.25


# Above this share of values beyond the reference's outer edges, the two are not the same
# quantity and PSI is meaningless rather than large.
OUT_OF_RANGE_THRESHOLD = 0.5


def _band(psi: float, out_of_range: float = 0.0) -> str:
    if out_of_range > OUT_OF_RANGE_THRESHOLD:
        return "reference mismatch"
    if psi > MATERIAL_THRESHOLD:
        return "material shift"
    return "watch" if psi > WATCH_THRESHOLD else "stable"


class DriftMonitor:
    """Compares a live batch against the reference profiles stored with the champion."""

    def __init__(self, bundle):
        self.reference = bundle.reference
        if not self.reference:
            raise ValueError(
                "bundle carries no reference profiles; rebuild it with scripts/build_artifacts.py"
            )

    def score_drift(self, scores) -> dict:
        """PSI of the live score distribution. The single most informative number here.

        `out_of_range` is reported alongside: a PSI computed against a reference the data
        does not overlap is a well-defined number that means nothing, and it looks exactly
        like severe drift.
        """
        values = pl.Series(scores)
        psi = psi_against_reference(values, self.reference["score"])
        out_of_range = outside_reference_range(values, self.reference["score"])
        return {
            "metric": "score",
            "psi": round(psi, 4),
            "out_of_range": round(out_of_range, 4),
            "band": _band(psi, out_of_range),
        }

    def feature_drift(self, df: pl.DataFrame) -> pl.DataFrame:
        """PSI per numeric feature, worst first, so a shift can be localised not just detected.

        Features absent from the batch are reported rather than skipped: a column that stops
        arriving is a drift event in its own right, and the more dangerous kind, because the
        model scores it as null without complaint.
        """
        rows = []
        for feature, profile in self.reference.get("features", {}).items():
            if feature not in df.columns:
                rows.append({"feature": feature, "psi": None, "band": "column missing"})
                continue
            psi = psi_against_reference(df[feature], profile)
            out_of_range = outside_reference_range(df[feature], profile)
            rows.append(
                {
                    "feature": feature,
                    "psi": round(psi, 4),
                    "out_of_range": round(out_of_range, 4),
                    "band": _band(psi, out_of_range),
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "feature": pl.Utf8,
                "psi": pl.Float64,
                "out_of_range": pl.Float64,
                "band": pl.Utf8,
            },
        ).sort("psi", descending=True, nulls_last=False)

    def report(self, df: pl.DataFrame, scores) -> dict:
        """Everything a monitoring job needs from one batch, in one call."""
        features = self.feature_drift(df)
        flagged = features.filter(pl.col("band") != "stable")
        return {
            "score": self.score_drift(scores),
            "n_features_checked": features.height,
            "n_features_flagged": flagged.height,
            "flagged": flagged.to_dicts(),
        }