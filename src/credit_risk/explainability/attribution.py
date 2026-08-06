"""Do both models rely on drivers a credit officer would recognise and defend?

After everything this project found about endogeneity, this is a substantive check rather
than a formality. Two questions, asked separately because the two models fail differently:

- The GBM is a black box. SHAP gives per-prediction attributions; aggregating them gives a
  global ranking, and correlating each feature's value against its own SHAP value gives the
  DIRECTION the model learned. A driver pointing the wrong way (higher income raising
  predicted risk) is a red flag no accuracy metric would surface.
- The scorecard is transparent but its coefficients are on WOE units, which nobody outside
  this repo reads. `scorecard_points` converts them into the points table that IS the
  scorecard deliverable: what each answer on the application form is worth.

Run SHAP on the CHAMPION (application-only). On the full-pool model it would simply report
that sub_grade dominates, which is already known and explains nothing.
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

_SAMPLE_SIZE = 20_000


def compute_shap_values(
    model, frame: pd.DataFrame, sample_size: int = _SAMPLE_SIZE, seed: int = 42
):
    """TreeExplainer SHAP values on a random subsample, returned with the rows used.

    Subsampled because exact tree SHAP is O(rows); 20k is ample for a global ranking and the
    ranking is stable well below that. Values are in log-odds space, so they add to the
    model's raw margin rather than to a probability.
    """
    import shap

    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    values = shap.TreeExplainer(model).shap_values(frame)
    if isinstance(values, list):  # some versions return one array per class
        values = values[1]
    return np.asarray(values), frame


def shap_global_importance(values: np.ndarray, frame: pd.DataFrame) -> pl.DataFrame:
    """Mean |SHAP| per feature, plus its share of total attribution."""
    mean_abs = np.abs(values).mean(axis=0)
    return (
        pl.DataFrame({"feature": list(frame.columns), "mean_abs_shap": mean_abs})
        .with_columns((pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum()).alias("share"))
        .sort("mean_abs_shap", descending=True)
    )


def shap_direction_report(values: np.ndarray, frame: pd.DataFrame) -> pl.DataFrame:
    """Sign of the relationship each feature has with its own SHAP contribution.

    Spearman between the feature's value and its SHAP value. Positive means higher values
    push predicted risk UP. Categorical columns are skipped: their codes have no order, so
    a correlation against them is meaningless rather than merely weak.
    """
    rows = []
    for i, name in enumerate(frame.columns):
        column = frame[name]
        if not pd.api.types.is_numeric_dtype(column):
            rows.append({"feature": name, "direction": None, "raises_risk": None})
            continue
        mask = column.notna().to_numpy()
        if mask.sum() < 100 or column[mask].nunique() < 2:
            rows.append({"feature": name, "direction": None, "raises_risk": None})
            continue
        rho = float(spearmanr(column[mask], values[mask, i]).statistic)
        rows.append({"feature": name, "direction": rho, "raises_risk": bool(rho > 0)})
    return pl.DataFrame(
        rows, schema={"feature": pl.Utf8, "direction": pl.Float64, "raises_risk": pl.Boolean}
    )


def rank_agreement(left: pl.DataFrame, right: pl.DataFrame, key: str = "feature") -> dict:
    """Spearman agreement between two feature rankings, over the features they share.

    Comparing SHAP importance against IV answers a specific question: IV is univariate, SHAP
    is not. Low agreement means the GBM is earning its keep through interactions a
    single-variable screen cannot see - which is the expected explanation for it beating the
    linear scorecard by more once sub_grade is removed.
    """
    shared = left.select(key).join(right.select(key), on=key, how="inner")[key].to_list()
    if len(shared) < 3:
        return {"n_shared": len(shared), "spearman": None}

    def rank(df: pl.DataFrame) -> dict:
        return {f: i for i, f in enumerate(df[key].to_list()) if f in shared}

    a, b = rank(left), rank(right)
    order = sorted(shared)
    return {
        "n_shared": len(shared),
        "spearman": float(spearmanr([a[f] for f in order], [b[f] for f in order]).statistic),
    }


def scorecard_points(
    encoder,
    model,
    features: list[str],
    pdo: int = 20,
    base_score: int = 600,
    base_odds: float = 50.0,
) -> pl.DataFrame:
    """Points awarded by every bin of every feature - the scorecard as a lookup table.

    Derivation. The logistic model gives logit(P(bad)) = b0 + sum(b_i * WOE_i), so
    ln(odds_good) = -b0 - sum(b_i * WOE_i), and with Score = offset + factor * ln(odds_good):

        factor = pdo / ln(2)
        offset = base_score - factor * ln(base_odds)
        points(feature i, bin j) = -factor * b_i * WOE_ij + (offset - factor * b0) / n_features

    The base term is spread evenly across features so the rows sum to the total score. Since
    correctly signed coefficients are negative and higher WOE means safer, -factor * b_i is
    positive: safer bins earn more points, which is the direction a reviewer expects.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    intercept = float(model.intercept_[0])
    base_per_feature = (offset - factor * intercept) / len(features)
    coefficients = dict(zip(features, model.coef_[0], strict=True))

    rows = []
    for feature in features:
        table = encoder.bin_tables_[feature]
        for record in table.to_dicts():
            rows.append(
                {
                    "feature": feature,
                    "bin": record["_bin"],
                    "n": record["n"],
                    "bad_rate": record["bad_rate"],
                    "woe": record["woe"],
                    "coefficient": coefficients[feature],
                    "points": -factor * coefficients[feature] * record["woe"] + base_per_feature,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("points").round(1))


def points_range(points_table: pl.DataFrame) -> pl.DataFrame:
    """Spread of points each feature can swing, worst bin to best - its practical weight.

    A feature with a high coefficient but bins that barely differ moves nobody's score. This
    is the ranking to show a credit committee, not the coefficient list.
    """
    return (
        points_table.group_by("feature")
        .agg(
            pl.col("points").min().alias("min_points"),
            pl.col("points").max().alias("max_points"),
            pl.len().alias("n_bins"),
        )
        .with_columns((pl.col("max_points") - pl.col("min_points")).alias("swing"))
        .sort("swing", descending=True)
    )
