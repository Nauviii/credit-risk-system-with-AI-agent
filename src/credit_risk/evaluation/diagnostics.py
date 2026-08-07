"""Scorecard diagnostics - coefficient sign sanity check and multicollinearity report."""

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from credit_risk.features.woe import DEFAULT_INITIAL_BINS, WOEEncoder


def coefficient_sign_report(model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    """Report each feature's coefficient sign against the expected direction.

    WOE = ln(dist_good/dist_bad), higher = safer. A model predicting P(default=1)
    should show NEGATIVE coefficients (higher WOE -> lower predicted default risk).
    A POSITIVE coefficient is the actual anomaly worth investigating, not the reverse.
    """
    coefs = dict(zip(features, model.coef_[0], strict=True))
    rows = [
        {
            "feature": f,
            "coefficient": round(c, 4),
            "flag": "UNEXPECTED POSITIVE - investigate" if c > 0 else "",
        }
        for f, c in coefs.items()
    ]
    return pd.DataFrame(rows).sort_values("coefficient").reset_index(drop=True)


def multicollinearity_report(
    train_woe: pl.DataFrame, features: list[str], threshold: float = 0.6
) -> pd.DataFrame:
    """Report WOE feature pairs correlated above threshold - candidates for pruning."""
    woe_cols = [f"{f}_woe" for f in features]
    corr = train_woe.select(woe_cols).to_pandas().corr()
    rows = []
    for i, a in enumerate(woe_cols):
        for b in woe_cols[i + 1 :]:
            c = corr.loc[a, b]
            if abs(c) > threshold:
                rows.append(
                    {
                        "feature_a": a.replace("_woe", ""),
                        "feature_b": b.replace("_woe", ""),
                        "correlation": round(c, 3),
                    }
                )
    result = pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation"])
    return result.reindex(
        result["correlation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def drop_until_signs_are_clean(
    features: list[str],
    train_df: pl.DataFrame,
    target: str = "default_flag",
    n_bins: int = DEFAULT_INITIAL_BINS,
    max_iterations: int = 10,
) -> tuple[list[str], "WOEEncoder", LogisticRegression]:
    """Iteratively refit and drop the worst positive-coefficient feature until none remain.

    Pairwise correlation pruning (woe.prune_correlated_features) only catches
    two-feature collinearity; three-or-more-feature collinearity can still leave
    a positive (wrong-sign) coefficient behind. This closes that gap by actually
    refitting after each drop, rather than assuming one pruning pass is enough.
    """

    current = list(features)
    for _ in range(max_iterations):
        encoder = WOEEncoder(features=current, target=target, n_bins=n_bins).fit(train_df)
        train_woe = encoder.transform(train_df)
        woe_cols = [f"{f}_woe" for f in current]
        model = LogisticRegression(max_iter=1000).fit(
            train_woe.select(woe_cols).to_pandas(), train_woe[target].to_pandas()
        )
        coefs = dict(zip(current, model.coef_[0], strict=True))
        positive = {f: c for f, c in coefs.items() if c > 0}
        if not positive:
            return current, encoder, model
        worst = max(positive, key=positive.get)
        print(f"dropping '{worst}' (coefficient {positive[worst]:.4f}, still positive)")
        current.remove(worst)
    raise RuntimeError(
        f"did not converge to all-negative coefficients within {max_iterations} iterations"
    )
