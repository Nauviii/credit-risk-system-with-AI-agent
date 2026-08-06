"""Turning a calibrated PD into a decision: cutoffs, approval curves and expected loss.

A PD model that stops at AUC has not answered any business question. What a credit
committee asks is: if we decline everyone below score X, what share of applicants do we
turn away, what bad rate does the surviving book run at, and what does it cost us. That
is the approval-rate versus bad-rate trade-off, and it is the artefact this module
produces.

Three project-specific corrections are built in rather than assumed away:

1. The model predicts a 24-MONTH PD. Expected loss over the life of a loan needs the
   lifetime PD, so `horizon_coverage` measures - on fully matured vintages - what share of
   a term's eventual defaults actually land inside the window, and `to_lifetime_pd` divides
   by it. The correction is larger for 60-month loans than 36-month ones, so applying a
   single factor to the whole book understates the long-term exposure.

2. LGD is estimated from the data, not assumed. `empirical_lgd` reads it off charged-off
   loans. This uses post-origination columns, which is legitimate for estimating a
   portfolio parameter from history and is NOT legitimate as model input - they stay in
   schema.LEAKAGE_COLUMNS.

3. Bad rate is reported on the APPROVED population at each cutoff, not on everyone. The
   two diverge quickly and only the first is the rate the book will actually run at.
"""

import numpy as np
import polars as pl

_TARGET = "default_flag"


def approval_curve(
    score: np.ndarray, y: np.ndarray, exposure: np.ndarray | None = None, n_points: int = 20
) -> pl.DataFrame:
    """Approval rate and approved-book bad rate as the cutoff sweeps down the score range.

    Cutoffs are score quantiles, so each step moves a similar number of applicants. Reading
    it: `approval_rate` is the share accepted at that cutoff; `bad_rate` is the default rate
    among those accepted; `bad_rate_declined` is what is being turned away, which is what
    justifies the cutoff to anyone arguing it is too tight.
    """
    exposure = np.ones_like(score, dtype=float) if exposure is None else exposure
    cutoffs = np.quantile(score, np.linspace(0.0, 0.95, n_points))
    rows = []
    for cutoff in np.unique(cutoffs):
        approved = score >= cutoff
        if approved.sum() == 0:
            continue
        declined = ~approved
        rows.append({
            "cutoff": float(cutoff),
            "n_approved": int(approved.sum()),
            "approval_rate": float(approved.mean()),
            "bad_rate": float(y[approved].mean()),
            "bad_rate_declined": float(y[declined].mean()) if declined.any() else None,
            "exposure_approved": float(exposure[approved].sum()),
        })
    return pl.DataFrame(rows).sort("cutoff", descending=True)


def horizon_coverage(
    df: pl.DataFrame, horizon: int, vintages: list[int], term_column: str = "term_months"
) -> pl.DataFrame:
    """Share of a fully matured vintage's defaults that occur within `horizon` months, per term.

    This is the factor the 24-month PD must be divided by to become a lifetime PD. Run it
    only on vintages old enough to have resolved completely, or the answer is circular.
    """
    return (
        df.filter(
            pl.col("issue_d").cast(pl.Utf8).str.strptime(pl.Date, "%b-%Y", strict=False)
            .dt.year().is_in(vintages)
            & (pl.col(_TARGET).is_not_null())
            & pl.col("mob_event").is_not_null()
        )
        .group_by(term_column)
        .agg(
            pl.len().alias("n"),
            (pl.col(_TARGET) == 1).sum().alias("n_default_in_horizon"),
        )
        .sort(term_column)
    )


def to_lifetime_pd(pd_horizon: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Scale a fixed-horizon PD up to lifetime by dividing by that segment's coverage.

    Crude by design: it assumes the ratio of in-window to eventual defaults is the same for
    every borrower within a term, which understates it for high-risk borrowers who default
    earlier. Good enough to stop expected loss being wrong by a factor, not good enough to
    price with - a discrete-time hazard model is the proper answer.
    """
    return np.clip(pd_horizon / np.clip(coverage, 1e-6, 1.0), 0.0, 1.0)


def empirical_lgd(df: pl.DataFrame) -> dict:
    """LGD from charged-off loans: unrecovered share of the principal still outstanding.

    Uses post-origination columns to estimate a portfolio parameter from history. They are
    banned as model features and stay banned; this is a different use.
    """
    losses = (
        df.filter(pl.col("loan_status") == "Charged Off")
        .with_columns(
            (pl.col("loan_amnt") - pl.col("total_rec_prncp")).alias("outstanding")
        )
        .filter(pl.col("outstanding") > 0)
        .with_columns(
            (1 - pl.col("recoveries").fill_null(0.0) / pl.col("outstanding")).clip(0.0, 1.0).alias("lgd")
        )
    )
    return {
        "n": losses.height,
        "lgd_mean": float(losses["lgd"].mean()),
        "lgd_median": float(losses["lgd"].median()),
        "exposure_weighted_lgd": float(
            (losses["lgd"] * losses["outstanding"]).sum() / losses["outstanding"].sum()
        ),
    }


def scheduled_gross_yield(df: pl.DataFrame) -> np.ndarray:
    """Contractual interest over the life of each loan, as a share of principal.

    Exact from the loan terms - (installment * term - principal) / principal - no rate
    formula needed. It is the CEILING on revenue, not the expectation: it assumes every
    payment is made on schedule. Two things pull realised yield below it, and only the
    first is modelled downstream. Defaults stop payments partway, which `cutoff_table`
    approximates with a (1 - PD) survival factor. Prepayment truncates interest on the
    good loans, and roughly half of LendingClub borrowers prepay - so a margin computed
    from this number is optimistic even after the default adjustment.
    """
    installment = df["installment"].to_numpy().astype(float)
    term = df["term_months"].to_numpy().astype(float)
    principal = df["loan_amnt"].to_numpy().astype(float)
    return (installment * term - principal) / np.clip(principal, 1e-9, None)


def cutoff_table(
    score: np.ndarray,
    y: np.ndarray,
    pd_lifetime: np.ndarray,
    ead: np.ndarray,
    lgd: float,
    n_points: int = 20,
    gross_yield: np.ndarray | None = None,
) -> pl.DataFrame:
    """Approval curve extended with expected loss - the table a cutoff is chosen from.

    expected_loss = sum(PD * LGD * EAD) over the approved book; `expected_loss_rate` divides
    it by approved exposure so cutoffs are comparable. `realised_loss_rate` uses the observed
    outcome instead of the predicted PD, so the two columns together show whether the model's
    loss forecast would have held up.

    Passing `gross_yield` adds the revenue side, which is what makes a break-even cutoff
    computable at all: `net_margin_rate` = yield * (1 - PD) - expected loss rate. Read it as
    an upper bound - prepayment is not modelled, and a cutoff chosen where this crosses zero
    is therefore too loose, not too tight.

    Three margin columns, and picking the wrong one picks the wrong cutoff:
      net_margin_rate     - margin per unit of approved exposure. Maximising it optimises
                            efficiency and ignores volume, so it always lands too tight.
      net_margin_total    - margin in currency. This is what a lender with capital to deploy
                            maximises, and it peaks far looser than the rate does.
      marginal_margin_rate- margin earned on the TRANCHE added by loosening from the previous
                            cutoff. Where it crosses zero is the economically correct stopping
                            point: past it, each extra approval destroys value even while the
                            average rate is still positive.
    """
    curve = approval_curve(score, y, exposure=ead, n_points=n_points)
    rows = []
    for row in curve.to_dicts():
        approved = score >= row["cutoff"]
        exposure = ead[approved].sum()
        el = float((pd_lifetime[approved] * lgd * ead[approved]).sum())
        realised = float((y[approved] * lgd * ead[approved]).sum())
        entry = {
            **row,
            "expected_loss": el,
            "expected_loss_rate": el / exposure if exposure else None,
            "realised_loss_rate": realised / exposure if exposure else None,
        }
        if gross_yield is not None and exposure:
            income = float(
                (gross_yield[approved] * (1 - pd_lifetime[approved]) * ead[approved]).sum()
            )
            entry["gross_yield_rate"] = float((gross_yield[approved] * ead[approved]).sum() / exposure)
            entry["net_margin_rate"] = (income - el) / exposure
        rows.append(entry)

    table = pl.DataFrame(rows).sort("approval_rate")
    if gross_yield is None:
        return table
    return table.with_columns(
        (pl.col("net_margin_rate") * pl.col("exposure_approved")).alias("net_margin_total")
    ).with_columns(
        (
            pl.col("net_margin_total").diff()
            / pl.col("exposure_approved").diff()
        ).alias("marginal_margin_rate")
    )