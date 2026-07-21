"""Schema contract and business constants for Lending Club accepted loans.

Column semantics are what make this dataset auditable (vs anonymized data):
every leakage/indeterminate decision below is documented for reviewers.
"""

import polars as pl
import pandera.polars as pa
from pandera.polars import Column, Check

# Columns only populated AFTER loan performance is observed.
# Must never be used as model features (would leak the label at scoring time).
LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d", "next_pymnt_d",
    "out_prncp", "out_prncp_inv", "hardship_flag", "debt_settlement_flag",
    "settlement_status", "settlement_date", "settlement_amount",
    # added after EDA (phase 3): all populated only during/after loan servicing
    "settlement_percentage", "settlement_term", "debt_settlement_flag_date",
    "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
    "hardship_amount", "hardship_start_date", "hardship_end_date",
    "hardship_length", "hardship_dpd", "hardship_loan_status",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "payment_plan_start_date", "orig_projected_additional_accrued_interest",
    "last_fico_range_high", "last_fico_range_low",  # FICO refreshed during loan life, not at origination
]

# Present in every raw row but carries zero information (verified during EDA phase 3).
ALWAYS_MISSING_COLUMNS = ["member_id"]

# "Missing" here means the event never happened (e.g. never delinquent), not unknown.
# Needs a has_history flag + sentinel imputation in features/, not mean/median fill.
STRUCTURALLY_MISSING_COLUMNS = [
    "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq", "mths_since_recent_inq",
]

# Populated only when application_type == "Joint App" (~5% of loans, verified in EDA).
# Missing = not applicable, not unknown - needs conditional handling, not naive imputation.
JOINT_APPLICATION_COLUMNS = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
]

# Free text or near-unique identifiers - drop, or need dedicated encoding (not one-hot).
HIGH_CARDINALITY_COLUMNS = ["emp_title", "desc", "title", "url", "zip_code"]

# LendingClub only started collecting these bureau trade-line fields around 2015-2016.
# Verified during EDA: 95-100% missing for issue_year <= 2015 (our train window),
# 0% missing from 2016 onward. Training would never see real values - excluded entirely
# for V1, not just deferred, since the train/serve feature availability would mismatch.
EXCLUDED_VINTAGE_COLUMNS = [
    "il_util", "open_acc_6m", "all_util", "open_act_il", "open_il_12m", "open_il_24m",
    "total_bal_il", "open_rv_12m", "open_rv_24m", "max_bal_bc", "inq_fi", "total_cu_tl",
    "inq_last_12m", "mths_since_rcnt_il",
]

# Known placeholder/sentinel values found during EDA phase 3 that must be handled
# explicitly before binning or imputation - they are not genuine extreme observations.
# dti max observed = 999.0, a clear placeholder against a realistic 0-40 range.
SENTINEL_VALUES = {"dti": [999.0]}

# p99.5 chosen from the observed distribution (p99.9=600k, p100=61M - an isolated,
# almost certainly erroneous outlier); winsorize rather than drop to keep the rows.
WINSORIZE_CAPS = {"annual_inc": 350_000.0}

# Near-duplicate of another kept feature (verified via IV + value comparison) or
# constant/near-constant with no modeling value. Kept in the raw file, excluded here.
REDUNDANT_OR_CONSTANT_COLUMNS = [
    "funded_amnt", "funded_amnt_inv",  # differ from loan_amnt in <0.1% of rows
    "fico_range_high",  # IV 0.124, near-identical to fico_range_low (IV 0.124) - same score, one kept
    "policy_code",  # constant (single value) across the entire sample
    "pymnt_plan",  # 99.97% one value; also ambiguous whether "y" can be set post-origination
]

# loan_status values with no final outcome yet (right-censored) -> excluded from training.
INDETERMINATE_STATUSES = [
    "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)",
]

RAW_ACCEPTED_SCHEMA = pa.DataFrameSchema(
    {
        "id": Column(pl.Utf8, nullable=True),
        "loan_amnt": Column(pl.Float64, Check.ge(0)),
        "term": Column(pl.Utf8),
        "int_rate": Column(pl.Float64, Check.ge(0)),
        "installment": Column(pl.Float64, Check.ge(0)),
        "grade": Column(pl.Utf8),
        "sub_grade": Column(pl.Utf8),
        "emp_length": Column(pl.Utf8, nullable=True),
        "home_ownership": Column(pl.Utf8),
        "annual_inc": Column(pl.Float64, Check.ge(0), nullable=True),
        "verification_status": Column(pl.Utf8),
        "issue_d": Column(pl.Utf8),
        "loan_status": Column(pl.Utf8),
        "purpose": Column(pl.Utf8),
        "dti": Column(pl.Float64, nullable=True),
        "delinq_2yrs": Column(pl.Float64, nullable=True),
        "earliest_cr_line": Column(pl.Utf8, nullable=True),
        "fico_range_low": Column(pl.Float64, nullable=True),
        "fico_range_high": Column(pl.Float64, nullable=True),
        "open_acc": Column(pl.Float64, nullable=True),
        "pub_rec": Column(pl.Float64, nullable=True),
        "revol_bal": Column(pl.Float64, nullable=True),
        "revol_util": Column(pl.Float64, nullable=True),
        "total_acc": Column(pl.Float64, nullable=True),
        "application_type": Column(pl.Utf8, nullable=True),
    },
    strict=False,  # raw file has ~150 cols; we only contract the ones we currently use
    coerce=True,
)