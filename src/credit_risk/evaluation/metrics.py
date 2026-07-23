"""Core discrimination metrics for PD models. Extended with calibration/stability in Phase 5."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true, y_pred_proba) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_pred_proba))


def gini(y_true, y_pred_proba) -> float:
    """Gini coefficient = 2*AUC - 1, the standard credit-scoring discrimination metric."""
    return 2 * auc(y_true, y_pred_proba) - 1


def ks_statistic(y_true, y_pred_proba) -> float:
    """Kolmogorov-Smirnov statistic - max separation between cumulative good/bad rates."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(np.abs(tpr - fpr)))


def discrimination_report(y_true, y_pred_proba) -> dict:
    """AUC/Gini/KS in one call - the standard trio reported for every model version."""
    return {
        "auc": auc(y_true, y_pred_proba),
        "gini": gini(y_true, y_pred_proba),
        "ks": ks_statistic(y_true, y_pred_proba),
    }