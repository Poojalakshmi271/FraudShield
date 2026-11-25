"""
FraudShield Threshold Optimization Module
==========================================
Searches for the decision threshold that maximises a chosen metric (default
F1) on *validation* data only. Test data is never touched here.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Threshold search parameters
_THRESHOLD_MIN = 0.05
_THRESHOLD_MAX = 0.95
_N_STEPS = 100

SUPPORTED_METRICS = ("f1", "precision", "recall")


# ---------------------------------------------------------------------------
# Core optimisation
# ---------------------------------------------------------------------------

def optimize_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float, pd.DataFrame]:
    """Find the decision threshold that maximises *metric* on the validation set.

    Searches 100 evenly-spaced thresholds in [0.05, 0.95]. This function
    operates exclusively on validation data – **never** on test data.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier with a ``predict_proba`` method.
    X_val : array-like, shape (n_samples, n_features)
        Validation feature matrix.
    y_val : array-like, shape (n_samples,)
        True binary labels for the validation set.
    metric : {'f1', 'precision', 'recall'}, default 'f1'
        Metric to maximise when selecting the best threshold.

    Returns
    -------
    best_threshold : float
        Threshold value that maximises *metric*.
    best_score : float
        Metric value at *best_threshold*.
    scores_df : pd.DataFrame
        DataFrame with columns ['threshold', 'precision', 'recall', 'f1']
        for every candidate threshold.

    Raises
    ------
    ValueError
        If *metric* is not one of the supported options.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"metric must be one of {SUPPORTED_METRICS}, got '{metric}'."
        )

    y_val = np.asarray(y_val)
    probas = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(_THRESHOLD_MIN, _THRESHOLD_MAX, _N_STEPS)

    rows = []
    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1})

    scores_df = pd.DataFrame(rows)

    best_idx = scores_df[metric].idxmax()
    best_threshold = float(scores_df.loc[best_idx, "threshold"])
    best_score = float(scores_df.loc[best_idx, metric])

    logger.info(
        "Threshold optimisation | metric='%s' | best_threshold=%.4f | best_score=%.4f.",
        metric,
        best_threshold,
        best_score,
    )

    return best_threshold, best_score, scores_df


# ---------------------------------------------------------------------------
# Apply threshold
# ---------------------------------------------------------------------------

def apply_threshold(probas: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probability scores to binary predictions using a fixed threshold.

    Parameters
    ----------
    probas : array-like, shape (n_samples,)
        Predicted positive-class probabilities.
    threshold : float
        Decision boundary; samples with probability >= threshold are
        classified as positive (fraud).

    Returns
    -------
    y_pred : np.ndarray of int, shape (n_samples,)
        Binary predictions (0 or 1).
    """
    probas = np.asarray(probas, dtype=float)
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
    return (probas >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_analysis(
    scores_df: pd.DataFrame,
    best_threshold: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot precision, recall, and F1 as a function of the decision threshold.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of :func:`optimize_threshold` (must have columns
        'threshold', 'precision', 'recall', 'f1').
    best_threshold : float
        The optimal threshold to highlight with a vertical line.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    metric_colors = {
        "precision": "#3498DB",
        "recall": "#E67E22",
        "f1": "#9B59B6",
    }

    for col, color in metric_colors.items():
        ax.plot(
            scores_df["threshold"],
            scores_df[col],
            lw=2,
            color=color,
            label=col.capitalize(),
        )

    ax.axvline(
        x=best_threshold,
        color="#E74C3C",
        linestyle="--",
        lw=1.8,
        label=f"Best threshold = {best_threshold:.3f}",
    )

    ax.set_xlim([_THRESHOLD_MIN, _THRESHOLD_MAX])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Threshold Analysis – Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend(loc="center left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax
