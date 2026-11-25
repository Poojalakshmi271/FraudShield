"""
FraudShield Probability Calibration Module
============================================
Wraps any fitted sklearn classifier with CalibratedClassifierCV and provides
utilities to compare, plot, and select the best calibration method.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate a fitted classifier's probability outputs.

    Parameters
    ----------
    model : sklearn estimator
        Already-fitted base classifier.
    X_val : array-like, shape (n_samples, n_features)
        Validation features used for calibration fitting.
    y_val : array-like, shape (n_samples,)
        True binary labels for the validation set.
    method : {'sigmoid', 'isotonic'}, default 'sigmoid'
        Calibration method – ``'sigmoid'`` (Platt scaling) or
        ``'isotonic'`` (non-parametric).

    Returns
    -------
    calibrated : CalibratedClassifierCV
        Fitted calibrated classifier ready for ``predict_proba``.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"method must be 'sigmoid' or 'isotonic', got '{method}'.")

    logger.info("Calibrating model with method='%s'.", method)

    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
    except (ImportError, TypeError, ValueError):
        calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")

    calibrated.fit(X_val, y_val)

    brier = brier_score_loss(y_val, calibrated.predict_proba(X_val)[:, 1])
    logger.info("Calibration complete. Brier score (val): %.5f", brier)

    return calibrated


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_calibration(
    uncalibrated_model: Any,
    calibrated_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Compare calibration quality between an uncalibrated and calibrated model.

    Parameters
    ----------
    uncalibrated_model : sklearn estimator
        Fitted but *not* calibrated classifier.
    calibrated_model : sklearn estimator
        Fitted *calibrated* classifier (e.g., output of :func:`calibrate_model`).
    X_val : array-like
        Validation features.
    y_val : array-like
        True binary labels.

    Returns
    -------
    dict with keys:
        brier_uncalibrated, brier_calibrated,
        fraction_positives_uncal, mean_predicted_uncal,
        fraction_positives_cal, mean_predicted_cal
    """
    y_val = np.asarray(y_val)

    probas_uncal = uncalibrated_model.predict_proba(X_val)[:, 1]
    probas_cal = calibrated_model.predict_proba(X_val)[:, 1]

    brier_uncal = brier_score_loss(y_val, probas_uncal)
    brier_cal = brier_score_loss(y_val, probas_cal)

    n_bins = 10
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_val, probas_uncal, n_bins=n_bins, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_val, probas_cal, n_bins=n_bins, strategy="uniform")

    logger.info(
        "Brier: uncalibrated=%.5f | calibrated=%.5f (improvement=%.5f).",
        brier_uncal,
        brier_cal,
        brier_uncal - brier_cal,
    )

    return {
        "brier_uncalibrated": float(brier_uncal),
        "brier_calibrated": float(brier_cal),
        "fraction_positives_uncal": frac_pos_uncal,
        "mean_predicted_uncal": mean_pred_uncal,
        "fraction_positives_cal": frac_pos_cal,
        "mean_predicted_cal": mean_pred_cal,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    uncal_model: Any,
    cal_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot reliability diagrams for an uncalibrated and calibrated model.

    Parameters
    ----------
    uncal_model : sklearn estimator
        Fitted uncalibrated classifier.
    cal_model : sklearn estimator
        Fitted calibrated classifier.
    X_val : array-like
        Validation features.
    y_val : array-like
        True binary labels.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    y_val = np.asarray(y_val)
    n_bins = 10

    for label, mdl, color in [
        ("Uncalibrated", uncal_model, "#E74C3C"),
        ("Calibrated", cal_model, "#2ECC71"),
    ]:
        probas = mdl.predict_proba(X_val)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_val, probas, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, color=color, label=label)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--", color="#95A5A6", label="Perfect calibration")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Reliability Diagram (Calibration Curves)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Best calibration selector
# ---------------------------------------------------------------------------

def select_best_calibration(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[str, CalibratedClassifierCV]:
    """Try both calibration methods and return the one with the lower Brier score.

    Parameters
    ----------
    model : sklearn estimator
        Already-fitted base classifier.
    X_val : array-like
        Validation features.
    y_val : array-like
        True binary labels.

    Returns
    -------
    best_method : str
        Either ``'sigmoid'`` or ``'isotonic'``.
    calibrated_model : CalibratedClassifierCV
        The calibrated model using the best method.
    """
    y_val = np.asarray(y_val)

    results: Dict[str, Tuple[float, CalibratedClassifierCV]] = {}

    for method in ("sigmoid", "isotonic"):
        cal = calibrate_model(model, X_val, y_val, method=method)
        brier = brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1])
        results[method] = (brier, cal)
        logger.info("Method='%s' | Brier=%.5f", method, brier)

    best_method = min(results, key=lambda m: results[m][0])
    best_brier, best_model = results[best_method]

    logger.info(
        "Selected calibration method: '%s' (Brier=%.5f).",
        best_method,
        best_brier,
    )

    return best_method, best_model
