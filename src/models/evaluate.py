"""
FraudShield Model Evaluation Module
=====================================
Comprehensive evaluation utilities for binary fraud-detection classifiers.
Produces metrics, curves, and formatted text reports.
"""

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a binary classifier and return a comprehensive metrics dict.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier with a ``predict_proba`` method.
    X : array-like, shape (n_samples, n_features)
        Feature matrix to evaluate on.
    y : array-like, shape (n_samples,)
        True binary labels.
    threshold : float, default 0.5
        Decision threshold applied to predicted probabilities.

    Returns
    -------
    dict with keys:
        precision, recall, f1, roc_auc, pr_auc,
        confusion_matrix, classification_report,
        fpr_array, tpr_array, precision_array, recall_array,
        thresholds_pr, thresholds_roc
    """
    y = np.asarray(y)
    probas = model.predict_proba(X)[:, 1]
    y_pred = (probas >= threshold).astype(int)

    # Scalar metrics
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, probas)
    pr_auc = average_precision_score(y, probas)

    cm = confusion_matrix(y, y_pred)
    cls_report = classification_report(y, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0)

    # Curves
    fpr_array, tpr_array, thresholds_roc = roc_curve(y, probas)
    precision_array, recall_array, thresholds_pr = precision_recall_curve(y, probas)

    logger.info(
        "Evaluation | threshold=%.3f | precision=%.4f | recall=%.4f | "
        "f1=%.4f | roc_auc=%.4f | pr_auc=%.4f",
        threshold,
        prec,
        rec,
        f1,
        roc_auc,
        pr_auc,
    )

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "fpr_array": fpr_array,
        "tpr_array": tpr_array,
        "precision_array": precision_array,
        "recall_array": recall_array,
        "thresholds_pr": thresholds_pr,
        "thresholds_roc": thresholds_roc,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a confusion matrix as a colour-annotated grid.

    Parameters
    ----------
    cm : ndarray, shape (2, 2)
        Confusion matrix from :func:`evaluate_model`.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Legitimate", "Fraud"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.figure.tight_layout()
    return ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot an ROC curve.

    Parameters
    ----------
    fpr : ndarray
        False positive rates from :func:`evaluate_model`.
    tpr : ndarray
        True positive rates from :func:`evaluate_model`.
    roc_auc : float
        Area under the ROC curve.
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, lw=2, color="#E74C3C", label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--", color="#95A5A6", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E74C3C")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a Precision-Recall curve.

    Parameters
    ----------
    precision : ndarray
        Precision values from :func:`evaluate_model`.
    recall : ndarray
        Recall values from :func:`evaluate_model`.
    pr_auc : float
        Area under the PR curve (average precision).
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, lw=2, color="#2ECC71", label=f"PR AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#2ECC71")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_evaluation_report(results_dict: Dict[str, Any]) -> str:
    """Generate a formatted plain-text evaluation report.

    Parameters
    ----------
    results_dict : dict
        Output of :func:`evaluate_model`.

    Returns
    -------
    str
        Multi-line formatted report suitable for logging or saving to disk.
    """
    sep = "=" * 65
    thin = "-" * 65

    cm = results_dict["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    lines = [
        sep,
        "  FRAUDSHIELD MODEL EVALUATION REPORT",
        sep,
        "",
        "  SCALAR METRICS",
        thin,
        f"  Precision        : {results_dict['precision']:.4f}",
        f"  Recall           : {results_dict['recall']:.4f}",
        f"  F1 Score         : {results_dict['f1']:.4f}",
        f"  ROC AUC          : {results_dict['roc_auc']:.4f}",
        f"  PR AUC           : {results_dict['pr_auc']:.4f}",
        "",
        "  CONFUSION MATRIX",
        thin,
        f"  True Negatives   : {tn:>10,}",
        f"  False Positives  : {fp:>10,}",
        f"  False Negatives  : {fn:>10,}",
        f"  True Positives   : {tp:>10,}",
        "",
        "  CLASSIFICATION REPORT",
        thin,
        results_dict["classification_report"],
        sep,
    ]

    report = "\n".join(lines)
    logger.debug("Generated evaluation report.")
    return report
