"""
FraudShield SMOTE Resampler
============================
Applies Synthetic Minority Over-sampling Technique (SMOTE) to the **training**
set only, correcting the extreme class imbalance present in credit-card fraud
data without touching validation or test splits.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    *,
    is_training_data: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE over-sampling to the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for the **training** split only.
    y_train : pd.Series
        Binary target labels for the training split (0 = genuine, 1 = fraud).
    config : dict
        Project configuration dict (from ``config/config.yaml``).  The
        ``resampling`` sub-key is used:

        .. code-block:: yaml

            resampling:
              smote_random_state: 42
              smote_k_neighbors: 5
              sampling_strategy: "auto"

    is_training_data : bool, optional
        Safety flag that **must** be ``True`` (the default) when calling with
        training data.  Passing ``False`` raises a ``ValueError`` immediately,
        preventing accidental leakage of SMOTE into validation / test splits.

    Returns
    -------
    X_resampled : pd.DataFrame
        Over-sampled feature matrix with the same column names as ``X_train``.
    y_resampled : pd.Series
        Corresponding over-sampled target labels (dtype preserved).

    Raises
    ------
    ValueError
        * If ``is_training_data=False`` – SMOTE must never be applied to
          validation or test data.
        * If ``X_train`` and ``y_train`` have mismatched lengths.
        * If ``y_train`` contains fewer than 2 distinct classes.
    """
    # --- Guard: refuse to run on non-training data ---
    if not is_training_data:
        raise ValueError(
            "apply_smote() must only be called with training data. "
            "Pass is_training_data=True when calling with the training split. "
            "SMOTE must NEVER be applied to validation or test data, as this "
            "would cause data leakage and invalidate evaluation metrics."
        )

    # --- Basic input validation ---
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} "
            "entries. They must have the same length."
        )

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(
            f"y_train contains only class(es) {unique_classes.tolist()}. "
            "SMOTE requires at least 2 classes to be present."
        )

    # --- Extract SMOTE hyper-parameters from config ---
    resamp_cfg: dict = config.get("resampling", {})
    random_state: int = int(resamp_cfg.get("smote_random_state", 42))
    k_neighbors: int = int(resamp_cfg.get("smote_k_neighbors", 5))
    sampling_strategy = resamp_cfg.get("sampling_strategy", "auto")

    # --- Log before-resampling class distribution ---
    before_counts: Dict[int, int] = _class_counts(y_train)
    logger.info(
        "Before SMOTE – class counts: %s  (fraud rate: %.4f%%)",
        before_counts,
        100.0 * before_counts.get(1, 0) / max(sum(before_counts.values()), 1),
    )

    # --- Fit-resample ---
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    col_names = list(X_train.columns)
    y_index_name = y_train.name

    X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)

    # --- Reconstruct DataFrames with original metadata ---
    X_resampled = pd.DataFrame(X_arr, columns=col_names)
    y_resampled = pd.Series(y_arr, name=y_index_name, dtype=y_train.dtype)

    # --- Log after-resampling class distribution ---
    after_counts: Dict[int, int] = _class_counts(y_resampled)
    logger.info(
        "After SMOTE  – class counts: %s  (fraud rate: %.4f%%)",
        after_counts,
        100.0 * after_counts.get(1, 0) / max(sum(after_counts.values()), 1),
    )
    logger.info(
        "SMOTE added %d synthetic minority samples (dataset grew from %d to %d rows).",
        len(X_resampled) - len(X_train),
        len(X_train),
        len(X_resampled),
    )

    return X_resampled, y_resampled


def compare_class_distributions(
    y_original: pd.Series,
    y_resampled: pd.Series,
) -> Dict[str, object]:
    """
    Compare class distributions before and after resampling.

    Parameters
    ----------
    y_original : pd.Series
        Target labels before SMOTE.
    y_resampled : pd.Series
        Target labels after SMOTE.

    Returns
    -------
    dict with keys:
        ``before_counts``, ``after_counts``, ``before_fraud_rate``,
        ``after_fraud_rate``, ``samples_added``, ``size_increase_pct``.
    """
    before_counts = _class_counts(y_original)
    after_counts = _class_counts(y_resampled)

    n_before = sum(before_counts.values())
    n_after = sum(after_counts.values())

    before_fraud_rate = before_counts.get(1, 0) / max(n_before, 1)
    after_fraud_rate = after_counts.get(1, 0) / max(n_after, 1)

    samples_added = n_after - n_before
    size_increase_pct = 100.0 * samples_added / max(n_before, 1)

    comparison = {
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_fraud_rate": before_fraud_rate,
        "after_fraud_rate": after_fraud_rate,
        "samples_added": samples_added,
        "size_increase_pct": round(size_increase_pct, 2),
    }

    logger.info(
        "Class distribution comparison:\n"
        "  Before: %s  (fraud rate %.4f%%)\n"
        "  After:  %s  (fraud rate %.4f%%)\n"
        "  Added %d synthetic samples (+%.2f%%)",
        before_counts,
        before_fraud_rate * 100,
        after_counts,
        after_fraud_rate * 100,
        samples_added,
        size_increase_pct,
    )

    return comparison


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _class_counts(y: pd.Series) -> Dict[int, int]:
    """Return {class_label: count} sorted by label."""
    counts = y.value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}
