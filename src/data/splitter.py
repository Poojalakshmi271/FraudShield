"""
FraudShield – Stratified Data Splitter
=======================================
Splits the raw Credit Card Fraud dataframe into train / validation / test
sets using stratified sampling to preserve class proportions.

No preprocessing is performed here – this module is a pure split utility.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config.logging_config import setup_logging
    logger = setup_logging(logger_name="fraudshield.splitter")
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraudshield.splitter")

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_ArrayLike = Union[pd.DataFrame, np.ndarray]
SplitResult = Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # X_train, X_val, X_test
    pd.Series, pd.Series, pd.Series,            # y_train, y_val, y_test
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TRAIN_RATIO = 0.70
_DEFAULT_VAL_RATIO   = 0.15
_DEFAULT_TEST_RATIO  = 0.15
_DEFAULT_SEED        = 42
_PROPORTION_TOLERANCE = 0.01   # 1 % tolerance for class proportion drift


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> SplitResult:
    """
    Split a dataframe into train, validation and test sets using stratified
    sampling.

    The split honours the class proportions present in ``df`` so that the
    minority (fraud) class is represented equally in every partition.

    Two-stage approach
    ------------------
    1. ``df`` → train  (70 %)  +  temp  (30 %)
    2. ``temp`` → val  (50 % of temp = 15 % of full)  +  test  (50 % of temp = 15 % of full)

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataframe.  Must contain a 'Class' column (or the column
        named in ``config['data']['target_column']``).
    config : dict, optional
        Parsed YAML config dict.  When omitted, default ratios and seeds are
        used.  Relevant keys:
        ``splitting.train_ratio``, ``splitting.val_ratio``,
        ``splitting.test_ratio``, ``splitting.stratify``,
        ``splitting.random_seed``, ``data.target_column``.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test)`` where X* are
        DataFrames (feature columns only) and y* are Series (target column).

    Raises
    ------
    KeyError
        If the target column is not found in ``df``.
    ValueError
        If the split ratios do not sum to 1 (within floating-point tolerance).
    """
    cfg = config or {}
    split_cfg = cfg.get("splitting", {})
    data_cfg  = cfg.get("data", {})

    # --- Resolve parameters ---
    train_ratio:  float = float(split_cfg.get("train_ratio",  _DEFAULT_TRAIN_RATIO))
    val_ratio:    float = float(split_cfg.get("val_ratio",    _DEFAULT_VAL_RATIO))
    test_ratio:   float = float(split_cfg.get("test_ratio",   _DEFAULT_TEST_RATIO))
    stratify_flag: bool = bool(split_cfg.get("stratify",      True))
    random_seed:   int  = int(split_cfg.get("random_seed",    _DEFAULT_SEED))
    target_col:    str  = data_cfg.get("target_column", "Class")

    # --- Validate ratios ---
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    # --- Validate target column ---
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )

    # --- Separate features and target ---
    y: pd.Series = df[target_col].copy()
    X: pd.DataFrame = df.drop(columns=[target_col])

    logger.info(
        "Splitting %d rows (train=%.0f%%, val=%.0f%%, test=%.0f%%), stratify=%s, seed=%d",
        len(df), train_ratio * 100, val_ratio * 100, test_ratio * 100,
        stratify_flag, random_seed,
    )

    stratify_arr = y if stratify_flag else None

    # Stage 1: train vs (val + test)
    temp_ratio = val_ratio + test_ratio          # fraction going to temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        stratify=stratify_arr,
        random_state=random_seed,
    )

    # Stage 2: val vs test  (each half of temp)
    val_of_temp = val_ratio / temp_ratio         # e.g. 0.15 / 0.30 = 0.50
    stratify_temp = y_temp if stratify_flag else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_of_temp),
        stratify=stratify_temp,
        random_state=random_seed,
    )

    # --- Log split sizes ---
    logger.info(
        "Split complete → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # --- Verify class proportions ---
    _verify_class_proportions(y, y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# verify_no_overlap
# ---------------------------------------------------------------------------
def verify_no_overlap(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> bool:
    """
    Verify that no row index appears in more than one split.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature splits as returned by :func:`split_data`.

    Returns
    -------
    bool
        ``True`` if the splits are mutually exclusive (no overlap), ``False``
        if any index collision is detected.

    Notes
    -----
    Uses pandas Index intersection rather than set operations so that the
    function works efficiently on both integer-indexed and custom-indexed
    DataFrames.
    """
    train_idx = X_train.index
    val_idx   = X_val.index
    test_idx  = X_test.index

    train_val_overlap  = train_idx.intersection(val_idx)
    train_test_overlap = train_idx.intersection(test_idx)
    val_test_overlap   = val_idx.intersection(test_idx)

    overlaps = {
        "train∩val":  len(train_val_overlap),
        "train∩test": len(train_test_overlap),
        "val∩test":   len(val_test_overlap),
    }

    has_overlap = any(v > 0 for v in overlaps.values())

    if has_overlap:
        logger.error(
            "Index overlap detected between splits: %s",
            {k: v for k, v in overlaps.items() if v > 0},
        )
    else:
        logger.info(
            "No index overlap detected – splits are mutually exclusive "
            "(train=%d, val=%d, test=%d).",
            len(train_idx), len(val_idx), len(test_idx),
        )

    return not has_overlap


# ---------------------------------------------------------------------------
# get_split_stats
# ---------------------------------------------------------------------------
def get_split_stats(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Dict[str, dict]:
    """
    Return a summary dictionary with class counts and fraud rates per split.

    Parameters
    ----------
    y_train, y_val, y_test : pd.Series
        Target arrays for each split.

    Returns
    -------
    dict
        Structure::

            {
                "train": {
                    "total": int,
                    "legitimate": int,
                    "fraud": int,
                    "fraud_rate": float,
                },
                "val":  { … },
                "test": { … },
            }
    """
    def _stats(y: pd.Series) -> dict:
        total      = len(y)
        fraud      = int((y == 1).sum())
        legitimate = int((y == 0).sum())
        fraud_rate = fraud / total if total > 0 else 0.0
        return {
            "total":      total,
            "legitimate": legitimate,
            "fraud":      fraud,
            "fraud_rate": fraud_rate,
        }

    stats = {
        "train": _stats(y_train),
        "val":   _stats(y_val),
        "test":  _stats(y_test),
    }

    for split_name, s in stats.items():
        logger.info(
            "Split %-5s | total=%6d | legitimate=%6d | fraud=%4d | fraud_rate=%.4f%%",
            split_name, s["total"], s["legitimate"], s["fraud"], s["fraud_rate"] * 100,
        )

    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _verify_class_proportions(
    y_full:  pd.Series,
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
) -> None:
    """
    Log a warning if the fraud rate in any split deviates by more than
    ``_PROPORTION_TOLERANCE`` (1 %) from the full-dataset fraud rate.
    """
    overall_fraud_rate = (y_full == 1).mean()

    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        split_fraud_rate = (y_split == 1).mean()
        drift = abs(split_fraud_rate - overall_fraud_rate)
        if drift > _PROPORTION_TOLERANCE:
            logger.warning(
                "Class proportion drift in '%s' split: overall=%.4f%%, "
                "split=%.4f%%, drift=%.4f%% (threshold=%.4f%%)",
                name,
                overall_fraud_rate * 100,
                split_fraud_rate * 100,
                drift * 100,
                _PROPORTION_TOLERANCE * 100,
            )
        else:
            logger.debug(
                "Class proportions maintained in '%s' split "
                "(overall=%.4f%%, split=%.4f%%, drift=%.4f%%).",
                name,
                overall_fraud_rate * 100,
                split_fraud_rate * 100,
                drift * 100,
            )
