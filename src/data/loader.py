"""
FraudShield – Data Loader
=========================
Loads the Credit Card Fraud Detection dataset, validates it, and
exposes helper utilities for downstream pipeline stages.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
# Allow running from project root or from any sub-directory by inserting the
# project root on sys.path so that "config.logging_config" is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config.logging_config import setup_logging
    logger = setup_logging(logger_name="fraudshield.loader")
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraudshield.loader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_PCA_COLS: List[str] = [f"V{i}" for i in range(1, 29)]
_REQUIRED_COLS: List[str] = ["Time", "Amount", "Class"] + _PCA_COLS
_EXPECTED_CLASSES = {0, 1}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config(config_path: Optional[Path] = None) -> dict:
    """Load YAML configuration, returning an empty dict on failure."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        logger.debug("Config loaded from %s", path)
        return cfg or {}
    except FileNotFoundError:
        logger.warning("Config file not found at %s – using defaults.", path)
        return {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse config YAML: %s – using defaults.", exc)
        return {}


# ---------------------------------------------------------------------------
# DataValidationReport
# ---------------------------------------------------------------------------
@dataclass
class DataValidationReport:
    """Summary of dataset validation checks."""

    shape: tuple
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    duplicate_count: int
    class_distribution: Dict[int, int]
    validation_passed: bool
    messages: List[str] = field(default_factory=list)

    # Convenience properties
    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        return self.shape[1]

    @property
    def fraud_rate(self) -> float:
        total = sum(self.class_distribution.values())
        fraud = self.class_distribution.get(1, 0)
        return fraud / total if total > 0 else 0.0

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"DataValidationReport",
            f"  Shape            : {self.shape}",
            f"  Missing values   : {sum(self.missing_counts.values())}",
            f"  Duplicates       : {self.duplicate_count}",
            f"  Class dist       : {self.class_distribution}",
            f"  Fraud rate       : {self.fraud_rate:.4%}",
            f"  Validation passed: {self.validation_passed}",
        ]
        for msg in self.messages:
            lines.append(f"  [msg] {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KaggleHub download helper
# ---------------------------------------------------------------------------
def _try_kagglehub_download(dataset: str) -> Optional[Path]:
    """
    Attempt to download the dataset via kagglehub.

    Returns the downloaded directory Path on success, None otherwise.
    Callers should locate the actual CSV inside the returned directory.
    """
    try:
        import kagglehub  # type: ignore
        logger.info("Attempting KaggleHub download for dataset '%s' …", dataset)
        raw_path = kagglehub.dataset_download(dataset)
        downloaded = Path(raw_path)
        logger.info("KaggleHub download complete: %s", downloaded)
        return downloaded
    except ImportError:
        logger.debug("kagglehub not installed – skipping automatic download.")
        return None
    except Exception as exc:  # network errors, auth errors, etc.
        logger.warning("KaggleHub download failed (%s) – falling back to manual path.", exc)
        return None


def _find_csv_in_dir(directory: Path, filename: str = "creditcard.csv") -> Optional[Path]:
    """Search recursively for a CSV file within a directory."""
    for candidate in directory.rglob(filename):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_raw_data(
    csv_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load the raw Credit Card Fraud dataset.

    Resolution order
    ----------------
    1. Explicit ``csv_path`` argument (if provided).
    2. Path specified in ``config/config.yaml`` → ``data.raw_path``.
    3. KaggleHub automatic download (if ``kagglehub`` is installed and credentials exist).
    4. ``data/raw/creditcard.csv`` relative to the project root (default fallback).

    Parameters
    ----------
    csv_path : str or Path, optional
        Direct path to the CSV file.  Skips config and KaggleHub lookup when provided.
    config_path : str or Path, optional
        Path to the YAML config file.  Defaults to ``config/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as stored on disk (no transformations applied).

    Raises
    ------
    FileNotFoundError
        When the CSV cannot be found through any resolution strategy.
    """
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    kaggle_dataset: str = data_cfg.get("kaggle_dataset", "mlg-ulb/creditcardfraud")

    # --- Strategy 1: explicit path ---
    if csv_path is not None:
        resolved = Path(csv_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        if resolved.exists():
            logger.info("Loading dataset from explicit path: %s", resolved)
            return _read_csv(resolved)
        raise FileNotFoundError(
            f"Explicit csv_path '{csv_path}' does not exist.\n"
            "Tip: provide the correct path or let the loader resolve it automatically."
        )

    # --- Strategy 2: config path ---
    config_csv_path_str: str = data_cfg.get("raw_path", "data/raw/creditcard.csv")
    config_csv_resolved = _PROJECT_ROOT / config_csv_path_str
    if config_csv_resolved.exists():
        logger.info("Loading dataset from config path: %s", config_csv_resolved)
        return _read_csv(config_csv_resolved)

    # --- Strategy 3: KaggleHub ---
    downloaded_dir = _try_kagglehub_download(kaggle_dataset)
    if downloaded_dir is not None:
        kaggle_csv = _find_csv_in_dir(downloaded_dir)
        if kaggle_csv:
            logger.info("Loading dataset from KaggleHub download: %s", kaggle_csv)
            # Also copy to expected location for future runs
            _ensure_raw_dir_and_copy(kaggle_csv, config_csv_resolved)
            return _read_csv(kaggle_csv)
        logger.warning(
            "KaggleHub returned directory %s but creditcard.csv not found inside.",
            downloaded_dir,
        )

    # --- Strategy 4: Default fallback ---
    default_path = _PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if default_path.exists():
        logger.info("Loading dataset from default fallback path: %s", default_path)
        return _read_csv(default_path)

    # --- All strategies exhausted ---
    raise FileNotFoundError(
        "Could not locate 'creditcard.csv'. Tried:\n"
        f"  1. Config path : {config_csv_resolved}\n"
        f"  2. KaggleHub   : dataset '{kaggle_dataset}' (requires kagglehub + credentials)\n"
        f"  3. Default path: {default_path}\n\n"
        "To fix this:\n"
        "  Option A – Install kagglehub and configure Kaggle API credentials:\n"
        "             pip install kagglehub\n"
        "             Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
        "  Option B – Download manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        f"             and place creditcard.csv at: {default_path}"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with appropriate dtypes and log basic info."""
    logger.info("Reading CSV: %s …", path)
    df = pd.read_csv(path)
    logger.info(
        "Loaded %d rows × %d columns (%.1f MB on disk)",
        len(df),
        df.shape[1],
        path.stat().st_size / 1_048_576,
    )
    return df


def _ensure_raw_dir_and_copy(src: Path, dst: Path) -> None:
    """Copy the downloaded CSV to the expected raw location if not already there."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            logger.info("Copied downloaded CSV to %s for future use.", dst)
    except Exception as exc:
        logger.debug("Could not copy CSV to raw dir: %s", exc)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------
def validate_dataset(
    df: pd.DataFrame,
    config_path: Optional[str | Path] = None,
    check_dimensions: bool = True,
) -> DataValidationReport:
    """
    Run a comprehensive suite of validation checks on the raw dataframe.

    Checks performed
    ----------------
    * Required columns present (Time, Amount, Class, V1–V28).
    * All columns are numeric (float/int).
    * Missing value counts per column.
    * Duplicate row count.
    * Target column contains only classes 0 and 1.
    * Dataset dimensions within expected bounds (280 k–290 k rows, 31 cols).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by :func:`load_raw_data`.
    config_path : str or Path, optional
        Path to the YAML config file.

    Returns
    -------
    DataValidationReport
        Structured report with ``validation_passed`` flag and ``messages`` list.
    """
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    expected_shape_cfg = data_cfg.get("expected_shape", {})
    min_rows: int = expected_shape_cfg.get("min_rows", 280_000)
    max_rows: int = expected_shape_cfg.get("max_rows", 290_000)
    n_cols_expected: int = expected_shape_cfg.get("n_cols", 31)

    # Required columns from config (also include PCA cols)
    required_from_config: List[str] = data_cfg.get("required_columns", ["Time", "Amount", "Class"])
    pca_from_config: List[str] = data_cfg.get("pca_features", _PCA_COLS)
    required_cols = list(dict.fromkeys(required_from_config + pca_from_config))  # dedup, preserve order

    messages: List[str] = []
    validation_passed = True

    # --- Shape ---
    shape = df.shape

    # --- dtypes ---
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    # --- Missing values ---
    missing_counts: Dict[str, int] = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # --- Duplicates ---
    duplicate_count = int(df.duplicated().sum())

    # --- Class distribution ---
    target_col = data_cfg.get("target_column", "Class")
    if target_col in df.columns:
        class_distribution: Dict[int, int] = df[target_col].value_counts().to_dict()
    else:
        class_distribution = {}

    # ---- Check 1: required columns ----
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        msg = f"FAIL – Missing required columns: {missing_required}"
        messages.append(msg)
        logger.error(msg)
        validation_passed = False
    else:
        messages.append(f"PASS – All {len(required_cols)} required columns present.")
        logger.info("All required columns present.")

    # ---- Check 2: numeric dtypes ----
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        msg = f"FAIL – Non-numeric columns detected: {non_numeric}"
        messages.append(msg)
        logger.error(msg)
        validation_passed = False
    else:
        messages.append("PASS – All columns are numeric.")
        logger.info("All columns are numeric.")

    # ---- Check 3: missing values ----
    if total_missing > 0:
        cols_with_missing = {c: v for c, v in missing_counts.items() if v > 0}
        msg = f"WARN – {total_missing} missing values detected in columns: {cols_with_missing}"
        messages.append(msg)
        logger.warning(msg)
        # Missing values are a warning, not a hard failure (they can be imputed downstream)
    else:
        messages.append("PASS – No missing values.")
        logger.info("No missing values.")

    # ---- Check 4: duplicate records ----
    if duplicate_count > 0:
        msg = f"WARN – {duplicate_count} duplicate rows detected."
        messages.append(msg)
        logger.warning(msg)
    else:
        messages.append("PASS – No duplicate rows.")
        logger.info("No duplicate rows.")

    # ---- Check 5: target classes ----
    if target_col in df.columns:
        observed_classes = set(df[target_col].unique())
        invalid_classes = observed_classes - _EXPECTED_CLASSES
        if invalid_classes:
            msg = f"FAIL – Unexpected classes in target column: {invalid_classes}. Expected {{0, 1}}."
            messages.append(msg)
            logger.error(msg)
            validation_passed = False
        else:
            msg = (
                f"PASS – Target classes valid: "
                f"{{0: {class_distribution.get(0, 0)}, 1: {class_distribution.get(1, 0)}}}"
            )
            messages.append(msg)
            logger.info(
                "Target classes valid – legitimate: %d, fraud: %d",
                class_distribution.get(0, 0),
                class_distribution.get(1, 0),
            )
    else:
        msg = f"FAIL – Target column '{target_col}' not found in dataframe."
        messages.append(msg)
        logger.error(msg)
        validation_passed = False

    # ---- Check 6: dataset dimensions ----
    n_rows, n_cols = shape
    if check_dimensions and not (min_rows <= n_rows <= max_rows):
        msg = (
            f"FAIL – Row count {n_rows} outside expected range [{min_rows}, {max_rows}]. "
            "Ensure the full Credit Card Fraud dataset was downloaded."
        )
        messages.append(msg)
        logger.error(msg)
        validation_passed = False
    elif not check_dimensions:
        messages.append(f"PASS – Row count {n_rows} (dimension check bypassed).")
    else:
        messages.append(f"PASS – Row count {n_rows} within expected range [{min_rows}, {max_rows}].")
        logger.info("Row count %d within expected range.", n_rows)

    if n_cols != n_cols_expected:
        msg = f"FAIL – Column count {n_cols} ≠ expected {n_cols_expected}."
        messages.append(msg)
        logger.error(msg)
        validation_passed = False
    else:
        messages.append(f"PASS – Column count {n_cols} matches expected {n_cols_expected}.")
        logger.info("Column count %d matches expected.", n_cols)

    # ---- Build report ----
    report = DataValidationReport(
        shape=shape,
        dtypes=dtypes,
        missing_counts=missing_counts,
        duplicate_count=duplicate_count,
        class_distribution=class_distribution,
        validation_passed=validation_passed,
        messages=messages,
    )

    if validation_passed:
        logger.info("Dataset validation PASSED.")
    else:
        logger.error("Dataset validation FAILED – see messages for details.")

    return report


# ---------------------------------------------------------------------------
# get_sample_transactions
# ---------------------------------------------------------------------------
def get_sample_transactions(
    df: pd.DataFrame,
    n: int = 5,
    fraud: bool = True,
    config_path: Optional[str | Path] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a random sample of transactions filtered by class.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or processed dataframe containing a 'Class' column.
    n : int
        Number of sample rows to return.
    fraud : bool
        If ``True`` (default) sample from fraudulent transactions (Class == 1).
        If ``False`` sample from legitimate transactions (Class == 0).
    config_path : str or Path, optional
        Path to the YAML config file (used to resolve the target column name).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled rows.  Returns fewer than ``n`` rows if the filtered subset is
        smaller than ``n``.
    """
    cfg = _load_config(config_path)
    target_col = cfg.get("data", {}).get("target_column", "Class")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )

    class_label = 1 if fraud else 0
    subset = df[df[target_col] == class_label]

    if subset.empty:
        logger.warning("No rows found for class=%d – returning empty DataFrame.", class_label)
        return subset

    sample_size = min(n, len(subset))
    if sample_size < n:
        logger.warning(
            "Requested %d samples but only %d rows available for class=%d.",
            n, sample_size, class_label,
        )

    sample = subset.sample(n=sample_size, random_state=random_state)
    logger.info(
        "Returning %d sample transaction(s) [class=%d, fraud=%s].",
        len(sample), class_label, fraud,
    )
    return sample
