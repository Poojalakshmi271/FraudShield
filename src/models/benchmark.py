"""
FraudShield Model Benchmarking Module
=======================================
Trains all 6 model variants (3 algorithms × 2 imbalance strategies),
evaluates each on the validation set, and identifies the best model by
PR-AUC.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.models.evaluate import evaluate_model
from src.models.threshold import optimize_threshold
from src.models.train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

logger = logging.getLogger(__name__)

# Ordered list of (display_name, trainer_fn, use_smote)
_VARIANT_SPECS = [
    ("LR+classweight", train_logistic_regression, False),
    ("LR+SMOTE", train_logistic_regression, True),
    ("RF+classweight", train_random_forest, False),
    ("RF+SMOTE", train_random_forest, True),
    ("XGB+classweight", train_xgboost, False),
    ("XGB+SMOTE", train_xgboost, True),
]


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train all 6 model variants and evaluate each on the validation set.

    Variants:
        * ``LR+classweight``  – Logistic Regression with class_weight='balanced'
        * ``LR+SMOTE``        – Logistic Regression after SMOTE oversampling
        * ``RF+classweight``  – Random Forest with class_weight='balanced'
        * ``RF+SMOTE``        – Random Forest after SMOTE oversampling
        * ``XGB+classweight`` – XGBoost with auto scale_pos_weight
        * ``XGB+SMOTE``       – XGBoost after SMOTE oversampling

    Parameters
    ----------
    X_train : array-like, shape (n_train, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_train,)
        Training labels.
    X_val : array-like, shape (n_val, n_features)
        Validation feature matrix (never used for training).
    y_val : array-like, shape (n_val,)
        Validation labels.
    config : dict
        Project configuration dictionary forwarded to each trainer.

    Returns
    -------
    dict with keys:
        ``results``         – dict mapping variant name -> metrics dict
        ``best_model_name`` – name of the variant with highest PR-AUC
        ``best_model``      – fitted estimator of the best variant
        ``models``          – dict mapping variant name -> fitted estimator
    """
    results: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Any] = {}

    for name, trainer_fn, use_smote in _VARIANT_SPECS:
        logger.info("--- Benchmarking variant: %s ---", name)

        try:
            estimator, best_params, cv_results = trainer_fn(
                X_train, y_train, config, use_smote=use_smote
            )

            # Optimise threshold on validation set
            best_thr, best_thr_score, _ = optimize_threshold(
                estimator, X_val, y_val, metric="f1"
            )

            # Evaluate at the optimised threshold
            eval_metrics = evaluate_model(estimator, X_val, y_val, threshold=best_thr)

            results[name] = {
                "pr_auc": eval_metrics["pr_auc"],
                "roc_auc": eval_metrics["roc_auc"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "best_threshold": best_thr,
                "confusion_matrix": eval_metrics.get("confusion_matrix"),
                "best_params": best_params,
                "cv_best_score": cv_results["best_score"],
            }
            models[name] = estimator

            logger.info(
                "%s | pr_auc=%.4f | roc_auc=%.4f | f1=%.4f | threshold=%.3f",
                name,
                eval_metrics["pr_auc"],
                eval_metrics["roc_auc"],
                eval_metrics["f1"],
                best_thr,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Variant '%s' failed: %s", name, exc, exc_info=True)
            results[name] = {
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "best_threshold": 0.5,
                "error": str(exc),
            }
            models[name] = None

    # Select best by PR-AUC
    best_model_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = models[best_model_name]

    logger.info(
        "Benchmark complete. Best variant: %s (PR-AUC=%.4f).",
        best_model_name,
        results[best_model_name]["pr_auc"],
    )

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Best model selector
# ---------------------------------------------------------------------------

def select_best_model(benchmark_results: Dict[str, Any]) -> Tuple[str, Any]:
    """Return the best model from a completed benchmark run.

    Selection criterion: highest PR-AUC on the validation set.

    Parameters
    ----------
    benchmark_results : dict
        Output of :func:`run_benchmark`.

    Returns
    -------
    model_name : str
        Name of the winning variant.
    model : sklearn estimator
        Fitted estimator for the winning variant.
    """
    results = benchmark_results["results"]
    models = benchmark_results["models"]

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = models[best_name]

    logger.info(
        "Selected best model: '%s' | PR-AUC=%.4f.",
        best_name,
        results[best_name]["pr_auc"],
    )

    return best_name, best_model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(
    benchmark_results: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot grouped bar charts comparing key metrics across all variants.

    Parameters
    ----------
    benchmark_results : dict
        Output of :func:`run_benchmark`.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    results = benchmark_results["results"]
    best_name = benchmark_results["best_model_name"]

    variant_names = list(results.keys())
    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
    metric_labels = ["PR AUC", "ROC AUC", "F1", "Precision", "Recall"]
    colors = ["#3498DB", "#2ECC71", "#9B59B6", "#E67E22", "#E74C3C"]

    n_variants = len(variant_names)
    n_metrics = len(metrics)
    x = np.arange(n_variants)
    bar_width = 0.14
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[v].get(metric, 0.0) for v in variant_names]
        bars = ax.bar(x + offsets[i], values, bar_width, label=label, color=color, alpha=0.85)

        # Annotate bar tops
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    # Highlight best model
    if best_name in variant_names:
        best_idx = variant_names.index(best_name)
        ax.axvspan(best_idx - 0.45, best_idx + 0.45, alpha=0.08, color="#F39C12", label=f"Best: {best_name}")

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim([0.0, 1.15])
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Benchmark Comparison – All 6 Model Variants", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_benchmark_results(results: Dict[str, Any], path: str) -> None:
    """Save benchmark results dict to disk using joblib.

    Parameters
    ----------
    results : dict
        Output of :func:`run_benchmark`.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(results, path)
    logger.info("Benchmark results saved to %s.", path)


def load_benchmark_results(path: str) -> Dict[str, Any]:
    """Load previously saved benchmark results from disk.

    Parameters
    ----------
    path : str
        Path to the joblib file produced by :func:`save_benchmark_results`.

    Returns
    -------
    dict
        Deserialized benchmark results.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark results file not found: {path}")
    results = joblib.load(path)
    logger.info("Benchmark results loaded from %s.", path)
    return results


# ---------------------------------------------------------------------------
# CLI Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from src.data.loader import load_raw_data, validate_dataset
    from src.data.splitter import split_data
    from src.features.preprocessor import FraudPreprocessor
    from src.models.calibration import calibrate_model
    from src.models.train import save_model

    print("==========================================================")
    print("FraudShield – End-to-End Training & Benchmarking Pipeline")
    print("==========================================================")

    # 1. Load config
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Load and validate data
    print("\n[Step 1/6] Loading raw dataset...")
    df = load_raw_data()
    print(f"Loaded {len(df):,} transactions.")

    print("\n[Step 2/6] Validating dataset schema...")
    report = validate_dataset(df)
    print(f"Validation status: {'PASSED' if report.validation_passed else 'FAILED'}")
    print(f"Class distribution: {report.class_distribution} (Fraud rate: {report.fraud_rate * 100:.3f}%)")

    # 3. Stratified Split (70/15/15) - BEFORE any preprocessing
    print("\n[Step 3/6] Performing Stratified Train/Val/Test Split (70% / 15% / 15%)...")
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df, config)
    print(f"Train size: {len(X_train_raw):,} | Val size: {len(X_val_raw):,} | Test size: {len(X_test_raw):,}")

    # 4. Feature Preprocessing (Fit on train ONLY)
    print("\n[Step 4/6] Fitting FraudPreprocessor on training data only...")
    preprocessor = FraudPreprocessor(config)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(str(models_dir / "preprocessor.joblib"))
    print("Preprocessor fitted and saved to models/preprocessor.joblib")

    # 5. Model Benchmarking
    print("\n[Step 5/6] Benchmarking all 6 model variants (LR, RF, XGB × class_weight / SMOTE)...")
    benchmark_res = run_benchmark(X_train, y_train, X_val, y_val, config)
    save_benchmark_results(benchmark_res, str(models_dir / "benchmark_results.joblib"))

    best_name = benchmark_res["best_model_name"]
    best_model = benchmark_res["best_model"]
    save_model(best_model, str(models_dir / "best_model.joblib"))
    print(f"\nBenchmark Complete! 🏆 Selected Best Model: {best_name} (PR-AUC: {benchmark_res['results'][best_name]['pr_auc']:.4f})")

    # 6. Probability Calibration & Final Test Set Evaluation
    print("\n[Step 6/6] Calibrating probabilities & evaluating on untouched test set...")
    calibrated_model = calibrate_model(best_model, X_val, y_val, method="sigmoid")
    save_model(calibrated_model, str(models_dir / "calibrated_model.joblib"))

    # Single final evaluation on untouched test set
    test_thr = benchmark_res["results"][best_name]["best_threshold"]
    test_metrics = evaluate_model(calibrated_model, X_test, y_test, threshold=test_thr)

    print("\n==========================================================")
    print("FINAL UNTOUCHED TEST SET RESULTS (Selected Final Model)")
    print("==========================================================")
    print(f"Model:               {best_name} (Calibrated)")
    print(f"Decision Threshold:  {test_thr:.4f}")
    print(f"PR-AUC:              {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:             {test_metrics['roc_auc']:.4f}")
    print(f"Precision:           {test_metrics['precision']:.4f}")
    print(f"Recall:              {test_metrics['recall']:.4f}")
    print(f"F1-Score:            {test_metrics['f1']:.4f}")
    print("==========================================================")
    print("All models and artifacts saved successfully in models/ directory!")

