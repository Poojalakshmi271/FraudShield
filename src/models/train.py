"""
FraudShield Model Training Module
==================================
Trains Logistic Regression, Random Forest, and XGBoost models with
hyperparameter tuning via RandomizedSearchCV and optional SMOTE resampling.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER = 20
SCORING = "average_precision"

LR_PARAM_GRID: Dict[str, Any] = {
    "C": [0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
    "max_iter": [200],
}

RF_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [50, 100],
    "max_depth": [6, 10],
    "min_samples_leaf": [2, 4],
}

XGB_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [80, 120],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_smote_if_needed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_smote: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally oversample the minority class with SMOTE."""
    if not use_smote:
        return X_train, y_train

    from imblearn.over_sampling import SMOTE  # lazy import – optional dep

    logger.info(
        "Applying SMOTE. Before: %d negatives, %d positives.",
        int((y_train == 0).sum()),
        int((y_train == 1).sum()),
    )
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE: %d negatives, %d positives.",
        int((y_res == 0).sum()),
        int((y_res == 1).sum()),
    )
    return X_res, y_res


def _run_random_search(
    estimator: Any,
    param_grid: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Run RandomizedSearchCV with stratified CV and return results."""
    tuning_cfg = (config or {}).get("tuning", {})
    cv_folds = int(tuning_cfg.get("cv_folds", CV_FOLDS))
    n_iter = int(tuning_cfg.get("n_iter", 3))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
    )

    logger.info("Starting RandomizedSearchCV for %s (n_iter=%d, cv=%d).", model_name, n_iter, cv_folds)
    search.fit(X, y)
    logger.info(
        "%s best CV %s: %.4f | params: %s",
        model_name,
        SCORING,
        search.best_score_,
        search.best_params_,
    )

    cv_results = {
        "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
        "std_test_score": search.cv_results_["std_test_score"].tolist(),
        "params": search.cv_results_["params"],
        "best_score": search.best_score_,
        "best_index": int(search.best_index_),
    }

    return search.best_estimator_, search.best_params_, cv_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    use_smote: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Train a Logistic Regression classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_samples,)
        Binary target labels.
    config : dict
        Project configuration dictionary (may include model-level keys).
    use_smote : bool, default False
        If True, apply SMOTE oversampling before training.

    Returns
    -------
    best_estimator : LogisticRegression
        Fitted model with best hyperparameters.
    best_params : dict
        Best hyperparameter values found by search.
    cv_results : dict
        Summary of cross-validation results.
    """
    X, y = _apply_smote_if_needed(X_train, y_train, use_smote)

    base_estimator = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    logger.info("Training LogisticRegression | SMOTE=%s | samples=%d.", use_smote, len(y))
    return _run_random_search(base_estimator, LR_PARAM_GRID, X, y, "LogisticRegression", config)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    use_smote: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Train a Random Forest classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_samples,)
        Binary target labels.
    config : dict
        Project configuration dictionary.
    use_smote : bool, default False
        If True, apply SMOTE oversampling before training.

    Returns
    -------
    best_estimator : RandomForestClassifier
        Fitted model with best hyperparameters.
    best_params : dict
        Best hyperparameter values found by search.
    cv_results : dict
        Summary of cross-validation results.
    """
    X, y = _apply_smote_if_needed(X_train, y_train, use_smote)

    class_weight = None if use_smote else "balanced"
    base_estimator = RandomForestClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training RandomForest | SMOTE=%s | samples=%d.", use_smote, len(y))
    return _run_random_search(base_estimator, RF_PARAM_GRID, X, y, "RandomForest", config)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    use_smote: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Train an XGBoost classifier with hyperparameter tuning.

    When *not* using SMOTE, the ``scale_pos_weight`` parameter is set to
    ``count(negative) / count(positive)`` to handle class imbalance.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_samples,)
        Binary target labels.
    config : dict
        Project configuration dictionary.
    use_smote : bool, default False
        If True, apply SMOTE oversampling before training (and
        scale_pos_weight is set to 1.0 since classes are balanced).

    Returns
    -------
    best_estimator : XGBClassifier
        Fitted model with best hyperparameters.
    best_params : dict
        Best hyperparameter values found by search.
    cv_results : dict
        Summary of cross-validation results.
    """
    X, y = _apply_smote_if_needed(X_train, y_train, use_smote)

    n_negative = int((np.asarray(y) == 0).sum())
    n_positive = int((np.asarray(y) == 1).sum())

    if use_smote or n_positive == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = n_negative / n_positive

    logger.info(
        "Training XGBoost | SMOTE=%s | scale_pos_weight=%.2f | samples=%d.",
        use_smote,
        scale_pos_weight,
        len(y),
    )

    base_estimator = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )

    return _run_random_search(base_estimator, XGB_PARAM_GRID, X, y, "XGBoost", config)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Persist a trained model to disk using joblib.

    Parameters
    ----------
    model : sklearn estimator or similar
        Fitted model object to save.
    path : str
        Destination file path (e.g. ``models/lr_classweight.joblib``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s.", path)


def load_model(path: str) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    path : str
        Path to the joblib file produced by :func:`save_model`.

    Returns
    -------
    model : object
        Deserialized model object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s.", path)
    return model
