"""
FraudShield SHAP Explainability Module
========================================
Provides global and local SHAP explanations for fraud detection models.
Supports tree-based models (RandomForest, XGBoost) and linear models (LogisticRegression).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")  # Non-interactive backend safe for server use

logger = logging.getLogger(__name__)


class FraudSHAPExplainer:
    """
    SHAP-based explainability wrapper for FraudShield models.

    Supports:
      - Tree models  : shap.TreeExplainer  (RandomForest, XGBoost, GradientBoosting)
      - Linear models: shap.LinearExplainer (LogisticRegression, LinearSVC, etc.)

    Parameters
    ----------
    model : sklearn estimator or XGBoost model
        The trained classification model to explain.
    X_background : pd.DataFrame or np.ndarray
        Background dataset used for explainer initialisation.
    model_type : str
        'tree' or 'linear'. When 'auto' is passed the type is inferred from the
        model class name. Defaults to 'tree'.
    """

    # Class names whose presence in the model type string signals a linear model
    _LINEAR_INDICATORS = ("logistic", "linear", "ridge", "lasso", "sgd", "linearsvr", "linearsvc")

    def __init__(
        self,
        model,
        X_background: Union[pd.DataFrame, np.ndarray],
        model_type: str = "tree",
    ) -> None:
        self.model = model
        self.X_background = X_background
        self.model_type = self._resolve_model_type(model, model_type)
        self.explainer: Optional[Union[shap.TreeExplainer, shap.LinearExplainer]] = None
        logger.info("FraudSHAPExplainer initialised (model_type=%s).", self.model_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_type(model, model_type: str) -> str:
        """Infer model type from class name when model_type == 'auto'."""
        if model_type not in ("auto", "tree", "linear"):
            logger.warning("Unknown model_type '%s'; defaulting to 'tree'.", model_type)
            return "tree"
        if model_type == "auto":
            class_name = type(model).__name__.lower()
            for indicator in FraudSHAPExplainer._LINEAR_INDICATORS:
                if indicator in class_name:
                    return "linear"
            return "tree"
        return model_type

    def _background_sample(self, X_background_sample=None) -> Union[pd.DataFrame, np.ndarray]:
        """Return a sample of the background data capped at 100 rows for speed."""
        data = X_background_sample if X_background_sample is not None else self.X_background
        n = min(100, len(data))
        if isinstance(data, pd.DataFrame):
            return data.sample(n=n, random_state=42)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=n, replace=False)
        return data[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_background_sample=None) -> "FraudSHAPExplainer":
        """
        Build the SHAP explainer.

        Parameters
        ----------
        X_background_sample : array-like, optional
            Alternative background sample. When None the background passed at
            construction is used (subsampled to ≤100 rows).

        Returns
        -------
        self
        """
        sample = self._background_sample(X_background_sample)
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("TreeExplainer created.")
        else:
            self.explainer = shap.LinearExplainer(self.model, sample)
            logger.info("LinearExplainer created with %d background samples.", len(sample))
        return self

    def explain_global(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
    ) -> dict:
        """
        Compute global SHAP values and feature importance.

        Parameters
        ----------
        X : array-like
            Dataset to explain (should be the validation or test set).
        max_display : int
            Maximum number of features to surface in summaries.

        Returns
        -------
        dict with keys:
            shap_values           – raw SHAP values (ndarray)
            feature_names         – list of feature column names
            mean_abs_shap         – mean |SHAP| per feature (ndarray)
            feature_importance_df – DataFrame[feature, importance] sorted desc.
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_global().")

        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]

        shap_values = self.explainer.shap_values(X)

        # For binary classifiers that return a list (class 0, class 1) take class-1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("Global SHAP explanation computed for %d samples, %d features.", len(X), len(feature_names))
        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "feature_importance_df": feature_importance_df,
        }

    def explain_local(
        self,
        x_single: Union[pd.DataFrame, np.ndarray],
        feature_names: list,
    ) -> dict:
        """
        Compute a local (single-instance) SHAP explanation.

        Parameters
        ----------
        x_single : array-like, shape (1, n_features)
            Single transaction to explain.
        feature_names : list of str
            Feature column names.

        Returns
        -------
        dict with keys:
            shap_values   – SHAP values for each feature (1-D ndarray)
            feature_names – list of feature names
            base_value    – SHAP expected value
            prediction    – model predicted probability
            waterfall_data – list of dicts {feature, shap_value, cumulative}
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_local().")

        shap_values = self.explainer.shap_values(x_single)

        # Flatten for single instance; handle binary list case
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values).flatten()

        # Base value
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            base_value = float(expected[1])
        else:
            base_value = float(expected)

        # Prediction probability
        x_arr = x_single.values if isinstance(x_single, pd.DataFrame) else np.asarray(x_single)
        if hasattr(self.model, "predict_proba"):
            prediction = float(self.model.predict_proba(x_arr.reshape(1, -1))[0, 1])
        else:
            prediction = float(self.model.predict(x_arr.reshape(1, -1))[0])

        # Waterfall data (sorted by |shap|)
        order = np.argsort(np.abs(shap_values))[::-1]
        cumulative = base_value
        waterfall_data = []
        for idx in order:
            cumulative += shap_values[idx]
            waterfall_data.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": float(shap_values[idx]),
                    "cumulative": float(cumulative),
                }
            )

        logger.info("Local SHAP explanation computed. Prediction=%.4f, base_value=%.4f.", prediction, base_value)
        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "base_value": base_value,
            "prediction": prediction,
            "waterfall_data": waterfall_data,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: list,
        max_display: int = 20,
    ) -> plt.Figure:
        """
        Beeswarm (summary) plot of global SHAP values.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, max(6, min(max_display // 2 + 2, 14))))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="dot",
            show=False,
        )
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("SHAP Summary – Feature Impact on Fraud Probability", fontsize=13, fontweight="bold")
        plt.tight_layout()
        logger.debug("Summary (beeswarm) plot created.")
        return fig

    def plot_waterfall(self, local_explanation: dict, title: str = "") -> plt.Figure:
        """
        Waterfall plot for a single-transaction SHAP explanation.

        Parameters
        ----------
        local_explanation : dict
            Output of explain_local().
        title : str
            Optional plot title suffix.

        Returns
        -------
        matplotlib.figure.Figure
        """
        waterfall_data = local_explanation["waterfall_data"]
        base_value = local_explanation["base_value"]
        prediction = local_explanation["prediction"]
        top_n = min(15, len(waterfall_data))

        features = [d["feature"] for d in waterfall_data[:top_n]]
        shap_vals = [d["shap_value"] for d in waterfall_data[:top_n]]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2 + 2)))
        bars = ax.barh(range(top_n), shap_vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(features[::-1], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=11)
        heading = f"Local SHAP – {title}" if title else "Local SHAP Explanation"
        ax.set_title(
            f"{heading}\nPredicted probability: {prediction:.4f}  |  Base value: {base_value:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        logger.debug("Waterfall plot created.")
        return fig

    def plot_global_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = 15,
    ) -> plt.Figure:
        """
        Horizontal bar chart of mean absolute SHAP values (global importance).

        Parameters
        ----------
        feature_importance_df : pd.DataFrame
            DataFrame with columns 'feature' and 'importance' sorted descending.
        top_n : int
            Number of top features to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = feature_importance_df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(10, max(5, top_n // 2 + 2)))
        bars = ax.barh(
            range(len(df)),
            df["importance"].values[::-1],
            color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df))),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(f"Top {len(df)} Features – Global SHAP Importance", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Annotate bar values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + max(df["importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        plt.tight_layout()
        logger.debug("Global importance bar chart created.")
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the explainer to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FraudSHAPExplainer saved to %s.", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FraudSHAPExplainer":
        """Load a persisted FraudSHAPExplainer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Explainer file not found: {path}")
        instance = joblib.load(path)
        logger.info("FraudSHAPExplainer loaded from %s.", path)
        return instance
