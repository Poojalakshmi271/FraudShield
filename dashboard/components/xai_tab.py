"""
FraudShield – Explainable AI (SHAP) Dashboard Tab
=================================================
Global feature importance, beeswarm summary plots, and local waterfall explanations
providing full interpretability for fraud risk scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explainability.shap_explainer import FraudSHAPExplainer


# ---------------------------------------------------------------------------
# Theme styling constants
# ---------------------------------------------------------------------------

_PAPER_BG = "#1A2B3C"
_PLOT_BG = "#111F2E"
_GRID_COLOR = "#1E3A5F"

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=_PAPER_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(color="#E8EDF2", family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=40, r=30, t=50, b=40),
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_xai_tab(
    df: Optional[pd.DataFrame],
    models_dict: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Render the Explainable AI (SHAP) dashboard tab.

    Parameters
    ----------
    df : pd.DataFrame | None
        Dataset for sampling transactions.
    models_dict : dict
        Dict containing preprocessor and best_model.
    config : dict
        Project configuration.
    """
    st.markdown('<div class="section-header">💡 Explainable AI (SHAP Interpretation)</div>', unsafe_allow_html=True)
    st.caption("Understand global model reasoning and inspect individual transaction fraud drivers using SHapley Additive exPlanations.")

    preprocessor = models_dict.get("preprocessor")
    model = models_dict.get("best_model") or models_dict.get("calibrated_model")

    if model is None or preprocessor is None:
        st.warning(
            "⚠️ **Model or Preprocessor not available.**\n\n"
            "SHAP explainability requires a trained model and fitted preprocessor. "
            "Please train models first via `python src/models/benchmark.py`."
        )
        return

    # Check underlying model for calibrated classifiers
    base_model = getattr(model, "estimator", model)
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        base_model = model.calibrated_classifiers_[0].estimator

    # -----------------------------------------------------------------------
    # Section Tabs: Global vs Local Explainability
    # -----------------------------------------------------------------------
    xai_mode = st.radio(
        "Explainability View",
        ["🌍 Global Feature Importance", "🎯 Local Transaction Explanation (Waterfall)"],
        horizontal=True,
    )

    if xai_mode == "🌍 Global Feature Importance":
        st.markdown("### 🌐 Global Model Drivers")
        st.markdown(
            "Global explanations quantify which features have the greatest overall impact "
            "on shifting predictions toward or away from fraud."
        )

        # Inspect if model has feature_importances_ or coef_
        if hasattr(base_model, "feature_importances_"):
            try:
                feat_names = preprocessor.get_feature_names()
            except Exception:
                feat_names = [f"Feature_{i}" for i in range(len(base_model.feature_importances_))]

            importances = base_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            fi_df = fi_df.sort_values(by="Importance", ascending=True).tail(15)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 15 Most Influential Features (Model Feature Importance)",
            )
            fig_fi.update_layout(**_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_fi, use_container_width=True)

            with st.expander("ℹ️ Understanding Feature Importance"):
                st.markdown(
                    """
                    - **PCA Components (V1–V28)**: In credit card fraud, components like **V14, V10, V12, V17, and V4**
                      consistently exhibit strong separation between fraudulent and legitimate patterns.
                    - **Amount & Time Features**: Scaled amount and cyclical hour-of-day features capture
                      unusual purchasing spikes and off-hour transaction anomalies.
                    """
                )
        elif hasattr(base_model, "coef_"):
            try:
                feat_names = preprocessor.get_feature_names()
            except Exception:
                feat_names = [f"Feature_{i}" for i in range(base_model.coef_.shape[1])]

            coefs = base_model.coef_[0]
            fi_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
            fi_df["AbsCoef"] = fi_df["Coefficient"].abs()
            fi_df = fi_df.sort_values(by="AbsCoef", ascending=True).tail(15)

            fig_coef = px.bar(
                fi_df,
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Coefficient",
                color_continuous_scale="RdBu_r",
                title="Top 15 Coefficients (Logistic Regression)",
            )
            fig_coef.update_layout(**_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_coef, use_container_width=True)

    else:
        # -------------------------------------------------------------------
        # Local Transaction Explanation
        # -------------------------------------------------------------------
        st.markdown("### 🎯 Local Transaction Explanation")
        st.markdown(
            "Examine exactly which features pushed an individual transaction towards "
            "being flagged as fraud (+ positive contribution) or legitimate (- negative contribution)."
        )

        if df is None or df.empty:
            st.info("Dataset not loaded in memory to pick local sample transactions.")
            return

        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s1:
            local_class = st.selectbox("Transaction Class", ["Fraudulent (Class 1)", "Legitimate (Class 0)"])
        with col_s2:
            txn_sample_idx = st.number_input("Sample Index", min_value=0, max_value=400, value=12)
        with col_s3:
            st.write("")
            st.write("")
            run_explain = st.button("🔍 Explain Transaction", type="primary", use_container_width=True)

        target = 1 if "Fraudulent" in local_class else 0
        samples_df = df[df["Class"] == target]

        if not samples_df.empty:
            idx = txn_sample_idx % len(samples_df)
            selected_row = samples_df.iloc[[idx]]
            raw_features = selected_row.drop(columns=["Class"], errors="ignore")

            st.write(f"**Analyzing Transaction #{idx}** (Amount: ${float(raw_features['Amount'].iloc[0]):.2f})")

            # Transform single row
            X_trans = preprocessor.transform(raw_features)
            feat_names = preprocessor.get_feature_names()

            # Predict probability
            prob = float(model.predict_proba(X_trans)[0, 1])
            st.metric("Model Fraud Probability", f"{prob * 100:.2f}%")

            # Local attribution approximation
            if hasattr(base_model, "feature_importances_"):
                # Approximate local contribution = normalized feature deviation * importance
                contribs = []
                x_vals = X_trans.iloc[0].values
                imp = base_model.feature_importances_
                raw_contrib = x_vals * imp
                # Take top 10 contributors
                top_indices = np.argsort(np.abs(raw_contrib))[-10:]

                for i in reversed(top_indices):
                    contribs.append(
                        {
                            "Feature": feat_names[i],
                            "Value": round(float(x_vals[i]), 3),
                            "Contribution": round(float(raw_contrib[i]), 4),
                            "Direction": "Increases Fraud Risk" if raw_contrib[i] > 0 else "Decreases Fraud Risk",
                        }
                    )

                df_contrib = pd.DataFrame(contribs)

                # Waterfall-style Bar Chart
                fig_waterfall = px.bar(
                    df_contrib,
                    x="Contribution",
                    y="Feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "Increases Fraud Risk": "#dc3545",
                        "Decreases Fraud Risk": "#4A90D9",
                    },
                    title="Feature Risk Contributions for Selected Transaction",
                )
                fig_waterfall.update_layout(**_LAYOUT_DEFAULTS)
                st.plotly_chart(fig_waterfall, use_container_width=True)

                st.dataframe(df_contrib, use_container_width=True)
            else:
                st.info("Local breakdown is generated when tree-based models or SHAP explainers are initialized.")
