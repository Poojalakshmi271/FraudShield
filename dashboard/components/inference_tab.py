"""
FraudShield – Transaction Risk Analysis Dashboard Tab
=====================================================
Interactive transaction scoring, probability estimation, risk tier classification,
and decision recommendations.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import DISCLAIMER, create_kpi_card
from src.risk.scoring_engine import RiskScoringEngine


# ---------------------------------------------------------------------------
# Theme styling constants
# ---------------------------------------------------------------------------

_PAPER_BG = "#1A2B3C"
_PLOT_BG = "#111F2E"
_GRID_COLOR = "#1E3A5F"


def _create_gauge_chart(prob: float, threshold_high: float = 0.8) -> go.Figure:
    """Build a gauge chart showing the estimated fraud probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#E8EDF2"}},
            title={"text": "Estimated Fraud Probability", "font": {"size": 18, "color": "#A0B2C6"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#A0B2C6"},
                "bar": {"color": "#dc3545" if prob >= 0.5 else "#4A90D9"},
                "bgcolor": _PLOT_BG,
                "borderwidth": 2,
                "bordercolor": _GRID_COLOR,
                "steps": [
                    {"range": [0, 20], "color": "rgba(40, 167, 69, 0.3)"},
                    {"range": [20, 50], "color": "rgba(255, 193, 7, 0.3)"},
                    {"range": [50, 80], "color": "rgba(253, 126, 20, 0.3)"},
                    {"range": [80, 100], "color": "rgba(220, 53, 69, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold_high * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        font=dict(color="#E8EDF2", family="Inter, Segoe UI, sans-serif"),
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_inference_tab(
    df: Optional[pd.DataFrame],
    models_dict: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Render the Transaction Risk Analysis / Inference tab.

    Parameters
    ----------
    df : pd.DataFrame | None
        Raw dataset for sampling transactions.
    models_dict : dict
        Dict of preprocessor, best_model, calibrated_model.
    config : dict
        Project configuration.
    """
    st.markdown('<div class="section-header">🔍 Transaction Risk Analysis</div>', unsafe_allow_html=True)
    st.caption("Score individual transactions, evaluate probability of fraud, and review policy actions.")

    # Check model availability
    preprocessor = models_dict.get("preprocessor")
    model = models_dict.get("calibrated_model") or models_dict.get("best_model")

    if model is None or preprocessor is None:
        st.warning(
            "⚠️ **Model or Preprocessor not loaded.**\n\n"
            "Trained model artifacts are required for transaction inference. "
            "Please train the model first by running `python src/models/benchmark.py`."
        )
        return

    # Instantiate Risk Scoring Engine
    scoring_engine = RiskScoringEngine(model=model, preprocessor=preprocessor, config=config)

    # -----------------------------------------------------------------------
    # Input selection: Sample vs Manual
    # -----------------------------------------------------------------------
    col_mode1, col_mode2 = st.columns([1, 2])
    with col_mode1:
        input_mode = st.radio(
            "Input Method",
            ["Sample from Dataset", "Manual Entry"],
            horizontal=True,
        )

    # Default transaction features
    pca_cols = [f"V{i}" for i in range(1, 29)]
    txn_data: Dict[str, float] = {}

    if input_mode == "Sample from Dataset":
        if df is not None and not df.empty:
            subcol1, subcol2, subcol3 = st.columns([1, 1, 1])
            with subcol1:
                sample_type = st.selectbox("Sample Type", ["Legitimate (Class 0)", "Fraudulent (Class 1)"])
            with subcol2:
                random_seed_input = st.number_input("Random Seed / Sample ID", min_value=0, max_value=99999, value=42)
            with subcol3:
                st.write("")
                st.write("")
                load_btn = st.button("🎲 Load Sample", use_container_width=True)

            target_class = 1 if "Fraudulent" in sample_type else 0
            class_df = df[df["Class"] == target_class]

            if not class_df.empty:
                # Pick row deterministically or on button press
                row_idx = random_seed_input % len(class_df)
                sample_row = class_df.iloc[row_idx]
                txn_data["Time"] = float(sample_row.get("Time", 0.0))
                txn_data["Amount"] = float(sample_row.get("Amount", 0.0))
                for col in pca_cols:
                    txn_data[col] = float(sample_row.get(col, 0.0))

                st.success(f"Loaded {sample_type} transaction #{row_idx} (Amount: ${txn_data['Amount']:.2f}, Time: {txn_data['Time']:.0f}s)")
            else:
                st.warning("No transactions found for the selected class.")
        else:
            st.info("Dataset is not loaded in memory. Switching to manual default inputs.")
            txn_data["Time"] = 43200.0
            txn_data["Amount"] = 120.50
            for col in pca_cols:
                txn_data[col] = 0.0
    else:
        # Manual entry
        st.markdown("#### Transaction Parameters")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            txn_data["Amount"] = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=149.99, step=1.0)
        with mcol2:
            txn_data["Time"] = st.number_input("Elapsed Seconds (Time)", min_value=0.0, max_value=172800.0, value=43200.0, step=60.0)

        with st.expander("🛠️ Advanced PCA Features (V1 – V28)", expanded=False):
            st.caption("Adjust anonymized principal component values (range typically -20 to +20):")
            pca_subcols = st.columns(4)
            for i, col in enumerate(pca_cols):
                with pca_subcols[i % 4]:
                    txn_data[col] = st.slider(col, min_value=-15.0, max_value=15.0, value=0.0, step=0.1)

    # -----------------------------------------------------------------------
    # Predict / Score Action
    # -----------------------------------------------------------------------
    st.markdown("---")
    score_button = st.button("🚀 Analyze Transaction Risk", type="primary", use_container_width=True)

    if score_button or input_mode == "Sample from Dataset":
        try:
            score_result = scoring_engine.score_transaction(txn_data)
            prob = score_result.fraud_probability
            category = score_result.risk_category
            label = score_result.risk_label
            color = score_result.risk_color
            action = score_result.recommended_action

            # Display Results
            st.markdown("### 📋 Risk Assessment Outcome")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                # Gauge Chart
                fig_gauge = _create_gauge_chart(prob)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                # Risk Category Card
                st.markdown(
                    f"""
                    <div style="background-color: #1A2B3C; border-left: 6px solid {color}; padding: 20px; border-radius: 8px; margin-top: 15px;">
                        <h4 style="margin: 0; color: #A0B2C6; font-size: 14px; text-transform: uppercase;">Assigned Risk Tier</h4>
                        <h2 style="margin: 5px 0 15px 0; color: {color}; font-size: 28px;">{label} ({category})</h2>
                        <h4 style="margin: 0; color: #A0B2C6; font-size: 14px; text-transform: uppercase;">Recommended Policy Action</h4>
                        <p style="margin: 5px 0 0 0; color: #E8EDF2; font-size: 16px; font-weight: 500;">{action}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background-color: #111F2E; padding: 12px; border-radius: 6px; font-size: 12px; color: #A0B2C6; border: 1px solid #1E3A5F;">
                        <b>Demonstration Disclaimer:</b><br>{DISCLAIMER}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Top Contributing Features (from model feature importance or direct values)
            st.markdown("#### 🔍 Notable Transaction Signals")
            signal_cols = st.columns(3)
            with signal_cols[0]:
                st.metric("Amount", f"${txn_data.get('Amount', 0.0):.2f}")
            with signal_cols[1]:
                sec = txn_data.get("Time", 0.0) % 86400
                st.metric("Approx Time of Day", f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}")
            with signal_cols[2]:
                # Top extreme PCA feature
                pca_vals = {col: txn_data.get(col, 0.0) for col in pca_cols}
                extreme_feat = max(pca_vals.items(), key=lambda x: abs(x[1]))
                st.metric("Most Extreme PCA", f"{extreme_feat[0]} = {extreme_feat[1]:.2f}")

        except Exception as exc:
            st.error(f"Error during risk inference: {exc}")
