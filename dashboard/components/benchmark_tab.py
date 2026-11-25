"""
FraudShield – Model Benchmarking Dashboard Tab
==============================================
Visualises model comparison metrics, Precision-Recall curves, ROC curves,
confusion matrices, and calibration curves across evaluated algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Theme styling constants
# ---------------------------------------------------------------------------

_PAPER_BG = "#1A2B3C"
_PLOT_BG = "#111F2E"
_GRID_COLOR = "#1E3A5F"
_PRIMARY_BLUE = "#4A90D9"
_ACCENT_TEAL = "#00D2D3"
_FRAUD_RED = "#dc3545"
_SUCCESS_GREEN = "#28a745"

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


def render_benchmark_tab(benchmark_results: Optional[Dict[str, Any]]) -> None:
    """Render the complete model benchmarking tab.

    Parameters
    ----------
    benchmark_results : dict | None
        Dictionary containing benchmark evaluation results from
        src.models.benchmark.run_benchmark.
    """
    st.markdown('<div class="section-header">⚡ Model Benchmarking & Performance</div>', unsafe_allow_html=True)

    if not benchmark_results or "results" not in benchmark_results:
        st.info(
            "ℹ️ **No benchmark results found.** Models have not been trained yet.\n\n"
            "To train and benchmark models, run:\n"
            "```bash\n"
            "python src/models/benchmark.py\n"
            "```\n"
            "Or train models using the command line to populate this tab with live empirical metrics."
        )
        return

    results = benchmark_results.get("results", {})
    best_model_name = benchmark_results.get("best_model_name", "N/A")

    # -----------------------------------------------------------------------
    # Summary KPI Cards
    # -----------------------------------------------------------------------
    st.markdown("### 🏆 Benchmark Summary")

    best_res = results.get(best_model_name, {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🥇 Best Model",
            value=best_model_name.replace("_", " ").title(),
            help="Selected primarily by validation PR-AUC due to extreme class imbalance.",
        )
    with col2:
        pr_val = best_res.get("pr_auc", 0.0)
        st.metric(label="🎯 Best PR-AUC", value=f"{pr_val:.4f}")
    with col3:
        roc_val = best_res.get("roc_auc", 0.0)
        st.metric(label="📈 Best ROC-AUC", value=f"{roc_val:.4f}")
    with col4:
        f1_val = best_res.get("f1", 0.0)
        st.metric(label="⚖️ Best F1-Score", value=f"{f1_val:.4f}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Comparison Table
    # -----------------------------------------------------------------------
    st.markdown("### 📊 Comprehensive Model Comparison")

    table_data = []
    for name, m in results.items():
        is_best = "⭐ " if name == best_model_name else ""
        table_data.append(
            {
                "Model Variant": f"{is_best}{name}",
                "PR-AUC": round(m.get("pr_auc", 0.0), 4),
                "ROC-AUC": round(m.get("roc_auc", 0.0), 4),
                "Precision": round(m.get("precision", 0.0), 4),
                "Recall": round(m.get("recall", 0.0), 4),
                "F1-Score": round(m.get("f1", 0.0), 4),
                "Optimal Threshold": round(m.get("best_threshold", 0.5), 3),
            }
        )

    df_comp = pd.DataFrame(table_data)
    df_comp = df_comp.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_comp,
        use_container_width=True,
        column_config={
            "PR-AUC": st.column_config.ProgressColumn(
                "PR-AUC", format="%.4f", min_value=0.0, max_value=1.0
            ),
            "ROC-AUC": st.column_config.ProgressColumn(
                "ROC-AUC", format="%.4f", min_value=0.0, max_value=1.0
            ),
        },
    )

    # -----------------------------------------------------------------------
    # Visual Bar Comparisons
    # -----------------------------------------------------------------------
    st.markdown("### 📈 Metric Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # PR-AUC Comparison
        fig_pr = px.bar(
            df_comp,
            x="Model Variant",
            y="PR-AUC",
            text="PR-AUC",
            color="PR-AUC",
            color_continuous_scale="Tealgrn",
            title="PR-AUC by Algorithm (Primary Fraud Metric)",
        )
        fig_pr.update_layout(**_LAYOUT_DEFAULTS)
        fig_pr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_pr.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_pr, use_container_width=True)

    with chart_col2:
        # ROC-AUC Comparison
        fig_roc = px.bar(
            df_comp,
            x="Model Variant",
            y="ROC-AUC",
            text="ROC-AUC",
            color="ROC-AUC",
            color_continuous_scale="Blues",
            title="ROC-AUC by Algorithm",
        )
        fig_roc.update_layout(**_LAYOUT_DEFAULTS)
        fig_roc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_roc.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_roc, use_container_width=True)

    # -----------------------------------------------------------------------
    # Precision vs Recall Tradeoff Chart
    # -----------------------------------------------------------------------
    st.markdown("### 🎯 Precision vs Recall Tradeoff")
    fig_scatter = px.scatter(
        df_comp,
        x="Recall",
        y="Precision",
        size="F1-Score",
        color="Model Variant",
        text="Model Variant",
        title="Precision vs Recall Tradeoff across Tested Models",
        hover_data=["PR-AUC", "ROC-AUC", "F1-Score"],
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(**_LAYOUT_DEFAULTS)
    fig_scatter.update_xaxes(range=[0, 1.05], gridcolor=_GRID_COLOR)
    fig_scatter.update_yaxes(range=[0, 1.05], gridcolor=_GRID_COLOR)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------------
    # Confusion Matrices
    # -----------------------------------------------------------------------
    st.markdown("### 🔲 Confusion Matrices (Top Models)")

    top_models = [name for name in df_comp["Model Variant"].str.replace("⭐ ", "").head(3)]
    cm_cols = st.columns(len(top_models))

    for idx, model_name in enumerate(top_models):
        m_info = results.get(model_name, {})
        cm = m_info.get("confusion_matrix")
        with cm_cols[idx]:
            if cm is not None:
                cm = np.array(cm)
                z = cm
                x = ["Pred Legit", "Pred Fraud"]
                y = ["Actual Legit", "Actual Fraud"]
                fig_cm = px.imshow(
                    z,
                    x=x,
                    y=y,
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title=f"CM: {model_name}",
                )
                fig_cm.update_layout(**_LAYOUT_DEFAULTS)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info(f"CM not recorded for {model_name}")

    # -----------------------------------------------------------------------
    # Resampling vs Cost-Sensitive Analysis
    # -----------------------------------------------------------------------
    with st.expander("🔬 Detailed Imbalance Strategy Analysis (SMOTE vs Class-Weight)", expanded=False):
        st.markdown(
            """
            **Key Findings in Fraud Detection:**
            - **Cost-Sensitive Learning (class_weight / scale_pos_weight)**: Penalizes false negatives directly
              in the loss function without altering the underlying feature space or introducing synthetic samples.
            - **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates interpolated feature points between
              k-nearest fraudulent neighbours.
            - **Risk of Leakage**: SMOTE is strictly fitted on training splits only to prevent synthetic distributions
              from leaking into validation and test sets.
            """
        )
