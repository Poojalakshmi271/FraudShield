"""
FraudShield – EDA Dashboard Tab
================================
Exploratory data analysis visualisations for the credit-card fraud dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

_PLOTLY_TEMPLATE = "plotly_dark"
_PAPER_BG = "#1A2B3C"
_PLOT_BG = "#111F2E"
_GRID_COLOR = "#1E3A5F"
_FRAUD_COLOR = "#dc3545"
_LEGIT_COLOR = "#4A90D9"


def _layout_defaults() -> dict:
    return dict(
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#E8EDF2", family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=50, r=30, t=50, b=50),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_eda_tab(df: pd.DataFrame) -> None:
    """Render the full EDA tab.

    Parameters
    ----------
    df : pd.DataFrame
        The raw credit-card transactions DataFrame (must contain at least
        'Class', 'Amount', 'Time', and 'V1'–'V28' columns).
    """
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No dataset loaded.  Please place `creditcard.csv` in `data/raw/` and restart.")
        return

    # ── Interactive filters ──────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=False):
        col_cls, col_amt = st.columns(2)
        with col_cls:
            class_filter = st.multiselect(
                "Class filter",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda x: "Legitimate (0)" if x == 0 else "Fraudulent (1)",
            )
        with col_amt:
            max_amount = float(df["Amount"].max())
            amount_range = st.slider(
                "Amount range (USD)",
                min_value=0.0,
                max_value=max_amount,
                value=(0.0, min(max_amount, 5000.0)),
                step=1.0,
            )

    # Apply filters
    mask = df["Class"].isin(class_filter) & df["Amount"].between(*amount_range)
    dff = df[mask].copy()

    if dff.empty:
        st.warning("No data matches the current filters.")
        return

    # ── Summary statistics ───────────────────────────────────────────────
    total = len(dff)
    fraud_count = int((dff["Class"] == 1).sum())
    legit_count = int((dff["Class"] == 0).sum())
    fraud_rate = fraud_count / total * 100 if total else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total (filtered)", f"{total:,}")
    col2.metric("Legitimate", f"{legit_count:,}")
    col3.metric("Fraudulent", f"{fraud_count:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.markdown("---")

    # ── Row 1: Class distribution donut + Amount histogram ───────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("#### Class Distribution")
        counts = dff["Class"].value_counts().sort_index()
        labels = ["Legitimate", "Fraudulent"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        fig_donut = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=[_LEGIT_COLOR, _FRAUD_COLOR],
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            )
        )
        fig_donut.update_layout(
            **_layout_defaults(),
            title_text="Legitimate vs Fraudulent",
            title_x=0.5,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            height=380,
        )
        # centre annotation
        fig_donut.add_annotation(
            text=f"<b>{total:,}</b><br>transactions",
            x=0.5, y=0.5,
            font=dict(size=14, color="#FFFFFF"),
            showarrow=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with r1c2:
        st.markdown("#### Transaction Amount Distribution (log scale)")
        dff_plot = dff.copy()
        dff_plot["log_amount"] = np.log1p(dff_plot["Amount"])
        dff_plot["Class_label"] = dff_plot["Class"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_hist = px.histogram(
            dff_plot,
            x="log_amount",
            color="Class_label",
            nbins=60,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"Legitimate": _LEGIT_COLOR, "Fraudulent": _FRAUD_COLOR},
            labels={"log_amount": "log(1 + Amount)", "Class_label": "Class"},
        )
        fig_hist.update_layout(
            **_layout_defaults(),
            title_text="Amount Distribution (log scale)",
            title_x=0.5,
            xaxis=dict(gridcolor=_GRID_COLOR),
            yaxis=dict(gridcolor=_GRID_COLOR),
            legend_title_text="Class",
            height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 2: Amount boxplot + Time-of-day patterns ─────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("#### Amount by Class (Boxplot)")
        dff_plot2 = dff.copy()
        dff_plot2["Class_label"] = dff_plot2["Class"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_box = px.box(
            dff_plot2,
            x="Class_label",
            y="Amount",
            color="Class_label",
            color_discrete_map={"Legitimate": _LEGIT_COLOR, "Fraudulent": _FRAUD_COLOR},
            points=False,
            log_y=True,
            labels={"Class_label": "Class", "Amount": "Amount (USD, log)"},
        )
        fig_box.update_layout(
            **_layout_defaults(),
            title_text="Amount Distribution by Class",
            title_x=0.5,
            showlegend=False,
            xaxis=dict(gridcolor=_GRID_COLOR),
            yaxis=dict(gridcolor=_GRID_COLOR),
            height=380,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with r2c2:
        st.markdown("#### Time-of-Day Patterns (Hour)")
        dff_time = dff.copy()
        dff_time["hour_of_day"] = (dff_time["Time"] % 86400) / 3600
        dff_time["Class_label"] = dff_time["Class"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_time = px.histogram(
            dff_time,
            x="hour_of_day",
            color="Class_label",
            nbins=24,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"Legitimate": _LEGIT_COLOR, "Fraudulent": _FRAUD_COLOR},
            labels={"hour_of_day": "Hour of Day (0-24)", "Class_label": "Class"},
        )
        fig_time.update_layout(
            **_layout_defaults(),
            title_text="Transaction Hour Distribution",
            title_x=0.5,
            xaxis=dict(gridcolor=_GRID_COLOR, dtick=2),
            yaxis=dict(gridcolor=_GRID_COLOR),
            legend_title_text="Class",
            height=380,
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Row 3: Correlation heatmap ───────────────────────────────────────
    st.markdown("#### Feature Correlations with Class")
    with st.expander("Show correlation heatmap", expanded=True):
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in dff.columns]
        num_cols = ["Amount", "Time"] + v_cols + ["Class"]
        corr_df = dff[num_cols].corr()

        # Pull top 15 features most correlated with Class
        top_features = (
            corr_df["Class"].drop("Class").abs().nlargest(15).index.tolist()
        )
        sub_corr = corr_df.loc[top_features + ["Class"], top_features + ["Class"]]

        fig_heat = go.Figure(
            go.Heatmap(
                z=sub_corr.values,
                x=sub_corr.columns.tolist(),
                y=sub_corr.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(sub_corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=9),
                hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            **_layout_defaults(),
            title_text="Correlation Heatmap (Top 15 Features + Class)",
            title_x=0.5,
            height=520,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Row 4: PCA feature violin plots (V1–V10) ─────────────────────────
    st.markdown("#### PCA Feature Distributions (V1–V10) by Class")
    with st.expander("Show violin plots", expanded=False):
        v_subset = [f"V{i}" for i in range(1, 11) if f"V{i}" in dff.columns]
        dff_melt = dff[v_subset + ["Class"]].melt(
            id_vars="Class", var_name="Feature", value_name="Value"
        )
        dff_melt["Class_label"] = dff_melt["Class"].map({0: "Legitimate", 1: "Fraudulent"})

        fig_violin = px.violin(
            dff_melt,
            x="Feature",
            y="Value",
            color="Class_label",
            box=True,
            points=False,
            color_discrete_map={"Legitimate": _LEGIT_COLOR, "Fraudulent": _FRAUD_COLOR},
            labels={"Class_label": "Class"},
        )
        fig_violin.update_layout(
            **_layout_defaults(),
            title_text="V1–V10 Distributions by Class",
            title_x=0.5,
            violingap=0.1,
            violingroupgap=0.05,
            xaxis=dict(gridcolor=_GRID_COLOR),
            yaxis=dict(gridcolor=_GRID_COLOR),
            height=480,
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Key statistics expandable ─────────────────────────────────────────
    with st.expander("📋 Key Statistics", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Legitimate Transactions**")
            legit_df = dff[dff["Class"] == 0]["Amount"]
            st.table(
                pd.DataFrame({
                    "Statistic": ["Count", "Mean Amount", "Median Amount", "Max Amount", "Std Amount"],
                    "Value": [
                        f"{len(legit_df):,}",
                        f"${legit_df.mean():.2f}",
                        f"${legit_df.median():.2f}",
                        f"${legit_df.max():.2f}",
                        f"${legit_df.std():.2f}",
                    ],
                })
            )
        with col_s2:
            st.markdown("**Fraudulent Transactions**")
            fraud_df = dff[dff["Class"] == 1]["Amount"]
            if len(fraud_df) > 0:
                st.table(
                    pd.DataFrame({
                        "Statistic": ["Count", "Mean Amount", "Median Amount", "Max Amount", "Std Amount"],
                        "Value": [
                            f"{len(fraud_df):,}",
                            f"${fraud_df.mean():.2f}",
                            f"${fraud_df.median():.2f}",
                            f"${fraud_df.max():.2f}",
                            f"${fraud_df.std():.2f}",
                        ],
                    })
                )
            else:
                st.info("No fraudulent transactions in filtered view.")
