"""
FraudShield – Main Streamlit Application
=========================================
Intelligent Financial Fraud Detection & Risk Scoring Platform.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from dashboard.components.benchmark_tab import render_benchmark_tab
from dashboard.components.eda_tab import render_eda_tab
from dashboard.components.inference_tab import render_inference_tab
from dashboard.components.xai_tab import render_xai_tab
from dashboard.utils import (
    DISCLAIMER,
    apply_custom_css,
    create_kpi_card,
    format_number,
    load_config,
    load_data,
    load_models,
)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FraudShield – Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified dark navy styling
apply_custom_css()


# ---------------------------------------------------------------------------
# Load Resources
# ---------------------------------------------------------------------------

config = load_config()
df = load_data()
models_dict = load_models()


# ---------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="color: #4A90D9; font-size: 26px; margin: 0;">🛡️ FraudShield</h1>
            <p style="color: #A0B2C6; font-size: 12px; margin: 5px 0 0 0;">Explainable AI Financial Fraud Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Navigation Menu
    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Exploratory Data Analysis",
            "⚡ Model Benchmarking",
            "🔍 Transaction Risk Analysis",
            "💡 Explainable AI (SHAP)",
        ],
        index=0,
    )

    st.markdown("---")

    # System Status Indicators
    st.markdown("#### ⚙️ System Status")

    # Dataset status
    if df is not None and not df.empty:
        st.markdown(
            f'<div class="status-badge status-loaded">● Dataset Loaded ({len(df):,} txns)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-not-loaded">○ Dataset Missing (data/raw/creditcard.csv)</div>',
            unsafe_allow_html=True,
        )

    # Models status
    best_model = models_dict.get("best_model")
    calibrated_model = models_dict.get("calibrated_model")
    if best_model is not None or calibrated_model is not None:
        st.markdown(
            '<div class="status-badge status-loaded">● Models Trained & Loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-not-loaded">○ Models Not Trained</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Configurable Risk Thresholds
    st.markdown("#### 🎚️ Risk Thresholds")
    st.caption("Adjust policy sensitivity for demonstration risk bands:")

    cfg_thresh = config.get("risk", {}).get("thresholds", {})
    t_low = st.slider("Low / Medium Threshold", 0.05, 0.40, float(cfg_thresh.get("low", 0.20)), 0.05)
    t_med = st.slider("Medium / High Threshold", 0.25, 0.70, float(cfg_thresh.get("medium", 0.50)), 0.05)
    t_high = st.slider("High / Critical Threshold", 0.55, 0.95, float(cfg_thresh.get("high", 0.80)), 0.05)

    if "risk" in config:
        config["risk"]["thresholds"] = {"low": t_low, "medium": t_med, "high": t_high}

    st.markdown("---")
    st.markdown(
        f'<div style="font-size: 10px; color: #7B8F9F; line-height: 1.4;">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Route to Selected Tab
# ---------------------------------------------------------------------------

if nav_selection == "🏠 Overview":
    st.markdown('<div class="section-header">🛡️ FraudShield Platform Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "Welcome to **FraudShield**, an end-to-end, explainable machine learning system designed to "
        "detect fraudulent financial transactions, evaluate calibrated risk probabilities, and explain model decisions."
    )

    # Top KPI Metrics Row
    if df is not None and not df.empty and "Class" in df.columns:
        total_txns = len(df)
        fraud_txns = int(df["Class"].sum())
        legit_txns = total_txns - fraud_txns
        fraud_rate = (fraud_txns / total_txns) * 100

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(
                create_kpi_card("Total Transactions", format_number(total_txns), "#4A90D9", "📁 Kaggle ULB"),
                unsafe_allow_html=True,
            )
        with kpi_col2:
            st.markdown(
                create_kpi_card("Legitimate Txns", format_number(legit_txns), "#28a745", "Class 0"),
                unsafe_allow_html=True,
            )
        with kpi_col3:
            st.markdown(
                create_kpi_card("Fraudulent Txns", format_number(fraud_txns), "#dc3545", "Class 1"),
                unsafe_allow_html=True,
            )
        with kpi_col4:
            st.markdown(
                create_kpi_card("Fraud Imbalance Rate", f"{fraud_rate:.3f}%", "#ffc107", "Extreme Imbalance"),
                unsafe_allow_html=True,
            )
    else:
        st.info("Dataset statistics will appear here once `creditcard.csv` is loaded.")

    st.markdown("---")

    # Architecture & Features
    col_arch, col_info = st.columns([1, 1])

    with col_arch:
        st.markdown("### 🏗️ System Architecture")
        st.markdown(
            """
            ```
            ┌──────────────────────────────────────────────┐
            │   Raw Credit Card Transactions (284,807)     │
            └──────────────────────┬───────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────┐
            │  Stratified Splitting (70% Train, 15% V, 15% T)
            └──────────────────────┬───────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────┐
            │ Feature Pipeline (RobustScaler, Cyclical Time)│
            └──────────────────────┬───────────────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
            [Cost-Sensitive Learning]            [SMOTE Resampling]
             (Balanced Weights)                  (Train Fold Only)
             ┌─────────────────────┬─────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────┐
            │ Model Benchmarking (LR, RF, XGBoost)         │
            │ PR-AUC Prioritized Model Selection          │
            └──────────────────────┬───────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────┐
            │ Calibrated Probabilities & Threshold Tuning  │
            └──────────────────────┬───────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────┐
            │ SHAP Explainability & Risk Scoring Engine    │
            └──────────────────────────────────────────────┘
            ```
            """
        )

    with col_info:
        st.markdown("### 🎯 Key Capabilities")
        st.markdown(
            """
            - **Strict Leakage Prevention**: Stratified splitting executed before any scaling, encoding, or resampling.
            - **Imbalance Strategies**: Systematic comparison between cost-sensitive algorithms (`class_weight`, `scale_pos_weight`) and synthetic oversampling (`SMOTE`).
            - **PR-AUC Prioritization**: Evaluated on Precision-Recall AUC rather than misleading accuracy metrics.
            - **Probability Calibration**: Uses isotonic and Platt sigmoid calibration to transform raw outputs into reliable probabilities.
            - **Explainable AI (SHAP)**: Global beeswarm importance and individual transaction waterfall charts for regulatory transparency.
            - **Tiered Risk Policy**: Configurable 4-tier risk decision engine (Low, Medium, High, Critical) with actionable remediation suggestions.
            """
        )

    st.markdown("---")

    # Getting Started Guidance
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown(
            """
            1. **Download Dataset**:
               Ensure `data/raw/creditcard.csv` is present or download via `kagglehub.dataset_download('mlg-ulb/creditcardfraud')`.
            2. **Train Models**:
               Run `python src/models/benchmark.py` in your terminal to train and evaluate all models.
            3. **Explore EDA**:
               Use the **Exploratory Data Analysis** tab in the sidebar to review fraud distributions and time patterns.
            4. **Inspect Risk & XAI**:
               Navigate to **Transaction Risk Analysis** to test individual transactions or examine **Explainable AI** for SHAP feature drivers.
            """
        )

elif nav_selection == "📊 Exploratory Data Analysis":
    render_eda_tab(df)

elif nav_selection == "⚡ Model Benchmarking":
    render_benchmark_tab(models_dict.get("benchmark_results"))

elif nav_selection == "🔍 Transaction Risk Analysis":
    render_inference_tab(df, models_dict, config)

elif nav_selection == "💡 Explainable AI (SHAP)":
    render_xai_tab(df, models_dict, config)
