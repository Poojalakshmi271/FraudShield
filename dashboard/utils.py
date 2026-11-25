"""
FraudShield Dashboard Utilities
================================
Helper functions, CSS theme injection, caching, and config loading
for the FraudShield Streamlit dashboard.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Optional

import pandas as pd
import streamlit as st
import yaml
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "⚠️ DEMONSTRATION RISK CLASSIFICATIONS – SIMULATION ONLY. "
    "This system is for research and educational purposes. "
    "It is NOT a certified financial decision system and must NOT be used "
    "as the sole basis for any real-world financial or compliance decisions. "
    "Always consult qualified fraud analysts and comply with applicable regulations."
)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load project configuration from config/config.yaml.

    Returns the parsed YAML as a plain dict.  Falls back to an empty dict
    when the file is missing so callers can handle absence gracefully.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s – using defaults.", config_path)
        return {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse config.yaml: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> Optional[pd.DataFrame]:
    """Load the raw credit-card transaction dataset.

    Returns a pd.DataFrame when successful, or None when the CSV is absent.
    Result is cached with st.cache_data so it is only read once per session.
    """
    config = load_config()
    raw_path = config.get("data", {}).get("raw_path", "data/raw/creditcard.csv")

    # Resolve relative to project root (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(__file__))
    abs_path = os.path.join(project_root, raw_path)

    if not os.path.exists(abs_path):
        logger.warning("Dataset CSV not found at %s", abs_path)
        return None

    try:
        df = pd.read_csv(abs_path)
        logger.info("Dataset loaded: %s rows × %s cols", *df.shape)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Error reading dataset: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_models() -> dict:
    """Load all saved model artefacts from the models/ directory.

    Returns a dict with keys:
        - preprocessor      : FraudPreprocessor | None
        - best_model        : sklearn estimator  | None
        - calibrated_model  : sklearn estimator  | None
        - benchmark_results : dict               | None

    Missing files are stored as None so callers can check gracefully.
    """
    config = load_config()
    storage = config.get("model_storage", {})
    base = storage.get("base_path", "models/")

    project_root = os.path.dirname(os.path.dirname(__file__))
    base_abs = os.path.join(project_root, base)

    names = {
        "preprocessor": storage.get("preprocessor_name", "preprocessor.joblib"),
        "best_model": storage.get("best_model_name", "best_model.joblib"),
        "calibrated_model": storage.get("calibrated_model_name", "calibrated_model.joblib"),
        "benchmark_results": storage.get("benchmark_results_name", "benchmark_results.joblib"),
    }

    models: dict = {}
    for key, filename in names.items():
        path = os.path.join(base_abs, filename)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
                logger.info("Loaded %s from %s", key, path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", key, exc)
                models[key] = None
        else:
            models[key] = None

    return models


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def format_number(n) -> str:
    """Format an integer or float with thousands separators.

    Examples
    --------
    >>> format_number(1234567)
    '1,234,567'
    >>> format_number(284807)
    '284,807'
    """
    try:
        if isinstance(n, float):
            if n == int(n):
                return f"{int(n):,}"
            return f"{n:,.2f}"
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


# ---------------------------------------------------------------------------
# KPI card HTML
# ---------------------------------------------------------------------------


def create_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    color: str = "#1E3A5F",
) -> str:
    """Generate HTML markup for a styled KPI card.

    Parameters
    ----------
    title   : Card label (shown above the value)
    value   : Primary metric value (string already formatted by caller)
    delta   : Optional sub-label / change indicator
    color   : Background colour (hex); defaults to dark-navy #1E3A5F

    Returns the raw HTML string.  Callers should render it with
    ``st.markdown(html, unsafe_allow_html=True)``.
    """
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="kpi-delta">{delta}</div>'

    html = f"""
<div class="kpi-card" style="background: {color};">
    <div class="kpi-title">{title}</div>
    <div class="kpi-value">{value}</div>
    {delta_html}
</div>
"""
    return html


# ---------------------------------------------------------------------------
# Custom CSS theme
# ---------------------------------------------------------------------------


def apply_custom_css() -> None:
    """Inject a dark navy/blue Streamlit CSS theme.

    Colours
    -------
    Primary background : #0F1923
    Card background    : #1A2B3C
    Primary accent     : #1E3A5F
    Blue accent        : #4A90D9
    Text               : #FFFFFF / #E8EDF2
    """
    css = """
<style>
/* ── Global base ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0F1923 !important;
    color: #E8EDF2 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stHeader"] {
    background-color: #0F1923 !important;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #111F2E !important;
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * {
    color: #E8EDF2 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #A8C0D6 !important;
    font-size: 0.85rem;
}

/* ── Main content area ──────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    background-color: #0F1923 !important;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* ── Headings ───────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}
h1 { font-size: 1.9rem; font-weight: 700; }
h2 { font-size: 1.5rem; font-weight: 600; }
h3 { font-size: 1.2rem; font-weight: 600; color: #4A90D9 !important; }

/* ── Tabs ───────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #111F2E;
    border-bottom: 2px solid #1E3A5F;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: #1A2B3C;
    color: #A8C0D6 !important;
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #1E3A5F !important;
    color: #FFFFFF !important;
    border-bottom: 2px solid #4A90D9 !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background-color: #1E3A5F !important;
    color: #FFFFFF !important;
    border: 1px solid #4A90D9 !important;
    border-radius: 6px !important;
    padding: 0.4rem 1.2rem !important;
    font-weight: 500 !important;
    transition: background 0.2s ease;
}
.stButton > button:hover {
    background-color: #4A90D9 !important;
    color: #FFFFFF !important;
}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1A2B3C;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #A8C0D6 !important; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricDelta"] { color: #4A90D9 !important; }

/* ── DataFrames ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    overflow: hidden;
}
.dvn-scroller { background-color: #1A2B3C !important; }

/* ── Selectboxes / inputs ───────────────────────────────── */
[data-baseweb="select"] {
    background-color: #1A2B3C !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] * { color: #E8EDF2 !important; }
.stSlider [data-baseweb="slider"] { background-color: #1E3A5F; }

/* ── Expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #1A2B3C !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #4A90D9 !important; }

/* ── Info / Warning / Success boxes ────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left: 4px solid #4A90D9 !important;
}

/* ── Dividers ───────────────────────────────────────────── */
hr { border-color: #1E3A5F !important; }

/* ── KPI card component ─────────────────────────────────── */
.kpi-card {
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    border: 1px solid #4A90D9;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #A8C0D6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1;
}
.kpi-delta {
    margin-top: 6px;
    font-size: 0.82rem;
    color: #4A90D9;
    font-weight: 500;
}

/* ── Risk badges ────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.risk-low      { background: #0d3321; color: #28a745; border: 1px solid #28a745; }
.risk-medium   { background: #332b00; color: #ffc107; border: 1px solid #ffc107; }
.risk-high     { background: #331800; color: #fd7e14; border: 1px solid #fd7e14; }
.risk-critical { background: #330a0a; color: #dc3545; border: 1px solid #dc3545; }

/* ── Section headers ────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #1E3A5F 0%, #0F1923 100%);
    padding: 10px 16px;
    border-left: 4px solid #4A90D9;
    border-radius: 4px;
    margin-bottom: 12px;
    font-weight: 600;
    color: #FFFFFF;
}

/* ── Disclaimer box ─────────────────────────────────────── */
.disclaimer-box {
    background-color: #1A1A0A;
    border: 1px solid #ffc107;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 12px 16px;
    color: #ffc107;
    font-size: 0.82rem;
    margin-top: 16px;
}

/* ── Plotly chart containers ────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background-color: #1A2B3C;
    border-radius: 8px;
    border: 1px solid #1E3A5F;
    padding: 4px;
}

/* ── Progress / spinner ─────────────────────────────────── */
[data-testid="stSpinner"] { color: #4A90D9 !important; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F1923; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4A90D9; }
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Feature name helper
# ---------------------------------------------------------------------------


def get_feature_names(preprocessor) -> list:
    """Return ordered list of feature names from a fitted FraudPreprocessor.

    Falls back to a canonical default list when the preprocessor is None
    or does not expose a ``get_feature_names()`` method.
    """
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names"):
        try:
            return list(preprocessor.get_feature_names())
        except Exception:  # noqa: BLE001
            pass

    # Default canonical feature order matching FraudPreprocessor output
    v_features = [f"V{i}" for i in range(1, 29)]
    return ["Amount_scaled", "hour_sin", "hour_cos"] + v_features
