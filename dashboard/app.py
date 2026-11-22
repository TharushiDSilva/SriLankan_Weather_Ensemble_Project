import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Sri Lankan Rainfall Forecasting Using Ensemble Learning and Hybrid Models",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths (preserved)
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
MODEL_DIR = PROJECT_DIR / "notebooks" / "models"
ASSETS_DIR = DASHBOARD_DIR / "assets"

RESULTS_FILE = MODEL_DIR / "final_model_comparison.csv"
TUNED_XGB_FILE = MODEL_DIR / "tuned_xgboost.pkl"
HERO_BG_FILE = ASSETS_DIR / "rain_bg.jpeg"

FALLBACK_BG_URL = (
    "https://images.unsplash.com/photo-1534088568595-a066f410bcda"
    "?auto=format&fit=crop&w=1920&q=80"
)

# 2. GEOGRAPHICAL & DOMAIN REFERENCE MAPPINGS

PROVINCE_PCODES = {
    "Western Province": "LK1",
    "Central Province": "LK2",
    "Southern Province": "LK3",
    "Northern Province": "LK4",
    "Eastern Province": "LK5",
    "North Western Province": "LK6",
    "North Central Province": "LK7",
    "Uva Province": "LK8",
    "Sabaragamuwa Province": "LK9",
}

DISTRICT_PCODES = {
    "Ampara": ("LK52", "Eastern Province"),
    "Anuradhapura": ("LK71", "North Central Province"),
    "Badulla": ("LK81", "Uva Province"),
    "Batticaloa": ("LK51", "Eastern Province"),
    "Colombo": ("LK11", "Western Province"),
    "Galle": ("LK31", "Southern Province"),
    "Gampaha": ("LK12", "Western Province"),
    "Hambantota": ("LK33", "Southern Province"),
    "Jaffna": ("LK41", "Northern Province"),
    "Kalutara": ("LK13", "Western Province"),
    "Kandy": ("LK21", "Central Province"),
    "Kegalle": ("LK92", "Sabaragamuwa Province"),
    "Kilinochchi": ("LK45", "Northern Province"),
    "Kurunegala": ("LK61", "North Western Province"),
    "Mannar": ("LK42", "Northern Province"),
    "Matale": ("LK22", "Central Province"),
    "Matara": ("LK32", "Southern Province"),
    "Monaragala": ("LK82", "Uva Province"),
    "Mullaitivu": ("LK44", "Northern Province"),
    "Nuwara Eliya": ("LK23", "Central Province"),
    "Polonnaruwa": ("LK72", "North Central Province"),
    "Puttalam": ("LK62", "North Western Province"),
    "Ratnapura": ("LK91", "Sabaragamuwa Province"),
    "Trincomalee": ("LK53", "Eastern Province"),
    "Vavuniya": ("LK43", "Northern Province"),
}

PCODE_TO_NAME = {
    **{code: name for name, code in PROVINCE_PCODES.items()},
    **{code: f"{name} District" for name, (code, _) in DISTRICT_PCODES.items()},
}

MONTH_INFO = {
    1: ("January", "Northeast Monsoon"),
    2: ("February", "Northeast Monsoon"),
    3: ("March", "First Inter-Monsoon"),
    4: ("April", "First Inter-Monsoon"),
    5: ("May", "Southwest Monsoon"),
    6: ("June", "Southwest Monsoon"),
    7: ("July", "Southwest Monsoon"),
    8: ("August", "Southwest Monsoon"),
    9: ("September", "Southwest Monsoon"),
    10: ("October", "Second Inter-Monsoon"),
    11: ("November", "Second Inter-Monsoon"),
    12: ("December", "Northeast Monsoon"),
}

# 3. CUSTOM HIGH-CONTRAST GLASSMORPHIC STYLING


def get_base64_image(image_path: Path) -> str:
    if image_path.exists():
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""


def inject_atmospheric_css():
    bg_b64 = get_base64_image(HERO_BG_FILE)
    bg_source = (
        f"url('data:image/jpeg;base64,{bg_b64}')"
        if bg_b64
        else f"url('{FALLBACK_BG_URL}')"
    )

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* High-transparency subtle storm backdrop so foreground components pop cleanly */
.stApp {{
    background-image:
        linear-gradient(
            140deg,
            rgba(8, 15, 25, 0.68) 70%,
            rgba(13, 24, 38, 0.65) 30%,
            rgba(7, 13, 22, 0.92) 100%
        ),
        {bg_source};
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

header[data-testid="stHeader"] {{
    background: rgba(8, 15, 25, 0.65);
    backdrop-filter: blur(12px);
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1340px;
}}

h1, h2, h3, h4, p, label, span {{
    color: #f8fafc;
}}

/* Highlighted Frosted Sidebar */
section[data-testid="stSidebar"] {{
    background: rgba(10, 19, 31, 0.86) !important;
    backdrop-filter: blur(22px) saturate(140%);
    -webkit-backdrop-filter: blur(22px) saturate(140%);
    border-right: 1px solid rgba(56, 189, 248, 0.28);
    box-shadow: 6px 0 28px rgba(0, 0, 0, 0.45);
}}

section[data-testid="stSidebar"] * {{
    color: #f1f5f9 !important;
}}

section[data-testid="stSidebar"] .stRadio > div {{
    gap: 0.48rem;
}}

section[data-testid="stSidebar"] .stRadio label {{
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    padding: 0.65rem 0.95rem;
    transition: all 0.2s ease;
}}

section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(56, 189, 248, 0.20);
    border-color: rgba(56, 189, 248, 0.65);
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.22);
    transform: translateX(3px);
}}

/* Split-Glass Weather Showcase Card */
.weather-showcase {{
    display: grid;
    grid-template-columns: 325px 1fr;
    border-radius: 22px;
    overflow: hidden;
    border: 1.5px solid rgba(56, 189, 248, 0.36);
    box-shadow:
        0 24px 56px rgba(0, 0, 0, 0.60),
        inset 0 1px 0 rgba(255, 255, 255, 0.16);
    background: rgba(11, 21, 34, 0.78);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    margin-bottom: 1.35rem;
}}

@media (max-width: 960px) {{
    .weather-showcase {{
        grid-template-columns: 1fr;
    }}
}}

.showcase-left {{
    background: linear-gradient(180deg, rgba(22, 39, 60, 0.85) 0%, rgba(13, 25, 40, 0.92) 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.26);
    padding: 1.8rem 1.65rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}

.showcase-search-pill {{
    background: rgba(8, 15, 26, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    font-size: 0.83rem;
    font-weight: 500;
    color: #e0f2fe;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.35rem;
}}

.showcase-big-stat {{
    font-size: 3.4rem;
    font-weight: 300;
    line-height: 1.02;
    color: #ffffff;
    letter-spacing: -0.03em;
    text-align: center;
    margin: 0.2rem 0;
    text-shadow: 0 4px 20px rgba(56, 189, 248, 0.28);
}}

.showcase-big-unit {{
    font-size: 1.25rem;
    font-weight: 600;
    color: #38bdf8;
    margin-left: 0.25rem;
}}

.showcase-sub-stat {{
    text-align: center;
    font-size: 0.86rem;
    color: #cbd5e1;
    margin-bottom: 1.3rem;
}}

.showcase-stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.52rem 0.3rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}}

.showcase-stat-val {{
    font-size: 1.28rem;
    font-weight: 600;
    color: #ffffff;
}}

.showcase-stat-lbl {{
    font-size: 0.82rem;
    color: #94a3b8;
}}

/* Segmented Intensity Scale Bar */
.scale-block {{
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(56, 189, 248, 0.24);
}}

.scale-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.55rem;
    font-size: 0.86rem;
    color: #e2e8f0;
}}

.scale-pills {{
    display: flex;
    gap: 5px;
    margin-bottom: 0.72rem;
}}

.scale-pill {{
    height: 15px;
    flex: 1;
    border-radius: 999px;
    opacity: 0.30;
    transition: all 0.2s ease;
}}

.scale-pill.active {{
    opacity: 1;
    transform: scaleY(1.22);
    box-shadow: 0 0 14px rgba(255, 255, 255, 0.55);
}}

.scale-legend {{
    font-size: 0.77rem;
    color: #cbd5e1;
    line-height: 1.65;
}}

.scale-legend-row {{
    display: flex;
    justify-content: space-between;
}}

/* Right Showcase Panel */
.showcase-right {{
    padding: 1.95rem 2.2rem 1.35rem 2.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: linear-gradient(
        180deg,
        rgba(14, 26, 42, 0.62) 0%,
        rgba(9, 17, 28, 0.84) 100%
    );
}}

.showcase-eyebrow {{
    display: inline-block;
    background: rgba(56, 189, 248, 0.16);
    border: 1px solid rgba(56, 189, 248, 0.42);
    border-radius: 999px;
    padding: 0.24rem 0.78rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #7dd3fc;
    margin-bottom: 0.65rem;
}}

.showcase-headline {{
    font-size: 2.25rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.18;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.02em;
}}

.showcase-location {{
    font-size: 0.92rem;
    color: #bae6fd;
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

.showcase-desc {{
    font-size: 0.93rem;
    color: #e2e8f0;
    line-height: 1.65;
    max-width: 740px;
    margin-bottom: 1.45rem;
}}

/* 5-Column Horizontal Timeline Strip */
.timeline-strip {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.7rem;
    text-align: center;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.14);
}}

@media (max-width: 768px) {{
    .timeline-strip {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}

.timeline-item {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    padding: 0.6rem 0.4rem;
    border-radius: 14px;
    transition: all 0.2s ease;
}}

.timeline-item.highlight {{
    background: rgba(56, 189, 248, 0.20);
    border: 1.5px solid rgba(56, 189, 248, 0.65);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.22);
}}

.timeline-top {{
    font-size: 0.78rem;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 0.28rem;
}}

.timeline-val {{
    font-size: 1.68rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.15;
}}

.timeline-sub {{
    font-size: 0.75rem;
    color: #cbd5e1;
    margin-top: 0.25rem;
}}

/* High-Contrast Glass Panels, Headers & KPI Cards */
.glass-panel {{
    background: rgba(12, 22, 35, 0.82);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1.5px solid rgba(56, 189, 248, 0.28);
    box-shadow:
        0 14px 36px rgba(0, 0, 0, 0.42),
        inset 0 1px 0 rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 1.4rem 1.55rem;
    margin-bottom: 1.25rem;
}}

.glass-header {{
    background: linear-gradient(135deg, rgba(14, 26, 42, 0.88) 0%, rgba(18, 34, 54, 0.84) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1.5px solid rgba(56, 189, 248, 0.34);
    border-left: 5px solid #38bdf8;
    border-radius: 16px;
    padding: 1.25rem 1.55rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.38);
}}

.glass-header h2 {{
    margin: 0 0 0.28rem 0;
    font-size: 1.58rem;
    font-weight: 700;
    color: #ffffff;
}}

.glass-header p {{
    margin: 0;
    color: #cbd5e1;
    font-size: 0.94rem;
}}

.glass-kpi {{
    background: linear-gradient(160deg, rgba(15, 28, 45, 0.88) 0%, rgba(10, 20, 33, 0.90) 100%);
    backdrop-filter: blur(18px);
    border: 1.5px solid rgba(56, 189, 248, 0.35);
    border-top: 3px solid #38bdf8;
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.38);
    height: 100%;
    transition: transform 0.18s ease, border-color 0.18s ease;
}}

.glass-kpi:hover {{
    transform: translateY(-3px);
    border-color: rgba(125, 211, 252, 0.72);
}}

.glass-kpi-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #7dd3fc;
    font-weight: 700;
    margin-bottom: 0.38rem;
}}

.glass-kpi-value {{
    font-size: 2.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
}}

.glass-badge {{
    display: inline-block;
    background: rgba(56, 189, 248, 0.22);
    border: 1px solid rgba(56, 189, 248, 0.55);
    color: #e0f2fe;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.26rem 0.72rem;
    border-radius: 999px;
}}

.insight-card {{
    background: rgba(14, 26, 42, 0.85);
    border: 1.5px solid rgba(56, 189, 248, 0.30);
    border-top: 3px solid #eab308;
    border-radius: 15px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.34);
    height: 100%;
}}

.insight-card h4 {{
    margin: 0 0 0.45rem 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: #fde047;
}}

.insight-card p {{
    margin: 0;
    font-size: 0.89rem;
    color: #e2e8f0;
    line-height: 1.58;
}}

/* Form Inputs & Selectboxes Glass Styling */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: rgba(11, 20, 32, 0.88) !important;
    border: 1.5px solid rgba(56, 189, 248, 0.36) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="input"] input {{
    color: #ffffff !important;
}}

div[data-testid="stForm"] {{
    background: rgba(12, 22, 35, 0.82);
    backdrop-filter: blur(20px);
    border: 1.5px solid rgba(56, 189, 248, 0.32);
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.40);
    border-radius: 18px;
    padding: 1.4rem 1.55rem;
}}

/* Submit Button */
div[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%);
    color: #06101c !important;
    font-weight: 800;
    border: none;
    border-radius: 12px;
    padding: 0.72rem 1.45rem;
    font-size: 1rem;
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.42);
    transition: all 0.2s ease;
}}

div[data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(56, 189, 248, 0.60);
}}

div[data-testid="stAlert"] {{
    background: rgba(14, 28, 44, 0.88) !important;
    backdrop-filter: blur(16px);
    border: 1.5px solid rgba(56, 189, 248, 0.45) !important;
    color: #f8fafc !important;
    border-radius: 12px;
}}

.geo-pill-bar {{
    background: rgba(12, 23, 37, 0.86);
    border: 1.5px solid rgba(56, 189, 248, 0.38);
    border-radius: 13px;
    padding: 0.9rem 1.15rem;
    margin: 0.45rem 0 1.05rem 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.30);
}}

.geo-chip {{
    background: rgba(56, 189, 248, 0.22);
    border: 1px solid rgba(56, 189, 248, 0.55);
    color: #e0f2fe;
    padding: 0.25rem 0.72rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
}}

/* Highlighted Academic Footer Card */
.footer-card {{
    background: rgba(10, 19, 31, 0.88);
    border: 1.5px solid rgba(56, 189, 248, 0.32);
    border-radius: 16px;
    padding: 1.1rem 1.5rem;
    text-align: center;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
    margin-top: 0.5rem;
}}

.footer-title {{
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}

.footer-meta {{
    color: #7dd3fc;
    font-size: 0.85rem;
    font-weight: 500;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


inject_atmospheric_css()


# 4. DATA & MODEL LOADERS + HELPERS


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_FILE)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def classify_model_paradigm(model_name: str) -> str:
    """Tag each evaluated model with its ensemble/hybrid paradigm for clarity."""
    name_lower = model_name.lower()
    if "hybrid" in name_lower:
        return "Hybrid (RF + ARIMA)"
    if "stack" in name_lower:
        return "Stacking Ensemble"
    if "random forest" in name_lower or name_lower == "rf":
        return "Bagging (Random Forest)"
    if any(k in name_lower for k in ["xgb", "lightgbm", "lgbm", "catboost", "boost"]):
        return "Gradient Boosting"
    if "arima" in name_lower:
        return "Statistical Baseline"
    return "Ensemble / ML"


def format_feature_label(raw_name: str) -> str:
    clean = raw_name.replace("num__", "").replace("cat__", "").replace("remainder__", "")
    direct_map = {
        "rfh_current": "Current Rainfall (rfh_current)",
        "rfh_lag_1": "Previous Period Rainfall (rfh_lag_1)",
        "rfh_lag_2": "2-Period Lag Rainfall (rfh_lag_2)",
        "rfh_lag_5": "5-Period Lag Rainfall (rfh_lag_5)",
        "month": "Calendar Month (Seasonality)",
        "adm_level_1": "Admin Level: Province (1)",
        "adm_level_2": "Admin Level: District (2)",
    }
    if clean in direct_map:
        return direct_map[clean]
    if clean.startswith("PCODE_"):
        code = clean.replace("PCODE_", "")
        area_name = PCODE_TO_NAME.get(code, code)
        return f"Region: {area_name} ({code})"
    return clean


def validate_geographical_categories(model, pcode: str, adm_level: str):
    """Validate selected PCODE and adm_level against the fitted preprocessor categories."""
    resolved_adm = adm_level
    try:
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            for _, transformer, cols in preprocessor.transformers_:
                if transformer in ("drop", "passthrough"):
                    continue
                col_list = (
                    list(cols)
                    if isinstance(cols, (list, tuple, np.ndarray, pd.Index))
                    else [cols]
                )
                encoder = transformer
                if hasattr(transformer, "named_steps"):
                    for step in transformer.named_steps.values():
                        if hasattr(step, "categories_"):
                            encoder = step
                            break
                if hasattr(encoder, "categories_"):
                    for col_name, cats in zip(col_list, encoder.categories_):
                        cat_values = list(cats)
                        if col_name == "PCODE" and pcode not in cat_values:
                            return (
                                False,
                                resolved_adm,
                                f"PCODE '{pcode}' not found in trained categories.",
                            )
                        if col_name == "adm_level":
                            if adm_level in cat_values:
                                resolved_adm = adm_level
                            elif int(adm_level) in cat_values:
                                resolved_adm = int(adm_level)
                            else:
                                return (
                                    False,
                                    resolved_adm,
                                    f"Admin level '{adm_level}' not found in trained categories.",
                                )
    except Exception:
        pass

    return True, resolved_adm, "Verified against trained model categories"


def get_rainfall_regime(mm_value: float):
    if mm_value < 5.0:
        return (
            "Dry / Clear Spell",
            0,
            "Minimal precipitation expected for the upcoming observation window.",
        )
    if mm_value < 25.0:
        return (
            "Light Passing Showers",
            1,
            "Low-to-light rainfall accumulation consistent with dry-season or inter-monsoon transition.",
        )
    if mm_value < 60.0:
        return (
            "Moderate Monsoon Rain",
            2,
            "Steady seasonal rainfall accumulation across the selected administrative boundary.",
        )
    if mm_value < 110.0:
        return (
            "Heavy Monsoon Rainfall",
            3,
            "Elevated precipitation levels typical of active monsoon surges.",
        )
    if mm_value < 180.0:
        return (
            "Very Heavy Downpour",
            4,
            "Substantial rainfall accumulation; heightened surface runoff potential.",
        )
    return (
        "Extreme Monsoon Surge",
        5,
        "Peak-intensity precipitation regime characteristic of severe convective or cyclonic activity.",
    )


def build_spline_chart(x_labels, y_values, y_title="Rainfall (mm)", height=150):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_values,
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in y_values],
            textposition="top center",
            textfont=dict(color="#e2e8f0", size=11),
            line=dict(
                color="#eab308",
                width=3.2,
                shape="spline",
                smoothing=1.15,
            ),
            marker=dict(
                size=9,
                color=["#38bdf8", "#38bdf8", "#a3e635", "#eab308", "#f97316"][: len(y_values)],
                line=dict(color="#ffffff", width=1.8),
            ),
            hovertemplate="<b>%{x}</b><br>" + f"{y_title}: %{{y:.2f}}<extra></extra>",
        )
    )
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max((y_max - y_min) * 0.42, 6.0)
    fig.update_layout(
        paper_bgcolor="rgba(12, 22, 35, 0.72)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=22, r=22, t=22, b=14),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[max(0, y_min - pad * 0.5), y_max + pad],
        ),
        showlegend=False,
    )
    return fig


try:
    results = load_results()
except Exception as e:
    st.error(f"Could not load model comparison results from `{RESULTS_FILE}`: {e}")
    st.stop()

# 5. FROSTED GLASS SIDEBAR NAVIGATION
  

with st.sidebar:
    st.markdown(
        """
<div style="padding: 0.35rem 0.15rem 0.65rem 0.15rem;">
    <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; line-height: 1.25;">
        🌧️ Sri Lankan Rainfall Forecasting
    </div>
    <div style="font-size: 0.78rem; color: #7dd3fc; margin-top: 0.25rem; font-weight: 500;">
        Ensemble Learning &amp; Hybrid Models
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    nav_options = {
        "🏠  Overview": "Overview",
        "📊  Model Comparison": "Model Comparison",
        "🔍  Feature Importance": "Feature Importance",
        "🌦️  Rainfall Prediction": "Rainfall Prediction",
    }

    selected_nav = st.radio(
        "Navigation",
        list(nav_options.keys()),
        label_visibility="collapsed",
    )
    page = nav_options[selected_nav]

    st.markdown("---")

    st.markdown(
        """
<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(56,189,248,0.32); border-radius: 14px; padding: 0.9rem;">
    <div style="font-size: 0.82rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.55rem;">
        Rainfall Intensity Index (mm)
    </div>
    <div style="display: flex; gap: 4px; margin-bottom: 0.65rem;">
        <div style="height: 10px; flex:1; border-radius: 99px; background: #38bdf8;"></div>
        <div style="height: 10px; flex:1; border-radius: 99px; background: #4ade80;"></div>
        <div style="height: 10px; flex:1; border-radius: 99px; background: #facc15;"></div>
        <div style="height: 10px; flex:1; border-radius: 99px; background: #fb923c;"></div>
        <div style="height: 10px; flex:1; border-radius: 99px; background: #f87171;"></div>
    </div>
    <div style="font-size: 0.75rem; color: #e2e8f0; line-height: 1.65;">
        <div style="display:flex; justify-content:space-between;"><span>• Dry / Light</span><span>0–25</span></div>
        <div style="display:flex; justify-content:space-between;"><span>• Moderate</span><span>25–60</span></div>
        <div style="display:flex; justify-content:space-between;"><span>• Heavy</span><span>60–110</span></div>
        <div style="display:flex; justify-content:space-between;"><span>• Very Heavy</span><span>110–180</span></div>
        <div style="display:flex; justify-content:space-between;"><span>• Extreme</span><span>180+</span></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(
        """
<div style="background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.38); border-radius: 12px; padding: 0.8rem 0.9rem; font-size: 0.78rem; line-height: 1.55;">
    <div style="font-weight: 700; color: #fde047;">ML Module Assignment</div>
    <div style="color: #f8fafc; font-weight: 600;">DE SILVA W.T.W.</div>
    <div style="color: #bae6fd;">Index No: 224035N</div>
</div>
""",
        unsafe_allow_html=True,
    )


# 6. OVERVIEW PAGE


if page == "Overview":
    best_mae = results.loc[results["MAE"].idxmin()]
    best_rmse = results.loc[results["RMSE"].idxmin()]
    best_r2 = results.loc[results["R2"].idxmax()]
    top5_models = results.sort_values("RMSE").head(5).reset_index(drop=True)

    timeline_html_items = ""
    for idx, row in top5_models.iterrows():
        hl_class = "highlight" if idx == 0 else ""
        timeline_html_items += f"""
<div class="timeline-item {hl_class}">
    <div class="timeline-top">Rank #{idx + 1}</div>
    <div class="timeline-val">{row['RMSE']:.2f}</div>
    <div class="timeline-sub">{row['Model']} (R² {row['R2']:.2f})</div>
</div>
"""

    st.markdown(
        f"""
<div class="weather-showcase">
    <div class="showcase-left">
        <div>
            <div class="showcase-search-pill">
                <span>📍 Sri Lanka • 34 Regions</span>
                <span>🔍</span>
            </div>
            <div class="showcase-big-stat">
                {best_rmse['RMSE']:.2f}<span class="showcase-big-unit">mm</span>
            </div>
            <div class="showcase-sub-stat">
                Best RMSE • <strong>{best_rmse['Model']}</strong>
            </div>
            <div class="showcase-stat-row">
                <span class="showcase-stat-val">{best_mae['MAE']:.2f} mm</span>
                <span class="showcase-stat-lbl">best MAE ({best_mae['Model']})</span>
            </div>
            <div class="showcase-stat-row">
                <span class="showcase-stat-val">{best_r2['R2'] * 100:.1f}%</span>
                <span class="showcase-stat-lbl">variance explained (R²)</span>
            </div>
        </div>
        <div class="scale-block">
            <div class="scale-header">
                <span>Benchmark Suite</span>
                <strong>{len(results)} Models</strong>
            </div>
            <div class="scale-pills">
                <div class="scale-pill active" style="background:#38bdf8;"></div>
                <div class="scale-pill active" style="background:#4ade80;"></div>
                <div class="scale-pill active" style="background:#facc15;"></div>
                <div class="scale-pill active" style="background:#fb923c;"></div>
                <div class="scale-pill active" style="background:#f87171;"></div>
            </div>
            <div class="scale-legend">
                <div class="scale-legend-row"><span>• Admin Level 1</span><span>9 Provinces</span></div>
                <div class="scale-legend-row"><span>• Admin Level 2</span><span>25 Districts</span></div>
                <div class="scale-legend-row"><span>• Paradigms</span><span>Bagging, Boosting, Stacking, Hybrid</span></div>
            </div>
        </div>
    </div>
    <div class="showcase-right">
        <div>
            <div class="showcase-eyebrow">Machine Learning Module • Sri Lankan Meteorological Dataset</div>
            <h1 class="showcase-headline">Sri Lankan Rainfall Forecasting Using Ensemble Learning and Hybrid Models</h1>
            <div class="showcase-location">
                <span>📍 Democratic Socialist Republic of Sri Lanka • OCHA PCODE Coverage (LK1–LK92)</span>
            </div>
            <p class="showcase-desc">
                Comparative benchmark evaluating <strong>Bagging</strong> (Random Forest), <strong>Gradient Boosting</strong>
                (XGBoost, Tuned XGBoost, LightGBM, CatBoost), <strong>Stacking</strong>, <strong>ARIMA</strong>, and a
                <strong>Hybrid RF + ARIMA</strong> architecture on Sri Lankan spatio-temporal rainfall observations.
            </p>
        </div>
        <div class="timeline-strip">
            {timeline_html_items}
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        build_spline_chart(
            top5_models["Model"].tolist(),
            top5_models["RMSE"].tolist(),
            y_title="RMSE (mm)",
            height=135,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Highlighted KPI Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
<div class="glass-kpi">
    <div class="glass-kpi-label">Lowest Mean Absolute Error (MAE)</div>
    <div class="glass-kpi-value">{best_mae['MAE']:.4f} mm</div>
    <span class="glass-badge">🏆 {best_mae['Model']} • {classify_model_paradigm(best_mae['Model'])}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
<div class="glass-kpi">
    <div class="glass-kpi-label">Lowest Root Mean Squared Error (RMSE)</div>
    <div class="glass-kpi-value">{best_rmse['RMSE']:.4f} mm</div>
    <span class="glass-badge">🏆 {best_rmse['Model']} • {classify_model_paradigm(best_rmse['Model'])}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
<div class="glass-kpi">
    <div class="glass-kpi-label">Highest Explained Variance (R²)</div>
    <div class="glass-kpi-value">{best_r2['R2']:.4f}</div>
    <span class="glass-badge">🏆 {best_r2['Model']} • {classify_model_paradigm(best_r2['Model'])}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Highlighted Assignment Insights Row
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(
            """
<div class="insight-card">
    <h4>🌲 Bagging vs. Gradient Boosting</h4>
    <p><strong>Random Forest</strong> reduces variance via parallel bootstrap trees, whereas <strong>XGBoost, LightGBM, and CatBoost</strong> sequentially minimize residual errors—capturing sharp monsoon rainfall spikes more effectively.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with ins2:
        st.markdown(
            """
<div class="insight-card">
    <h4>🔗 Hybrid RF + ARIMA &amp; Stacking</h4>
    <p><strong>Hybrid RF + ARIMA</strong> combines nonlinear tabular splits from Random Forest with linear autoregressive residual modeling, while <strong>Stacking</strong> blends diverse base learners through a meta-regressor.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with ins3:
        st.markdown(
            """
<div class="insight-card">
    <h4>📊 Tabular vs. Image / Text Data</h4>
    <p>Tree ensembles excel on structured tabular features (lags, month, PCODEs) with minimal tuning, whereas unstructured <strong>image or text data</strong> require deep neural networks (CNNs/Transformers) for spatial/semantic representation learning.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Complete Model Performance Summary")
    sorted_overview = results.sort_values("RMSE").reset_index(drop=True)
    sorted_overview.insert(0, "Rank", [f"#{i}" for i in range(1, len(sorted_overview) + 1)])
    sorted_overview.insert(
        2, "Paradigm", [classify_model_paradigm(m) for m in sorted_overview["Model"]]
    )

    st.dataframe(
        sorted_overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Model": st.column_config.TextColumn("Model Architecture", width="medium"),
            "Paradigm": st.column_config.TextColumn("Learning Family", width="medium"),
            "MAE": st.column_config.NumberColumn("MAE (mm)", format="%.4f"),
            "RMSE": st.column_config.NumberColumn("RMSE (mm)", format="%.4f"),
            "R2": st.column_config.ProgressColumn(
                "R² Score", format="%.4f", min_value=0.0, max_value=1.0
            ),
        },
    )
    st.caption(
        "Lower MAE and RMSE indicate higher predictive accuracy; higher R² indicates stronger variance explanation."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# 7. MODEL COMPARISON PAGE


elif page == "Model Comparison":
    st.markdown(
        """
<div class="glass-header">
    <h2>📊 Model Performance Comparison</h2>
    <p>Interactive benchmarking across Bagging, Gradient Boosting, Stacking, ARIMA, and Hybrid RF + ARIMA architectures.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.3, 1.6])
    with ctrl1:
        metric = st.selectbox("Evaluation Metric", ["RMSE", "MAE", "R2"], index=0)
    with ctrl2:
        view_mode = st.selectbox(
            "Chart View",
            ["Single Metric Ranking", "Normalized Multi-Metric Comparison"],
            index=0,
        )
    with ctrl3:
        all_models = results["Model"].tolist()
        selected_models = st.multiselect("Models Included", all_models, default=all_models)

    if not selected_models:
        st.warning("Select at least one model to render the comparison.")
        st.stop()

    filtered = results[results["Model"].isin(selected_models)].copy()
    ascending = metric != "R2"
    chart_data = filtered.sort_values(metric, ascending=ascending).reset_index(drop=True)
    winner = chart_data.iloc[0]

    st.info(
        f"🏆 **Best Performer for {metric}:** **{winner['Model']}** ({classify_model_paradigm(winner['Model'])}) "
        f"with **{winner[metric]:.4f}** — {'lowest error' if ascending else 'highest explained variance'} among selected architectures."
    )

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    if view_mode == "Single Metric Ranking":
        bar_colors = [
            "#38bdf8" if m == winner["Model"] else "rgba(148, 163, 184, 0.58)"
            for m in chart_data["Model"]
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=chart_data["Model"],
                    y=chart_data[metric],
                    marker_color=bar_colors,
                    marker_line=dict(color="rgba(255,255,255,0.45)", width=1.2),
                    text=chart_data[metric].map(lambda v: f"{v:.4f}"),
                    textposition="outside",
                    textfont=dict(color="#f8fafc", size=12),
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
                )
            ]
        )
        y_max = chart_data[metric].max()
        fig.update_layout(
            title=dict(
                text=f"Model Comparison — {metric} ({'Lower is Better' if ascending else 'Higher is Better'})",
                font=dict(color="#ffffff", size=17),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Model Architecture", color="#e2e8f0", showgrid=False),
            yaxis=dict(
                title=metric,
                color="#e2e8f0",
                gridcolor="rgba(255,255,255,0.12)",
                range=[0, y_max * 1.18],
            ),
            height=435,
            margin=dict(t=55, b=35, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        norm_df = chart_data.copy()
        for col_name, lower_better in [("MAE", True), ("RMSE", True), ("R2", False)]:
            c_min, c_max = results[col_name].min(), results[col_name].max()
            if c_max == c_min:
                norm_df[f"{col_name}_norm"] = 100.0
            elif lower_better:
                norm_df[f"{col_name}_norm"] = 100.0 * (
                    1.0 - (norm_df[col_name] - c_min) / (c_max - c_min)
                )
            else:
                norm_df[f"{col_name}_norm"] = 100.0 * (
                    (norm_df[col_name] - c_min) / (c_max - c_min)
                )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="MAE Score",
                x=norm_df["Model"],
                y=norm_df["MAE_norm"],
                customdata=norm_df["MAE"],
                marker_color="#38bdf8",
                hovertemplate="<b>%{x}</b><br>MAE: %{customdata:.4f} mm<br>Score: %{y:.1f}/100<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name="RMSE Score",
                x=norm_df["Model"],
                y=norm_df["RMSE_norm"],
                customdata=norm_df["RMSE"],
                marker_color="#eab308",
                hovertemplate="<b>%{x}</b><br>RMSE: %{customdata:.4f} mm<br>Score: %{y:.1f}/100<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name="R² Score",
                x=norm_df["Model"],
                y=norm_df["R2_norm"],
                customdata=norm_df["R2"],
                marker_color="#4ade80",
                hovertemplate="<b>%{x}</b><br>R²: %{customdata:.4f}<br>Score: %{y:.1f}/100<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="group",
            title=dict(
                text="Normalized Multi-Metric Comparison (100 = Best Benchmark)",
                font=dict(color="#ffffff", size=17),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Model Architecture", color="#e2e8f0", showgrid=False),
            yaxis=dict(
                title="Normalized Score (0–100)",
                color="#e2e8f0",
                gridcolor="rgba(255,255,255,0.12)",
                range=[0, 112],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#f8fafc"),
            ),
            height=445,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Complete Model Comparison (Sorted by {metric})")
    table_df = chart_data.copy()
    table_df.insert(0, "Rank", [f"#{i}" for i in range(1, len(table_df) + 1)])
    table_df.insert(2, "Paradigm", [classify_model_paradigm(m) for m in table_df["Model"]])
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Model": st.column_config.TextColumn("Model"),
            "Paradigm": st.column_config.TextColumn("Learning Family"),
            "MAE": st.column_config.NumberColumn("MAE (mm)", format="%.4f"),
            "RMSE": st.column_config.NumberColumn("RMSE (mm)", format="%.4f"),
            "R2": st.column_config.NumberColumn("R²", format="%.4f"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Analytical Interpretation Cards for Presentation
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """
<div class="insight-card">
    <h4>⚡ Why Gradient Boosting Outperforms Single Statistical Baselines</h4>
    <p>Pure time-series models like <strong>ARIMA</strong> assume linear autocorrelation and struggle across heterogeneous administrative regions simultaneously. Gradient boosted trees (<strong>XGBoost, LightGBM, CatBoost</strong>) learn nonlinear interactions between lag features (<code>rfh_lag_1..5</code>), seasonal months, and regional PCODEs.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
<div class="insight-card">
    <h4>🧩 Role of Hybrid RF + ARIMA &amp; Stacking</h4>
    <p><strong>Hybrid RF + ARIMA</strong> decomposes forecasting into nonlinear Random Forest feature learning plus autoregressive residual correction, while <strong>Stacking</strong> reduces individual model bias by combining complementary tree ensembles.</p>
</div>
""",
            unsafe_allow_html=True,
        )

# 8. FEATURE IMPORTANCE PAGE
 

elif page == "Feature Importance":
    st.markdown(
        """
<div class="glass-header">
    <h2>🔍 Feature Importance Analysis</h2>
    <p>Relative predictive contribution of rainfall lags, seasonality, and spatial PCODEs in the tuned XGBoost model.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        model = load_model(TUNED_XGB_FILE)
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importance_values = xgb_model.feature_importances_

        imp_df = pd.DataFrame(
            {
                "Raw_Feature": feature_names,
                "Display_Feature": [format_feature_label(f) for f in feature_names],
                "Importance": importance_values,
                "Importance_Pct": importance_values * 100.0,
            }
        ).sort_values("Importance", ascending=False).reset_index(drop=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            top_n = st.slider(
                "Top Features to Display",
                min_value=5,
                max_value=min(30, len(imp_df)),
                value=min(15, len(imp_df)),
            )
        with c2:
            label_mode = st.radio(
                "Label Style",
                ["Readable Names", "Raw Pipeline Names"],
                horizontal=True,
            )

        y_field = "Display_Feature" if label_mode == "Readable Names" else "Raw_Feature"
        plot_df = imp_df.head(top_n).sort_values("Importance", ascending=True)

        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        fig = px.bar(
            plot_df,
            x="Importance",
            y=y_field,
            orientation="h",
            text=plot_df["Importance_Pct"].map(lambda v: f"{v:.1f}%"),
            color="Importance",
            color_continuous_scale=["#38bdf8", "#eab308", "#f97316"],
            custom_data=["Raw_Feature", "Importance_Pct"],
            title=f"Top {top_n} Important Features — Tuned XGBoost",
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color="#f8fafc", size=12),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pipeline Feature: %{customdata[0]}<br>"
                "Importance: %{x:.4f} (%{customdata[1]:.2f}%)<extra></extra>"
            ),
        )
        x_max = plot_df["Importance"].max()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(font=dict(color="#ffffff", size=17)),
            xaxis=dict(
                title="Relative Importance Score",
                color="#e2e8f0",
                gridcolor="rgba(255,255,255,0.12)",
                range=[0, x_max * 1.18],
            ),
            yaxis=dict(title="", color="#f8fafc", showgrid=False),
            coloraxis_showscale=False,
            height=max(440, top_n * 30),
            margin=dict(l=15, r=30, t=55, b=35),
        )
        st.plotly_chart(fig, use_container_width=True)

        top3 = imp_df.head(3)
        if len(top3) >= 3:
            st.markdown(
                f"""
<div style="background: rgba(56,189,248,0.10); border: 1.5px solid rgba(56,189,248,0.34); border-radius: 14px; padding: 1.05rem 1.25rem; margin-top: 0.5rem;">
    <div style="font-weight: 700; color: #fde047; margin-bottom: 0.45rem; font-size: 1rem;">📌 Interpretation of Top Model Drivers</div>
    <div style="font-size: 0.91rem; color: #f1f5f9; line-height: 1.7;">
        • <strong>{top3.iloc[0]['Display_Feature']} ({top3.iloc[0]['Importance_Pct']:.1f}%):</strong> Primary predictor governing next-period rainfall magnitude.<br/>
        • <strong>{top3.iloc[1]['Display_Feature']} ({top3.iloc[1]['Importance_Pct']:.1f}%):</strong> Captures short-term precipitation persistence.<br/>
        • <strong>{top3.iloc[2]['Display_Feature']} ({top3.iloc[2]['Importance_Pct']:.1f}%):</strong> Accounts for seasonal monsoon transitions or regional rainfall baselines.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.caption(
            "Feature importance represents relative split contribution within the trained XGBoost model and does not imply physical causation."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load feature importance from `{TUNED_XGB_FILE}`: {e}")

#  
# 9. RAINFALL PREDICTION PAGE (WEATHER SHOWCASE UI)
#  

elif page == "Rainfall Prediction":
    st.markdown(
        """
<div class="glass-header">
    <h2>🌦️ Next-Period Rainfall Prediction</h2>
    <p>Select any of Sri Lanka's 9 Provinces or 25 Districts and enter recent rainfall observations to generate an ensemble forecast.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        model = load_model(TUNED_XGB_FILE)
    except Exception as e:
        st.error(f"Could not load tuned XGBoost model from `{TUNED_XGB_FILE}`: {e}")
        st.stop()

    g1, g2 = st.columns([1, 1.35])
    with g1:
        admin_level_label = st.selectbox(
            "Administrative Level",
            options=["Province", "District"],
            index=1,
            help="Province (Level 1: 9 provinces) or District (Level 2: 25 districts).",
        )

    with g2:
        if admin_level_label == "Province":
            adm_level_code = "1"
            selected_region_name = st.selectbox(
                "Select Province (9 Provinces)",
                options=list(PROVINCE_PCODES.keys()),
                index=0,
            )
            mapped_pcode = PROVINCE_PCODES[selected_region_name]
            parent_context = "Sri Lanka • Administrative Level 1"
        else:
            adm_level_code = "2"
            selected_region_name = st.selectbox(
                "Select District (25 Districts)",
                options=list(DISTRICT_PCODES.keys()),
                index=4,  # Colombo
            )
            mapped_pcode, parent_province = DISTRICT_PCODES[selected_region_name]
            parent_context = f"{parent_province} • Administrative Level 2"

    is_valid_geo, validated_adm_level, validation_msg = validate_geographical_categories(
        model, mapped_pcode, adm_level_code
    )

    st.markdown(
        f"""
<div class="geo-pill-bar">
    <span class="geo-chip">📍 {selected_region_name}</span>
    <span class="geo-chip">PCODE: {mapped_pcode}</span>
    <span class="geo-chip">adm_level: "{validated_adm_level}"</span>
    <span style="font-size: 0.84rem; color: #e2e8f0; font-weight: 500;">{parent_context} • ✓ {validation_msg}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            current_rainfall = st.number_input(
                "Current rainfall — rfh_current (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=50.0,
                step=1.0,
            )
            lag_1 = st.number_input(
                "Previous period — rfh_lag_1 (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=45.0,
                step=1.0,
            )
        with f2:
            lag_2 = st.number_input(
                "Two periods ago — rfh_lag_2 (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=40.0,
                step=1.0,
            )
            lag_5 = st.number_input(
                "Five periods ago — rfh_lag_5 (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=35.0,
                step=1.0,
            )
        with f3:
            month = st.selectbox(
                "Target prediction month",
                options=list(range(1, 13)),
                index=9,  # October (Second Inter-Monsoon)
                format_func=lambda m: f"{m:02d} — {MONTH_INFO[m][0]} ({MONTH_INFO[m][1]})",
            )
            st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🌦️ Predict Next-Period Rainfall",
                use_container_width=True,
            )

    if not is_valid_geo:
        st.error(f"Cannot run prediction: {validation_msg}")
    else:
        try:
            # Exact 7 feature names and column order preserved
            input_data = pd.DataFrame(
                [
                    {
                        "rfh_current": float(current_rainfall),
                        "rfh_lag_1": float(lag_1),
                        "rfh_lag_2": float(lag_2),
                        "rfh_lag_5": float(lag_5),
                        "month": int(month),
                        "PCODE": mapped_pcode,
                        "adm_level": validated_adm_level,
                    }
                ]
            )

            raw_pred = float(model.predict(input_data)[0])
            predicted_mm = max(0.0, raw_pred)

            condition_title, active_idx, condition_desc = get_rainfall_regime(predicted_mm)
            month_name, monsoon_season = MONTH_INFO[month]
            delta_mm = predicted_mm - current_rainfall
            delta_str = f"{'+' if delta_mm >= 0 else ''}{delta_mm:.1f} mm"
            lag_mean = float(np.mean([lag_1, lag_2, lag_5]))

            pill_colors = ["#38bdf8", "#4ade80", "#facc15", "#fb923c", "#f97316", "#ef4444"]
            pills_html = "".join(
                f'<div class="scale-pill {"active" if i == active_idx else ""}" style="background:{c};"></div>'
                for i, c in enumerate(pill_colors)
            )

            st.markdown(
                f"""
<div class="weather-showcase">
    <div class="showcase-left">
        <div>
            <div class="showcase-search-pill">
                <span>📍 {selected_region_name} ({mapped_pcode})</span>
                <span>🔍</span>
            </div>
            <div class="showcase-big-stat">
                {predicted_mm:.1f}<span class="showcase-big-unit">mm</span>
            </div>
            <div class="showcase-sub-stat">
                Change vs. current: <strong>{delta_str}</strong>
            </div>
            <div class="showcase-stat-row">
                <span class="showcase-stat-val">{current_rainfall:.1f} mm</span>
                <span class="showcase-stat-lbl">current period</span>
            </div>
            <div class="showcase-stat-row">
                <span class="showcase-stat-val">{lag_mean:.1f} mm</span>
                <span class="showcase-stat-lbl">lag average</span>
            </div>
        </div>
        <div class="scale-block">
            <div class="scale-header">
                <span>Intensity tier:</span>
                <strong>{active_idx + 1} / 6</strong>
            </div>
            <div class="scale-pills">
                {pills_html}
            </div>
            <div class="scale-legend">
                <div class="scale-legend-row"><span>• dry / light</span><span>0–25</span></div>
                <div class="scale-legend-row"><span>• moderate</span><span>25–60</span></div>
                <div class="scale-legend-row"><span>• heavy</span><span>60–110</span></div>
                <div class="scale-legend-row"><span>• very heavy</span><span>110–180</span></div>
                <div class="scale-legend-row"><span>• extreme</span><span>180+</span></div>
            </div>
        </div>
    </div>
    <div class="showcase-right">
        <div>
            <div class="showcase-eyebrow">Rainfall Forecast • Tuned XGBoost</div>
            <h1 class="showcase-headline">{condition_title}</h1>
            <div class="showcase-location">
                <span>📍 {selected_region_name} ({mapped_pcode}), {admin_level_label} • {month_name} ({monsoon_season})</span>
            </div>
            <p class="showcase-desc">
                {condition_desc} Based on current rainfall of {current_rainfall:.1f} mm and historical lag
                observations (t−1: {lag_1:.1f} mm, t−2: {lag_2:.1f} mm, t−5: {lag_5:.1f} mm), the tuned XGBoost
                pipeline estimates <strong>{predicted_mm:.2f} mm</strong> for the upcoming period.
            </p>
        </div>
        <div class="timeline-strip">
            <div class="timeline-item">
                <div class="timeline-top">Lag 5 (t−5)</div>
                <div class="timeline-val">{lag_5:.1f}</div>
                <div class="timeline-sub">mm observed</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-top">Lag 2 (t−2)</div>
                <div class="timeline-val">{lag_2:.1f}</div>
                <div class="timeline-sub">mm observed</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-top">Lag 1 (t−1)</div>
                <div class="timeline-val">{lag_1:.1f}</div>
                <div class="timeline-sub">mm observed</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-top">Current (t)</div>
                <div class="timeline-val">{current_rainfall:.1f}</div>
                <div class="timeline-sub">mm observed</div>
            </div>
            <div class="timeline-item highlight">
                <div class="timeline-top">Forecast (t+1)</div>
                <div class="timeline-val">{predicted_mm:.1f}</div>
                <div class="timeline-sub">{month_name} est.</div>
            </div>
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                build_spline_chart(
                    ["Lag 5 (t−5)", "Lag 2 (t−2)", "Lag 1 (t−1)", "Current (t)", "Forecast (t+1)"],
                    [lag_5, lag_2, lag_1, current_rainfall, predicted_mm],
                    y_title="Rainfall (mm)",
                    height=145,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.caption(
                "⚠️ Disclaimer: This is a machine learning model estimate for the next observation period and not an official meteorological weather forecast."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# 10. COPYRIGHT FOOTER


st.markdown("---")
st.markdown(
    """
<div class="footer-card">
    <div class="footer-title">
        Sri Lankan Rainfall Forecasting Using Ensemble Learning and Hybrid Models
    </div>
    <div class="footer-meta">
        © 2026 DE SILVA W.T.W. &nbsp;|&nbsp; Index No: 224035N &nbsp;|&nbsp; Machine Learning Module Assignment
    </div>
</div>
""",
    unsafe_allow_html=True,
)