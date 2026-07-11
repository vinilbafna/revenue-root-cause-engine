"""
app/app.py

Streamlit app for the Revenue Root Cause Diagnostic Engine.

Run from project root with the venv active:
    streamlit run app/app.py
"""

import sys
import os
from pathlib import Path

# Ensure the project root (not app/) is on the import path, so
# "from src.xxx import ..." resolves correctly regardless of how
# Streamlit was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detector import detect_revenue_anomalies
from src.decompose_drop import decompose_revenue_drop
from src.drilldown import run_full_drilldown, find_combination_effects, DIMENSIONS
from src.memo_generator import generate_memo, polish_memo_with_claude

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
WEEKLY_PATH = PROCESSED_DIR / "weekly_revenue.parquet"
MERGED_PATH = PROCESSED_DIR / "merged_orders.parquet"

st.set_page_config(
    page_title="Revenue Root Cause Diagnostic Engine",
    page_icon="◈",
    layout="wide",
)

BG = "#0B1220"
SURFACE = "#121A2B"
BORDER = "#223049"
TEXT = "#E8ECF4"
MUTED = "#8592AD"
ACCENT = "#3DD9C2"
SEVERE = "#EF5B5B"
MODERATE = "#F2A65A"
STABLE = "#59C48B"
CREDIBLE_GRAY = "#3A4A66"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {BG};
        color: {TEXT};
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stHeader"] {{ background-color: transparent; }}
    [data-testid="stToolbar"] {{ right: 1rem; }}

    h1, h2, h3, h4 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {TEXT} !important;
        letter-spacing: -0.01em;
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 0.35rem;
    }}

    .masthead {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        padding: 1.1rem 1.4rem;
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-bottom: 1.6rem;
    }}
    .masthead-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {TEXT};
    }}
    .chip-row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
    .chip {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        padding: 0.32rem 0.7rem;
        border-radius: 20px;
        border: 1px solid {BORDER};
        color: {MUTED};
        background: {BG};
        white-space: nowrap;
    }}
    .chip b {{ color: {TEXT}; font-weight: 600; }}

    .metric-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid var(--accent-color, {ACCENT});
        border-radius: 8px;
        padding: 0.95rem 1.1rem;
        height: 100%;
    }}
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 600;
        color: {TEXT};
    }}

    .severity-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 0.22rem 0.6rem;
        border-radius: 5px;
        display: inline-block;
    }}

    .memo-box {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left-width: 4px;
        border-radius: 8px;
        padding: 1.3rem 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
        line-height: 1.65;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: {TEXT};
    }}

    .combo-box {{
        background: {SURFACE};
        border: 1px solid {ACCENT};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: {TEXT};
        font-size: 0.95rem;
    }}

    .section-gap {{ margin-top: 2.1rem; }}

    .footer-note {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: {MUTED};
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER};
    }}

    [data-testid="stSelectbox"] label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: {MUTED};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_dark_layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    base_xaxis = dict(gridcolor=BORDER, zerolinecolor=BORDER)
    base_yaxis = dict(gridcolor=BORDER, zerolinecolor=BORDER)
    if "xaxis" in kwargs:
        base_xaxis.update(kwargs.pop("xaxis"))
    if "yaxis" in kwargs:
        base_yaxis.update(kwargs.pop("yaxis"))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        xaxis=base_xaxis,
        yaxis=base_yaxis,
        **kwargs,
    )
    return fig


@st.cache_data
def load_weekly() -> pd.DataFrame:
    return pd.read_parquet(WEEKLY_PATH)


@st.cache_data
def load_merged() -> pd.DataFrame:
    return pd.read_parquet(MERGED_PATH)


@st.cache_data
def compute_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    return detect_revenue_anomalies(weekly)


@st.cache_data
def precompute_all_memos(_weekly: pd.DataFrame, _merged: pd.DataFrame, _anomalies: pd.DataFrame) -> dict:
    """
    Pre-computes decomposition, drilldown, and memo for every flagged
    anomaly week ONCE at load time, instead of recomputing on every
    dropdown click - the dataset is static, so there's no reason to pay
    this cost repeatedly for the same inputs.
    """
    flagged = _anomalies[_anomalies["severity"].isin(["moderate_anomaly", "severe_anomaly"])]
    results = {}
    for _, arow in flagged.iterrows():
        week_start = arow["week_start"]
        decomp = decompose_revenue_drop(_weekly, week_start)
        if not decomp.get("has_baseline", False):
            continue
        drilldown_results = run_full_drilldown(_merged, week_start)
        combo_result = find_combination_effects(
            _merged, week_start, "product_category_name_english", "payment_type_mode"
        )
        raw_memo = generate_memo(arow, decomp, drilldown_results, combo_result)
        results[week_start] = {
            "anomaly_row": arow,
            "decomposition": decomp,
            "drilldown": drilldown_results,
            "combo": combo_result,
            "raw_memo": raw_memo,
        }
    return results


@st.cache_data
def get_polished_memo(raw_memo: str) -> str:
    return polish_memo_with_claude(raw_memo)


# Load data and precompute memos
weekly = load_weekly()
merged = load_merged()
anomalies = compute_anomalies(weekly)
memo_data = precompute_all_memos(weekly, merged, anomalies)
flagged_weeks = sorted(memo_data.keys())

severe_count = sum(1 for w in flagged_weeks if memo_data[w]["anomaly_row"]["severity"] == "severe_anomaly")
moderate_count = len(flagged_weeks) - severe_count
date_min = weekly["week_start"].min().strftime("%b %Y")
date_max = weekly["week_start"].max().strftime("%b %Y")

st.markdown(
    f"""
    <div class="masthead">
        <div class="masthead-title">◈ Revenue Root Cause Diagnostic Engine</div>
        <div class="chip-row">
            <div class="chip">{date_min} → {date_max}</div>
            <div class="chip"><b>{len(weekly)}</b> weeks analyzed</div>
            <div class="chip"><b>{len(flagged_weeks)}</b> anomalies flagged</div>
            <div class="chip" style="color:{SEVERE};border-color:{SEVERE}66;">
                <b>{severe_count}</b> severe
            </div>
            <div class="chip" style="color:{MODERATE};border-color:{MODERATE}66;">
                <b>{moderate_count}</b> moderate
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "This tool scans weekly revenue from the Olist Brazilian e-commerce dataset, "
    "flags weeks that fall statistically below their recent trend, and breaks down "
    "*why* — separating whether the drop came from fewer orders, lower average order "
    "value, or a concentrated problem in a specific category, region, seller base, or "
    "payment method. Where no single cause explains the drop, it says so honestly "
    "rather than forcing an explanation."
)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">TREND</div>', unsafe_allow_html=True)
st.subheader("Weekly Revenue")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly["week_start"], y=weekly["revenue"],
    mode="lines", name="Weekly Revenue",
    line=dict(color=ACCENT, width=2),
))

moderate = anomalies[anomalies["severity"] == "moderate_anomaly"]
severe = anomalies[anomalies["severity"] == "severe_anomaly"]

fig.add_trace(go.Scatter(
    x=moderate["week_start"], y=moderate["revenue"],
    mode="markers", name="Moderate anomaly",
    marker=dict(color=MODERATE, size=11, symbol="circle", line=dict(color=BG, width=1.5)),
))
fig.add_trace(go.Scatter(
    x=severe["week_start"], y=severe["revenue"],
    mode="markers", name="Severe anomaly",
    marker=dict(color=SEVERE, size=15, symbol="diamond", line=dict(color=BG, width=1.5)),
))

fig = plotly_dark_layout(
    fig, height=420,
    xaxis_title="Week starting (Monday)",
    yaxis_title="Revenue (BRL)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

if not flagged_weeks:
    st.warning("No anomaly weeks were detected in this dataset.")
    st.stop()

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">DIAGNOSIS</div>', unsafe_allow_html=True)
st.subheader("Investigate an Anomaly Week")

week_labels = {
    w: f"{w.date()}  —  {memo_data[w]['anomaly_row']['severity'].replace('_anomaly', '').upper()}"
    for w in flagged_weeks
}
selected_week = st.selectbox(
    "Select a flagged anomaly week",
    options=flagged_weeks,
    format_func=lambda w: week_labels[w],
)

data = memo_data[selected_week]
decomp = data["decomposition"]
drilldown_results = data["drilldown"]
combo_result = data["combo"]
severity = data["anomaly_row"]["severity"]
severity_color = SEVERE if severity == "severe_anomaly" else MODERATE

st.markdown(
    f'<span class="severity-badge" style="background:{severity_color}22;color:{severity_color};'
    f'border:1px solid {severity_color}55;">{severity.replace("_", " ").upper()}</span>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown("#### Volume vs. AOV Decomposition")

def metric_card(label, value, accent):
    return (
        f'<div class="metric-card" style="--accent-color:{accent};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Revenue Change", f"{decomp['revenue_pct_change']:+.1f}%", SEVERE), unsafe_allow_html=True)
c2.markdown(metric_card("Volume Change", f"{decomp['volume_pct_change']:+.1f}%", ACCENT), unsafe_allow_html=True)
c3.markdown(metric_card("AOV Change", f"{decomp['aov_pct_change']:+.1f}%", ACCENT), unsafe_allow_html=True)
c4.markdown(metric_card("Classification", decomp["classification"].replace("-", " ").title(), STABLE), unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">SEGMENT ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("#### Segment Drilldown")

dim_cols = st.columns(len(DIMENSIONS))
for col, dim in zip(dim_cols, DIMENSIONS):
    df = drilldown_results[dim].head(6).copy()
    colors = [SEVERE if credible else CREDIBLE_GRAY for credible in df["is_credible_root_cause"]]
    bar_fig = go.Figure(go.Bar(
        x=df["contribution_pct_of_drop"],
        y=df["segment"].astype(str),
        orientation="h",
        marker=dict(color=colors),
    ))
    bar_fig = plotly_dark_layout(
        bar_fig, height=260,
        title=dict(text=dim.replace("_", " "), font=dict(size=13, color=TEXT)),
        xaxis_title="% of total drop",
        yaxis=dict(autorange="reversed", gridcolor=BORDER),
        showlegend=False,
    )
    col.plotly_chart(bar_fig, use_container_width=True)

st.markdown(
    f'<span style="color:{MUTED};font-size:0.85rem;">'
    f'<span style="color:{SEVERE};">■</span> meets credibility bar '
    f'(≥3% baseline share, &gt;1.3x overall decline rate, real dollar decline) &nbsp;&nbsp;'
    f'<span style="color:{CREDIBLE_GRAY};">■</span> does not'
    f'</span>',
    unsafe_allow_html=True,
)

if combo_result is not None and not combo_result.empty:
    credible_combo = combo_result[combo_result["is_credible_root_cause"]]
    if not credible_combo.empty:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("#### Top Combination Effect (Category × Payment Type)")
        top_combo = credible_combo.iloc[0]
        st.markdown(
            f'<div class="combo-box">'
            f'<b>{top_combo["product_category_name_english"]}</b> paid via '
            f'<b>{top_combo["payment_type_mode"]}</b>: {top_combo["baseline_share_pct"]:.1f}% of baseline '
            f'revenue, fell {top_combo["segment_pct_change"]:.1f}% '
            f'({top_combo["decline_ratio_vs_overall"]:.2f}x overall decline rate), '
            f'contributing {top_combo["contribution_pct_of_drop"]:.1f}% of the total drop.'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">REPORT</div>', unsafe_allow_html=True)
st.markdown("#### Analyst Memo")

api_key_available = os.environ.get("ANTHROPIC_API_KEY") is not None

if api_key_available:
    view = st.radio("Memo version", ["Raw (deterministic)", "Claude-polished"], horizontal=True)
    memo_text = get_polished_memo(data["raw_memo"]) if view == "Claude-polished" else data["raw_memo"]
else:
    memo_text = data["raw_memo"]

st.markdown(
    f'<div class="memo-box" style="border-left-color:{severity_color};">{memo_text}</div>',
    unsafe_allow_html=True,
)

if not api_key_available:
    st.caption("ℹ️ Set ANTHROPIC_API_KEY as an environment variable to enable the Claude-polished memo version.")

with st.expander("Methodology"):
    st.markdown(
        f"""
- **Anomaly detection** — each week's revenue is compared to a trailing 8-week rolling
  mean and standard deviation, computed from the prior 8 weeks only (the current week
  is excluded from its own baseline). A z-score below -1.5 is flagged **moderate**,
  below -2.5 is flagged **severe**. Weeks with fewer than 4 prior weeks of history are
  never flagged.
- **Volume vs. AOV decomposition** — Revenue = Volume × AOV. The revenue change is
  split exactly into a volume effect `(volume_now − volume_baseline) × aov_baseline`
  and an AOV effect `(aov_now − aov_baseline) × volume_now`, which sum exactly to the
  total revenue change with no residual.
- **Segment drilldown** — a segment is flagged as a credible root cause only if it
  represents at least 3% of baseline revenue AND declined more than 1.3x faster than
  the overall week AND genuinely lost revenue — preventing small segments with dramatic
  but economically insignificant swings from being mistaken for real causes.
        """
    )

st.markdown(
    '<div class="footer-note">Data: Olist Brazilian E-Commerce Public Dataset (Kaggle) · '
    'Built with pandas, Plotly &amp; Streamlit</div>',
    unsafe_allow_html=True,
)
