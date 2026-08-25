import os
import textwrap
import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from datetime import datetime, timedelta

from live_features import get_live_input, HISTORY_FILE

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "temp", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "aqi", "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
    "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
]

AQI_BANDS = [
    (0, 50, "Good", "#0ca30c", "🙂",
     "Air quality is satisfactory, and air pollution poses little or no risk."),
    (51, 100, "Moderate", "#fab219", "😐",
     "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion."),
    (101, 150, "Unhealthy for Sensitive Groups", "#eb6834", "😷",
     "Sensitive groups (children, elderly, respiratory conditions) may experience health effects."),
    (151, 200, "Unhealthy", "#d03b3b", "🚫",
     "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."),
    (201, 300, "Very Unhealthy", "#8e44ad", "☣️",
     "Health alert: everyone may experience more serious health effects."),
    (301, 10_000, "Hazardous", "#7e0023", "☠️",
     "Health warning of emergency conditions — the entire population is more likely to be affected."),
]

MODEL_DIRS = {"24h": "model_24h", "48h": "model_48h", "72h": "model_72h"}
HORIZON_LABELS = {"24h": "Day 1 (0-24h)", "48h": "Day 2 (24-48h)", "72h": "Day 3 (48-72h)"}
SHAP_BACKGROUND_FILE = "shap_background.csv"
ALERTS_LOG_FILE = "alerts_log.csv"
ALERT_THRESHOLD = 151   # "Unhealthy" and worse

FEATURE_DISPLAY_NAMES = {
    "hour": "Hour of day", "day": "Day of month", "month": "Month", "day_of_week": "Day of week",
    "temp": "Temperature", "humidity": "Humidity", "wind_speed": "Wind speed",
    "pm2_5": "PM2.5", "pm10": "PM10", "co": "CO", "no2": "NO2", "so2": "SO2", "o3": "O3",
    "aqi": "Current AQI", "aqi_24h_ago": "AQI 24h ago", "aqi_48h_ago": "AQI 48h ago",
    "aqi_72h_ago": "AQI 72h ago", "aqi_change_24h": "24h AQI change",
    "aqi_rolling_mean_24h": "24h rolling avg AQI", "aqi_rolling_std_24h": "24h rolling AQI volatility",
}

# Metrics from the last registered training run (register_models.py) -
# not recomputed live since that needs a full Hopsworks historical read.
TRAINING_METRICS = {
    "24h": {"rmse": 11.49, "mae": 8.26, "r2": 0.569, "persistence_rmse": 16.76},
    "48h": {"rmse": 14.48, "mae": 10.78, "r2": 0.285, "persistence_rmse": 21.44},
    "72h": {"rmse": 15.02, "mae": 11.18, "r2": 0.176, "persistence_rmse": 23.12},
}

INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
SURFACE = "#1a1a19"
BLUE = "#3987e5"


def aqi_band(aqi_value):
    for lo, hi, label, color, icon, desc in AQI_BANDS:
        if lo <= aqi_value <= hi:
            return label, color, icon, desc
    return AQI_BANDS[-1][2], AQI_BANDS[-1][3], AQI_BANDS[-1][4], AQI_BANDS[-1][5]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    textwrap.dedent("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  { font-family: 'Inter', system-ui, sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseRing {
        0%   { box-shadow: 0 0 0 0 rgba(208,59,59,0.45); }
        70%  { box-shadow: 0 0 0 14px rgba(208,59,59,0); }
        100% { box-shadow: 0 0 0 0 rgba(208,59,59,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    .fade-card {
        animation: fadeInUp 0.55s ease-out both;
        background: #1a1a19;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }
    .fade-card:nth-of-type(2)  { animation-delay: 0.06s; }
    .fade-card:nth-of-type(3)  { animation-delay: 0.12s; }
    .fade-card:nth-of-type(4)  { animation-delay: 0.18s; }
    .stat-label { color: #898781; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .stat-value { color: #ffffff; font-size: 1.55rem; font-weight: 700; margin-top: 2px; }
    .stat-unit  { color: #c3c2b7; font-size: 0.85rem; font-weight: 500; margin-left: 4px; }
    .hero-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.01em; margin-bottom: 0; }
    .hero-sub   { color: #c3c2b7; font-size: 0.95rem; margin-top: 2px; }
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.82rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .forecast-card { text-align: center; }
    .forecast-day  { color: #898781; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .forecast-aqi  { font-size: 2.3rem; font-weight: 800; margin: 6px 0 2px 0; }
    .forecast-cat  { font-size: 0.88rem; font-weight: 600; }
    .forecast-delta{ font-size: 0.82rem; margin-top: 6px; font-weight: 600; }
    .pulse { animation: pulseRing 2.2s infinite; border-radius: 999px; }
    .footer-note { color: #898781; font-size: 0.8rem; line-height: 1.6; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-family: 'Inter', sans-serif; }
    </style>
    """).strip(),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for horizon, model_dir in MODEL_DIRS.items():
        path = os.path.join(model_dir, "model.pkl")
        if os.path.exists(path):
            models[horizon] = joblib.load(path)
        else:
            models[horizon] = None
    return models


@st.cache_data(ttl=300, show_spinner=False)
def load_live_data(_cache_key):
    features_df, as_of_ts, gaps = get_live_input()
    return features_df, as_of_ts, gaps


@st.cache_resource(show_spinner=False)
def load_explainers():
    if not os.path.exists(SHAP_BACKGROUND_FILE):
        return {}, None
    background = pd.read_csv(SHAP_BACKGROUND_FILE)
    models = load_models()
    explainers = {}
    for horizon, model in models.items():
        if model is not None:
            explainers[horizon] = shap.LinearExplainer(model, background)
    return explainers, background


@st.cache_data(ttl=300, show_spinner=False)
def load_history(_cache_key):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df.sort_values("datetime")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

left, right = st.columns([5, 1])
with left:
    st.markdown('<div class="hero-title">🌫️ Karachi AQI Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Live conditions and a 3-day forecast, served by three '
        'independently-trained Ridge regression models.</div>',
        unsafe_allow_html=True,
    )
with right:
    refresh = st.button("↻ Refresh", use_container_width=True)

if refresh:
    st.cache_data.clear()
    st.toast("Live data refreshed.", icon="✅")

if refresh:
    cache_key = datetime.now().strftime("%Y%m%d%H%M")
else:
    cache_key = datetime.now().strftime("%Y%m%d%H")

models = load_models()

try:
    with st.spinner("Fetching live weather & pollution data..."):
        features_df, as_of_ts, gaps = load_live_data(cache_key)
except Exception as e:
    st.error(f"Couldn't fetch live data right now: {e}")
    st.stop()

history_df = load_history(cache_key)

current_aqi = int(features_df["aqi"].iloc[0])
cat_label, cat_color, cat_icon, cat_desc = aqi_band(current_aqi)
as_of_dt = datetime.fromtimestamp(as_of_ts)

forecast_values = {}
for horizon, model in models.items():
    if model is not None:
        forecast_values[horizon] = float(model.predict(features_df[FEATURE_COLUMNS])[0])

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hazardous AQI alert banner
# ---------------------------------------------------------------------------

alert_points = [("Now", current_aqi)]
for horizon, value in forecast_values.items():
    alert_points.append((HORIZON_LABELS[horizon], value))

worst_label = None
worst_aqi = None
for label, value in alert_points:
    if worst_aqi is None or value > worst_aqi:
        worst_label = label
        worst_aqi = value

if worst_aqi >= ALERT_THRESHOLD:
    w_label, w_color, w_icon, _ = aqi_band(worst_aqi)
    triggered = []
    for label, value in alert_points:
        if value >= ALERT_THRESHOLD:
            triggered.append(f"{label} ({value:.0f})")
    st.markdown(
        f'<div class="fade-card pulse" style="border-color:{w_color}; '
        f'background:linear-gradient(0deg, rgba(0,0,0,0) 0%, {w_color}22 100%); margin-bottom:1rem;">'
        f'<span class="badge"><span class="dot" style="background:{w_color};"></span>'
        f'{w_icon} Air quality alert &mdash; {w_label}</span>'
        f'<p style="color:{INK_SECONDARY}; margin-top:10px; font-size:0.92rem;">'
        f'AQI reaches <b style="color:{w_color};">{worst_aqi:.0f}</b> at <b>{worst_label}</b>. '
        f'Triggered by: {", ".join(triggered)}. Limit outdoor exposure, especially for children, '
        f'the elderly, and anyone with respiratory conditions.</p></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero row: gauge + category summary
# ---------------------------------------------------------------------------

hero_l, hero_r = st.columns([1, 1.4])

with hero_l:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={"suffix": "", "font": {"size": 46, "color": INK_PRIMARY, "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 400], "tickvals": [0, 50, 100, 150, 200, 300, 400],
                         "tickcolor": INK_MUTED, "tickfont": {"color": INK_MUTED, "size": 10}},
                "bar": {"color": cat_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(12,163,12,0.16)"},
                    {"range": [50, 100], "color": "rgba(250,178,25,0.16)"},
                    {"range": [100, 150], "color": "rgba(235,104,52,0.16)"},
                    {"range": [150, 200], "color": "rgba(208,59,59,0.16)"},
                    {"range": [200, 300], "color": "rgba(142,68,173,0.16)"},
                    {"range": [300, 400], "color": "rgba(126,0,35,0.16)"},
                ],
                "threshold": {"line": {"color": cat_color, "width": 3}, "thickness": 0.9, "value": current_aqi},
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": INK_PRIMARY},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with hero_r:
    if current_aqi > 150:
        pulse_class = "pulse"
    else:
        pulse_class = ""
    st.markdown(
        textwrap.dedent(f"""
        <div class="fade-card" style="margin-top: 1.2rem;">
            <span class="badge {pulse_class}">
                <span class="dot" style="background:{cat_color};"></span>{cat_icon} {cat_label}
            </span>
            <p style="color:{INK_SECONDARY}; margin-top:12px; font-size:0.95rem; line-height:1.5;">{cat_desc}</p>
            <p style="color:{INK_MUTED}; font-size:0.8rem; margin-top:14px;">
                As of {as_of_dt.strftime('%b %d, %Y - %I:%M %p')} &middot; dominant pollutant:
                <b style="color:{INK_SECONDARY};">{features_df['dominant_pollutant'].iloc[0]}</b>
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Current conditions - stat tiles
# ---------------------------------------------------------------------------

st.markdown("##### Current conditions")
c1, c2, c3, c4, c5, c6 = st.columns(6)
tiles = [
    (c1, "🌡️ Temp", f"{features_df['temp'].iloc[0]:.1f}", "°C"),
    (c2, "💧 Humidity", f"{features_df['humidity'].iloc[0]:.0f}", "%"),
    (c3, "💨 Wind", f"{features_df['wind_speed'].iloc[0]:.1f}", "m/s"),
    (c4, "PM2.5", f"{features_df['pm2_5'].iloc[0]:.1f}", "µg/m³"),
    (c5, "PM10", f"{features_df['pm10'].iloc[0]:.1f}", "µg/m³"),
    (c6, "O₃", f"{features_df['o3'].iloc[0]:.1f}", "µg/m³"),
]
for col, label, value, unit in tiles:
    with col:
        st.markdown(
            f'<div class="fade-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}<span class="stat-unit">{unit}</span></div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3-day forecast cards
# ---------------------------------------------------------------------------

st.markdown("##### 3-day forecast")
f1, f2, f3 = st.columns(3)
for col, horizon in zip([f1, f2, f3], ["24h", "48h", "72h"]):
    pred = forecast_values.get(horizon)
    with col:
        if pred is None:
            st.markdown(
                f'<div class="fade-card forecast-card">'
                f'<div class="forecast-day">{HORIZON_LABELS[horizon]}</div>'
                f'<div style="color:{INK_MUTED}; margin-top:1rem;">Model not found</div></div>',
                unsafe_allow_html=True,
            )
            continue
        p_label, p_color, p_icon, _ = aqi_band(pred)
        delta = pred - current_aqi
        if delta < -1:
            delta_color = "#0ca30c"
            delta_arrow = "▼"
            delta_word = "improving"
        elif delta > 1:
            delta_color = "#d03b3b"
            delta_arrow = "▲"
            delta_word = "worsening"
        else:
            delta_color = INK_MUTED
            delta_arrow = "▬"
            delta_word = "steady"
        st.markdown(
            f'<div class="fade-card forecast-card">'
            f'<div class="forecast-day">{HORIZON_LABELS[horizon]}</div>'
            f'<div class="forecast-aqi" style="color:{p_color};">{pred:.0f}</div>'
            f'<span class="badge"><span class="dot" style="background:{p_color};"></span>{p_icon} {p_label}</span>'
            f'<div class="forecast-delta" style="color:{delta_color};">{delta_arrow} {abs(delta):.0f} pts &middot; {delta_word}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SHAP explanation panel
# ---------------------------------------------------------------------------

st.markdown("##### Why this forecast?")
explainers, shap_background = load_explainers()

if not explainers:
    st.info(
        "No SHAP background sample found — run `compute_shap_background.py` once "
        "(reads a spread of historical rows from Hopsworks) to enable this panel."
    )
else:
    def horizon_label(horizon):
        return HORIZON_LABELS[horizon]

    explain_horizon = st.radio(
        "Explain the prediction for:",
        options=["24h", "48h", "72h"],
        format_func=horizon_label,
        horizontal=True,
        label_visibility="collapsed",
    )
    explainer = explainers.get(explain_horizon)
    if explainer is not None:
        input_row = features_df[FEATURE_COLUMNS]
        shap_row = explainer.shap_values(input_row)[0]
        base_value = float(explainer.expected_value)
        predicted = base_value + shap_row.sum()

        feature_labels = []
        feature_values = []
        for column_name in FEATURE_COLUMNS:
            feature_labels.append(FEATURE_DISPLAY_NAMES.get(column_name, column_name))
            feature_values.append(input_row.iloc[0][column_name])

        contrib = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "label": feature_labels,
            "value": feature_values,
            "shap": shap_row,
        })
        contrib["abs_shap"] = contrib["shap"].abs()
        contrib = contrib.sort_values("abs_shap", ascending=False)

        TOP_N = 8
        top = contrib.iloc[:TOP_N].copy()
        rest = contrib.iloc[TOP_N:]
        if len(rest) > 0:
            other_row = pd.DataFrame([{
                "feature": "_other", "label": f"All other features ({len(rest)})",
                "value": None, "shap": rest["shap"].sum(), "abs_shap": rest["shap"].sum(),
            }])
            top = pd.concat([top, other_row], ignore_index=True)
        top = top.sort_values("shap")

        bar_colors = []
        for shap_value in top["shap"]:
            if shap_value > 0:
                bar_colors.append("#d03b3b")
            else:
                bar_colors.append("#0ca30c")

        text_labels = []
        for _, row in top.iterrows():
            if pd.isna(row["value"]):
                text_labels.append(row["label"])
            else:
                text_labels.append(f"{row['label']} = {row['value']:.1f}")

        fig3 = go.Figure(
            go.Bar(
                x=top["shap"], y=text_labels, orientation="h",
                marker=dict(color=bar_colors),
                hovertemplate="%{y}<br>impact %{x:+.1f} AQI pts<extra></extra>",
            )
        )
        fig3.add_vline(x=0, line=dict(color=BASELINE, width=1))
        fig3.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": INK_SECONDARY, "family": "Inter"},
            xaxis=dict(title="Impact on predicted AQI (points)", showgrid=True, gridcolor=GRIDLINE,
                       zeroline=False, color=INK_MUTED),
            yaxis=dict(showgrid=False, color=INK_PRIMARY),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"Baseline (typical predicted AQI for this horizon): **{base_value:.0f}** → "
            f"this forecast: **{predicted:.0f}**. Red bars push the forecast up (worse air "
            f"quality), green bars pull it down (better air quality). Computed with "
            f"`shap.LinearExplainer` against a 150-row sample spread across the training period."
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Trend chart: recent history -> now -> forecast
# ---------------------------------------------------------------------------

st.markdown("##### AQI trend: recent history & forecast")

fig2 = go.Figure()

if not history_df.empty:
    fig2.add_trace(
        go.Scatter(
            x=history_df["datetime"], y=history_df["aqi"],
            mode="lines", name="Actual (recent history)",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy", fillcolor="rgba(57,135,229,0.08)",
            hovertemplate="%{x|%b %d, %I:%M %p}<br>AQI %{y:.0f}<extra></extra>",
        )
    )

# Build the forecast line point by point: it starts at "now" (the current
# reading) and then adds one point per horizon (24h, 48h, 72h).
forecast_x = [as_of_dt]
forecast_y = [current_aqi]
marker_colors = [cat_color]
for horizon in ["24h", "48h", "72h"]:
    hours_ahead = int(horizon[:-1])
    forecast_x.append(as_of_dt + timedelta(hours=hours_ahead))

    predicted_value = forecast_values.get(horizon, None)
    forecast_y.append(predicted_value)

    if predicted_value is not None:
        marker_colors.append(aqi_band(predicted_value)[1])
    else:
        marker_colors.append(INK_MUTED)

fig2.add_trace(
    go.Scatter(
        x=forecast_x, y=forecast_y,
        mode="lines+markers", name="Forecast",
        line=dict(color=INK_SECONDARY, width=2, dash="dash"),
        marker=dict(size=13, color=marker_colors, line=dict(width=2, color=SURFACE)),
        hovertemplate="%{x|%b %d}<br>Predicted AQI %{y:.0f}<extra></extra>",
    )
)

for threshold in [50, 100, 150, 200, 300]:
    fig2.add_hline(y=threshold, line=dict(color=GRIDLINE, width=1, dash="dot"))

fig2.add_vline(x=as_of_dt, line=dict(color=BASELINE, width=1.5, dash="dot"))
fig2.add_annotation(x=as_of_dt, y=1.05, yref="paper", text="now", showarrow=False,
                     font=dict(color=INK_MUTED, size=11))

fig2.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": INK_SECONDARY, "family": "Inter"},
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_SECONDARY)),
    xaxis=dict(showgrid=False, color=INK_MUTED, linecolor=BASELINE),
    yaxis=dict(showgrid=True, gridcolor=GRIDLINE, color=INK_MUTED, title="AQI", zeroline=False),
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

if history_df.empty:
    st.info(
        "No local history yet — `recent_history.csv` fills in one row per hour as the "
        "automation runs, so the trend line will appear once a few hours have accumulated."
    )
else:
    max_gap = max(gaps.values())
    if max_gap > 6:
        st.caption(
            f"⚠️ Lookback data is still sparse — the furthest lag lookup was "
            f"{max_gap:.1f}h off its exact target. Accuracy improves as more hourly "
            f"history accumulates (24h/48h/72h ago need 1/2/3 days respectively)."
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Alert history
# ---------------------------------------------------------------------------

st.markdown("##### Recent hazardous-AQI alerts")
if os.path.exists(ALERTS_LOG_FILE):
    alerts_df = pd.read_csv(ALERTS_LOG_FILE)
    alerts_df["when"] = pd.to_datetime(alerts_df["timestamp"], unit="s")
    alerts_df = alerts_df.sort_values("when", ascending=False)
    for _, row in alerts_df.head(10).iterrows():
        a_label, a_color, a_icon, _ = aqi_band(row["aqi"])
        st.markdown(
            f'<div class="fade-card" style="padding:0.7rem 1.1rem; margin-bottom:8px;">'
            f'<span class="badge"><span class="dot" style="background:{a_color};"></span>'
            f'{a_icon} {a_label} &middot; AQI {row["aqi"]:.0f}</span>'
            f'<span style="color:{INK_MUTED}; font-size:0.82rem; margin-left:10px;">'
            f'{row["when"].strftime("%b %d, %Y - %I:%M %p")}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Logged automatically by the hourly pipeline whenever AQI ≥ {ALERT_THRESHOLD} (Unhealthy or worse).")
else:
    st.caption(
        f"No hazardous readings logged yet — the hourly pipeline records an entry here "
        f"whenever AQI ≥ {ALERT_THRESHOLD} (Unhealthy or worse)."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Architecture / model performance detail
# ---------------------------------------------------------------------------

with st.expander("📐 How this works — architecture & model performance"):
    st.markdown(
        """
Three independent **Ridge regression** models — one per horizon — each predict the
**average AQI over that day's window** (Day 1 = hours 1-24, Day 2 = 25-48, Day 3 = 49-72),
trained on ~3 years of hourly Karachi weather + pollution data backfilled via Open-Meteo
and served live via OpenWeather.

**Pipeline:** OpenWeather → feature engineering → Hopsworks Feature Store (hourly, via
GitHub Actions) → Hopsworks Model Registry. Live serving reads the current reading
directly from OpenWeather and its own rolling local history cache (`recent_history.csv`,
also written by the same hourly job) rather than querying Hopsworks for recent rows —
Hopsworks' materialization job runs behind the free tier's hourly insert cadence, so
very recent rows aren't reliably queryable yet. Training still reads exclusively from
Hopsworks, since older data has fully materialized.

**Registration bar:** beat the persistence baseline ("tomorrow = today"), not an
absolute R² target — R² is naturally capped in low-variance (calm AQI) periods
regardless of model quality.
        """
    )
    metrics_df = pd.DataFrame(TRAINING_METRICS).T
    metrics_df.index.name = "Horizon"
    metrics_df = metrics_df.rename(columns={
        "rmse": "RMSE", "mae": "MAE", "r2": "R²", "persistence_rmse": "Persistence RMSE",
    })
    metrics_df["Beats persistence"] = metrics_df["RMSE"] < metrics_df["Persistence RMSE"]
    st.dataframe(metrics_df.style.format({
        "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}", "Persistence RMSE": "{:.2f}",
    }), use_container_width=True)

st.markdown(
    '<div class="footer-note">Data: OpenWeather (live) + Open-Meteo (historical backfill) &middot; '
    'Feature Store &amp; Model Registry: Hopsworks &middot; '
    'Location: Karachi, Pakistan (24.86&deg;N, 67.00&deg;E) &middot; '
    'Auto-refreshes cached data every 5 minutes.</div>',
    unsafe_allow_html=True,
)
