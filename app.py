import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import numpy as np
from datetime import datetime, timedelta, timezone

from backend.cleaning import clean_and_merge
from backend.map import build_network_map, KPI_CONFIGS
from backend.charts import (
    build_operator_speed_bar,
    build_latency_comparison,
    build_operator_radar_chart,
    build_connection_type_bar,
)
from backend.forecast import build_kpi_forecast_chart, FORECAST_KPI_OPTIONS
from backend.clustering import build_region_clustering_chart


UPLOADS_DIR = "uploads"
ASSETS_DIR = "assets"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_OUTPUT = os.path.join(APP_DIR, "output", "forecast_evaluation.csv")


def launch_forecast_eval():
    """Runs the forecast validation in a separate process.
    Results are written to output/forecast_evaluation.csv (not shown in the dashboard)."""
    old = st.session_state.get("_eval_proc")
    if old is not None and old.poll() is None:
        old.terminate()
    os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
    if os.path.exists(EVAL_OUTPUT):
        try:
            os.remove(EVAL_OUTPUT)
        except PermissionError:
            pass  # file open in Excel: it will be overwritten when possible
    print("[forecast_eval] started...", flush=True)
    # stdout and stderr are inherited, so results print in the terminal
    st.session_state["_eval_proc"] = subprocess.Popen(
        [sys.executable, "-u", os.path.join(APP_DIR, "backend", "forecast_eval.py")],
        cwd=APP_DIR,
    )


st.set_page_config(page_title="Network Performance Dashboard", layout="wide")


st.markdown("""
<style>
:root {
    --tt-red: #E2231A;
    --tt-yellow: #F7B500;
    --tt-green: #2DBE60;
    --tt-blue: #2F80ED;
    --tt-bg: #F5F7FB;
    --tt-surface: rgba(255, 255, 255, 0.92);
    --tt-text: #1F2937;
    --tt-muted: #64748B;
    --tt-border: rgba(15, 23, 42, 0.08);
    --tt-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    --tt-radius: 18px;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(226, 35, 26, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(45, 190, 96, 0.09), transparent 26%),
        radial-gradient(circle at bottom left, rgba(47, 128, 237, 0.08), transparent 24%),
        linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
.tt-brand-line {
    height: 4px;
    width: 100%;
    background: linear-gradient(90deg, var(--tt-red), var(--tt-yellow), var(--tt-green), var(--tt-blue));
    margin-top: 8px;
    margin-bottom: 12px;
    border-radius: 999px;
    box-shadow: 0 6px 18px rgba(226, 35, 26, 0.14);
}
.header-wrap { display: flex; justify-content: center; width: 100%; }
.header-card {
    width: 100%;
    max-width: 1100px;
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    padding: 0.75rem 1rem;
    backdrop-filter: blur(10px);
}
.brand-title {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    color: #1F2937;
    margin: 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em;
    text-align: left;
}
.logo-holder { display: flex; justify-content: center; align-items: center; height: 100%; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}
.stMetric {
    background: var(--tt-surface);
    border: 1px solid var(--tt-border);
    border-radius: var(--tt-radius);
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}
.stPlotlyChart {
    background: var(--tt-surface);
    border: 1px solid var(--tt-border);
    border-radius: var(--tt-radius);
    box-shadow: var(--tt-shadow);
    overflow: hidden;
}
.content-card {
    background: var(--tt-surface);
    border: 1px solid var(--tt-border);
    border-radius: var(--tt-radius);
    box-shadow: var(--tt-shadow);
    padding: 0.75rem;
    backdrop-filter: blur(8px);
}
.panel-title {
    font-size: 1.05rem;
    color: var(--tt-muted);
    font-weight: 700;
    margin-bottom: 0.35rem;
    text-align: left;
}
.section-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #1F2937;
    margin: 1.8rem 0 1rem 0;
    text-align: center;
    letter-spacing: -0.01em;
}
[data-testid="stFileUploader"] { display: flex; justify-content: center; }
[data-testid="stFileUploader"] section { width: 100%; }
[data-testid="stFileUploaderFile"], .stFileUploaderFile, [data-testid="stFileUploaderPagination"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


if 'merged_df' not in st.session_state:
    st.session_state['merged_df'] = None
if 'uploader_center_mode' not in st.session_state:
    st.session_state['uploader_center_mode'] = True
if 'last_processed_fingerprint' not in st.session_state:
    st.session_state['last_processed_fingerprint'] = None


df = st.session_state['merged_df']


LOCAL_LOGO = os.path.join(ASSETS_DIR, "tunisie_telecom_logo.webp")
REMOTE_FALLBACK = "https://upload.wikimedia.org/wikipedia/commons/e/e3/Tunisie_Telecom_Logo.png"


def render_header():
    st.markdown('<div class="header-wrap"><div class="header-card">', unsafe_allow_html=True)
    center_col = st.columns([1, 2.8, 1])[1]
    with center_col:
        logo_col, title_col = st.columns([0.55, 2.0])
        with logo_col:
            st.markdown('<div class="logo-holder">', unsafe_allow_html=True)
            if os.path.exists(LOCAL_LOGO):
                st.image(LOCAL_LOGO, width=160)
            else:
                try:
                    st.image(REMOTE_FALLBACK, width=160)
                except Exception:
                    pass
            st.markdown('</div>', unsafe_allow_html=True)
        with title_col:
            st.markdown("<h1 class='brand-title'>Network Performance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown("<div class='tt-brand-line'></div>", unsafe_allow_html=True)


def render_uploader_main():
    upload_col = st.columns([0.1, 0.8, 0.1])[1]
    with upload_col:
        return st.file_uploader(
            "Upload Network Log CSVs",
            accept_multiple_files=True,
            type=['csv'],
            key="landing_uploader"
        )


if st.session_state['uploader_center_mode'] and df is None:
    render_header()
    ui_files = render_uploader_main()
else:
    with st.sidebar:
        ui_files = st.file_uploader(
            "Upload Network Log CSVs",
            accept_multiple_files=True,
            type=['csv'],
            key="sidebar_uploader"
        )
        st.markdown("---")
        if df is not None:
            final_csv_stream = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=final_csv_stream,
                file_name="cleaned_network_performance.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.button("Download Cleaned Data (CSV)", disabled=True, width="stretch")


if ui_files:
    st.session_state['uploader_center_mode'] = False
    file_fingerprint = "-".join([f"{f.name}_{f.size}" for f in ui_files])

    if st.session_state.get('last_processed_fingerprint') != file_fingerprint:
        with st.sidebar.spinner("Processing logs..."):
            raw_dataframes = []

            for filename in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            for file in ui_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{file.name}")
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                raw_df = pd.read_csv(save_path)
                raw_dataframes.append(raw_df)

            merged_df, cleaning_report = clean_and_merge(raw_dataframes)
            st.session_state['merged_df'] = merged_df
            st.session_state['last_processed_fingerprint'] = file_fingerprint
            launch_forecast_eval()
        st.rerun()


if not st.session_state['uploader_center_mode']:
    render_header()


if df is None:
    st.stop()


operator_col = next((c for c in ['operator', 'attr_isp_common_name'] if c in df.columns), None)
region_col = next((c for c in ['attr_place_region', 'region'] if c in df.columns), None)


time_col = "ts_result"
if time_col not in df.columns:
    st.error(f"Time column '{time_col}' not found in data. Cannot apply time filter.")
    st.stop()


if "_ts_result_dt" not in st.session_state or st.session_state.get("_ts_result_dt_cached_df_id") != id(df):
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    st.session_state["_ts_result_dt"] = df[time_col]
    st.session_state["_ts_result_dt_cached_df_id"] = id(df)


ts_series = st.session_state["_ts_result_dt"]


min_ts = ts_series.min()
max_ts = ts_series.max()


default_from = min_ts.to_pydatetime() if pd.notna(min_ts) else datetime.now(timezone.utc) - timedelta(days=7)
default_to = max_ts.to_pydatetime() if pd.notna(max_ts) else datetime.now(timezone.utc)


filter_container = st.container()
with filter_container:
    time_col1, time_col2 = st.columns(2)
    with time_col1:
        filter_start = st.datetime_input(
            "From",
            value=default_from,
            key="time_from"
        )
    with time_col2:
        filter_end = st.datetime_input(
            "To",
            value=default_to,
            key="time_to"
        )


if filter_start is not None and filter_start.tzinfo is None:
    filter_start = filter_start.replace(tzinfo=timezone.utc)
if filter_end is not None and filter_end.tzinfo is None:
    filter_end = filter_end.replace(tzinfo=timezone.utc)


time_filter_error = None
if filter_start and filter_end and filter_start > filter_end:
    time_filter_error = "Start time must be before end time."
    st.error(time_filter_error)


time_filtered_df = df.copy()

if not time_filter_error:
    if filter_start and filter_end:
        time_mask = (ts_series >= filter_start) & (ts_series <= filter_end)
        time_filtered_df = time_filtered_df[time_mask].copy()
    elif filter_start:
        time_mask = ts_series >= filter_start
        time_filtered_df = time_filtered_df[time_mask].copy()
    elif filter_end:
        time_mask = ts_series <= filter_end
        time_filtered_df = time_filtered_df[time_mask].copy()


filter_left, filter_right = st.columns(2)
with filter_left:
    if operator_col:
        operator_options = ["All Operators"] + sorted(
            op for op in df[operator_col].dropna().astype(str).unique().tolist() if op != "Other"
        )
        selected_operator = st.selectbox("Operator", operator_options, index=0, key="operator_filter", label_visibility="collapsed")
    else:
        selected_operator = "All Operators"
with filter_right:
    if region_col:
        region_options = ["All Regions"] + sorted(df[region_col].dropna().astype(str).unique().tolist())
        selected_region = st.selectbox("Region", region_options, index=0, key="region_filter", label_visibility="collapsed")
    else:
        selected_region = "All Regions"


base_filtered_df = time_filtered_df.copy()
if region_col and selected_region != "All Regions":
    base_filtered_df = base_filtered_df[base_filtered_df[region_col].astype(str) == selected_region]

main_filtered_df = base_filtered_df.copy()
if operator_col and selected_operator != "All Operators":
    main_filtered_df = main_filtered_df[main_filtered_df[operator_col].astype(str) == selected_operator]


st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


total_rows = len(df)


def render_data_caption(active_df, active_operator=None):
    if time_filter_error:
        return

    op_str = f" **{active_operator}**" if active_operator and active_operator != "All Operators" else " **All Operators**"
    reg_str = f" in **{selected_region}**" if selected_region != "All Regions" else " across **All Regions**"

    time_str = ""
    if filter_start and filter_end:
        time_str = f" from *{filter_start.strftime('%Y-%m-%d %H:%M')}* to *{filter_end.strftime('%Y-%m-%d %H:%M')}*"

    st.caption(
        f"Showing **{len(active_df):,} rows** (out of {total_rows:,} total) — Filtered by{op_str}{reg_str}{time_str}."
    )


def _clean_to_numeric_series(series):
    cleaned = pd.to_numeric(series.astype(str).str.replace(r'[^\d\.]', '', regex=True), errors="coerce")
    return cleaned.dropna()


def _interquartile_mean(series):
    clean_series = _clean_to_numeric_series(series)
    if clean_series.empty:
        return np.nan
    q25 = np.percentile(clean_series, 25)
    q75 = np.percentile(clean_series, 75)
    iqr_filtered = clean_series[(clean_series >= q25) & (clean_series <= q75)]
    return iqr_filtered.mean() if not iqr_filtered.empty else clean_series.median()


def render_kpi_cards(page_df):
    try:
        download_col = next((c for c in ['download_mbps', 'val_download_kbps', 'download'] if c in page_df.columns), None)
        upload_col = next((c for c in ['upload_mbps', 'val_upload_kbps', 'upload'] if c in page_df.columns), None)
        latency_col = next((c for c in ['latency_ms', 'val_latency_iqm_ms', 'latency'] if c in page_df.columns), None)

        if download_col:
            clean_dl = _clean_to_numeric_series(page_df[download_col])
            dl_scale = 1 / 1000.0 if download_col == 'val_download_kbps' else 1.0
            g_download = clean_dl.median() * dl_scale if not clean_dl.empty else 0.0
        else:
            g_download = 0.0

        if upload_col:
            clean_ul = _clean_to_numeric_series(page_df[upload_col])
            ul_scale = 1 / 1000.0 if upload_col == 'val_upload_kbps' else 1.0
            g_upload = clean_ul.median() * ul_scale if not clean_ul.empty else 0.0
        else:
            g_upload = 0.0

        g_latency = _interquartile_mean(page_df[latency_col]) if latency_col else np.nan
        if pd.isna(g_latency):
            g_latency = 0.0

        g_samples = len(page_df)
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        g_download = g_upload = g_latency = 0.0
        g_samples = 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median Download Speed", f"{g_download:.2f} Mbps")
    k2.metric("Median Upload Speed", f"{g_upload:.2f} Mbps")
    k3.metric("Average Latency (Delay)", f"{g_latency:.1f} ms")
    k4.metric("Total Data Samples", f"{int(g_samples):,}")


def render_predictive_section(page_df, key_prefix):
    st.markdown("<div class='section-title'>Predictive &amp; Segmentation Analytics</div>", unsafe_allow_html=True)
    forecast_col, cluster_col = st.columns(2)

    with forecast_col:
        st.markdown("<div class='content-card'><h4 class='panel-title'>KPI Evolution Forecast</h4>", unsafe_allow_html=True)
        forecast_kpi_label = st.selectbox(
            "Forecast KPI",
            list(FORECAST_KPI_OPTIONS.values()),
            index=0,
            key=f"{key_prefix}_forecast_kpi_filter",
            label_visibility="collapsed"
        )
        forecast_kpi_key = [k for k, v in FORECAST_KPI_OPTIONS.items() if v == forecast_kpi_label][0]
        fig_forecast = build_kpi_forecast_chart(page_df, kpi=forecast_kpi_key, horizon_days=7)
        fig_forecast.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_forecast, width="stretch", key=f"{key_prefix}_forecast_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with cluster_col:
        st.markdown("<div class='content-card'><h4 class='panel-title'>Region Clustering by KPI Profile</h4>", unsafe_allow_html=True)
        fig_cluster, cluster_summary, best_k = build_region_clustering_chart(page_df)
        fig_cluster.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cluster, width="stretch", key=f"{key_prefix}_cluster_chart")
        st.markdown("</div>", unsafe_allow_html=True)


def render_main_page(page_df):
    render_data_caption(page_df, selected_operator)
    render_kpi_cards(page_df)
    st.markdown("<hr>", unsafe_allow_html=True)

    view_left, view_right = st.columns([0.45, 0.55])

    with view_left:
        st.markdown("<div class='content-card'><h4 class='panel-title'>Geographic Coverage Map</h4>", unsafe_allow_html=True)
        kpi_options = {cfg["label"]: key for key, cfg in KPI_CONFIGS.items()}
        selected_kpi_label = st.selectbox("Map KPI", list(kpi_options.keys()), index=0, key="main_map_kpi_filter", label_visibility="collapsed")
        selected_kpi_key = kpi_options[selected_kpi_label]

        highlight = selected_region if selected_region != "All Regions" else None
        map_fig = build_network_map(page_df, kpi=selected_kpi_key, highlight_region=highlight)
        map_fig.update_layout(height=450, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(map_fig, width="stretch", key="main_map_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with view_right:
        chart_height = 230
        grid_row1_left, grid_row1_right = st.columns(2)

        with grid_row1_left:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Network Speed Comparison</h4>", unsafe_allow_html=True)
            fig_speed = build_operator_speed_bar(page_df)
            fig_speed.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_speed, width="stretch", key="main_speed_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        with grid_row1_right:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Multi-KPI Operator Profile</h4>", unsafe_allow_html=True)
            fig_radar = build_operator_radar_chart(page_df)
            fig_radar.update_layout(height=chart_height, margin={"t": 5, "b": 5})
            st.plotly_chart(fig_radar, width="stretch", key="main_radar_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        grid_row2_left, grid_row2_right = st.columns(2)

        with grid_row2_left:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Network Response Time</h4>", unsafe_allow_html=True)
            fig_lat = build_latency_comparison(page_df)
            fig_lat.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lat, width="stretch", key="main_latency_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        with grid_row2_right:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Mobile Technology Generation Share</h4>", unsafe_allow_html=True)
            fig_tech = build_connection_type_bar(page_df)
            fig_tech.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tech, width="stretch", key="main_tech_chart")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    render_predictive_section(page_df, key_prefix="main")


render_main_page(main_filtered_df)