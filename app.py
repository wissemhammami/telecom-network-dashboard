import streamlit as st
import pandas as pd
import os
import numpy as np
from datetime import datetime


from backend.eda import run_eda
from backend.cleaning import clean_and_merge
from backend.map import build_network_map
from backend.charts import (
    build_operator_speed_bar,
    build_latency_comparison,
    build_signal_strength_comparison,
    build_connection_type_bar
)


UPLOADS_DIR = "uploads"
ASSETS_DIR = "assets"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


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

        .header-wrap {
            display: flex;
            justify-content: center;
            width: 100%;
        }

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

        .logo-holder {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }

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

        [data-testid="stFileUploader"] {
            display: flex;
            justify-content: center;
        }

        [data-testid="stFileUploader"] section {
            width: 100%;
        }

        [data-testid="stFileUploaderFile"],
        .stFileUploaderFile,
        [data-testid="stFileUploaderPagination"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


if 'merged_df' not in st.session_state:
    st.session_state['merged_df'] = None

if 'uploader_center_mode' not in st.session_state:
    st.session_state['uploader_center_mode'] = True


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
                use_container_width=True
            )
        else:
            st.button("Download Cleaned Data (CSV)", disabled=True, use_container_width=True)


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

            eda_report = run_eda(raw_dataframes)
            eda_report.log()

            merged_df, cleaning_report = clean_and_merge(raw_dataframes)
            st.session_state['merged_df'] = merged_df
            st.session_state['last_processed_fingerprint'] = file_fingerprint
            st.rerun()


if not st.session_state['uploader_center_mode']:
    render_header()

    # ── Filters: Operator & Region ──────────────────────────────────────
    operator_col = next((c for c in ['operator', 'attr_isp_common_name'] if c in df.columns), None)
    region_col = next((c for c in ['attr_place_region', 'region'] if c in df.columns), None)

    filter_left, filter_right = st.columns(2)

    with filter_left:
        if operator_col:
            operator_options = ["All Operators"] + sorted(df[operator_col].dropna().unique().tolist())
            selected_operator = st.selectbox("Operator", operator_options, index=0, key="operator_filter", label_visibility="collapsed")
        else:
            selected_operator = "All Operators"

    with filter_right:
        if region_col:
            region_options = ["All Regions"] + sorted(df[region_col].dropna().unique().tolist())
            selected_region = st.selectbox("Region", region_options, index=0, key="region_filter", label_visibility="collapsed")
        else:
            selected_region = "All Regions"

    if operator_col and selected_operator != "All Operators":
        df = df[df[operator_col] == selected_operator]
    if region_col and selected_region != "All Regions":
        df = df[df[region_col] == selected_region]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    # ── End filters ──────────────────────────────────────────────────────

    try:
        download_col = None
        for col in ['download_mbps', 'val_download_kbps', 'download']:
            if col in df.columns:
                download_col = col
                break

        upload_col = None
        for col in ['upload_mbps', 'val_upload_kbps', 'upload']:
            if col in df.columns:
                upload_col = col
                break

        latency_col = None
        for col in ['latency_ms', 'val_latency_iqm_ms', 'latency']:
            if col in df.columns:
                latency_col = col
                break

        def clean_to_numeric_series(series):
            cleaned = pd.to_numeric(
                series.astype(str).str.replace(r'[^\d\.]', '', regex=True),
                errors='coerce'
            )
            return cleaned.dropna()

        def interquartile_mean(series):
            clean_series = clean_to_numeric_series(series)
            if clean_series.empty:
                return np.nan
            q25 = np.percentile(clean_series, 25)
            q75 = np.percentile(clean_series, 75)
            iqr_filtered = clean_series[(clean_series >= q25) & (clean_series <= q75)]
            return iqr_filtered.mean() if not iqr_filtered.empty else clean_series.median()

        if download_col:
            clean_dl = clean_to_numeric_series(df[download_col])
            dl_scale = 1 / 1000.0 if download_col == 'val_download_kbps' else 1.0
            global_download = clean_dl.median() * dl_scale if not clean_dl.empty else 0.0
        else:
            global_download = 0.0

        if upload_col:
            clean_ul = clean_to_numeric_series(df[upload_col])
            ul_scale = 1 / 1000.0 if upload_col == 'val_upload_kbps' else 1.0
            global_upload = clean_ul.median() * ul_scale if not clean_ul.empty else 0.0
        else:
            global_upload = 0.0

        if latency_col:
            global_latency = interquartile_mean(df[latency_col])
            if pd.isna(global_latency):
                global_latency = 0.0
        else:
            global_latency = 0.0

        global_samples = len(df)

    except Exception:
        global_download = 0.0
        global_upload = 0.0
        global_latency = 0.0
        global_samples = 0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Median Download Speed", f"{global_download:.2f} Mbps")
    kpi_col2.metric("Median Upload Speed", f"{global_upload:.2f} Mbps")
    kpi_col3.metric("Average Latency (Delay)", f"{global_latency:.1f} ms")
    kpi_col4.metric("Total Data Samples", f"{int(global_samples):,}")

    st.markdown("<hr>", unsafe_allow_html=True)

    view_left, view_right = st.columns([0.45, 0.55])

    with view_left:
        st.markdown("<div class='content-card'><h4 class='panel-title'>Geographic Coverage Map</h4>", unsafe_allow_html=True)
        map_fig = build_network_map(df)
        map_fig.update_layout(
            height=490,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(map_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with view_right:
        chart_height = 230

        grid_row1_left, grid_row1_right = st.columns(2)
        with grid_row1_left:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Network Speed Comparison</h4>", unsafe_allow_html=True)
            fig_speed = build_operator_speed_bar(df)
            fig_speed.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_speed, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with grid_row1_right:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Signal Strength Distribution</h4>", unsafe_allow_html=True)
            fig_sig = build_signal_strength_comparison(df)
            fig_sig.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        grid_row2_left, grid_row2_right = st.columns(2)
        with grid_row2_left:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Network Response Time</h4>", unsafe_allow_html=True)
            fig_lat = build_latency_comparison(df)
            fig_lat.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with grid_row2_right:
            st.markdown("<div class='content-card'><h4 class='panel-title'>Mobile Technology Generation Share</h4>", unsafe_allow_html=True)
            fig_tech = build_connection_type_bar(df)
            fig_tech.update_layout(height=chart_height, margin={"t": 5, "b": 5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tech, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)