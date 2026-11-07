import numpy as np
import pandas as pd
import plotly.graph_objects as go

OPERATOR_COLORS = {
    "Tunisie Telecom": "#005BB6",
    "Ooredoo": "#E11A22",
    "Orange": "#FF6600"
}

OPERATOR_COLORS_LIGHT = {
    "Tunisie Telecom": "rgba(0, 91, 182, 0.35)",
    "Ooredoo": "rgba(225, 26, 34, 0.35)",
    "Orange": "rgba(255, 102, 0, 0.35)"
}

_THEME = {
    "bg_color": "rgba(0,0,0,0)", 
    "text_color": "#1F2937",
    "grid_color": "#E2E8F0",
    "font_family": "Inter, system-ui, Sans-Serif"
}

def _resolve_column(df, prioritized_list):
    if df is None:
        return None
    for col in prioritized_list:
        if col in df.columns:
            return col
    return None

def _get_valid_operators(df):
    if df is None:
        return [], None
    op_col = _resolve_column(df, ['operator', 'attr_isp_common_name', 'attr_sim_operator_common_name', 'attr_network_operator_common_name'])
    if not op_col:
        return [], None
    available_ops = [op for op in OPERATOR_COLORS.keys() if op in df[op_col].unique()]
    return available_ops, op_col

def _interquartile_mean(series):
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if clean_series.empty:
        return np.nan
    q25 = np.percentile(clean_series, 25)
    q75 = np.percentile(clean_series, 75)
    iqr_filtered = clean_series[(clean_series >= q25) & (clean_series <= q75)]
    return iqr_filtered.mean() if not iqr_filtered.empty else clean_series.median()

def _base_layout(y_title="", x_title="", barmode=None, height=360):
    layout = {
        "paper_bgcolor": _THEME["bg_color"],
        "plot_bgcolor": _THEME["bg_color"],
        "font": {"color": _THEME["text_color"], "family": _THEME["font_family"]},
        "margin": {"l": 55, "r": 20, "t": 35, "b": 45},
        "height": height,
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11, "color": "#64748B"}
        },
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12, "color": "#64748B"}}, 
            "gridcolor": _THEME["grid_color"], 
            "zeroline": False, 
            "tickfont": {"size": 11, "color": "#64748B"}
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12, "color": "#64748B"}}, 
            "gridcolor": _THEME["grid_color"], 
            "zeroline": False, 
            "tickfont": {"size": 11, "color": "#64748B"}
        }
    }
    if barmode:
        layout["barmode"] = barmode
    return layout

def compute_kpis(df):
    if df is None:
        return {}
    operators, op_col = _get_valid_operators(df)
    kpi_dict = {}
    if not op_col:
        return kpi_dict
        
    dl_col = _resolve_column(df, ['download_mbps', 'val_download_kbps', 'download'])
    ul_col = _resolve_column(df, ['upload_mbps', 'val_upload_kbps', 'upload'])
    lat_col = _resolve_column(df, ['latency_ms', 'val_latency_iqm_ms', 'latency'])
    
    for op in operators:
        op_df = df[df[op_col] == op]
        dl_scale = 1/1000.0 if dl_col == 'val_download_kbps' else 1.0
        ul_scale = 1/1000.0 if ul_col == 'val_upload_kbps' else 1.0
        
        download_med = op_df[dl_col].median() * dl_scale if dl_col else 0.0
        upload_med   = op_df[ul_col].median() * ul_scale if ul_col else 0.0
        latency_iqm  = _interquartile_mean(op_df[lat_col]) if lat_col else 0.0
        
        kpi_dict[op] = {
            "download": download_med if pd.notna(download_med) else 0.0,
            "upload": upload_med if pd.notna(upload_med) else 0.0,
            "latency": latency_iqm if pd.notna(latency_iqm) else 0.0,
            "sample_count": int(len(op_df))
        }
    return kpi_dict

def build_operator_speed_bar(df):
    fig = go.Figure()
    if df is None:
        fig.update_layout(**_base_layout())
        return fig
    operators, op_col = _get_valid_operators(df)
    if not op_col:
        fig.update_layout(**_base_layout())
        return fig
    
    dl_col = _resolve_column(df, ['download_mbps', 'val_download_kbps', 'download'])
    ul_col = _resolve_column(df, ['upload_mbps', 'val_upload_kbps', 'upload'])
    dl_scale = 1/1000.0 if dl_col == 'val_download_kbps' else 1.0
    ul_scale = 1/1000.0 if ul_col == 'val_upload_kbps' else 1.0
    
    download_medians = []
    upload_medians = []
    
    for op in operators:
        op_df = df[df[op_col] == op]
        download_medians.append((op_df[dl_col].median() * dl_scale) if dl_col else 0)
        upload_medians.append((op_df[ul_col].median() * ul_scale) if ul_col else 0)
        
    fig.add_trace(go.Bar(
        x=operators, y=download_medians, name="Download",
        marker_color=[OPERATOR_COLORS[op] for op in operators],
        text=[f"{v:.1f}" for v in download_medians], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=operators, y=upload_medians, name="Upload",
        marker_color=[OPERATOR_COLORS_LIGHT[op] for op in operators],
        text=[f"{v:.1f}" for v in upload_medians], textposition='auto',
        marker_line_width=1.5, marker_line_color=[OPERATOR_COLORS[op] for op in operators]
    ))
    
    fig.update_layout(**_base_layout(y_title="Data Rate (Mbps)", barmode="group"))
    return fig

def build_latency_comparison(df):
    fig = go.Figure()
    if df is None:
        fig.update_layout(**_base_layout())
        return fig
    operators, op_col = _get_valid_operators(df)
    if not op_col:
        fig.update_layout(**_base_layout())
        return fig
    
    lat_col = _resolve_column(df, ['latency_ms', 'val_latency_iqm_ms', 'latency'])
    iqm_values = []
    
    for op in operators:
        op_df = df[df[op_col] == op]
        val = _interquartile_mean(op_df[lat_col]) if lat_col else 0
        iqm_values.append((op, val if pd.notna(val) else 0))
        
    iqm_values.sort(key=lambda x: x[1])
    sorted_ops = [x[0] for x in iqm_values]
    sorted_metrics = [x[1] for x in iqm_values]
    
    fig.add_trace(go.Bar(
        x=sorted_ops, y=sorted_metrics,
        marker_color=[OPERATOR_COLORS[op] for op in sorted_ops],
        text=[f"{v:.1f} ms" for v in sorted_metrics], textposition='auto', showlegend=False
    ))
    fig.update_layout(**_base_layout(y_title="Latency (ms)"))
    return fig

def build_signal_strength_comparison(df):
    """Kept for backward compatibility / potential reuse elsewhere. No
    longer called from the main dashboard layout, which now uses
    build_operator_radar_chart() in this slot instead."""
    fig = go.Figure()
    if df is None:
        fig.update_layout(**_base_layout())
        return fig
    operators, op_col = _get_valid_operators(df)
    rsrp_col = _resolve_column(df, ['rsrp', 'val_signal_rsrp_dbm', 'signal_strength'])
    if not op_col or not rsrp_col:
        fig.update_layout(**_base_layout())
        return fig
    
    good_pct, fair_pct, poor_pct = [], [], []
    
    for op in operators:
        op_df = df[df[op_col] == op]
        total = len(op_df)
        
        if total > 0:
            good = len(op_df[op_df[rsrp_col] >= -85])
            fair = len(op_df[(op_df[rsrp_col] < -85) & (op_df[rsrp_col] >= -105)])
            poor = len(op_df[op_df[rsrp_col] < -105])
            
            good_pct.append((good / total) * 100)
            fair_pct.append((fair / total) * 100)
            poor_pct.append((poor / total) * 100)
        else:
            good_pct.append(0); fair_pct.append(0); poor_pct.append(0)
            
    fig.add_trace(go.Bar(
        y=operators, x=good_pct, name="Excellent (>= -85 dBm)",
        orientation='h', marker_color='#059669',
        text=[f"{v:.1f}%" if v > 0 else "" for v in good_pct], textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=operators, x=fair_pct, name="Degraded (-85 to -105 dBm)",
        orientation='h', marker_color='#D97706',
        text=[f"{v:.1f}%" if v > 0 else "" for v in fair_pct], textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=operators, x=poor_pct, name="Critical (< -105 dBm)",
        orientation='h', marker_color='#DC2626',
        text=[f"{v:.1f}%" if v > 0 else "" for v in poor_pct], textposition='inside'
    ))
    
    fig.update_layout(**_base_layout(x_title="Sample Distribution (%)", barmode="stack"))
    fig.update_xaxes(range=[0, 100])
    return fig

def build_connection_type_bar(df):
    fig = go.Figure()
    if df is None:
        fig.update_layout(**_base_layout())
        return fig
    operators, op_col = _get_valid_operators(df)
    tech_col = _resolve_column(df, ['network_type', 'connection_type', 'attr_connection_type_start_string', 'attr_connection_type_end_string'])
    if not op_col or not tech_col:
        fig.update_layout(**_base_layout())
        return fig
    
    gen_mapping = {
        '5G': ['5G', 'NR', 'NRNSA'],
        '4G': ['4G', 'LTE', 'LTE_CA'],
        '3G/Legacy': ['3G', 'HSPA+', 'HSPA', 'UMTS', 'EDGE', 'GPRS']
    }
    
    generation_data = {'5G': [], '4G': [], '3G/Legacy': []}
    
    for op in operators:
        op_df = df[df[op_col] == op]
        cellular_df = op_df[~op_df[tech_col].astype(str).str.upper().str.contains('WIFI', na=False)]
        total = len(cellular_df)
        
        if total > 0:
            def categorize_gen(val):
                val_str = str(val).upper().strip()
                for gen, aliases in gen_mapping.items():
                    if any(alias in val_str for alias in aliases):
                        return gen
                return '3G/Legacy'
            
            mapped_series = cellular_df[tech_col].apply(categorize_gen)
            counts = mapped_series.value_counts()
            
            for gen in generation_data.keys():
                generation_data[gen].append((counts.get(gen, 0) / total) * 100)
        else:
            for gen in generation_data.keys():
                generation_data[gen].append(0)
                
    gen_colors = {'5G': '#1E3A8A', '4G': '#3B82F6', '3G/Legacy': '#64748B'}
    
    for gen, percentages in generation_data.items():
        fig.add_trace(go.Bar(
            x=operators, y=percentages, name=gen,
            marker_color=gen_colors[gen],
            text=[f"{v:.1f}%" if v > 0 else "" for v in percentages], textposition='inside'
        ))
        
    fig.update_layout(**_base_layout(y_title="Spectral Allocation Share (%)", barmode="stack"))
    fig.update_yaxes(range=[0, 100])
    return fig

# ==========================================================
# 5-AXIS OPERATOR RADAR / SPIDER CHART
# ==========================================================

RADAR_AXES = [
    {"key": "download", "label": "Download", "cols": ['download_mbps', 'val_download_kbps', 'download'], "unit": "Mbps", "lower_is_better": False, "kbps_col": "val_download_kbps"},
    {"key": "upload", "label": "Upload", "cols": ['upload_mbps', 'val_upload_kbps', 'upload'], "unit": "Mbps", "lower_is_better": False, "kbps_col": "val_upload_kbps"},
    {"key": "latency", "label": "Latency", "cols": ['latency_ms', 'val_latency_iqm_ms', 'latency'], "unit": "ms", "lower_is_better": True, "kbps_col": None},
    {"key": "jitter", "label": "Jitter", "cols": ['val_jitter_ms', 'jitter_ms', 'jitter'], "unit": "ms", "lower_is_better": True, "kbps_col": None},
    {"key": "rsrp", "label": "RSRP", "cols": ['val_signal_rsrp_dbm', 'rsrp', 'signal_strength'], "unit": "dBm", "lower_is_better": False, "kbps_col": None},
]


def _axis_operator_values(df, op_col, operators, axis):
    """Returns {operator: median_value} for one radar axis, scaling
    kbps->Mbps where relevant, using the interquartile mean for
    latency/jitter (matches the rest of the dashboard's convention)."""
    col = _resolve_column(df, axis["cols"])
    values = {}
    if not col:
        return values
    scale = 1 / 1000.0 if col == axis["kbps_col"] else 1.0
    for op in operators:
        op_series = df[df[op_col] == op][col]
        if axis["key"] in ("latency", "jitter"):
            val = _interquartile_mean(op_series)
        else:
            numeric = pd.to_numeric(op_series, errors="coerce").dropna()
            val = numeric.median() * scale if not numeric.empty else np.nan
        if pd.notna(val):
            values[op] = float(val)
    return values


def build_operator_radar_chart(df):
    """5-axis radar/spider chart (Download, Upload, Latency, Jitter,
    Signal/RSRP) comparing operators. Each axis is normalized using
    fixed absolute reference bounds based on realistic p5-p95 ranges,
    so outliers don't distort the scale. Raw values are clipped to
    these bounds before normalization."""
    fig = go.Figure()
    if df is None:
        fig.update_layout(**_base_layout())
        return fig

    operators, op_col = _get_valid_operators(df)
    if not op_col or not operators:
        fig.update_layout(**_base_layout())
        return fig

    # Fixed reference bounds per axis (min, max)
    AXIS_BOUNDS = {
        "download": (0, 100),   # Mbps
        "upload": (0, 30),      # Mbps
        "latency": (15, 100),   # ms (lower is better)
        "jitter": (0, 30),      # ms (lower is better)
        "rsrp": (-115, -70),    # dBm (higher is better)
    }

    axis_labels = [axis["label"] for axis in RADAR_AXES]
    raw_by_axis = {}
    for axis in RADAR_AXES:
        raw_by_axis[axis["key"]] = _axis_operator_values(df, op_col, operators, axis)

    scores = {op: [] for op in operators}
    for axis in RADAR_AXES:
        key = axis["key"]
        raw_values = raw_by_axis[key]
        lo, hi = AXIS_BOUNDS[key]
        
        for op in operators:
            if op not in raw_values:
                scores[op].append(0.0)
                continue
            
            # Clip raw value to bounds
            clipped = max(lo, min(hi, raw_values[op]))
            
            # Normalize to 0-100
            span = hi - lo
            if span == 0:
                scores[op].append(100.0)
                continue
            
            norm = (clipped - lo) / span
            if axis["lower_is_better"]:
                norm = 1.0 - norm
            scores[op].append(round(norm * 100, 1))

    for op in operators:
        r_values = scores[op] + scores[op][:1]
        theta_values = axis_labels + axis_labels[:1]

        hover_lines = []
        for axis in RADAR_AXES:
            raw_val = raw_by_axis[axis["key"]].get(op)
            if raw_val is None:
                hover_lines.append(f"{axis['label']}: no data")
            else:
                hover_lines.append(f"{axis['label']}: {raw_val:.1f} {axis['unit']}")
        hover_text = "<br>".join(hover_lines)

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself',
            name=op,
            line=dict(color=OPERATOR_COLORS.get(op, "#64748B"), width=2),
            fillcolor=OPERATOR_COLORS_LIGHT.get(op, "rgba(100,116,139,0.25)"),
            hovertext=[hover_text] * len(r_values),
            hoverinfo="text+name",
        ))

    fig.update_layout(
        paper_bgcolor=_THEME["bg_color"],
        font={"color": _THEME["text_color"], "family": _THEME["font_family"]},
        polar=dict(
            bgcolor=_THEME["bg_color"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor=_THEME["grid_color"],
            ),
            angularaxis=dict(
                gridcolor=_THEME["grid_color"],
                tickfont={"size": 11, "color": "#64748B"},
            ),
        ),
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11, "color": "#64748B"}
        },
        margin={"l": 60, "r": 60, "t": 35, "b": 40},
        height=420,
    )
    return fig