import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# ENTERPRISE TELECOM PALETTE DEFINITIONS
# ==========================================
OPERATOR_COLORS = {
    "Tunisie Telecom": "#005BB6",  # Deep Corporate Telecom Blue
    "Ooredoo": "#E11A22",          # Calibrated Matte Crimson
    "Orange": "#FF6600"            # Signal Orange
}

OPERATOR_COLORS_LIGHT = {
    "Tunisie Telecom": "rgba(0, 91, 182, 0.35)",
    "Ooredoo": "rgba(225, 26, 34, 0.35)",
    "Orange": "rgba(255, 102, 0, 0.35)"
}

# NOC-Style Neutral Interface Palette
_THEME = {
    "bg_color": "rgba(0,0,0,0)", 
    "text_color": "#F8FAFC",       # Off-white slate text
    "grid_color": "#1E293B",       # Subtle deep slate grid lines
    "font_family": "Inter, system-ui, Sans-Serif"
}

# ==========================================
# INTERNAL SELF-HEALING UTILITIES
# ==========================================
def _resolve_column(df, prioritized_list):
    for col in prioritized_list:
        if col in df.columns:
            return col
    return None

def _get_valid_operators(df):
    op_col = _resolve_column(df, ['operator', 'attr_sim_operator_common_name', 'attr_network_operator_common_name'])
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
            "font": {"size": 11, "color": "#94A3B8"}
        },
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12, "color": "#94A3B8"}}, 
            "gridcolor": _THEME["grid_color"], 
            "zeroline": False, 
            "tickfont": {"size": 11, "color": "#94A3B8"}
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12, "color": "#94A3B8"}}, 
            "gridcolor": _THEME["grid_color"], 
            "zeroline": False, 
            "tickfont": {"size": 11, "color": "#94A3B8"}
        }
    }
    if barmode:
        layout["barmode"] = barmode
    return layout

# ==========================================
# CORE KPI COMPUTATION METRICS
# ==========================================
def compute_kpis(df):
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

# ==========================================
# VISUALIZATION LAYER CONFIGURATIONS
# ==========================================
def build_operator_speed_bar(df):
    fig = go.Figure()
    operators, op_col = _get_valid_operators(df)
    if not op_col: return fig
    
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
    operators, op_col = _get_valid_operators(df)
    if not op_col: return fig
    
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
    fig = go.Figure()
    operators, op_col = _get_valid_operators(df)
    rsrp_col = _resolve_column(df, ['rsrp', 'val_signal_rsrp_dbm', 'signal_strength'])
    if not op_col or not rsrp_col: return fig
    
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
            
    # Telecom Standards Compliance Color Scale (Desaturated Mattes)
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
    operators, op_col = _get_valid_operators(df)
    tech_col = _resolve_column(df, ['network_type', 'connection_type', 'attr_connection_type_start_string', 'attr_connection_type_end_string'])
    if not op_col or not tech_col: return fig
    
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
                
    # Cohesive Telecom Blues & Grays Scale
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