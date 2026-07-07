from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


BACKGROUND = "#070b16"
PANEL = "#0d1629"
GRID = "rgba(140, 178, 208, 0.18)"
TEXT = "#dff2ff"
MUTED = "#8fb3d0"
ACCENT = "#26d7ff"
GREEN = "#36e3a5"
AMBER = "#ffcb63"
RED = "#ff6d7a"

OPERATOR_COLORS = {
    "Ooredoo": RED,
    "Tunisie Telecom": "#1e88e5",
    "Orange": "#fb8c00",
}


def build_map(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        return fig

    map_frame = df.copy()
    if "attr_location_latitude" not in map_frame.columns or "attr_location_longitude" not in map_frame.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        return fig

    map_frame = map_frame.dropna(subset=["attr_location_latitude", "attr_location_longitude"])
    if map_frame.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        return fig

    map_frame["attr_location_latitude"] = pd.to_numeric(map_frame["attr_location_latitude"], errors="coerce")
    map_frame["attr_location_longitude"] = pd.to_numeric(map_frame["attr_location_longitude"], errors="coerce")
    map_frame = map_frame.dropna(subset=["attr_location_latitude", "attr_location_longitude"])

    map_frame["operator"] = map_frame.get("operator", "Other").fillna("Other").astype(str)
    if "download_mbps" in map_frame.columns:
        map_frame["download_mbps"] = pd.to_numeric(map_frame["download_mbps"], errors="coerce")
    else:
        map_frame["download_mbps"] = np.nan

    if "upload_mbps" in map_frame.columns:
        map_frame["upload_mbps"] = pd.to_numeric(map_frame["upload_mbps"], errors="coerce")
    else:
        map_frame["upload_mbps"] = np.nan

    if "latency_ms" in map_frame.columns:
        map_frame["latency_ms"] = pd.to_numeric(map_frame["latency_ms"], errors="coerce")
    else:
        map_frame["latency_ms"] = np.nan

    if "val_signal_rsrp_dbm" in map_frame.columns:
        map_frame["val_signal_rsrp_dbm"] = pd.to_numeric(map_frame["val_signal_rsrp_dbm"], errors="coerce")
    else:
        map_frame["val_signal_rsrp_dbm"] = np.nan

    size_values = map_frame["download_mbps"].fillna(map_frame["download_mbps"].median())
    if size_values.isna().all():
        size_values = pd.Series([12.0] * len(map_frame), index=map_frame.index)
    size_min = float(size_values.min())
    size_max = float(size_values.max())
    if size_max > size_min:
        bubble_sizes = 10 + (size_values - size_min) * (24 / (size_max - size_min))
    else:
        bubble_sizes = pd.Series([18.0] * len(map_frame), index=map_frame.index)

    fig = go.Figure()
    for operator, color in OPERATOR_COLORS.items():
        operator_frame = map_frame[map_frame["operator"] == operator]
        if operator_frame.empty:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=operator_frame["attr_location_latitude"],
                lon=operator_frame["attr_location_longitude"],
                mode="markers",
                name=operator,
                marker=dict(
                    size=bubble_sizes.loc[operator_frame.index],
                    color=color,
                    opacity=0.82,
                    sizemode="area",
                ),
                customdata=operator_frame[["operator", "download_mbps", "upload_mbps", "latency_ms", "val_signal_rsrp_dbm"]],
                hovertemplate=(
                    "Operator: %{customdata[0]}<br>"
                    "Download: %{customdata[1]:.2f} Mbps<br>"
                    "Upload: %{customdata[2]:.2f} Mbps<br>"
                    "Latency: %{customdata[3]:.1f} ms<br>"
                    "RSRP: %{customdata[4]:.1f} dBm<extra></extra>"
                ),
            )
        )

    other_frame = map_frame[~map_frame["operator"].isin(OPERATOR_COLORS)]
    if not other_frame.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=other_frame["attr_location_latitude"],
                lon=other_frame["attr_location_longitude"],
                mode="markers",
                name="Other",
                marker=dict(
                    size=bubble_sizes.loc[other_frame.index],
                    color=MUTED,
                    opacity=0.78,
                    sizemode="area",
                ),
                customdata=other_frame[["operator", "download_mbps", "upload_mbps", "latency_ms", "val_signal_rsrp_dbm"]],
                hovertemplate=(
                    "Operator: %{customdata[0]}<br>"
                    "Download: %{customdata[1]:.2f} Mbps<br>"
                    "Upload: %{customdata[2]:.2f} Mbps<br>"
                    "Latency: %{customdata[3]:.1f} ms<br>"
                    "RSRP: %{customdata[4]:.1f} dBm<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=5.6,
            center=dict(lat=34.0, lon=9.0),
            bearing=0,
            pitch=0,
        ),
        dragmode="pan",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        showlegend=True,
    )
    return fig


def build_network_map(frame):
    return build_map(frame)
