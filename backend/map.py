from __future__ import annotations

import json
import os
import unicodedata
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point, shape

PANEL = "#0d1629"
NO_DATA_COLOR = "#E2E8F0"
BORDER_COLOR = "#94A3B8"
HIGHLIGHT_BORDER_COLOR = "#1F2937"

GEOJSON_PATH = os.path.join("assets", "tunisia_governorates.geojson")
REPAIRED_GEOJSON_PATH = os.path.join("output", "tunisia_governorates_repaired.geojson")

_GEOJSON_CACHE: dict | None = None
_GOV_CACHE: dict | None = None
_SHAPELY_CACHE: dict | None = None

REGION_ALIASES = {
    "ariana": "Ariana",
    "beja": "Beja",
    "ben arous": "Ben Arous",
    "bizerte": "Bizerte",
    "gabes": "Gabes",
    "gafsa": "Gafsa",
    "jendouba": "Jendouba",
    "jandouba": "Jendouba",
    "kairouan": "Kairouan",
    "kasserine": "Kasserine",
    "kebili": "Kebili",
    "kef": "Le Kef",
    "le kef": "Le Kef",
    "mahdia": "Mahdia",
    "la manouba": "Mannouba",
    "manouba": "Mannouba",
    "manubah": "Mannouba",
    "mannouba": "Mannouba",
    "medenine": "Mednine",
    "mednine": "Mednine",
    "monastir": "Monastir",
    "nabeul": "Nabeul",
    "sfax": "Sfax",
    "sidi bou zid": "Sidi Bou Zid",
    "sidi bouzid": "Sidi Bou Zid",
    "siliana": "Siliana",
    "sousse": "Sousse",
    "tataouine": "Tataouine",
    "tozeur": "Tozeur",
    "tunis": "Tunis",
    "zaghouan": "Zaghouan",
}

OPERATOR_COLORS = {
    "Tunisie Telecom": "#005BB6",
    "Ooredoo": "#E11A22",
    "Orange": "#FF6600",
}

KPI_CONFIGS = {
    "download_mbps": {"label": "Download Speed", "unit": "Mbps", "lower_is_better": False},
    "upload_mbps": {"label": "Upload Speed", "unit": "Mbps", "lower_is_better": False},
    "latency_ms": {"label": "Latency", "unit": "ms", "lower_is_better": True},
    "val_signal_rsrp_dbm": {"label": "Signal Strength (RSRP)", "unit": "dBm", "lower_is_better": False},
}

# ==========================================================
# Absolute 4-band thresholds (green/yellow/orange/red), used by the
# Page 2 (single-operator) choropleth instead of the operator-winner
# coloring used on the main page. Band 0 in each list is always "best"
# for that KPI's direction.
# ==========================================================
THRESHOLD_BANDS = {
    "download_mbps": [
        (100, float("inf"), "#2ecc71", "Excellent (>100 Mbps)"),
        (50, 100, "#f1c40f", "Good (50-100 Mbps)"),
        (20, 50, "#e67e22", "Fair (20-50 Mbps)"),
        (float("-inf"), 20, "#e74c3c", "Poor (<20 Mbps)"),
    ],
    "upload_mbps": [
        (50, float("inf"), "#2ecc71", "Excellent (>50 Mbps)"),
        (20, 50, "#f1c40f", "Good (20-50 Mbps)"),
        (10, 20, "#e67e22", "Fair (10-20 Mbps)"),
        (float("-inf"), 10, "#e74c3c", "Poor (<10 Mbps)"),
    ],
    "latency_ms": [
        (float("-inf"), 20, "#2ecc71", "Excellent (<20 ms)"),
        (20, 50, "#f1c40f", "Good (20-50 ms)"),
        (50, 100, "#e67e22", "Fair (50-100 ms)"),
        (100, float("inf"), "#e74c3c", "Poor (>100 ms)"),
    ],
    "val_jitter_ms": [
        (float("-inf"), 10, "#2ecc71", "Excellent (<10 ms)"),
        (10, 20, "#f1c40f", "Good (10-20 ms)"),
        (20, 40, "#e67e22", "Fair (20-40 ms)"),
        (40, float("inf"), "#e74c3c", "Poor (>40 ms)"),
    ],
    "val_signal_rsrp_dbm": [
        (-80, float("inf"), "#2ecc71", "Excellent (>-80 dBm)"),
        (-95, -80, "#f1c40f", "Good (-95 to -80 dBm)"),
        (-105, -95, "#e67e22", "Fair (-105 to -95 dBm)"),
        (float("-inf"), -105, "#e74c3c", "Poor (<-105 dBm)"),
    ],
}

# Column candidates + display config for every KPI selectable on the
# Page 2 absolute-threshold map (adds Jitter, which the winner-mode
# KPI_CONFIGS above does not include).
THRESHOLD_KPI_OPTIONS = {
    "download_mbps": {"label": "Download Speed", "unit": "Mbps", "cols": ["download_mbps", "val_download_kbps", "download"], "kbps_col": "val_download_kbps"},
    "upload_mbps": {"label": "Upload Speed", "unit": "Mbps", "cols": ["upload_mbps", "val_upload_kbps", "upload"], "kbps_col": "val_upload_kbps"},
    "latency_ms": {"label": "Latency", "unit": "ms", "cols": ["latency_ms", "val_latency_iqm_ms", "latency"], "kbps_col": None},
    "val_jitter_ms": {"label": "Jitter", "unit": "ms", "cols": ["val_jitter_ms", "jitter_ms", "jitter"], "kbps_col": None},
    "val_signal_rsrp_dbm": {"label": "Signal Strength (RSRP)", "unit": "dBm", "cols": ["val_signal_rsrp_dbm", "rsrp", "signal_strength"], "kbps_col": None},
}


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_region_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = _strip_accents(name).strip().lower()
    return REGION_ALIASES.get(key, name.strip())


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_geojson() -> dict:
    global _GEOJSON_CACHE
    if _GEOJSON_CACHE is not None:
        return _GEOJSON_CACHE

    path = REPAIRED_GEOJSON_PATH if os.path.exists(REPAIRED_GEOJSON_PATH) else GEOJSON_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        _GEOJSON_CACHE = json.load(f)

    _GEOJSON_CACHE["_id_key"] = "gouv_fr"
    return _GEOJSON_CACHE


def _extract_rings(geometry: dict | None):
    if not geometry:
        return []
    rings = []
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            if len(ring) >= 3:
                lon, lat = zip(*ring)
                rings.append({"lon": list(lon), "lat": list(lat)})
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                if len(ring) >= 3:
                    lon, lat = zip(*ring)
                    rings.append({"lon": list(lon), "lat": list(lat)})
    return rings


def _load_governorates() -> dict:
    global _GOV_CACHE
    if _GOV_CACHE is not None:
        return _GOV_CACHE

    gj = _load_geojson()
    id_key = gj.get("_id_key", "gouv_fr")
    out = {}

    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        name = props.get(id_key)
        if not name:
            continue

        geom = feat.get("geometry")
        if not geom:
            continue

        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            if not coords or len(coords[0]) < 3:
                continue
        elif geom.get("type") == "MultiPolygon":
            if not coords or any(len(poly[0]) < 3 for poly in coords):
                continue

        out[name] = {
            "name": name,
            "props": props,
            "geometry": geom,
            "rings": _extract_rings(geom),
            "id_key": id_key,
        }

    _GOV_CACHE = out
    return _GOV_CACHE


def _load_shapely_polygons():
    global _SHAPELY_CACHE
    if _SHAPELY_CACHE is not None:
        return _SHAPELY_CACHE

    out = {}
    for name, entry in _load_governorates().items():
        geom = entry.get("geometry")
        if not geom:
            continue
        try:
            s = shape(geom)
            if not s.is_valid:
                continue
            out[name] = s
        except Exception:
            continue

    _SHAPELY_CACHE = out
    return _SHAPELY_CACHE


def get_region_for_point(lon: float, lat: float) -> str | None:
    try:
        pt = Point(float(lon), float(lat))
    except Exception:
        return None

    for name, poly in _load_shapely_polygons().items():
        try:
            if poly.contains(pt):
                return name
        except Exception:
            pass
    return None


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")


def _compute_region_operator_winners(df: pd.DataFrame, kpi: str) -> dict:
    """Groups rows by (normalized region, operator), takes the median KPI
    per group, and picks the winning operator per region. Region names are
    normalized (accents/casing/aliases) via normalize_region_name() before
    grouping, so raw CSV values like 'TUNIS' or 'tunis' correctly match the
    canonical GeoJSON governorate name 'Tunis' used for map lookups.
    Used only by the main-page, multi-operator winner-coloring map."""
    region_col = _resolve_column(df, ["attr_place_region", "region"])
    operator_col = _resolve_column(df, ["operator", "attr_isp_common_name"])
    if region_col is None or operator_col is None or df.empty:
        return {}

    kpi_col = kpi if kpi in df.columns else _resolve_column(df, [kpi])
    if kpi_col is None:
        return {}

    temp = df[[region_col, operator_col, kpi_col]].copy()
    temp["_kpi"] = _clean_numeric(temp[kpi_col])
    temp["_region_norm"] = temp[region_col].map(normalize_region_name)
    temp = temp.dropna(subset=["_kpi", "_region_norm"])
    if temp.empty:
        return {}

    grouped = temp.groupby(["_region_norm", operator_col], dropna=True, observed=False)["_kpi"].median().reset_index()
    results = {}
    lower_is_better = KPI_CONFIGS.get(kpi, {}).get("lower_is_better", False)

    for region, grp in grouped.groupby("_region_norm", observed=False):
        by_operator = {str(row[operator_col]): float(row["_kpi"]) for _, row in grp.iterrows()}
        if not by_operator:
            continue
        if lower_is_better:
            winner = min(by_operator.items(), key=lambda kv: kv[1])[0]
        else:
            winner = max(by_operator.items(), key=lambda kv: kv[1])[0]
        results[str(region)] = {
            "winner": winner,
            "winner_value": by_operator[winner],
            "by_operator": by_operator,
        }
    return results


def _build_hover_text(region_name: str, region_results: dict, config: dict) -> str:
    result = region_results.get(region_name)
    if result is None:
        return f"<b>{region_name}</b><br>No data<extra></extra>"

    winner = result["winner"]
    winner_value = result["winner_value"]
    lines = [f"<b>{region_name}</b>", f"Best: {winner} ({winner_value:.1f} {config['unit']})", "—"]
    ordered = sorted(result["by_operator"].items(), key=lambda kv: kv[1], reverse=not config["lower_is_better"])
    for op, val in ordered:
        lines.append(f"{op}: {val:.1f} {config['unit']}")
    return "<br>".join(lines) + "<extra></extra>"


def get_region_info_text(region_name: str, df: pd.DataFrame, kpi: str = "download_mbps") -> str:
    region_col = _resolve_column(df, ["attr_place_region", "region"])
    if region_col is not None:
        region_df = df[df[region_col].astype(str).map(normalize_region_name) == str(region_name)]
    else:
        region_df = df.copy()

    kpi_col = kpi if kpi in region_df.columns else _resolve_column(region_df, [kpi])
    if kpi_col is None or region_df.empty:
        return f"**{region_name}**\nNo KPI data available."

    vals = _clean_numeric(region_df[kpi_col]).dropna()
    if vals.empty:
        return f"**{region_name}**\nNo KPI samples."

    median = vals.median()
    label = KPI_CONFIGS.get(kpi, {}).get("label", kpi)
    unit = KPI_CONFIGS.get(kpi, {}).get("unit", "")
    return f"**{region_name}**\nSamples: {len(vals)}\nMedian {label}: {median:.2f} {unit}"


def build_choropleth_map(
    df: pd.DataFrame,
    kpi: str = "download_mbps",
    highlight_region: str | None = None,
) -> go.Figure:
    """Main-page map: colors each governorate by whichever operator has
    the best median value for the selected KPI in that region (operator
    winner-coloring). Unchanged from the previous version."""
    if kpi not in KPI_CONFIGS:
        kpi = "download_mbps"
    config = KPI_CONFIGS[kpi]

    governorates = _load_governorates()
    region_results = _compute_region_operator_winners(df, kpi)

    highlight_gouv = None
    if highlight_region and highlight_region != "All Regions":
        highlight_gouv = normalize_region_name(highlight_region)

    fig = go.Figure()

    for gouv_name, entry in governorates.items():
        result = region_results.get(gouv_name)
        if result is None:
            color = NO_DATA_COLOR
        else:
            winner = result["winner"]
            color = OPERATOR_COLORS.get(winner, NO_DATA_COLOR)

        is_highlighted = highlight_gouv is not None and gouv_name == highlight_gouv
        line_color = HIGHLIGHT_BORDER_COLOR if is_highlighted else BORDER_COLOR
        line_width = 3 if is_highlighted else 1

        hover = _build_hover_text(gouv_name, region_results, config)

        for ring in entry["rings"]:
            fig.add_trace(go.Scatter(
                x=ring["lon"],
                y=ring["lat"],
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=line_color, width=line_width),
                hoveron="fills",
                hovertemplate=hover,
                name=gouv_name,
                showlegend=False,
            ))

    for operator, color in OPERATOR_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=operator,
            showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color=NO_DATA_COLOR),
        name="No data",
        showlegend=True,
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
    )
    return fig


def build_network_map(frame: pd.DataFrame, kpi: str = "download_mbps", highlight_region: str | None = None) -> go.Figure:
    return build_choropleth_map(frame, kpi=kpi, highlight_region=highlight_region)


# ==========================================================
# PAGE 2 (single-operator) ABSOLUTE THRESHOLD CHOROPLETH
# ==========================================================

def _region_median_values(df: pd.DataFrame, kpi: str) -> dict[str, float]:
    """Median raw KPI value per normalized region, across whatever rows
    are in df (caller is expected to have already filtered to a single
    operator, e.g. Tunisie Telecom, before calling this)."""
    region_col = _resolve_column(df, ["attr_place_region", "region"])
    kpi_conf = THRESHOLD_KPI_OPTIONS.get(kpi)
    if region_col is None or kpi_conf is None or df is None or df.empty:
        return {}

    kpi_col = _resolve_column(df, kpi_conf["cols"])
    if kpi_col is None:
        return {}

    temp = df[[region_col, kpi_col]].copy()
    temp["_kpi"] = _clean_numeric(temp[kpi_col])
    if kpi_col == kpi_conf["kbps_col"]:
        temp["_kpi"] = temp["_kpi"] / 1000.0
    temp["_region_norm"] = temp[region_col].map(normalize_region_name)
    temp = temp.dropna(subset=["_kpi", "_region_norm"])
    if temp.empty:
        return {}

    medians = temp.groupby("_region_norm", observed=False)["_kpi"].median()
    return medians.to_dict()


def _bucket_for_value(value: float, kpi: str) -> tuple[str, str]:
    for low, high, color, label in THRESHOLD_BANDS.get(kpi, []):
        if low <= value < high:
            return color, label
    return NO_DATA_COLOR, "No data"


def build_absolute_threshold_map(
    df: pd.DataFrame,
    kpi: str = "download_mbps",
    highlight_region: str | None = None,
) -> go.Figure:
    """Page 2 map: colors each governorate using fixed green/yellow/orange/
    red performance bands for the selected KPI's median value in that
    region, instead of comparing operators against each other. Intended
    to be called with `df` already filtered to a single operator (e.g.
    Tunisie Telecom) upstream in app.py."""
    if kpi not in THRESHOLD_KPI_OPTIONS:
        kpi = "download_mbps"
    kpi_conf = THRESHOLD_KPI_OPTIONS[kpi]

    governorates = _load_governorates()
    region_values = _region_median_values(df, kpi)

    highlight_gouv = None
    if highlight_region and highlight_region != "All Regions":
        highlight_gouv = normalize_region_name(highlight_region)

    fig = go.Figure()

    for gouv_name, entry in governorates.items():
        value = region_values.get(gouv_name)
        if value is None:
            color, bucket_label = NO_DATA_COLOR, "No data"
            value_text = "No data"
        else:
            color, bucket_label = _bucket_for_value(value, kpi)
            value_text = f"{value:.1f} {kpi_conf['unit']}"

        is_highlighted = highlight_gouv is not None and gouv_name == highlight_gouv
        line_color = HIGHLIGHT_BORDER_COLOR if is_highlighted else BORDER_COLOR
        line_width = 3 if is_highlighted else 1

        hover = f"<b>{gouv_name}</b><br>{kpi_conf['label']}: {value_text}<br>{bucket_label}<extra></extra>"

        for ring in entry["rings"]:
            fig.add_trace(go.Scatter(
                x=ring["lon"],
                y=ring["lat"],
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=line_color, width=line_width),
                hoveron="fills",
                hovertemplate=hover,
                name=gouv_name,
                showlegend=False,
            ))

    for _, _, color, label in THRESHOLD_BANDS.get(kpi, []):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,
            showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color=NO_DATA_COLOR),
        name="No data",
        showlegend=True,
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
    )
    return fig