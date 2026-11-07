import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_SAMPLES_PER_REGION = 10   # regions with fewer tests have an unreliable median
MIN_REGIONS = 3
MIN_K = 2
MAX_K = 7

# Tier names and colors for every possible k, ordered fastest -> slowest
_TIER_NAMES = {
    2: ["High speed", "Low speed"],
    3: ["High speed", "Medium speed", "Low speed"],
    4: ["High speed", "Upper-medium speed", "Lower-medium speed", "Low speed"],
    5: ["Very high speed", "High speed", "Medium speed", "Low speed", "Very low speed"],
    6: ["Very high speed", "High speed", "Upper-medium speed",
        "Lower-medium speed", "Low speed", "Very low speed"],
    7: ["Top speed", "Very high speed", "High speed", "Medium speed",
        "Low speed", "Very low speed", "Lowest speed"],
}
_TIER_COLORS = {
    2: ["#10B981", "#EF4444"],
    3: ["#10B981", "#F59E0B", "#EF4444"],
    4: ["#10B981", "#84CC16", "#F59E0B", "#EF4444"],
    5: ["#059669", "#10B981", "#F59E0B", "#F97316", "#EF4444"],
    6: ["#059669", "#10B981", "#84CC16", "#F59E0B", "#F97316", "#EF4444"],
    7: ["#047857", "#10B981", "#84CC16", "#FACC15", "#F59E0B", "#F97316", "#EF4444"],
}
_BLANK = {"", "nan", "none", "null", "<na>"}


def _resolve_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color="#94A3B8"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig, pd.DataFrame(), 0


def _place_labels(points_xy, base_radius=22, radius_step=8, n_angles=12, max_rings=4):
    """Greedy search: for each point, find an (xshift, yshift) offset in px that
    avoids other points and already placed labels."""
    placed = []
    results = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for (px, py) in points_xy:
        best = None
        for ring in range(max_rings):
            radius = base_radius + ring * radius_step
            for a in angles:
                lx = px + radius * np.cos(a)
                ly = py + radius * np.sin(a)
                ok = all((ox - lx) ** 2 + (oy - ly) ** 2 >= (base_radius * 0.6) ** 2
                         for (ox, oy) in points_xy)
                if ok:
                    ok = all((lxp - lx) ** 2 + (lyp - ly) ** 2 >= (base_radius * 0.9) ** 2
                             for (lxp, lyp) in placed)
                if ok:
                    best = (radius * np.cos(a), radius * np.sin(a))
                    placed.append((lx, ly))
                    break
            if best is not None:
                break
        if best is None:
            radius = base_radius + max_rings * radius_step
            best = (0, -radius)
            placed.append((px, py - radius))
        results.append(best)
    return results


def _choose_k(X):
    """Best number of clusters (MIN_K..MAX_K) by silhouette score.
    k must stay below the number of regions for the silhouette to be defined."""
    k_max = min(MAX_K, len(X) - 1)
    best_k, best_score = MIN_K, -1.0
    for k in range(MIN_K, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def build_region_clustering_chart(df):
    if df is None or df.empty:
        return _empty_figure("Missing columns")

    region_col = _resolve_column(df, ["attr_place_region", "region"])
    dl_col = _resolve_column(df, ["download_mbps", "val_download_kbps", "download"])
    ul_col = _resolve_column(df, ["upload_mbps", "val_upload_kbps", "upload"])

    if not region_col or not dl_col or not ul_col:
        return _empty_figure("Missing columns")

    work = df[[region_col, dl_col, ul_col]].copy()
    for c in (dl_col, ul_col):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[region_col] = work[region_col].astype(str).str.strip()
    work = work[~work[region_col].str.lower().isin(_BLANK)]
    work = work.dropna(subset=[dl_col, ul_col])

    if "kbps" in dl_col:
        work[dl_col] = work[dl_col] / 1000.0
    if "kbps" in ul_col:
        work[ul_col] = work[ul_col] / 1000.0

    # One row per region: median speeds + number of tests
    agg = work.groupby(region_col).agg(**{
        dl_col: (dl_col, "median"),
        ul_col: (ul_col, "median"),
        "sample_count": (dl_col, "size"),
    }).reset_index()
    agg = agg[agg["sample_count"] >= MIN_SAMPLES_PER_REGION].reset_index(drop=True)

    if len(agg) < MIN_REGIONS:
        return _empty_figure(
            f"Not enough regions (need {MIN_REGIONS} with at least "
            f"{MIN_SAMPLES_PER_REGION} tests each)")

    # Clip outliers, then standardize
    for c in (dl_col, ul_col):
        q1, q99 = agg[c].quantile([0.01, 0.99])
        agg[c] = agg[c].clip(q1, q99)
    X = StandardScaler().fit_transform(agg[[dl_col, ul_col]].values)

    best_k = _choose_k(X)
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)

    # Rank clusters from fastest to slowest so colors and names are meaningful
    strength = pd.Series(X.sum(axis=1)).groupby(labels).mean().sort_values(ascending=False)
    rank = {old: new for new, old in enumerate(strength.index)}
    agg["cluster"] = [rank[l] for l in labels]
    names = _TIER_NAMES[best_k]
    colors = _TIER_COLORS[best_k]
    agg["cluster_name"] = agg["cluster"].map(lambda i: names[i])

    # Label placement in an approximate pixel space
    x_range = (agg[dl_col].max() - agg[dl_col].min()) or 1.0
    y_range = (agg[ul_col].max() - agg[ul_col].min()) or 1.0
    plot_w, plot_h = 700, 500
    pts_px = [
        ((row[dl_col] - agg[dl_col].min()) / x_range * plot_w,
         (1 - (row[ul_col] - agg[ul_col].min()) / y_range) * plot_h)
        for _, row in agg.iterrows()
    ]
    label_offsets = _place_labels(pts_px)

    annotations = []
    for (_, row), (offset_x, offset_y) in zip(agg.iterrows(), label_offsets):
        annotations.append(dict(
            x=row[dl_col], y=row[ul_col],
            text=f"<b>{str(row[region_col]).title()}</b>",
            showarrow=False,
            font=dict(size=11, color="#1F2937", family="Inter, system-ui, sans-serif"),
            xshift=offset_x, yshift=offset_y,
            bgcolor="rgba(255, 255, 255, 0.88)",
            bordercolor="rgba(15, 23, 42, 0.12)",
            borderwidth=1, borderpad=5,
        ))

    fig = go.Figure()
    hover = ("<b>%{text}</b><br>"
             "Download: %{customdata[0]:.1f} Mbps<br>"
             "Upload: %{customdata[1]:.1f} Mbps<br>"
             "Samples: %{customdata[2]}<extra></extra>")
    for cid in range(best_k):
        sub = agg[agg["cluster"] == cid]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[dl_col], y=sub[ul_col], mode="markers",
            text=sub[region_col].astype(str).str.title(),
            customdata=sub[[dl_col, ul_col, "sample_count"]].values,
            hovertemplate=hover,
            marker=dict(size=16, color=colors[cid]),
            name=names[cid],
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1F2937", family="Inter"),
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis=dict(title="Download (Mbps)", gridcolor="#E2E8F0",
                   tickfont=dict(color="#64748B"), title_font=dict(color="#64748B")),
        yaxis=dict(title="Upload (Mbps)", gridcolor="#E2E8F0",
                   tickfont=dict(color="#64748B"), title_font=dict(color="#64748B")),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    font=dict(color="#64748B")),
        annotations=annotations,
        updatemenus=[dict(
            buttons=[
                dict(args=[{"annotations": annotations}], label="<b>Show Labels</b>", method="relayout"),
                dict(args=[{"annotations": []}], label="<b>Hide Labels</b>", method="relayout"),
            ],
            direction="down", showactive=True, active=0,
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.9)",
            font=dict(color="#1F2937", size=11, family="Inter, system-ui, sans-serif"),
            bordercolor="rgba(15, 23, 42, 0.15)", borderwidth=1,
            pad=dict(r=8, t=6, b=6, l=8),
        )],
    )

    return fig, agg, best_k