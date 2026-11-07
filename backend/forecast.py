from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta

logger = logging.getLogger(__name__)

# Light theme: same palette as charts.py / clustering.py (the app is light).
_THEME = {
    "bg_color": "rgba(0,0,0,0)",
    "text_color": "#1F2937",
    "grid_color": "#E2E8F0",
    "muted_color": "#64748B",
    "font_family": "Inter, system-ui, Sans-Serif",
}

FORECAST_KPI_OPTIONS = {
    "download_mbps": "Download Speed (Mbps)",
    "upload_mbps": "Upload Speed (Mbps)",
    "latency_ms": "Latency (ms)",
    "val_jitter_ms": "Jitter (ms)",
}

# ---------------------------------------------------------------------------
# Forecasting parameters (design choices, documented in one place)
# ---------------------------------------------------------------------------
DAILY_AGGREGATION = "median"   # robust to outliers; consistent with the KPI cards
MIN_SAMPLES_PER_DAY = 5        # days with fewer tests are too noisy and are dropped
MAX_FILL_GAP_DAYS = 2          # interpolate at most 2 missing days in a row
SEASON_LENGTH = 7              # weekly cycle for daily data
MIN_POINTS_FORECAST = 8        # below this: observed points only, no forecast
SEASON_MIN_POINTS = 14         # two full weekly cycles needed to use seasonality
CONFIDENCE_LEVEL = 80          # prediction interval level (%)
CALIBRATION_WINDOWS = 4        # rolling windows used to calibrate the interval

LO_COL = f"AutoTheta-lo-{CONFIDENCE_LEVEL}"
HI_COL = f"AutoTheta-hi-{CONFIDENCE_LEVEL}"
NATIVE_LO = f"{LO_COL}-native"   # AutoTheta's original interval
NATIVE_HI = f"{HI_COL}-native"


def _resolve_column(df, prioritized_list):
    for col in prioritized_list:
        if col in df.columns:
            return col
    return None


def _base_layout(y_title="", x_title="", height=380, bottom_margin=45):
    return {
        "paper_bgcolor": _THEME["bg_color"],
        "plot_bgcolor": _THEME["bg_color"],
        "font": {"color": _THEME["text_color"], "family": _THEME["font_family"]},
        "margin": {"l": 55, "r": 20, "t": 35, "b": bottom_margin},
        "height": height,
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.06, "xanchor": "center", "x": 0.5,
                   "font": {"size": 11, "color": _THEME["muted_color"]}},
        "xaxis": {"title": {"text": x_title, "font": {"size": 12, "color": _THEME["muted_color"]}},
                  "gridcolor": _THEME["grid_color"], "zeroline": False,
                  "tickfont": {"size": 11, "color": _THEME["muted_color"]}},
        "yaxis": {"title": {"text": y_title, "font": {"size": 12, "color": _THEME["muted_color"]}},
                  "gridcolor": _THEME["grid_color"], "zeroline": False,
                  "tickfont": {"size": 11, "color": _THEME["muted_color"]}},
    }


def _message_figure(y_label, message, observed=None):
    """Chart with a short message, optionally showing the observed points only."""
    fig = go.Figure()
    if observed is not None and not observed.empty:
        fig.add_trace(go.Scatter(x=observed["day"], y=observed["y"], mode="markers",
                                 name="Observed", marker=dict(color="#3B82F6", size=9)))
        fig.update_layout(**_base_layout(y_title=y_label, x_title="Date"))
        fig.add_annotation(text=message, showarrow=False,
                           font=dict(color=_THEME["muted_color"]), yshift=30)
    else:
        fig.update_layout(**_base_layout(y_title=y_label))
        fig.add_annotation(text=message, showarrow=False, font=dict(color=_THEME["muted_color"]))
    return fig


# ---------------------------------------------------------------------------
# Series preparation (shared by the dashboard chart and the evaluation script)
# ---------------------------------------------------------------------------
@dataclass
class PreparedSeries:
    daily: pd.DataFrame | None = None     # columns: day, y  (None = not usable)
    median_gap: int = 1                   # median gap in days between observed days
    n_filled: int = 0                     # days interpolated to complete the calendar
    message: str | None = None            # why `daily` is None
    observed: pd.DataFrame | None = None  # points to show when no forecast is possible


def _detect_gap_days(days: pd.Series) -> int:
    """Median gap (in days) between consecutive observed days, at least 1."""
    diffs = days.diff().dropna().dt.days
    if diffs.empty:
        return 1
    return max(int(round(diffs.median())), 1)


def _regularize_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Puts a daily series on a complete calendar so that the weekly cycle is
    not shifted by missing days. Short gaps are interpolated; if a longer gap
    remains, only the most recent unbroken stretch is kept.
    Returns (series with columns day, y) and the number of interpolated days."""
    observed_days = set(daily["day"])
    s = daily.set_index("day")["y"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).interpolate(limit=MAX_FILL_GAP_DAYS, limit_area="inside")
    if s.isna().any():
        s = s[s.index > s[s.isna()].index.max()]
    n_filled = int(len(s) - s.index.isin(list(observed_days)).sum())
    out = s.rename_axis("day").reset_index(name="y")
    return out, n_filled


def prepare_daily_series(df, kpi: str) -> PreparedSeries:
    """Raw rows -> one value per day (median of days with enough samples), put on a
    complete daily calendar when the data is daily. Day boundaries use Tunisian local
    time. Returns a PreparedSeries; `daily` is None (with a message) if unusable."""
    if df is None or df.empty:
        return PreparedSeries(message="No timestamp/KPI data available")

    ts_col = _resolve_column(df, ["ts_result", "ts_result_received", "timestamp", "date"])
    kpi_col = _resolve_column(df, [kpi])
    if not ts_col or not kpi_col:
        return PreparedSeries(message="No timestamp/KPI data available")

    work = df[[ts_col, kpi_col]].copy()
    ts = pd.to_datetime(work[ts_col], errors="coerce", utc=True)
    work[ts_col] = ts.dt.tz_convert("Africa/Tunis").dt.tz_localize(None)
    work[kpi_col] = pd.to_numeric(work[kpi_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return PreparedSeries(message="No valid data after cleaning")

    work["day"] = work[ts_col].dt.floor("D")
    grouped = work.groupby("day")[kpi_col].agg(y=DAILY_AGGREGATION, n="size")
    grouped = grouped[grouped["n"] >= MIN_SAMPLES_PER_DAY]
    daily = grouped.reset_index()[["day", "y"]]
    if len(daily) < 2:
        return PreparedSeries(
            observed=daily,
            message=f"Not enough days with at least {MIN_SAMPLES_PER_DAY} samples for a forecast")

    median_gap = _detect_gap_days(daily["day"])
    n_filled = 0
    if median_gap == 1:
        daily, n_filled = _regularize_daily(daily)

    if len(daily) < MIN_POINTS_FORECAST:
        return PreparedSeries(
            observed=daily, median_gap=median_gap, n_filled=n_filled,
            message=f"Only {len(daily)} usable data points: at least {MIN_POINTS_FORECAST} needed for a forecast")

    return PreparedSeries(daily=daily, observed=daily, median_gap=median_gap, n_filled=n_filled)


def choose_season_length(median_gap: int, n_points: int) -> int:
    """Weekly seasonality only for daily data with at least two full cycles."""
    return SEASON_LENGTH if (median_gap == 1 and n_points >= SEASON_MIN_POINTS) else 1


# ---------------------------------------------------------------------------
# Forecast with calibrated interval (shared by the dashboard chart and the evaluation)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _conformal_half_width(values: tuple, ds: tuple, season_length: int, horizon: int, freq: str):
    """Interval half-width from out-of-sample errors over rolling windows
    (conformal quantile). Returns None if the history is too short."""
    n = len(values)
    n_cal = min(CALIBRATION_WINDOWS, (n - SEASON_MIN_POINTS) // horizon)
    if n_cal < 1:
        return None
    sf_df = pd.DataFrame({
        "unique_id": "series",
        "ds": pd.DatetimeIndex(ds),
        "y": np.asarray(values, dtype=float),
    })
    try:
        sf = StatsForecast(models=[AutoTheta(season_length=season_length)], freq=freq, n_jobs=1)
        cv = sf.cross_validation(df=sf_df, h=horizon, step_size=horizon, n_windows=n_cal)
    except Exception:
        logger.exception("Interval calibration failed")
        return None
    errors = np.sort((cv["y"] - cv["AutoTheta"].clip(lower=0)).abs().to_numpy(dtype=float))
    if errors.size == 0:
        return None
    k = min(int(np.ceil((errors.size + 1) * CONFIDENCE_LEVEL / 100)), errors.size)
    return float(errors[k - 1])


def forecast_with_intervals(sf_df, horizon: int, season_length: int, freq: str, fitted: bool = False):
    """AutoTheta forecast with the wider of its native and the calibrated interval,
    clipped at 0. Returns (forecast, fitted or None, half_width or None)."""
    sf = StatsForecast(models=[AutoTheta(season_length=season_length)], freq=freq, n_jobs=1)
    forecast = sf.forecast(df=sf_df, h=horizon, level=[CONFIDENCE_LEVEL], fitted=fitted)
    fit = sf.forecast_fitted_values() if fitted else None

    q = _conformal_half_width(
        tuple(np.round(sf_df["y"].to_numpy(dtype=float), 4)),
        tuple(pd.to_datetime(sf_df["ds"])),
        int(season_length), int(horizon), freq)

    forecast[NATIVE_LO] = forecast[LO_COL]
    forecast[NATIVE_HI] = forecast[HI_COL]
    if q is not None:
        point = forecast["AutoTheta"]
        forecast[LO_COL] = np.minimum(forecast[LO_COL], point - q)
        forecast[HI_COL] = np.maximum(forecast[HI_COL], point + q)

    cols = ["AutoTheta", LO_COL, HI_COL, NATIVE_LO, NATIVE_HI]
    forecast[cols] = forecast[cols].clip(lower=0)
    if fit is not None:
        fit["AutoTheta"] = fit["AutoTheta"].clip(lower=0)
    return forecast, fit, q


# ---------------------------------------------------------------------------
# Dashboard chart
# ---------------------------------------------------------------------------
def build_kpi_forecast_chart(df, kpi="download_mbps", horizon_days=7):
    """Daily KPI series -> AutoTheta forecast with an 80% prediction interval.

    Steps: daily median (days with too few samples dropped) -> frequency from
    the median gap between days -> daily data put on a complete calendar ->
    weekly seasonality only if daily data and at least 14 points -> AutoTheta
    -> forecast with a calibrated interval, clipped at 0. The validation of the model
    (rolling backtest) lives in backend/forecast_eval.py.
    """
    y_label = FORECAST_KPI_OPTIONS.get(kpi, kpi)

    prep = prepare_daily_series(df, kpi)
    if prep.daily is None:
        return _message_figure(y_label, prep.message, prep.observed)

    daily = prep.daily
    n_days = len(daily)
    freq_str = f"{prep.median_gap}D"
    season_length = choose_season_length(prep.median_gap, n_days)

    # statsforecast requires columns named exactly 'unique_id', 'ds', 'y'
    sf_df = daily.rename(columns={"day": "ds"})
    sf_df["unique_id"] = "series"
    sf_df = sf_df[["unique_id", "ds", "y"]]

    lo_col, hi_col = LO_COL, HI_COL
    try:
        forecast, fitted, _ = forecast_with_intervals(
            sf_df, horizon_days, season_length, freq_str, fitted=True)
    except Exception:
        logger.exception("AutoTheta forecast failed")
        return _message_figure(y_label, "Forecast model could not be fit on this data", daily)

    fig = go.Figure()

    # Observed daily values (dots only)
    fig.add_trace(go.Scatter(
        x=sf_df["ds"], y=sf_df["y"], mode="markers",
        name="Observed", marker=dict(color="#3B82F6", size=9)))

    # In-sample fitted curve of the model over the history
    fig.add_trace(go.Scatter(
        x=fitted["ds"], y=fitted["AutoTheta"],
        mode="lines", name="Fitted values", line=dict(color="#3B82F6", width=2.5, shape="spline")))

    # Forecast, bridged to the last fitted point
    bridge_x = pd.concat([fitted["ds"].iloc[[-1]], forecast["ds"]])
    bridge_y = pd.concat([fitted["AutoTheta"].iloc[[-1]], forecast["AutoTheta"]])
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines", name="Forecast", line=dict(color="#F59E0B", width=2.5, dash="dash", shape="spline")))

    # Prediction interval band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast[hi_col], forecast[lo_col][::-1]]),
        fill="toself", fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(0,0,0,0)"), name=f"{CONFIDENCE_LEVEL}% prediction interval",
        hoverinfo="skip"))

    fig.update_layout(**_base_layout(y_title=y_label, x_title="Date"))

    # Top note: data-quality caveats
    notes = []
    if n_days < SEASON_MIN_POINTS:
        notes.append(f"{n_days} data points: limited history, no weekly seasonality")
    if prep.median_gap > 1:
        notes.append(f"data spaced ~{prep.median_gap} days apart")
    if prep.n_filled:
        notes.append(f"{prep.n_filled} missing day(s) interpolated")
    if notes:
        fig.add_annotation(
            text=" · ".join(notes) + " — forecast is indicative",
            showarrow=False, font=dict(color=_THEME["muted_color"], size=11),
            xref="paper", yref="paper", x=0.5, y=1.12)

    return fig