"""Rolling-origin validation of the KPI forecast against a seasonal-naive baseline.

Run from the project root: python -m backend.forecast_eval
Output: output/forecast_evaluation.csv and output/forecast_evaluation.txt
(one row per KPI, human-readable).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.cleaning import clean_and_merge
from backend.forecast import (
    CONFIDENCE_LEVEL,
    FORECAST_KPI_OPTIONS,
    HI_COL,
    LO_COL,
    NATIVE_HI,
    NATIVE_LO,
    SEASON_MIN_POINTS,
    choose_season_length,
    forecast_with_intervals,
    prepare_daily_series,
)

logger = logging.getLogger("forecast_eval")

UPLOADS_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
HORIZON = 7        # days, as in the dashboard
MAX_WINDOWS = 8
CSV_SEP = ";"      # use "," if your Excel uses English regional settings

COLUMNS = [
    "kpi", "status", "note", "n_days", "n_windows", "n_test_points", "mean_level",
    "mae_autotheta", "mae_baseline", "rel_mae_pct", "improvement_pct",
    "coverage", "coverage_native", "nominal_coverage", "n_windows_widened", "caveat",
]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic write, Excel-friendly (UTF-8 with BOM, ';' separator)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, sep=CSV_SEP, encoding="utf-8-sig")
    os.replace(tmp, path)


def _row(kpi: str, status: str, note: str = "", **values) -> dict:
    row = {c: np.nan for c in COLUMNS}
    row.update(kpi=kpi, status=status, note=note, nominal_coverage=CONFIDENCE_LEVEL / 100)
    row.update(values)
    return row


def _to_readable(results: pd.DataFrame) -> pd.DataFrame:
    """Same columns as the table that used to be in the dashboard."""
    return pd.DataFrame({
        "KPI": results["kpi"].map(lambda k: FORECAST_KPI_OPTIONS.get(k, k)),
        "MAE (model)": results["mae_autotheta"],
        "MAE (last week)": results["mae_baseline"],
        "MAE % of mean": results["rel_mae_pct"],
        "Gain vs last week %": results["improvement_pct"],
        "80% interval coverage %": (results["coverage"] * 100).round(1),
    })


def _write_txt(readable: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    """Aligned text table with borders, easy to read in any editor."""
    headers = list(readable.columns)
    rows = [["" if pd.isna(v) else str(v) for v in r]
            for r in readable.itertuples(index=False)]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def fmt(cells):
        return "| " + " | ".join(
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = ["FORECAST VALIDATION - rolling backtest on held-out weeks, horizon 7 days", "",
             sep, fmt(headers), sep, *[fmt(r) for r in rows], sep, "",
             "Legend",
             "  MAE (model)               mean absolute error of the forecast (same unit as the KPI, lower = better)",
             "  MAE (last week)           error of the baseline 'same as last week'",
             "  MAE % of mean             model MAE as % of the average KPI level",
             "  Gain vs last week %       improvement vs baseline (negative = model is worse)",
             f"  {CONFIDENCE_LEVEL}% interval coverage %   share of real values inside the {CONFIDENCE_LEVEL}% interval "
             f"(target ~{CONFIDENCE_LEVEL}%, noisy with few test points)"]
    notes = [f"  {r['kpi']}: {r['note']}" for _, r in results.iterrows()
             if pd.notna(r["note"]) and str(r["note"]).strip()]
    if notes:
        lines += ["", "Notes", *notes]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save(results: pd.DataFrame) -> None:
    readable = _to_readable(results)
    _write_csv(readable, OUTPUT_DIR / "forecast_evaluation.csv")
    _write_txt(readable, results, OUTPUT_DIR / "forecast_evaluation.txt")


def load_uploads() -> pd.DataFrame:
    files = sorted(UPLOADS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{UPLOADS_DIR}'")
    merged, _report = clean_and_merge([pd.read_csv(p) for p in files])
    return merged


def backtest(daily: pd.DataFrame):
    """Backtest using only data before each cutoff. Returns (cv, n_windows, n_widened)."""
    n = len(daily)
    n_windows = min(MAX_WINDOWS, (n - SEASON_MIN_POINTS) // HORIZON)
    if n_windows < 1:
        return None, 0, 0

    season_length = choose_season_length(1, n - n_windows * HORIZON)
    y = daily["y"].to_numpy(dtype=float)
    sf_all = pd.DataFrame({
        "unique_id": "series",
        "ds": pd.to_datetime(daily["day"]).to_numpy(),
        "y": y,
    })

    frames, widened = [], 0
    for w in range(n_windows):
        end = n - (n_windows - w) * HORIZON
        fc, _, q = forecast_with_intervals(sf_all.iloc[:end], HORIZON, season_length, "D")
        widened += q is not None
        # Seasonal-naive baseline
        base = [y[end - season_length + (k % season_length)] for k in range(HORIZON)]
        frames.append(pd.DataFrame({
            "cutoff": sf_all["ds"].iloc[end - 1],
            "ds": fc["ds"].to_numpy(),
            "y": y[end:end + HORIZON],
            "AutoTheta": fc["AutoTheta"].to_numpy(),
            LO_COL: fc[LO_COL].to_numpy(),
            HI_COL: fc[HI_COL].to_numpy(),
            NATIVE_LO: fc[NATIVE_LO].to_numpy(),
            NATIVE_HI: fc[NATIVE_HI].to_numpy(),
            "SeasonalNaive": base,
        }))
    return pd.concat(frames, ignore_index=True), n_windows, widened


def evaluate_kpi(df: pd.DataFrame, kpi: str) -> dict:
    prep = prepare_daily_series(df, kpi)
    if prep.daily is None:
        return _row(kpi, "skipped", prep.message or "no usable series")
    if prep.median_gap > 1:
        return _row(kpi, "skipped", f"data spaced ~{prep.median_gap} days apart: daily backtest not applicable",
                    n_days=len(prep.daily))

    n_days = len(prep.daily)
    try:
        cv, n_windows, n_widened = backtest(prep.daily)
    except Exception:
        logger.exception("Backtest failed for %s", kpi)
        return _row(kpi, "failed", "backtest raised an error (see log)", n_days=n_days)
    if cv is None:
        return _row(kpi, "skipped",
                    f"only {n_days} daily points: at least {SEASON_MIN_POINTS + HORIZON} needed",
                    n_days=n_days)

    mean_level = float(cv["y"].mean())
    mae_model = float((cv["y"] - cv["AutoTheta"]).abs().mean())
    mae_base = float((cv["y"] - cv["SeasonalNaive"]).abs().mean())
    inside = (cv["y"] >= cv[LO_COL]) & (cv["y"] <= cv[HI_COL])
    inside_native = (cv["y"] >= cv[NATIVE_LO]) & (cv["y"] <= cv[NATIVE_HI])

    notes = []
    if prep.n_filled:
        notes.append(f"{prep.n_filled} interpolated day(s) in the series; some may fall in test windows")

    return _row(
        kpi, "ok", "; ".join(notes),
        n_days=n_days,
        n_windows=n_windows,
        n_test_points=int(len(cv)),
        mean_level=round(mean_level, 3),
        mae_autotheta=round(mae_model, 3),
        mae_baseline=round(mae_base, 3),
        rel_mae_pct=round(100 * mae_model / mean_level, 1) if mean_level > 1e-9 else np.nan,
        improvement_pct=round(100 * (1 - mae_model / mae_base), 1) if mae_base > 1e-9 else np.nan,
        coverage=round(float(inside.mean()), 3),
        coverage_native=round(float(inside_native.mean()), 3),
        n_windows_widened=n_widened,
        caveat=(f"{len(cv)} test points: coverage is noisy, read it as close to "
                f"{CONFIDENCE_LEVEL}%, not calibrated"),
    )


def run() -> pd.DataFrame:
    try:
        df = load_uploads()
    except Exception as exc:
        logger.exception("Could not load data")
        results = pd.DataFrame([_row("-", "failed", f"could not load data: {exc}")], columns=COLUMNS)
        _save(results)
        return results

    results = pd.DataFrame([evaluate_kpi(df, k) for k in FORECAST_KPI_OPTIONS], columns=COLUMNS)
    _save(results)

    shown = ["kpi", "status", "n_windows", "n_test_points", "mae_autotheta", "mae_baseline",
             "rel_mae_pct", "improvement_pct", "coverage_native", "coverage", "n_windows_widened"]
    print(results[shown].to_string(index=False))
    for _, r in results[results["status"] != "ok"].iterrows():
        print(f"  [{r['status']}] {r['kpi']}: {r['note']}")
    print(f"\nCoverage rests on few test points: read it as close to {CONFIDENCE_LEVEL}%, not calibrated.")
    print(f"Saved: {OUTPUT_DIR / 'forecast_evaluation.csv'} and {OUTPUT_DIR / 'forecast_evaluation.txt'}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(0 if (run()["status"] == "ok").any() else 1)