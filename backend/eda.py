from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EDAReport:
    row_count: int = 0
    column_count: int = 0
    missing_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_rows: int = 0
    numeric_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    skewness: pd.Series = field(default_factory=pd.Series)
    outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_coordinates: int = 0

    def summary(self) -> str:
        return (
            f"{self.row_count:,} rows, {self.column_count} columns · "
            f"{self.duplicate_rows:,} duplicate rows · "
            f"{self.invalid_coordinates:,} invalid coordinates"
        )

    def log(self) -> None:
        logger.info("EDA (raw data): %s", self.summary())
        if not self.missing_summary.empty:
            top_missing = self.missing_summary[self.missing_summary["missing_count"] > 0]
            for col, row in top_missing.iterrows():
                logger.info("  missing: %s -> %s (%.2f%%)", col, row["missing_count"], row["missing_pct"])
        if not self.outlier_summary.empty:
            top_outliers = self.outlier_summary[self.outlier_summary["outlier_count"] > 0]
            for _, row in top_outliers.iterrows():
                logger.info("  outliers: %s -> %s (%.2f%%)", row["column"], row["outlier_count"], row["outlier_pct"])
        if not self.skewness.empty:
            skewed = self.skewness[self.skewness.abs() > 1]
            for col, val in skewed.items():
                logger.info("  skewness: %s -> %.2f", col, val)


def _iqr_outliers(series: pd.Series) -> tuple[int, float]:
    s = series.dropna()
    if s.empty:
        return 0, 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < lower) | (s > upper)).sum())
    pct = round(100 * count / len(s), 2)
    return count, pct


def run_eda(dataframes: list[pd.DataFrame]) -> EDAReport:
    """Profiles raw uploaded data BEFORE cleaning. Read-only — does not
    modify the dataframes or affect clean_and_merge()."""
    report = EDAReport()
    if not dataframes:
        return report

    raw = pd.concat(dataframes, ignore_index=True)
    report.row_count = len(raw)
    report.column_count = raw.shape[1]

    missing = raw.isna().sum()
    missing_pct = (missing / len(raw) * 100).round(2) if len(raw) else missing
    report.missing_summary = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    ).sort_values("missing_pct", ascending=False)

    report.duplicate_rows = int(raw.duplicated().sum())

    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report.numeric_summary = raw[numeric_cols].describe().T
        report.skewness = raw[numeric_cols].skew().sort_values(ascending=False)

        rows = []
        for col in numeric_cols:
            count, pct = _iqr_outliers(raw[col])
            rows.append({"column": col, "outlier_count": count, "outlier_pct": pct})
        report.outlier_summary = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)

        if len(numeric_cols) > 1:
            report.correlation = raw[numeric_cols].corr(method="pearson")

    lat_col, lon_col = "attr_location_latitude", "attr_location_longitude"
    if lat_col in raw.columns and lon_col in raw.columns:
        lat = pd.to_numeric(raw[lat_col], errors="coerce")
        lon = pd.to_numeric(raw[lon_col], errors="coerce")
        invalid = (
            (~lat.between(-90, 90))
            | (~lon.between(-180, 180))
            | ((lat == 0) & (lon == 0))
            | lat.isna()
            | lon.isna()
        )
        report.invalid_coordinates = int(invalid.sum())

    return report


# ── Manual dev-only entry point ─────────────────────────────────────────────
def run_eda_on_file(filename: str, uploads_dir: str = "uploads") -> EDAReport:
    """Run EDA on a single CSV chosen by name from the uploads folder.
    Not called by app.py — run this file directly: `python backend/eda.py`."""
    path = os.path.join(uploads_dir, filename)
    df = pd.read_csv(path)
    report = run_eda([df])
    report.log()
    print(report.summary())
    if not report.missing_summary.empty:
        print("\nMissing values:\n", report.missing_summary[report.missing_summary["missing_count"] > 0])
    if not report.outlier_summary.empty:
        print("\nOutliers:\n", report.outlier_summary[report.outlier_summary["outlier_count"] > 0])
    if not report.numeric_summary.empty:
        print("\nNumeric summary:\n", report.numeric_summary)
    return report


if __name__ == "__main__":
    FILENAME = "your_file.csv"  # <- change this to the file you want to inspect
    run_eda_on_file(FILENAME)