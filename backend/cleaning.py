from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY_TOKENS = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>"})

_CONNECTION_ALIASES = {
    "LTE": "LTE",
    "4G": "LTE",
    "NR": "NR",
    "5G": "NR",
    "NRNSA": "NRNSA",
    "NSA": "NRNSA",
    "5G NSA": "NRNSA",
    "HSPA+": "HSPA+",
    "HSPA": "HSPA+",
    "HSDPA": "HSPA+",
}


@dataclass
class CleaningReport:
    initial_rows: int = 0
    final_rows: int = 0
    duplicates_removed: int = 0
    invalid_coordinates_removed: int = 0
    invalid_values_corrected: int = 0
    empty_values_normalized: int = 0
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.final_rows:,} rows ready ({self.initial_rows:,} ingested)"]
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed:,} duplicates removed")
        if self.invalid_coordinates_removed:
            parts.append(f"{self.invalid_coordinates_removed:,} invalid coordinates removed")
        if self.invalid_values_corrected:
            parts.append(f"{self.invalid_values_corrected:,} invalid values corrected")
        if self.empty_values_normalized:
            parts.append(f"{self.empty_values_normalized:,} empty values normalized")
        return " · ".join(parts)

    def log(self) -> None:
        logger.info("Data cleaning: %s", self.summary())
        for line in self.details:
            logger.info("  %s", line)


def _map_operator_name(value: object) -> str:
    if pd.isna(value):
        return "Other"
    text = str(value).strip().lower()
    if not text:
        return "Other"
    if "ooredoo" in text:
        return "Ooredoo"
    if "tunisie" in text and "telecom" in text:
        return "Tunisie Telecom"
    if "orange" in text:
        return "Orange"
    return "Other"


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _is_blank(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in _EMPTY_TOKENS


def _normalize_categorical(series: pd.Series) -> tuple[pd.Series, int]:
    normalized = series.copy()
    corrected = 0
    for idx, value in normalized.items():
        if _is_blank(value):
            normalized.at[idx] = np.nan
            corrected += 1
        else:
            text = str(value).strip()
            if text != value:
                normalized.at[idx] = text
                corrected += 1
    return normalized, corrected


def _sanitize_numeric(series: pd.Series, *, min_val: float | None = None, max_val: float | None = None, positive: bool = False) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    corrected = int((series.notna() & numeric.isna()).sum())
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    corrected += int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
    if positive:
        mask = numeric <= 0
        corrected += int(mask.sum())
        numeric = numeric.mask(mask)
    if min_val is not None:
        mask = numeric < min_val
        corrected += int(mask.sum())
        numeric = numeric.mask(mask)
    if max_val is not None:
        mask = numeric > max_val
        corrected += int(mask.sum())
        numeric = numeric.mask(mask)
    return numeric, corrected


def _normalize_connection_type(value: object) -> object:
    if _is_blank(value):
        return np.nan
    key = str(value).strip().upper()
    return _CONNECTION_ALIASES.get(key, key if key in _CONNECTION_ALIASES.values() else np.nan)


def _valid_coordinate_mask(frame: pd.DataFrame) -> pd.Series:
    if "attr_location_latitude" not in frame.columns or "attr_location_longitude" not in frame.columns:
        return pd.Series(True, index=frame.index)
    lat = pd.to_numeric(frame["attr_location_latitude"], errors="coerce")
    lon = pd.to_numeric(frame["attr_location_longitude"], errors="coerce")
    valid = lat.notna() & lon.notna()
    valid &= lat.between(-90.0, 90.0)
    valid &= lon.between(-180.0, 180.0)
    valid &= ~((lat == 0) & (lon == 0))
    return valid | lat.isna() | lon.isna()


def _finalize_for_dashboard(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    finalized = frame.copy()
    numeric_columns = [
        "val_download_kbps", "val_upload_kbps", "download_mbps", "upload_mbps",
        "val_latency_iqm_ms", "latency_ms", "val_signal_rsrp_dbm", "val_signal_rsrq_db",
        "metric_packet_loss_percent",
    ]
    for column in numeric_columns:
        if column not in finalized.columns:
            continue
        finalized[column], corrected = _sanitize_numeric(finalized[column])
        report.invalid_values_corrected += corrected

    categorical_columns = ["attr_place_region", "attr_place_locality_type", "attr_connection_type_end_string", "attr_isp_common_name"]
    for column in categorical_columns:
        if column not in finalized.columns:
            continue
        finalized[column], corrected = _normalize_categorical(finalized[column])
        report.empty_values_normalized += corrected

    if "attr_connection_type_end_string" in finalized.columns:
        finalized["attr_connection_type_end_string"] = finalized["attr_connection_type_end_string"].map(_normalize_connection_type)

    if "attr_isp_common_name" in finalized.columns:
        finalized["operator"] = finalized["attr_isp_common_name"].map(_map_operator_name)
    elif "operator" in finalized.columns:
        finalized["operator"] = finalized["operator"].map(_map_operator_name)

    return finalized.reset_index(drop=True)


def clean_and_merge(dataframes: list[pd.DataFrame]) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    if not dataframes:
        return pd.DataFrame(), report

    report.initial_rows = sum(len(frame) for frame in dataframes)
    cleaned_frames = [clean_uploaded_frame(frame) for frame in dataframes]
    merged = pd.concat(cleaned_frames, ignore_index=True)

    before_dedup = len(merged)
    dedup_subset = ["guid_result"] if "guid_result" in merged.columns else None
    merged = merged.drop_duplicates(subset=dedup_subset, keep="last").reset_index(drop=True)
    report.duplicates_removed = before_dedup - len(merged)

    before_coords = len(merged)
    coord_mask = _valid_coordinate_mask(merged)
    merged = merged[coord_mask].reset_index(drop=True)
    report.invalid_coordinates_removed = before_coords - len(merged)

    merged = _finalize_for_dashboard(merged, report)
    report.final_rows = len(merged)
    report.log()
    return merged, report


def clean_uploaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()

    download_source = _first_existing_column(cleaned, ("val_download_kbps", "download_kbps", "download"))
    upload_source = _first_existing_column(cleaned, ("val_upload_kbps", "upload_kbps", "upload"))
    latency_source = _first_existing_column(cleaned, ("val_latency_iqm_ms", "latency_ms", "latency", "Latency", "val_latency_ms", "latency_milliseconds", "ping_ms"))
    operator_source = _first_existing_column(cleaned, ("attr_isp_common_name", "attr_sim_operator_common_name", "operator"))
    region_source = _first_existing_column(cleaned, ("attr_place_region", "region"))
    locality_source = _first_existing_column(cleaned, ("attr_place_locality_type", "attr_location_locality_type", "region"))
    connection_source = _first_existing_column(cleaned, ("attr_connection_type_end_string", "technology"))
    loss_source = _first_existing_column(cleaned, ("metric_packet_loss_percent", "packet_loss_pct"))

    if download_source and download_source != "val_download_kbps":
        cleaned["val_download_kbps"] = cleaned[download_source]
    if upload_source and upload_source != "val_upload_kbps":
        cleaned["val_upload_kbps"] = cleaned[upload_source]
    if latency_source and latency_source != "val_latency_iqm_ms":
        cleaned["val_latency_iqm_ms"] = cleaned[latency_source]
    if operator_source and operator_source != "attr_isp_common_name":
        cleaned["attr_isp_common_name"] = cleaned[operator_source]
    if region_source and region_source != "attr_place_region":
        cleaned["attr_place_region"] = cleaned[region_source]
    if locality_source and locality_source != "attr_place_locality_type":
        cleaned["attr_place_locality_type"] = cleaned[locality_source]
    if connection_source and connection_source != "attr_connection_type_end_string":
        cleaned["attr_connection_type_end_string"] = cleaned[connection_source]
    if loss_source and loss_source != "metric_packet_loss_percent":
        cleaned["metric_packet_loss_percent"] = cleaned[loss_source]

    if "guid_result" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["guid_result"], keep="last")

    if "ts_result" in cleaned.columns:
        cleaned["ts_result"] = pd.to_datetime(cleaned["ts_result"], errors="coerce")

    for column in ["val_download_kbps", "val_upload_kbps"]:
        if column in cleaned.columns:
            cleaned[column], _ = _sanitize_numeric(cleaned[column], positive=True)

    if "val_signal_rsrp_dbm" in cleaned.columns:
        cleaned["val_signal_rsrp_dbm"], _ = _sanitize_numeric(cleaned["val_signal_rsrp_dbm"], min_val=-140, max_val=-44)

    if "val_signal_rsrq_db" in cleaned.columns:
        cleaned["val_signal_rsrq_db"], _ = _sanitize_numeric(cleaned["val_signal_rsrq_db"], min_val=-35, max_val=-3)

    if "latency_ms" in cleaned.columns:
        cleaned["latency_ms"], _ = _sanitize_numeric(cleaned["latency_ms"], positive=True)

    if "val_download_kbps" in cleaned.columns:
        cleaned["download_mbps"] = cleaned["val_download_kbps"] / 1000.0
        cleaned["download_mbps"] = cleaned["download_mbps"].replace([np.inf, -np.inf], np.nan)
    if "val_upload_kbps" in cleaned.columns:
        cleaned["upload_mbps"] = cleaned["val_upload_kbps"] / 1000.0
        cleaned["upload_mbps"] = cleaned["upload_mbps"].replace([np.inf, -np.inf], np.nan)
    if "val_latency_iqm_ms" in cleaned.columns:
        cleaned["val_latency_iqm_ms"], _ = _sanitize_numeric(cleaned["val_latency_iqm_ms"], positive=True)
        cleaned["latency_ms"] = cleaned["val_latency_iqm_ms"]

    if operator_source:
        cleaned["operator"] = cleaned[operator_source].map(_map_operator_name)

    if "metric_packet_loss_percent" in cleaned.columns:
        cleaned["metric_packet_loss_percent"], _ = _sanitize_numeric(cleaned["metric_packet_loss_percent"], min_val=0, max_val=100)

    if "attr_location_latitude" in cleaned.columns:
        cleaned["attr_location_latitude"], _ = _sanitize_numeric(cleaned["attr_location_latitude"], min_val=-90, max_val=90)
    if "attr_location_longitude" in cleaned.columns:
        cleaned["attr_location_longitude"], _ = _sanitize_numeric(cleaned["attr_location_longitude"], min_val=-180, max_val=180)

    for column in ["attr_place_region", "attr_place_locality_type", "attr_connection_type_end_string", "attr_isp_common_name"]:
        if column in cleaned.columns:
            cleaned[column], _ = _normalize_categorical(cleaned[column])

    if "attr_connection_type_end_string" in cleaned.columns:
        cleaned["attr_connection_type_end_string"] = cleaned["attr_connection_type_end_string"].map(_normalize_connection_type)

    return cleaned.reset_index(drop=True)