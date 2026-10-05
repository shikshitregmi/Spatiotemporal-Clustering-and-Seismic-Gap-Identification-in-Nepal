"""
Phase 2 — Data ingestion, magnitude homogenisation, and catalogue merging.

Loads the local USGS ComCat extract (`query.csv`) and programmatically
downloads the Kaggle Nepal seismicity dataset. Events are standardised,
converted to Mw (Scordilis, 2006), restricted to the study window, and
deduplicated with a ±90 s / 50 km Haversine rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEDUP_DATEONLY_HOURS,
    DEDUP_DIST_KM,
    DEDUP_TIME_SEC,
    KAGGLE_DATASET,
    OUTPUT_DIR,
    SOURCE_PRIORITY,
    USGS_CSV,
    ensure_output_dirs,
)
from src.utils.geo import filter_study_region, haversine_km  # noqa: E402
from src.utils.io import parse_utc  # noqa: E402
from src.utils.magnitudes import homogenize_magnitudes  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402

COLUMN_ALIASES = {
    "Timestamp": [
        "time",
        "datetime",
        "date_time",
        "origintime",
        "origin_time",
        "eventtime",
        "timestamp",
        "date",
    ],
    "Latitude": ["latitude", "lat", "epicenter_lat", "y"],
    "Longitude": ["longitude", "lon", "long", "epicenter_lon", "x"],
    "Magnitude": [
        "magnitude",
        "mag",
        "mw",
        "ml",
        "mb",
        "ms",
        "m",
        "mag_value",
    ],
    "Depth": ["depth", "dep", "focaldepth", "depth_km", "focdepth"],
    "MagnitudeType": ["magtype", "mag_type", "magnitude_type", "mtype", "type_mag"],
    "EventID": ["id", "eventid", "event_id", "usgs_id", "evid"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _pick_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    return None


def _combine_date_time(df: pd.DataFrame) -> pd.Series | None:
    cols = {_norm(c): c for c in df.columns}
    date_col = cols.get("date")
    time_col = cols.get("time") or cols.get("origintime")
    if date_col and time_col and date_col != time_col:
        return (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        )
    return None


def standardize_catalog(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map heterogeneous schemas onto the common catalogue columns."""
    raw_cols = list(df.columns)
    mapping: dict[str, str] = {}
    for dest, aliases in COLUMN_ALIASES.items():
        found = _pick_column(raw_cols, aliases)
        if found:
            mapping[dest] = found

    out = pd.DataFrame()
    if "Timestamp" in mapping:
        out["Timestamp"] = df[mapping["Timestamp"]]
    else:
        combined = _combine_date_time(df)
        if combined is None:
            raise ValueError(f"{source}: no timestamp column in {raw_cols}")
        out["Timestamp"] = combined

    for dest in ("Latitude", "Longitude", "Magnitude", "Depth"):
        if dest in mapping:
            out[dest] = pd.to_numeric(df[mapping[dest]], errors="coerce")
        else:
            out[dest] = np.nan

    if "MagnitudeType" in mapping:
        out["MagnitudeType"] = df[mapping["MagnitudeType"]].astype(str)
    else:
        out["MagnitudeType"] = ""

    if "EventID" in mapping:
        out["EventID"] = df[mapping["EventID"]].astype(str)
    else:
        out["EventID"] = [f"{source}_{i}" for i in range(len(df))]

    out["Source"] = source
    out["Priority"] = SOURCE_PRIORITY.get(source, 9)

    # Drop obvious non-earthquakes when a USGS-style type column exists.
    type_col = _pick_column(raw_cols, ["type", "eventtype", "event_type"])
    if type_col is not None:
        kind = df[type_col].astype(str).str.lower()
        keep = kind.str.contains("earthquake") | kind.eq("nan") | kind.eq("")
        out = out.loc[keep.values].copy()

    out["Timestamp"] = parse_utc(out["Timestamp"])
    out = out.dropna(subset=["Timestamp", "Latitude", "Longitude", "Magnitude"])
    out = filter_study_region(out)
    return out.reset_index(drop=True)


def load_usgs() -> pd.DataFrame:
    print(f"[ingest] loading USGS catalogue: {USGS_CSV}")
    df = pd.read_csv(USGS_CSV)
    return standardize_catalog(df, "USGS")


def _kaggle_load_csv(dataset_dir: Path, filename: str) -> pd.DataFrame | None:
    matches = list(dataset_dir.rglob(filename))
    if not matches:
        # Case-insensitive / partial match (Kaggle filenames sometimes vary).
        needle = _norm(filename)
        matches = [p for p in dataset_dir.rglob("*.csv") if needle in _norm(p.name)]
    if not matches:
        print(f"[ingest] Kaggle file not found: {filename}")
        return None
    path = matches[0]
    print(f"[ingest] reading {path}")
    return pd.read_csv(path, low_memory=False)


def load_kaggle() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the Kaggle dataset and return (NSC, historical USGS dump).

    ``nepal_seismicity_master.csv`` is intentionally ignored: it is already
    a merged product and would double-count USGS events.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("kagglehub is required for Phase 2") from exc

    print(f"[ingest] downloading Kaggle dataset {KAGGLE_DATASET}")
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[ingest] Kaggle cache: {dataset_dir}")
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file():
            print(f"         {p.relative_to(dataset_dir)}")

    nsc_raw = _kaggle_load_csv(dataset_dir, "nsc_nepal_earthquakes_raw.csv")
    usgs_dump = _kaggle_load_csv(dataset_dir, "nepal_earthquakes_1990_2026.csv")

    nsc = (
        standardize_catalog(nsc_raw, "NSC")
        if nsc_raw is not None and len(nsc_raw)
        else pd.DataFrame()
    )
    kusgs = (
        standardize_catalog(usgs_dump, "KAGGLE_USGS")
        if usgs_dump is not None and len(usgs_dump)
        else pd.DataFrame()
    )
    return nsc, kusgs


def _is_date_only(ts: pd.Series) -> pd.Series:
    """True when the clock time is exactly midnight (typical date-only rows)."""
    clock = ts.dt.hour + ts.dt.minute + ts.dt.second + ts.dt.microsecond
    return clock.eq(0)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy space–time duplicate removal.

    Events are scanned in time order. A later event within the time window
    and 50 km of an earlier one is dropped unless it has higher source
    priority (lower Priority value), in which case the earlier event is
    dropped instead. Date-only NSC rows use a ±12 h window.
    """
    if df.empty:
        return df

    work = df.sort_values(["Timestamp", "Priority"]).reset_index(drop=True)
    n = len(work)
    keep = np.ones(n, dtype=bool)
    times = work["Timestamp"]
    tsec = times.astype("int64").to_numpy() / 1e9  # seconds since epoch
    lat = work["Latitude"].to_numpy()
    lon = work["Longitude"].to_numpy()
    prio = work["Priority"].to_numpy()
    date_only = _is_date_only(times).to_numpy()

    j = 0
    max_window = max(DEDUP_TIME_SEC, DEDUP_DATEONLY_HOURS * 3600.0)
    for i in range(n):
        if not keep[i]:
            continue
        while j < n and (tsec[j] - tsec[i]) <= max_window:
            j += 1
        for k in range(i + 1, j):
            if not keep[k]:
                continue
            dt = tsec[k] - tsec[i]
            window = (
                DEDUP_DATEONLY_HOURS * 3600.0
                if (date_only[i] or date_only[k])
                else DEDUP_TIME_SEC
            )
            if dt > window:
                continue
            if haversine_km(lat[i], lon[i], lat[k], lon[k]) > DEDUP_DIST_KM:
                continue
            # Same event: keep the higher-priority (usually USGS) record.
            if prio[k] < prio[i]:
                keep[i] = False
                break
            keep[k] = False

    removed = int((~keep).sum())
    print(f"[ingest] removed {removed} spatial-temporal duplicates")
    return work.loc[keep].reset_index(drop=True)


def main() -> dict:
    ensure_output_dirs()
    usgs = load_usgs()
    print(f"[ingest] USGS events in study region: {len(usgs)}")

    try:
        nsc, kusgs = load_kaggle()
    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] Kaggle download failed ({exc}). Continuing with USGS only.")
        nsc, kusgs = pd.DataFrame(), pd.DataFrame()

    print(f"[ingest] NSC events: {len(nsc)}")
    print(f"[ingest] Kaggle USGS-dump events: {len(kusgs)}")

    frames = [f for f in (usgs, nsc, kusgs) if f is not None and len(f)]
    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    merged = deduplicate(combined)
    merged = homogenize_magnitudes(merged)
    merged = merged.dropna(subset=["Magnitude"]).reset_index(drop=True)
    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    out_path = OUTPUT_DIR / "merged_catalog.csv"
    merged.to_csv(out_path, index=False)
    print(f"[ingest] wrote {out_path}  (n={len(merged)})")

    stats = update_stats(
        n_usgs_raw=int(len(usgs)),
        n_kaggle_nsc=int(len(nsc)),
        n_kaggle_usgs=int(len(kusgs)),
        n_before_dedup=int(n_before),
        n_after_dedup=int(len(merged)),
        dedup_time_sec=DEDUP_TIME_SEC,
        dedup_dist_km=DEDUP_DIST_KM,
    )
    return stats


if __name__ == "__main__":
    main()
