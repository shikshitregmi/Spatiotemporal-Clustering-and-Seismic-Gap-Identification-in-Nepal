"""Catalogue I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse mixed ISO-8601 timestamps (with or without fractional seconds)."""
    return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")


def read_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = parse_utc(df["Timestamp"])
    return df
