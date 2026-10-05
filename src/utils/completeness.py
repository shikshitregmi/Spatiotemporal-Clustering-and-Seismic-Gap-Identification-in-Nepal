"""
Magnitude of completeness via the Maximum Curvature (MAXC) method.

MAXC identifies Mc as the magnitude bin with the highest non-cumulative
event count (the mode of the frequency–magnitude distribution). It is
simple, stable for modest catalogues, and is the method requested for
this study (Wiemer & Wyss, 2000; Woessner & Wiemer, 2005).

A 0.1 magnitude bin width is used throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAG_BIN_WIDTH, PERIODS
from src.utils.io import parse_utc


def maxc_mc(magnitudes: np.ndarray, bin_width: float = MAG_BIN_WIDTH) -> float:
    """
    Maximum-curvature magnitude of completeness.

    Returns NaN if fewer than 10 valid magnitudes are supplied.
    """
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 10:
        return float("nan")

    mag_min = np.floor(mag.min() / bin_width) * bin_width
    mag_max = np.ceil(mag.max() / bin_width) * bin_width
    edges = np.arange(mag_min, mag_max + bin_width * 0.5, bin_width)
    counts, edges = np.histogram(mag, bins=edges)
    if counts.size == 0 or counts.max() == 0:
        return float("nan")
    # Bin centre of the modal (non-cumulative) bin.
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


def gutenberg_richter(magnitudes: np.ndarray, bin_width: float = MAG_BIN_WIDTH):
    """Return bin centres, non-cumulative counts, and cumulative N(≥M)."""
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        empty = np.array([])
        return empty, empty, empty
    mag_min = np.floor(mag.min() / bin_width) * bin_width
    mag_max = np.ceil(mag.max() / bin_width) * bin_width
    edges = np.arange(mag_min, mag_max + bin_width * 0.5, bin_width)
    counts, edges = np.histogram(mag, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cumulative = np.cumsum(counts[::-1])[::-1]
    return centres, counts.astype(float), cumulative.astype(float)


def split_periods(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a catalogue into the three analysis windows (right-open)."""
    ts = parse_utc(df["Timestamp"])
    out: dict[str, pd.DataFrame] = {}
    for key, (start, end) in PERIODS.items():
        t0 = pd.Timestamp(start, tz="UTC")
        t1 = pd.Timestamp(end, tz="UTC")
        out[key] = df.loc[(ts >= t0) & (ts < t1)].copy()
    return out
