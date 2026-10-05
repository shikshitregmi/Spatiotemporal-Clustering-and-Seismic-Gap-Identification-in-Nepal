"""
Homogenise reported magnitudes onto the moment-magnitude (Mw) scale.

Conversion relations follow Scordilis (2006), which are the standard
empirical formulae used for Nepal Himalaya catalogues (e.g. Rajaure &
Paudel, 2018, J. Nepal Geol. Soc.).

References
----------
Scordilis, E. M. (2006). Empirical global relations converting MS and mb
    to moment magnitude. Journal of Seismology, 10, 225–236.
Rajaure, S., & Paudel, L. P. (2018). A comprehensive earthquake catalogue
    for Nepal and its adjoining region. Journal of Nepal Geological Society.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Magnitude types already expressed as (or derived from) seismic moment.
MW_TYPES = {
    "mw",
    "mww",
    "mwc",
    "mwb",
    "mwr",
    "mwp",
    "mw_mww",
    "moment",
}

MB_TYPES = {"mb", "mb_lg", "mblg", "mb_lg"}
MS_TYPES = {"ms", "msz", "ms_20"}
ML_TYPES = {"ml", "mlv", "mlg", "md", "m", "local", "mdl"}


def _to_float(mag) -> float:
    try:
        val = float(mag)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(val):
        return np.nan
    return val


def scordilis_mb_to_mw(mb: float) -> float:
    """Mw = 0.85 mb + 1.03  (3.5 ≤ mb ≤ 6.2; applied with clamping)."""
    mb = float(np.clip(mb, 3.5, 6.2))
    return 0.85 * mb + 1.03


def scordilis_ms_to_mw(ms: float) -> float:
    """Two-segment Scordilis (2006) Ms → Mw relation."""
    if ms <= 6.1:
        return 0.67 * ms + 2.07
    return 0.99 * ms + 0.08


def to_mw(magnitude, mag_type: str | None = None) -> float:
    """
    Convert a single magnitude to Mw.

    Unknown or missing types are treated as local/body-wave magnitudes
    (NSC practice) and converted with the mb relation, matching Rajaure
    & Paudel (2018) who applied Eq. (3) of Scordilis to ML.
    """
    mag = _to_float(magnitude)
    if np.isnan(mag):
        return np.nan

    key = (mag_type or "").strip().lower().replace(" ", "")
    if key in MW_TYPES:
        return mag
    if key in MS_TYPES:
        return scordilis_ms_to_mw(mag)
    if key in MB_TYPES or key in ML_TYPES or key == "":
        return scordilis_mb_to_mw(mag)
    # Unrecognised agency codes (e.g. 'mh', 'unk') — conservative mb relation.
    return scordilis_mb_to_mw(mag)


def homogenize_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add MagnitudeMw from Magnitude + MagnitudeType; copy into Magnitude."""
    out = df.copy()
    types = out["MagnitudeType"] if "MagnitudeType" in out.columns else ""
    out["MagnitudeMw"] = [
        to_mw(m, t) for m, t in zip(out["Magnitude"], types, strict=False)
    ]
    out["MagnitudeOriginal"] = out["Magnitude"]
    out["Magnitude"] = out["MagnitudeMw"]
    return out
