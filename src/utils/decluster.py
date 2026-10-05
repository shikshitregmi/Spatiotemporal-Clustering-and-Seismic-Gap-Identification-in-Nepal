"""
Gardner–Knopoff (1974) aftershock declustering.

ObsPy does not ship a Gardner–Knopoff implementation. This module follows
the space–time windows tabulated by Van Stiphout et al. (2011) in the
CORSSA community review (the same windows used by OpenQuake HMTK).

Windows (M = magnitude):
    distance (km) = 10 ** (0.1238 * M + 0.983)
    time (days)   = 10 ** (0.032 * M + 2.7389)           if M < 6.5
                  = 10 ** (0.5409 * M − 0.547)            if M ≥ 6.5

Algorithm: events are processed from largest to smallest. Each unflagged
event is treated as a candidate mainshock; any unflagged neighbour inside
its distance window and its foreshock/aftershock time window is marked as
a dependent event. The mainshock itself is retained.

Reference
---------
Gardner, J. K., & Knopoff, L. (1974). Is the sequence of aftershocks in
    southern California, with aftershocks removed, Poissonian?
    Bulletin of the Seismological Society of America, 64(5), 1363–1367.
Van Stiphout, T., Zhuang, J., & Marsan, D. (2011). Seismicity declustering.
    Community Online Resource for Statistical Seismicity Analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.geo import haversine_km
from src.utils.io import parse_utc
from src.config import GK_SPATIAL_SCALE, GK_TIME_CAP_DAYS


def gardner_knopoff_windows(
    magnitude: np.ndarray,
    spatial_scale: float = GK_SPATIAL_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (distance_km, time_days) windows for each magnitude.

    Spatial windows are **kilometres**, never radians. They follow
    Van Stiphout et al. (2011) / CORSSA:

        L(km) = 10 ** (0.1238 M + 0.983)

    Time windows are **days**. The M ≥ 6.5 branch is capped at
    ``GK_TIME_CAP_DAYS`` (1000 d), matching the saturation of the original
    Gardner–Knopoff (1974) graphical windows.
    """
    mag = np.asarray(magnitude, dtype=float)
    dist_km = (10.0 ** (0.1238 * mag + 0.983)) * float(spatial_scale)
    time_days = np.where(
        mag >= 6.5,
        10.0 ** (0.5409 * mag - 0.547),
        10.0 ** (0.032 * mag + 2.7389),
    )
    time_days = np.minimum(time_days, GK_TIME_CAP_DAYS)
    return dist_km, time_days


def gardner_knopoff_decluster(
    df: pd.DataFrame,
    mag_col: str = "Magnitude",
    fs_time_prop: float = 1.0,
    spatial_scale: float = GK_SPATIAL_SCALE,
) -> pd.DataFrame:
    """
    Flag dependent events and return the catalogue with extra columns.

    Parameters
    ----------
    fs_time_prop :
        Fraction of the time window applied *before* the mainshock
        (foreshocks). 1.0 is the classical Gardner–Knopoff choice.
    """
    work = df.copy().reset_index(drop=True)
    n = len(work)
    if n == 0:
        work["cluster_id"] = []
        work["is_dependent"] = []
        work["is_mainshock"] = []
        return work

    mag = work[mag_col].to_numpy(dtype=float)
    lat = work["Latitude"].to_numpy(dtype=float)
    lon = work["Longitude"].to_numpy(dtype=float)
    times = parse_utc(work["Timestamp"])
    # Decimal days relative to the first event (stable, avoids leap issues).
    t0 = times.min()
    t_days = (times - t0).dt.total_seconds().to_numpy() / 86400.0

    dist_win, time_win = gardner_knopoff_windows(mag, spatial_scale=spatial_scale)
    # Sanity log: confirm windows are km / days (not radians).
    if mag.size:
        i_med = int(np.nanmedian(np.arange(mag.size)))
        print(
            f"[decluster] windows in km & days | spatial_scale={spatial_scale:.2f} | "
            f"example M={mag[i_med]:.1f} → {dist_win[i_med]:.1f} km, "
            f"{time_win[i_med]:.0f} d (cap={GK_TIME_CAP_DAYS:.0f} d)"
        )

    order = np.argsort(-mag, kind="mergesort")  # largest first; stable
    vcl = np.zeros(n, dtype=int)
    flag = np.zeros(n, dtype=int)  # 0 mainshock / independent; 1 aftershock; -1 foreshock
    cluster_index = 0

    for pos in range(n):
        i = order[pos]
        if vcl[i] != 0:
            continue
        dt = t_days - t_days[i]
        in_time = (dt >= -time_win[i] * fs_time_prop) & (dt <= time_win[i])
        unassigned = vcl == 0
        candidates = np.where(in_time & unassigned)[0]
        if candidates.size <= 1:
            continue
        dist = haversine_km(lat[i], lon[i], lat[candidates], lon[candidates])
        members = candidates[dist <= dist_win[i]]  # dist_win is kilometres
        if members.size <= 1:
            continue
        cluster_index += 1
        vcl[members] = cluster_index
        flag[members] = 1
        flag[members[dt[members] < 0]] = -1
        flag[i] = 0  # mainshock of this cluster

    work["cluster_id"] = vcl
    work["is_dependent"] = flag != 0
    work["is_mainshock"] = ~work["is_dependent"]
    return work


def mainshocks(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Poissonian (independent) sub-catalogue."""
    if "is_mainshock" not in df.columns:
        df = gardner_knopoff_decluster(df)
    return df.loc[df["is_mainshock"]].copy().reset_index(drop=True)
