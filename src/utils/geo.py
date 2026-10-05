"""Haversine geometry, study-region filters, and Himalayan thrust overlays."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    EARTH_RADIUS_KM,
    EDGE_BUFFER_DEG,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    SEIS_LAT_MAX,
    SEIS_LAT_MIN,
    SEIS_LON_MAX,
    SEIS_LON_MIN,
    MBT_LONLAT,
    MFT_LONLAT,
)

GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/"
    "gem-global-active-faults/master/geojson/gem_active_faults.geojson"
)
NE_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in kilometres (vectorised)."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def spherical_rectangle_area_km2(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> float:
    """Area (km²) of a longitude/latitude rectangle on a sphere."""
    dlon = np.radians(float(lon_max) - float(lon_min))
    dsin = np.sin(np.radians(float(lat_max))) - np.sin(np.radians(float(lat_min)))
    return float(abs(EARTH_RADIUS_KM**2 * dlon * dsin))


def boundary_edge_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    buffer_deg: float = EDGE_BUFFER_DEG,
    lon_min: float = SEIS_LON_MIN,
    lon_max: float = SEIS_LON_MAX,
    lat_min: float = SEIS_LAT_MIN,
    lat_max: float = SEIS_LAT_MAX,
) -> np.ndarray:
    """
    True for cells that intersect the outer frame of the analysis box.

    ``buffer_deg`` is measured inward from each side. On the 0.5° grid a
    0.5° buffer removes the outer ring of cells, which is where a KDE loses
    mass simply because there are no events outside the window.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    dlon = float(lon[1] - lon[0]) if lon.size > 1 else float(buffer_deg)
    dlat = float(lat[1] - lat[0]) if lat.size > 1 else float(buffer_deg)
    cell_lon_min = lon - dlon / 2.0
    cell_lon_max = lon + dlon / 2.0
    cell_lat_min = lat - dlat / 2.0
    cell_lat_max = lat + dlat / 2.0
    # A cell flush with the inner edge of the buffer does not enter the frame.
    col_edge = (cell_lon_min < lon_min + buffer_deg) | (cell_lon_max > lon_max - buffer_deg)
    row_edge = (cell_lat_min < lat_min + buffer_deg) | (cell_lat_max > lat_max - buffer_deg)
    return row_edge[:, None] | col_edge[None, :]


def in_seismogenic_zone(df: pd.DataFrame) -> pd.Series:
    """Events inside the closed seismogenic box (27.5–30.5°N, 80.0–88.5°E)."""
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    return lat.between(SEIS_LAT_MIN, SEIS_LAT_MAX) & lon.between(SEIS_LON_MIN, SEIS_LON_MAX)


def filter_seismogenic(df: pd.DataFrame) -> pd.DataFrame:
    """Drop epicentres outside the seismogenic belt before clustering or KDE."""
    if df.empty:
        return df.copy()
    return df.loc[in_seismogenic_zone(df)].reset_index(drop=True)


def in_study_region(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for events inside the Nepal Himalaya analysis window."""
    return (
        df["Latitude"].between(LAT_MIN, LAT_MAX)
        & df["Longitude"].between(LON_MIN, LON_MAX)
    )


def filter_study_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[in_study_region(df)].copy()


def _download(url: str, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        urlretrieve(url, dest)
        return dest
    except Exception as exc:  # noqa: BLE001 — network is optional
        print(f"[geo] download failed ({url}): {exc}")
        return None


def load_countries_clip():
    """Natural Earth countries clipped to the study window, or None."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    path = _download(NE_COUNTRIES_URL, CACHE_DIR / "ne_110m_admin_0_countries.geojson")
    if path is None:
        return None
    try:
        world = gpd.read_file(path)
        return world.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]
    except Exception as exc:  # noqa: BLE001
        print(f"[geo] could not clip Natural Earth: {exc}")
        return None


def load_nepal_outline():
    """Natural Earth admin-0 polygon for Nepal, or None if unavailable."""
    try:
        import geopandas as gpd
    except ImportError:
        return None

    path = _download(NE_COUNTRIES_URL, CACHE_DIR / "ne_110m_admin_0_countries.geojson")
    if path is None:
        return None
    try:
        world = gpd.read_file(path)
        name_col = "ADMIN" if "ADMIN" in world.columns else "name"
        nepal = world[world[name_col].str.contains("Nepal", case=False, na=False)]
        return nepal if len(nepal) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[geo] could not read Natural Earth: {exc}")
        return None


_THRUST_CACHE = None


def load_thrust_traces():
    """
    Return a GeoDataFrame of MFT/MBT linework.

    Preference order:
    1. GEM Global Active Faults clipped to the study window and named as
       Main Frontal / Himalayan Frontal / Main Boundary (``name`` or ``fs_name``).
    2. Approximate polylines in ``src.config`` (always available).
    """
    global _THRUST_CACHE
    if _THRUST_CACHE is not None:
        return _THRUST_CACHE.copy()

    import geopandas as gpd
    from shapely.geometry import LineString

    fallback = gpd.GeoDataFrame(
        {
            "name": ["Main Frontal Thrust (approx.)", "Main Boundary Thrust (approx.)"],
            "geometry": [
                LineString([(lon, lat) for lon, lat in MFT_LONLAT]),
                LineString([(lon, lat) for lon, lat in MBT_LONLAT]),
            ],
        },
        crs="EPSG:4326",
    )

    path = _download(GEM_FAULTS_URL, CACHE_DIR / "gem_active_faults.geojson")
    if path is None:
        print("[geo] using approximate MFT/MBT traces")
        _THRUST_CACHE = fallback
        return fallback.copy()

    try:
        faults = gpd.read_file(path)
        faults = faults.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]
        label = faults.get("name", "").fillna("").astype(str)
        if "fs_name" in faults.columns:
            label = label.where(label.str.strip().ne(""), faults["fs_name"].fillna("").astype(str))
        pattern = r"main frontal|himalayan frontal|main boundary|\bMFT\b|\bMBT\b|frontal thrust"
        matched = faults[label.str.contains(pattern, case=False, regex=True)]
        if len(matched) == 0:
            print("[geo] no named MFT/MBT in GEM clip; using approximate traces")
            _THRUST_CACHE = fallback
            return fallback.copy()
        print(f"[geo] loaded {len(matched)} GEM thrust segments")
        matched = matched.copy()
        matched["name"] = label.loc[matched.index]
        # Rebuild a clean EPSG:4326 frame (GEM GeoJSON reports "WGS 84",
        # which GeoPandas 1.x treats as distinct from EPSG:4326).
        out = gpd.GeoDataFrame(
            {"name": list(matched["name"]), "geometry": list(matched.geometry)},
            crs="EPSG:4326",
        )
        names = out["name"].str.lower()
        if not names.str.contains("boundary|mbt").any():
            mbt = gpd.GeoDataFrame(
                {"name": [fallback.iloc[1]["name"]], "geometry": [fallback.iloc[1].geometry]},
                crs="EPSG:4326",
            )
            out = gpd.GeoDataFrame(pd.concat([out, mbt], ignore_index=True), crs="EPSG:4326")
        _THRUST_CACHE = out
        return out.copy()
    except Exception as exc:  # noqa: BLE001
        print(f"[geo] GEM parse failed ({exc}); using approximate traces")
        _THRUST_CACHE = fallback
        return fallback.copy()


def nepal_mask(lon_m: np.ndarray, lat_m: np.ndarray, buffer_deg: float = 0.5) -> np.ndarray:
    """
    Boolean mask of grid cells along the Himalayan seismogenic front.

    A ~0.5° (~55 km) corridor about the Main Frontal Thrust is used so that
    the lowest-density tail is an along-strike quiescence, not empty Tibet
    or the Gangetic plain south of the Siwaliks.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    geoms = []
    try:
        traces = load_thrust_traces()
        for _, row in traces.iterrows():
            name = str(row.get("name", "")).lower()
            if "frontal" in name or "mft" in name:
                geoms.append(row.geometry)
    except Exception:  # noqa: BLE001
        geoms = []
    if geoms:
        corridor = unary_union(geoms).buffer(buffer_deg)
    else:
        corridor = LineString([(lon, lat) for lon, lat in MFT_LONLAT]).buffer(buffer_deg)

    mask = np.zeros(lon_m.shape, dtype=bool)
    for i in range(lon_m.shape[0]):
        for j in range(lon_m.shape[1]):
            mask[i, j] = corridor.contains(Point(float(lon_m[i, j]), float(lat_m[i, j])))
    print(f"[geo] MFT-corridor mask covers {int(mask.sum())}/{mask.size} grid cells")
    return mask
