"""
Project-wide paths and analysis constants.

Study region approximates Nepal and the adjoining Himalayan front used by the
USGS ComCat extract and the Kaggle historical dump (roughly 26–31°N, 79–89°E).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
USGS_CSV = PROJECT_ROOT / "query.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"
STATS_PATH = OUTPUT_DIR / "pipeline_stats.json"

KAGGLE_DATASET = "amansinghnp/nepal-earthquake-seismicity-dataset-1990-2026"

# ---------------------------------------------------------------------------
# Geographic window
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 26.0, 31.0
LON_MIN, LON_MAX = 79.0, 89.0

# Seismogenic belt used for clustering, KDE, and the Monte Carlo null.
# The southern edge sits on the Himalayan front so the Gangetic foreland
# (Terai), which is quiet because it is south of the faults, is excluded.
SEIS_LAT_MIN, SEIS_LAT_MAX = 27.5, 30.5
SEIS_LON_MIN, SEIS_LON_MAX = 80.0, 88.5

# 0.5° analysis grid (Phase 5 / 6) — coarser than 0.25° so sparse
# decade catalogues still have a usable events-per-cell ratio.
GRID_RES_DEG = 0.5

# Mean Earth radius used for Haversine distances (km)
EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# Catalog merging
# ---------------------------------------------------------------------------
# Mid-point of the 60–120 s window requested in the project brief.
DEDUP_TIME_SEC = 90.0
DEDUP_DIST_KM = 50.0
# Date-only NSC rows (midnight timestamps) use a same-calendar-day match.
DEDUP_DATEONLY_HOURS = 12.0

# Source priority when two catalogs report the same event (lower = preferred).
SOURCE_PRIORITY = {
    "USGS": 0,
    "KAGGLE_USGS": 1,
    "NSC": 2,
    "KAGGLE": 3,
}

# ---------------------------------------------------------------------------
# Completeness periods (right-open intervals)
# ---------------------------------------------------------------------------
PERIODS = {
    "1990_2000": ("1990-01-01", "2000-01-01"),
    "2000_2015": ("2000-01-01", "2015-01-01"),
    "2015_2026": ("2015-01-01", "2027-01-01"),
}
PERIOD_LABELS = {
    "1990_2000": "1990–2000",
    "2000_2015": "2000–2015",
    "2015_2026": "2015–2026",
}
MAG_BIN_WIDTH = 0.1

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
DBSCAN_MIN_SAMPLES = 5
# min_samples < min_cluster_size lets HDBSCAN react to local density
# (core distance from 5 neighbours) while still requiring a real cluster.
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 5
# Himalayan MFT clusters are local (tens of km). The unrestricted kneedle
# on a sparse Mc-screened catalogue jumped to ~66 km and merged the arc.
DBSCAN_EPS_MIN_KM = 15.0
DBSCAN_EPS_MAX_KM = 25.0

# ---------------------------------------------------------------------------
# Gardner–Knopoff windows (distances in km, times in days)
# ---------------------------------------------------------------------------
# Standard CORSSA / Van Stiphout (2011) spatial formula is already in km.
# Original GK (1974) time windows saturate near ~1000 days for M ≳ 6.5;
# leaving the M≥6.5 power-law uncapped lets the 2015 Gorkha mainshock
# swallow ~13 years of seismicity. Spatial scale < 1 is applied only if
# the removal fraction still exceeds 70%.
GK_TIME_CAP_DAYS = 1000.0
GK_SPATIAL_SCALE = 1.0
GK_MAX_REMOVAL_FRACTION = 0.70
GK_SPATIAL_SCALE_FALLBACK = 0.70

# ---------------------------------------------------------------------------
# KDE
# ---------------------------------------------------------------------------
# Multiply Scott's bandwidth. Sparse decade catalogues need stronger smoothing.
KDE_BW_FACTOR_FULL = 1.6
KDE_BW_FACTOR_SPARSE = 2.5
KDE_SPARSE_N = 150

# ---------------------------------------------------------------------------
# Gap definition
# ---------------------------------------------------------------------------
GAP_DENSITY_PERCENTILE = 10.0
# A candidate low-density cell is retained if this many of its 8-neighbours
# sit above the low-density threshold (interior troughs, not empty borders).
# Three neighbours is appropriate for an along-strike gap on a linear belt.
MIN_HIGH_DENSITY_NEIGHBORS = 3

# Drop every 0.5° cell that intersects this inward frame of the study
# bounding box, so KDE mass lost at the border is not called a gap.
EDGE_BUFFER_DEG = GRID_RES_DEG

# Inhomogeneous Monte Carlo null. The sampling surface is a broad KDE of
# the real catalogue (Scott factor below, and at least 1.5× the analysis
# bandwidth) so draws follow the tectonic arc rather than empty space.
MC_N_ITER = 1000
MC_ALPHA = 0.05
MC_SEED = 42
MC_INTENSITY_BW_FACTOR = 3.0

# Western Nepal Seismic Gap (approx. along the MFT).
WESTERN_GAP_LON = (79.0, 82.0)
WESTERN_GAP_LAT = (27.3, 30.0)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DPI = 300
MAP_CRS = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

# Approximate Main Frontal Thrust (southern Siwalik front) and Main Boundary
# Thrust traces digitised from standard Himalayan tectonic maps. Used when the
# GEM Global Active Faults extract does not yield named MFT/MBT segments.
MFT_LONLAT = [
    (80.05, 28.98),
    (80.40, 28.78),
    (80.80, 28.55),
    (81.20, 28.32),
    (81.60, 28.08),
    (82.00, 27.92),
    (82.50, 27.78),
    (83.00, 27.62),
    (83.45, 27.52),
    (84.00, 27.38),
    (84.50, 27.22),
    (85.00, 27.05),
    (85.50, 26.95),
    (86.00, 26.85),
    (86.50, 26.72),
    (87.00, 26.62),
    (87.50, 26.55),
    (88.15, 26.50),
]
MBT_LONLAT = [
    (80.25, 29.35),
    (80.70, 29.10),
    (81.20, 28.85),
    (81.70, 28.58),
    (82.20, 28.35),
    (82.80, 28.12),
    (83.40, 27.95),
    (84.00, 27.75),
    (84.60, 27.58),
    (85.20, 27.48),
    (85.80, 27.32),
    (86.40, 27.15),
    (87.00, 27.00),
    (87.60, 26.88),
    (88.10, 26.80),
]


def configure_stdio() -> None:
    """Use UTF-8 on Windows consoles so magnitude/arrow glyphs print."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def ensure_output_dirs() -> None:
    """Create output, figure, and cache directories if they do not exist."""
    configure_stdio()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
