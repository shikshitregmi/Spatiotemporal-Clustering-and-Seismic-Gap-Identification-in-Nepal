"""
Phase 4 — Spatial clustering with DBSCAN (haversine) and HDBSCAN.

A k-distance plot (k = min_samples) is used to locate the DBSCAN eps
elbow via a normalised-curve kneedle (maximum deviation from the
end-to-end chord). HDBSCAN is run on the same Haversine distance matrix
as a density-robustness check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DBSCAN_EPS_MAX_KM,
    DBSCAN_EPS_MIN_KM,
    DBSCAN_MIN_SAMPLES,
    EARTH_RADIUS_KM,
    FIGURE_DIR,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    OUTPUT_DIR,
    SEIS_LAT_MAX,
    SEIS_LAT_MIN,
    SEIS_LON_MAX,
    SEIS_LON_MIN,
    ensure_output_dirs,
)
from src.utils.geo import filter_seismogenic  # noqa: E402
from src.utils.io import read_catalog  # noqa: E402
from src.utils.plotting import apply_style, plot_cluster_map, savefig  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402


def kneedle_eps(distances_km: np.ndarray, lo: float = DBSCAN_EPS_MIN_KM, hi: float = DBSCAN_EPS_MAX_KM) -> float:
    """
    Knee of the k-distance curve, restricted to [lo, hi] km.

    The unrestricted kneedle on a sparse Himalayan catalogue sat at ~66 km
    and merged the entire MFT into one cluster. We therefore search for the
    elbow only among 5-distance values inside the 15–25 km band expected
    for local MFT seismicity.
    """
    y_all = np.sort(np.asarray(distances_km, dtype=float))[::-1]
    in_band = (y_all >= lo) & (y_all <= hi)
    y = y_all[in_band] if in_band.any() else y_all
    n = y.size
    if n < 3:
        return float(np.clip(np.median(y_all) if y_all.size else hi, lo, hi))
    x = np.linspace(0.0, 1.0, n)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1 = 0.0, y_n[0]
    x2, y2 = 1.0, y_n[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_n + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1) + 1e-12
    knee_i = int(np.argmax(num / den))
    return float(np.clip(y[knee_i], lo, hi))


def plot_k_distance(k_dist_km: np.ndarray, eps_km: float, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    y = np.sort(k_dist_km)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(np.arange(len(y)), y, color="#1f4e79", lw=1.8, label=f"{DBSCAN_MIN_SAMPLES}-distance")
    ax.axhspan(DBSCAN_EPS_MIN_KM, DBSCAN_EPS_MAX_KM, color="#f4e4c1", alpha=0.55, zorder=0, label="15–25 km search band")
    ax.axhline(eps_km, color="#c0392b", ls="--", lw=1.4, label=f"eps = {eps_km:.1f} km")
    ax.set_ylim(0, max(80.0, float(np.percentile(y, 95))))
    ax.set_xlabel("Events sorted by decreasing k-distance")
    ax.set_ylabel("Distance to 5th nearest neighbour (km)")
    ax.set_title("DBSCAN k-distance plot (haversine, km)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(framealpha=0.92)
    return savefig(fig, path)


def main() -> dict:
    ensure_output_dirs()
    screened = read_catalog(OUTPUT_DIR / "screened_catalog.csv")
    cat = filter_seismogenic(screened)
    print(
        f"[cluster] seismogenic box {SEIS_LAT_MIN:.1f}–{SEIS_LAT_MAX:.1f}°N, "
        f"{SEIS_LON_MIN:.1f}–{SEIS_LON_MAX:.1f}°E: {len(cat)} of {len(screened)} events"
    )
    if len(cat) < DBSCAN_MIN_SAMPLES:
        raise RuntimeError("Too few screened events for clustering")

    # sklearn haversine metric expects (lat, lon) in radians; eps in radians.
    coords_rad = np.radians(cat[["Latitude", "Longitude"]].to_numpy())

    nn = NearestNeighbors(
        n_neighbors=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    nn.fit(coords_rad)
    dists_rad, _ = nn.kneighbors(coords_rad)
    k_dist_km = dists_rad[:, -1] * EARTH_RADIUS_KM

    eps_km = kneedle_eps(k_dist_km)
    eps_rad = eps_km / EARTH_RADIUS_KM
    print(
        f"[cluster] k=5 elbow in 15–25 km band → eps = {eps_km:.2f} km "
        f"({eps_rad:.6f} rad)"
    )
    plot_k_distance(k_dist_km, eps_km, FIGURE_DIR / "k_distance_plot.png")

    db = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    cat["dbscan_label"] = db.fit_predict(coords_rad)
    n_db = int(len(set(cat["dbscan_label"]) - {-1}))
    n_db_noise = int((cat["dbscan_label"] == -1).sum())
    print(f"[cluster] DBSCAN: {n_db} clusters, {n_db_noise} noise points")

    # HDBSCAN does not implement haversine; use a precomputed km matrix.
    try:
        import hdbscan
        from sklearn.metrics.pairwise import haversine_distances

        dist_km = haversine_distances(coords_rad) * EARTH_RADIUS_KM
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="precomputed",
        )
        cat["hdbscan_label"] = hdb.fit_predict(dist_km)
    except Exception as exc:  # noqa: BLE001
        print(f"[cluster] HDBSCAN failed ({exc}); marking all as noise")
        cat["hdbscan_label"] = -1

    n_hdb = int(len(set(cat["hdbscan_label"]) - {-1}))
    n_hdb_noise = int((cat["hdbscan_label"] == -1).sum())
    print(
        f"[cluster] HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
        f"min_samples={HDBSCAN_MIN_SAMPLES}): {n_hdb} clusters, "
        f"{n_hdb_noise} noise points"
    )

    out_csv = OUTPUT_DIR / "clustered_catalog.csv"
    cat.to_csv(out_csv, index=False)

    plot_cluster_map(
        cat,
        "dbscan_label",
        FIGURE_DIR / "dbscan_clusters.png",
        "DBSCAN clusters (haversine metric)",
    )
    plot_cluster_map(
        cat,
        "hdbscan_label",
        FIGURE_DIR / "hdbscan_clusters.png",
        f"HDBSCAN clusters (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES})",
    )

    stats = update_stats(
        dbscan_eps_km=float(eps_km),
        dbscan_eps_rad=float(eps_rad),
        dbscan_min_samples=int(DBSCAN_MIN_SAMPLES),
        n_dbscan_clusters=n_db,
        n_dbscan_noise=n_db_noise,
        n_hdbscan_clusters=n_hdb,
        n_hdbscan_noise=n_hdb_noise,
        hdbscan_min_cluster_size=int(HDBSCAN_MIN_CLUSTER_SIZE),
        hdbscan_min_samples=int(HDBSCAN_MIN_SAMPLES),
        n_cluster_events=int(len(cat)),
    )
    return stats


if __name__ == "__main__":
    main()
