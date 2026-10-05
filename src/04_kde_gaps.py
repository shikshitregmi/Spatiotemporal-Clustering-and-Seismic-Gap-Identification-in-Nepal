"""
Phase 5 — KDE gaps inside the seismogenic belt, tested against the arc.

Events outside 27.5–30.5°N, 80.0–88.5°E are dropped. A 0.5° Gaussian KDE
is compared with 1000 catalogues drawn from a broad KDE of the same events
(inhomogeneous null). Exact one-sided p-values are corrected with
Benjamini–Hochberg FDR at alpha = 0.05. Only interior cells that survive
FDR are retained as significant gaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FIGURE_DIR,
    GRID_RES_DEG,
    MC_ALPHA,
    MC_INTENSITY_BW_FACTOR,
    MC_N_ITER,
    MC_SEED,
    OUTPUT_DIR,
    SEIS_LAT_MAX,
    SEIS_LAT_MIN,
    SEIS_LON_MAX,
    SEIS_LON_MIN,
    ensure_output_dirs,
)
from src.utils.geo import boundary_edge_mask, filter_seismogenic  # noqa: E402
from src.utils.io import read_catalog  # noqa: E402
from src.utils.kde import (  # noqa: E402
    benjamini_hochberg,
    evaluate_kde,
    monte_carlo_null,
    resolve_bw_factor,
    study_grid,
)
from matplotlib.lines import Line2D  # noqa: E402

from src.utils.plotting import apply_style, overlay_thrusts, prepare_map_ax, savefig  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402


GAP_COLUMNS = [
    "cell_i",
    "cell_j",
    "lon_center",
    "lat_center",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "density",
    "p_value",
]


def identify_gap_cells(
    density: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    edge: np.ndarray,
    p_values: np.ndarray,
    alpha: float = MC_ALPHA,
) -> tuple[pd.DataFrame, dict]:
    """
    Retain interior cells whose low-density p-values survive BH-FDR.

    Edge cells are excluded from the multiple-testing family so the
    correction is applied only where the KDE is not a boundary artifact.
    """
    interior = ~edge
    p_test = p_values[interior]
    reject = benjamini_hochberg(p_test, alpha=alpha)
    fdr = np.zeros(density.shape, dtype=bool)
    fdr[interior] = reject
    dlon = float(lon[1] - lon[0]) if len(lon) > 1 else GRID_RES_DEG
    dlat = float(lat[1] - lat[0]) if len(lat) > 1 else GRID_RES_DEG
    records = []
    ny, nx = density.shape
    for i in range(ny):
        for j in range(nx):
            if not fdr[i, j]:
                continue
            lon0 = lon[j] - dlon / 2.0
            lat0 = lat[i] - dlat / 2.0
            records.append(
                {
                    "cell_i": i,
                    "cell_j": j,
                    "lon_center": float(lon[j]),
                    "lat_center": float(lat[i]),
                    "lon_min": float(lon0),
                    "lon_max": float(lon0 + dlon),
                    "lat_min": float(lat0),
                    "lat_max": float(lat0 + dlat),
                    "density": float(density[i, j]),
                    "p_value": float(p_values[i, j]),
                }
            )
    gaps = pd.DataFrame.from_records(records, columns=GAP_COLUMNS)
    if len(gaps):
        gaps = gaps.sort_values(["p_value", "lat_center", "lon_center"]).reset_index(drop=True)
    info = {
        "n_edge_cells_excluded": int(np.count_nonzero(edge)),
        "n_fdr_tests": int(np.count_nonzero(interior)),
        "n_significant_interior": int(len(gaps)),
        "fdr_alpha": float(alpha),
    }
    print(
        f"[kde] FDR at alpha={alpha:.2f} on {info['n_fdr_tests']} interior cells; "
        f"{len(gaps)} survive"
    )
    return gaps, info


def plot_kde_gaps(
    density: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    gaps: pd.DataFrame,
    path: Path,
    title: str = "Kernel density and candidate seismic gaps",
    vmin=None,
    vmax=None,
    ax=None,
) -> object:
    """Draw KDE heatmap + gap rectangles + MFT/MBT. Returns ax or saved path."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_style()
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
    else:
        fig = ax.figure

    prepare_map_ax(ax, title)
    dlon = float(lon[1] - lon[0]) if len(lon) > 1 else GRID_RES_DEG
    dlat = float(lat[1] - lat[0]) if len(lat) > 1 else GRID_RES_DEG
    lon_e = np.append(lon - dlon / 2.0, lon[-1] + dlon / 2.0)
    lat_e = np.append(lat - dlat / 2.0, lat[-1] + dlat / 2.0)
    mesh = ax.pcolormesh(
        lon_e,
        lat_e,
        density,
        cmap="YlOrRd",
        shading="flat",
        alpha=0.78,
        zorder=2,
        vmin=vmin,
        vmax=vmax,
    )
    if created_fig:
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Event density (KDE)")

    for _, row in gaps.iterrows():
        ax.add_patch(
            Rectangle(
                (row["lon_min"], row["lat_min"]),
                dlon,
                dlat,
                fill=False,
                edgecolor="#1f78b4",
                linewidth=1.1,
                zorder=5,
            )
        )
    thrust_handles = overlay_thrusts(ax)
    if created_fig:
        gap_handle = Line2D(
            [0],
            [0],
            color="#1f78b4",
            lw=1.4,
            marker="s",
            mfc="none",
            markersize=8,
            label="Significant gap cell",
        )
        ax.legend(
            handles=list(thrust_handles) + [gap_handle],
            loc="lower left",
            framealpha=0.92,
        )
        return savefig(fig, path)
    return ax


def compute_density_and_gaps(
    events: pd.DataFrame,
    bw_factor: float | None = None,
    n_iter: int = MC_N_ITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, str, dict]:
    """Seismogenic-zone KDE, inhomogeneous null, FDR gaps."""
    n_in = int(len(events))
    events = filter_seismogenic(events)
    n_events = int(len(events))
    print(
        f"[kde] seismogenic box {SEIS_LAT_MIN:.1f}–{SEIS_LAT_MAX:.1f}°N, "
        f"{SEIS_LON_MIN:.1f}–{SEIS_LON_MAX:.1f}°E: {n_events} of {n_in} events"
    )
    lon, lat, lon_m, lat_m = study_grid()
    edge = boundary_edge_mask(lon, lat)
    bw = resolve_bw_factor(n_events, bw_factor)
    density, backend = evaluate_kde(
        events["Longitude"].to_numpy(),
        events["Latitude"].to_numpy(),
        lon_m,
        lat_m,
        bw_factor=bw,
    )
    print(f"[kde] backend = {backend}; analysis bandwidth × {bw}")
    print(
        f"[kde] Monte Carlo null: {n_iter} iterations, N={n_events}, "
        f"seed={MC_SEED}"
    )
    intensity_bw = max(float(MC_INTENSITY_BW_FACTOR), float(bw) * 1.5)
    intensity, null_stack, mc_backend = monte_carlo_null(
        events["Longitude"].to_numpy(),
        events["Latitude"].to_numpy(),
        lon_m,
        lat_m,
        bw_factor=bw,
        n_iter=n_iter,
        intensity_bw_factor=intensity_bw,
    )
    # Exact one-sided p: rank of the real density among the simulations / Nsim.
    # Edge cells are withheld from FDR (p set to NaN).
    p_values = np.full(density.shape, np.nan, dtype=float)
    interior = ~edge
    n_iter = int(null_stack.shape[0])
    flat_null = null_stack.reshape(n_iter, -1)
    flat_obs = density.ravel()
    flat_interior = interior.ravel()
    p_flat = np.full(flat_obs.shape, np.nan)
    p_flat[flat_interior] = np.mean(
        flat_null[:, flat_interior] <= flat_obs[flat_interior], axis=0
    )
    p_values = p_flat.reshape(density.shape)
    gaps, gap_info = identify_gap_cells(density, lon, lat, edge, p_values)
    info = {
        **gap_info,
        "p_values": p_values,
        "intensity": intensity,
        "mc_iterations": int(null_stack.shape[0]),
        "mc_n_points": n_events,
        "mc_alpha": float(MC_ALPHA),
        "mc_seed": int(MC_SEED),
        "mc_backend": mc_backend,
        "kde_bw_factor": float(bw),
        "mc_intensity_bw_factor": float(intensity_bw),
        "n_events_input": n_in,
        "n_edge_cells_excluded": int(np.count_nonzero(edge)),
    }
    print(
        f"[kde] edge buffer excluded {info['n_edge_cells_excluded']} cells; "
        f"Monte Carlo backend = {mc_backend}"
    )
    return lon, lat, density, gaps, backend, info


def main() -> dict:
    ensure_output_dirs()
    cat = read_catalog(OUTPUT_DIR / "screened_catalog.csv")
    print(f"[kde] events: {len(cat)}")

    lon, lat, density, gaps, backend, info = compute_density_and_gaps(cat)
    gaps.to_csv(OUTPUT_DIR / "gap_cells.csv", index=False)
    np.savez(
        OUTPUT_DIR / "kde_surface.npz",
        lon=lon,
        lat=lat,
        density=density,
        intensity=info["intensity"],
        p_value=info["p_values"],
    )
    plot_kde_gaps(
        density,
        lon,
        lat,
        gaps,
        FIGURE_DIR / "kde_gaps_map.png",
        title="FDR-significant gaps vs the arc-intensity null",
    )

    coords = [
        {
            "lon_center": float(r.lon_center),
            "lat_center": float(r.lat_center),
            "lon_min": float(r.lon_min),
            "lat_min": float(r.lat_min),
            "lon_max": float(r.lon_max),
            "lat_max": float(r.lat_max),
            "density": float(r.density),
            "p_value": float(r.p_value),
        }
        for r in gaps.itertuples(index=False)
    ]
    stats = update_stats(
        kde_backend=backend,
        n_gap_cells=int(len(gaps)),
        gap_cell_coordinates=coords,
        n_edge_cells_excluded=int(info["n_edge_cells_excluded"]),
        n_fdr_tests=int(info["n_fdr_tests"]),
        n_significant_interior=int(info["n_significant_interior"]),
        mc_iterations=int(info["mc_iterations"]),
        mc_n_points=int(info["mc_n_points"]),
        mc_alpha=float(info["mc_alpha"]),
        mc_seed=int(info["mc_seed"]),
        mc_backend=info["mc_backend"],
        kde_bw_factor=float(info["kde_bw_factor"]),
        mc_intensity_bw_factor=float(info["mc_intensity_bw_factor"]),
        n_kde_events=int(info["mc_n_points"]),
        seismogenic_lat=(SEIS_LAT_MIN, SEIS_LAT_MAX),
        seismogenic_lon=(SEIS_LON_MIN, SEIS_LON_MAX),
    )
    return stats


if __name__ == "__main__":
    main()
