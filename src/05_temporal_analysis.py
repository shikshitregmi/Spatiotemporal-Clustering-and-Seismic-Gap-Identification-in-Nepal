"""
Phase 6 — Repeat KDE / gap mapping independently for each Mc-screened period.

Produces a 1×3 panel figure so that gaps can be classified visually as
stable, emerging, or disappearing across 1990–2000, 2000–2015 and 2015–2026.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURE_DIR, KDE_BW_FACTOR_SPARSE, OUTPUT_DIR, PERIOD_LABELS, PERIODS, ensure_output_dirs  # noqa: E402
from src.utils.io import read_catalog  # noqa: E402
from src.utils.plotting import apply_style, savefig  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402


def main() -> dict:
    # Import from the Phase-5 module (numeric filename).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase04_kde", ROOT / "src" / "04_kde_gaps.py"
    )
    phase04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase04)

    ensure_output_dirs()
    apply_style()

    surfaces = {}
    period_gap_counts = {}
    vmax = 0.0
    for key in PERIODS:
        path = OUTPUT_DIR / f"screened_{key}.csv"
        cat = read_catalog(path)
        lon, lat, density, gaps, backend, _info = phase04.compute_density_and_gaps(
            cat, bw_factor=KDE_BW_FACTOR_SPARSE
        )
        gaps.to_csv(OUTPUT_DIR / f"gap_cells_{key}.csv", index=False)
        surfaces[key] = (lon, lat, density, gaps)
        period_gap_counts[key] = int(len(gaps))
        vmax = max(vmax, float(np.nanmax(density)) if density.size else 0.0)
        print(
            f"[temporal] {PERIOD_LABELS[key]}: n={len(cat)}, "
            f"gaps={len(gaps)}, kde={backend}"
            + (
                " (sparse sub-catalogue: jitter + Scott×"
                f"{KDE_BW_FACTOR_SPARSE} bandwidth)"
                if len(cat) < 150
                else ""
            )
        )
        # Compact individual map (avoids a tall 1×3 combined figure).
        phase04.plot_kde_gaps(
            density,
            lon,
            lat,
            gaps,
            path=FIGURE_DIR / f"temporal_{key}.png",
            title=f"KDE gaps  ·  {PERIOD_LABELS[key]}",
        )

    # Compact 1×3 strip (not a tall stacked figure). No web map tiles.
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.4), sharex=True, sharey=True, layout="constrained"
    )
    mesh = None
    for ax, key in zip(axes, PERIODS, strict=True):
        lon, lat, density, gaps = surfaces[key]
        phase04.plot_kde_gaps(
            density,
            lon,
            lat,
            gaps,
            path=FIGURE_DIR / f"temporal_{key}.png",
            title=PERIOD_LABELS[key],
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
            ax=ax,
        )
        for coll in ax.collections:
            if coll.__class__.__name__ == "QuadMesh":
                mesh = coll
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=list(axes), shrink=0.78, pad=0.02, fraction=0.025)
        cbar.set_label("KDE density")
    fig.suptitle(
        "Temporal KDE gaps (seismogenic belt, arc null, FDR 0.05)",
        fontsize=11,
    )
    savefig(fig, FIGURE_DIR / "temporal_analysis_panels.png")

    keys = list(PERIODS)
    cell_sets = {}
    for key in keys:
        g = surfaces[key][3]
        if g is None or g.empty:
            cell_sets[key] = set()
        else:
            cell_sets[key] = set(zip(g["cell_i"].astype(int), g["cell_j"].astype(int)))
    stable = cell_sets[keys[0]] & cell_sets[keys[1]] & cell_sets[keys[2]]
    emerging = cell_sets[keys[2]] - (cell_sets[keys[0]] | cell_sets[keys[1]])
    disappearing = (cell_sets[keys[0]] | cell_sets[keys[1]]) - cell_sets[keys[2]]
    print(
        f"[temporal] stable={len(stable)}, emerging={len(emerging)}, "
        f"disappearing={len(disappearing)}"
    )

    stats = update_stats(
        n_gap_cells_by_period=period_gap_counts,
        n_gaps_stable=int(len(stable)),
        n_gaps_emerging=int(len(emerging)),
        n_gaps_disappearing=int(len(disappearing)),
    )
    return stats


if __name__ == "__main__":
    main()
