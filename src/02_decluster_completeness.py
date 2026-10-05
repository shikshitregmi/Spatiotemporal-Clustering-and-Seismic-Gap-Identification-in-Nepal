"""
Phase 3 — Gardner–Knopoff declustering and MAXC completeness screening.

The merged Mw catalogue is declustered with Gardner–Knopoff (1974) windows
as parameterised by Van Stiphout et al. (2011). The independent catalogue
is then split into 1990–2000, 2000–2015 and 2015–2026, Mc is estimated
with Maximum Curvature, and each sub-catalogue is truncated at its Mc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FIGURE_DIR,
    GK_MAX_REMOVAL_FRACTION,
    GK_SPATIAL_SCALE,
    GK_SPATIAL_SCALE_FALLBACK,
    OUTPUT_DIR,
    PERIOD_LABELS,
    ensure_output_dirs,
)
from src.utils.completeness import maxc_mc, split_periods  # noqa: E402
from src.utils.decluster import gardner_knopoff_decluster, mainshocks  # noqa: E402
from src.utils.io import read_catalog  # noqa: E402
from src.utils.plotting import plot_gr_figures  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402


def main() -> dict:
    ensure_output_dirs()
    merged_path = OUTPUT_DIR / "merged_catalog.csv"
    cat = read_catalog(merged_path)
    print(f"[decluster] loaded {len(cat)} merged events")

    spatial_scale = float(GK_SPATIAL_SCALE)
    flagged = gardner_knopoff_decluster(cat, spatial_scale=spatial_scale)
    independent = mainshocks(flagged)
    n_before = len(flagged)
    n_after = len(independent)
    n_dependent = int(n_before - n_after)
    removal = n_dependent / max(n_before, 1)
    while removal > GK_MAX_REMOVAL_FRACTION and spatial_scale > 0.45:
        next_scale = max(0.45, round(spatial_scale * float(GK_SPATIAL_SCALE_FALLBACK), 3))
        if next_scale >= spatial_scale:
            break
        print(
            f"[decluster] removal {removal:.0%} exceeds "
            f"{GK_MAX_REMOVAL_FRACTION:.0%}; reducing spatial window × "
            f"{next_scale:.2f}"
        )
        spatial_scale = next_scale
        flagged = gardner_knopoff_decluster(cat, spatial_scale=spatial_scale)
        independent = mainshocks(flagged)
        n_after = len(independent)
        n_dependent = int(n_before - n_after)
        removal = n_dependent / max(n_before, 1)

    print(
        f"[decluster] Gardner–Knopoff: {n_before} → {n_after} "
        f"independent events ({n_dependent} removed, {removal:.0%}; "
        f"spatial_scale={spatial_scale:.2f})"
    )

    flagged_path = OUTPUT_DIR / "declustered_catalog_flagged.csv"
    indep_path = OUTPUT_DIR / "declustered_catalog.csv"
    flagged.to_csv(flagged_path, index=False)
    independent.to_csv(indep_path, index=False)

    periods = split_periods(independent)
    mc_values: dict[str, float] = {}
    period_mags: dict[str, object] = {}
    screened_counts: dict[str, int] = {}

    for key, sub in periods.items():
        mag = sub["Magnitude"].to_numpy()
        mc = maxc_mc(mag)
        mc_values[key] = mc
        period_mags[key] = mag
        if pd.notna(mc):
            screened = sub.loc[sub["Magnitude"] >= mc].copy()
        else:
            screened = sub.copy()
        screened_counts[key] = int(len(screened))
        out = OUTPUT_DIR / f"screened_{key}.csv"
        screened.to_csv(out, index=False)
        label = PERIOD_LABELS[key]
        print(
            f"[completeness] {label}: n={len(sub)}, Mc={mc:.2f}, "
            f"n(M≥Mc)={len(screened)}"
        )

    plot_gr_figures(period_mags, mc_values, FIGURE_DIR)
    stale = FIGURE_DIR / "mc_plots.png"
    if stale.exists():
        stale.unlink()

    # Concatenate Mc-screened events for later spatial clustering / KDE.
    screened_frames = [
        read_catalog(OUTPUT_DIR / f"screened_{key}.csv")
        for key in periods
    ]
    screened_all = pd.concat(screened_frames, ignore_index=True)
    screened_all.to_csv(OUTPUT_DIR / "screened_catalog.csv", index=False)

    stats = update_stats(
        n_before_decluster=int(n_before),
        n_after_decluster=int(n_after),
        n_dependent=int(n_dependent),
        decluster_removal_fraction=round(float(removal), 3),
        gk_spatial_scale=float(spatial_scale),
        mc_values={k: (None if pd.isna(v) else round(float(v), 2)) for k, v in mc_values.items()},
        n_screened=screened_counts,
        n_screened_total=int(len(screened_all)),
    )
    return stats


if __name__ == "__main__":
    main()
