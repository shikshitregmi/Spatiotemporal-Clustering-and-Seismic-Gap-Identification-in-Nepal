"""
Phase 7 — Validate candidate gaps against the Western Nepal Seismic Gap
and write ``outputs/summary_report.txt``.
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
    OUTPUT_DIR,
    PERIOD_LABELS,
    PERIODS,
    SEIS_LAT_MAX,
    SEIS_LAT_MIN,
    SEIS_LON_MAX,
    SEIS_LON_MIN,
    WESTERN_GAP_LAT,
    WESTERN_GAP_LON,
    ensure_output_dirs,
)
from src.utils.geo import spherical_rectangle_area_km2  # noqa: E402
from src.utils.kde import study_grid  # noqa: E402
from src.utils.plotting import apply_style, overlay_thrusts, prepare_map_ax, savefig  # noqa: E402
from src.utils.stats import update_stats  # noqa: E402


def cells_in_western_gap(gaps: pd.DataFrame) -> pd.DataFrame:
    """Cells whose centre (or polygon) intersects the Western Nepal box."""
    if gaps.empty:
        return gaps
    lon0, lon1 = WESTERN_GAP_LON
    lat0, lat1 = WESTERN_GAP_LAT
    hit = (
        (gaps["lon_max"] >= lon0)
        & (gaps["lon_min"] <= lon1)
        & (gaps["lat_max"] >= lat0)
        & (gaps["lat_min"] <= lat1)
    )
    return gaps.loc[hit].copy()


def plot_validation(gaps: pd.DataFrame, western: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    prepare_map_ax(ax, "Candidate gaps vs Western Nepal Seismic Gap")
    overlay_thrusts(ax)

    lon0, lon1 = WESTERN_GAP_LON
    lat0, lat1 = WESTERN_GAP_LAT
    ax.add_patch(
        Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            fill=True,
            facecolor="#6a51a3",
            alpha=0.18,
            edgecolor="#3f007d",
            linewidth=1.4,
            zorder=3,
            label="Western Nepal Seismic Gap",
        )
    )

    for _, row in gaps.iterrows():
        ax.add_patch(
            Rectangle(
                (row["lon_min"], row["lat_min"]),
                row["lon_max"] - row["lon_min"],
                row["lat_max"] - row["lat_min"],
                fill=False,
                edgecolor="#1f78b4",
                linewidth=1.0,
                zorder=4,
            )
        )
    if len(western):
        ax.scatter(
            western["lon_center"],
            western["lat_center"],
            s=28,
            c="#e31a1c",
            zorder=7,
            label="Gap cells inside western box",
            edgecolors="k",
            linewidths=0.3,
        )

    ax.legend(loc="lower left", framealpha=0.92)
    return savefig(fig, path)


def chance_metrics() -> dict:
    """
    Areas of the reference box and the study window, plus the chance that
    a random 0.5° analysis cell intersects the reference box.
    """
    study_km2 = spherical_rectangle_area_km2(
        SEIS_LON_MIN, SEIS_LON_MAX, SEIS_LAT_MIN, SEIS_LAT_MAX
    )
    west_km2 = spherical_rectangle_area_km2(
        WESTERN_GAP_LON[0],
        WESTERN_GAP_LON[1],
        WESTERN_GAP_LAT[0],
        WESTERN_GAP_LAT[1],
    )
    lon, lat, _, _ = study_grid()
    dlon = float(lon[1] - lon[0])
    dlat = float(lat[1] - lat[0])
    lon_min = lon - dlon / 2.0
    lon_max = lon + dlon / 2.0
    lat_min = lat - dlat / 2.0
    lat_max = lat + dlat / 2.0
    lon_hit = (lon_max >= WESTERN_GAP_LON[0]) & (lon_min <= WESTERN_GAP_LON[1])
    lat_hit = (lat_max >= WESTERN_GAP_LAT[0]) & (lat_min <= WESTERN_GAP_LAT[1])
    hit = lat_hit[:, None] & lon_hit[None, :]
    n_cells = int(hit.size)
    n_overlap = int(hit.sum())
    return {
        "study_area_km2": study_km2,
        "western_gap_area_km2": west_km2,
        "chance_area_percent": 100.0 * west_km2 / study_km2,
        "n_grid_cells": n_cells,
        "n_grid_cells_in_western_box": n_overlap,
        "chance_cell_percent": 100.0 * n_overlap / n_cells,
    }


def locate_gorkha() -> dict:
    """Confirm the 25 April 2015 Mw 7.8 Gorkha mainshock survived cleaning."""
    event_id = "us20002926"
    out = {
        "event_id": event_id,
        "timestamp": "n/a",
        "magnitude": None,
        "mag_type": "",
        "in_merged": False,
        "in_declustered": False,
        "is_mainshock": False,
        "in_screened": False,
    }
    merged = OUTPUT_DIR / "merged_catalog.csv"
    decl = OUTPUT_DIR / "declustered_catalog.csv"
    screened = OUTPUT_DIR / "screened_catalog.csv"

    def _hit(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "EventID" not in df.columns:
            return pd.DataFrame()
        return df.loc[df["EventID"].astype(str) == event_id]

    hit_m = _hit(merged)
    hit_d = _hit(decl)
    hit_s = _hit(screened)
    out["in_merged"] = not hit_m.empty
    out["in_declustered"] = not hit_d.empty
    out["in_screened"] = not hit_s.empty
    src = hit_d if not hit_d.empty else hit_m
    if not src.empty:
        row = src.iloc[0]
        out["timestamp"] = str(row["Timestamp"])
        out["magnitude"] = float(row["Magnitude"])
        out["mag_type"] = str(row.get("MagnitudeType", ""))
        if "is_mainshock" in src.columns:
            out["is_mainshock"] = bool(row["is_mainshock"])
    return out


def gap_markdown_table(gaps: pd.DataFrame) -> str:
    """Final FDR gaps with the sub-periods in which the same cell is a gap."""
    header = (
        "| Cell center (lat, lon) | Exact p-value | Density | Temporal persistence |\n"
        "| --- | ---: | ---: | --- |"
    )
    period_sets: dict[str, set[tuple[int, int]]] = {}
    for key in PERIODS:
        path = OUTPUT_DIR / f"gap_cells_{key}.csv"
        if not path.exists():
            period_sets[key] = set()
            continue
        try:
            sub = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            sub = pd.DataFrame()
        if sub.empty or "cell_i" not in sub.columns:
            period_sets[key] = set()
        else:
            period_sets[key] = set(
                zip(sub["cell_i"].astype(int), sub["cell_j"].astype(int))
            )
    if gaps.empty:
        return header + "\n| — | — | — | no cell survived FDR |"
    lines = [header]
    for row in gaps.itertuples(index=False):
        present = [
            PERIOD_LABELS[key]
            for key in PERIODS
            if (int(row.cell_i), int(row.cell_j)) in period_sets[key]
        ]
        persist = ", ".join(present) if present else "none"
        lines.append(
            f"| {row.lat_center:.2f}°N, {row.lon_center:.2f}°E "
            f"| {float(row.p_value):.3f} | {float(row.density):.4g} | {persist} |"
        )
    return "\n".join(lines)


def _fmt_mc(stats: dict) -> str:
    lines = []
    mc = stats.get("mc_values") or {}
    n_scr = stats.get("n_screened") or {}
    for key, label in PERIOD_LABELS.items():
        val = mc.get(key)
        n = n_scr.get(key, "n/a")
        if val is None:
            lines.append(f"  {label}: Mc = n/a  (n screened = {n})")
        else:
            lines.append(f"  {label}: Mc = {float(val):.2f}  (n screened = {n})")
    return "\n".join(lines)


def write_report(stats: dict, gaps: pd.DataFrame, western: pd.DataFrame) -> Path:
    recovered = len(western) > 0
    eps_km = stats.get("dbscan_eps_km")
    eps_rad = stats.get("dbscan_eps_rad")

    gap_table = gap_markdown_table(gaps)
    gorkha = locate_gorkha()
    if gorkha["magnitude"] is None:
        gorkha_mag = "n/a"
    else:
        gorkha_mag = f"Mw {gorkha['magnitude']:.1f}"
        if gorkha["mag_type"] and gorkha["mag_type"].lower() not in {"", "nan"}:
            gorkha_mag += f" ({gorkha['mag_type']})"
    gorkha_time = str(gorkha["timestamp"]).replace("+00:00", " UTC")
    gorkha_dup = "NO" if gorkha["in_merged"] else "YES"
    gorkha_as = "NO" if gorkha["is_mainshock"] else "YES"

    period_gap = stats.get("n_gap_cells_by_period") or {}
    period_gap_txt = "\n".join(
        f"  {PERIOD_LABELS.get(k, k)}: {v} cells" for k, v in period_gap.items()
    ) or "  n/a"

    n_gaps = int(stats.get("n_gap_cells", len(gaps)))
    if n_gaps > 0:
        hit_rate = 100.0 * len(western) / n_gaps
        hit_txt = f"{hit_rate:.1f}% ({len(western)} of {n_gaps} gap cells)"
    else:
        hit_txt = "n/a (no significant gap cells)"

    study_area = stats.get("study_area_km2")
    west_area = stats.get("western_gap_area_km2")
    cell_chance = stats.get("chance_cell_percent")
    area_chance = stats.get("chance_area_percent")
    alpha = stats.get("mc_alpha", 0.05)
    alpha_txt = "0.05" if alpha is None else f"{float(alpha):.2f}"

    study_txt = "n/a" if study_area is None else f"{float(study_area):,.0f} km²"
    west_txt = "n/a" if west_area is None else f"{float(west_area):,.0f} km²"
    cell_chance_txt = "n/a" if cell_chance is None else (
        f"{float(cell_chance):.1f}% "
        f"({stats.get('n_grid_cells_in_western_box', 'n/a')} of "
        f"{stats.get('n_grid_cells', 'n/a')} cells)"
    )
    area_chance_txt = "n/a" if area_chance is None else f"{float(area_chance):.1f}%"

    text = f"""Spatiotemporal Clustering and Seismic Gap Identification in Nepal
Density-based machine learning catalogue analysis, 1990–2026
================================================================

1. Catalogue merging (Phase 2)
   USGS events (study region)     : {stats.get('n_usgs_raw', 'n/a')}
   Kaggle NSC events              : {stats.get('n_kaggle_nsc', 'n/a')}
   Kaggle USGS-dump events        : {stats.get('n_kaggle_usgs', 'n/a')}
   Total before deduplication     : {stats.get('n_before_dedup', 'n/a')}
   Total after deduplication      : {stats.get('n_after_dedup', 'n/a')}
   Dedup rule                     : ±{stats.get('dedup_time_sec', 90)} s, {stats.get('dedup_dist_km', 50)} km Haversine

2. Declustering (Phase 3, Gardner–Knopoff 1974)
   Events before declustering     : {stats.get('n_before_decluster', 'n/a')}
   Independent events retained    : {stats.get('n_after_decluster', 'n/a')}
   Foreshocks/aftershocks removed : {stats.get('n_dependent', 'n/a')}
   Removal fraction               : {stats.get('decluster_removal_fraction', 'n/a')}
   GK spatial-window scale        : {stats.get('gk_spatial_scale', 'n/a')}

3. Magnitude of completeness (MAXC)
{_fmt_mc(stats)}
   Screened catalogue total       : {stats.get('n_screened_total', 'n/a')}

4. Spatial clustering (Phase 4)
   Analysis window                : {SEIS_LAT_MIN:.1f}–{SEIS_LAT_MAX:.1f}°N, {SEIS_LON_MIN:.1f}–{SEIS_LON_MAX:.1f}°E
   Events inside seismogenic zone : {stats.get('n_cluster_events', 'n/a')}
   DBSCAN min_samples             : {stats.get('dbscan_min_samples', 5)}
   Optimal DBSCAN eps             : {eps_km if eps_km is None else f'{eps_km:.2f} km'} ({eps_rad if eps_rad is None else f'{eps_rad:.6f} rad'})
   DBSCAN clusters / noise        : {stats.get('n_dbscan_clusters', 'n/a')} / {stats.get('n_dbscan_noise', 'n/a')}
   HDBSCAN min_cluster_size       : {stats.get('hdbscan_min_cluster_size', 5)}
   HDBSCAN min_samples            : {stats.get('hdbscan_min_samples', 5)}
   HDBSCAN clusters / noise       : {stats.get('n_hdbscan_clusters', 'n/a')} / {stats.get('n_hdbscan_noise', 'n/a')}

5. KDE gaps, inhomogeneous null, and FDR (Phase 5)
   KDE backend                    : {stats.get('kde_backend', 'n/a')}
   Monte Carlo backend            : {stats.get('mc_backend', 'n/a')}
   Monte Carlo iterations         : {stats.get('mc_iterations', 1000)}
   Null catalogue size N          : {stats.get('mc_n_points', 'n/a')}
   Null model                     : points drawn from a broad KDE of the real catalogue (arc intensity)
   Intensity bandwidth factor     : {stats.get('mc_intensity_bw_factor', 'n/a')}
   Exact p-value                  : (simulations with density <= observed) / {stats.get('mc_iterations', 1000)}
   Multiple-testing correction    : Benjamini–Hochberg FDR, alpha = {alpha_txt}
   Interior cells tested          : {stats.get('n_fdr_tests', 'n/a')}
   Edge buffer                    : 0.5° inward from the seismogenic box
   Edge cells excluded            : {stats.get('n_edge_cells_excluded', 'n/a')}
   FDR-significant gap cells      : {stats.get('n_gap_cells', len(gaps))}

   Significant gap cells:

{gap_table}

6. Temporal gaps (Phase 6)
{period_gap_txt}
   Stable (all three periods)     : {stats.get('n_gaps_stable', 'n/a')}
   Emerging (2015–2026 only)      : {stats.get('n_gaps_emerging', 'n/a')}
   Disappearing (absent 2015–2026): {stats.get('n_gaps_disappearing', 'n/a')}
   Each period is tested on its own arc-intensity null and FDR threshold.

7. Validation against the Western Nepal Seismic Gap
   Reference box                  : {WESTERN_GAP_LON[0]}–{WESTERN_GAP_LON[1]}°E, {WESTERN_GAP_LAT[0]}–{WESTERN_GAP_LAT[1]}°N (along the MFT)
   Western reference-box area     : {west_txt}
   Seismogenic-box area           : {study_txt}
   Chance (area ratio)            : {area_chance_txt}
   Chance a random grid cell hits : {cell_chance_txt}
   Overlapping significant cells  : {len(western)}
   Observed gap hit rate          : {hit_txt}
   Western Nepal gap recovered    : {'YES' if recovered else 'NO'}

8. 2015 Gorkha mainshock check
   Event ID                       : {gorkha['event_id']}
   Timestamp                      : {gorkha_time}
   Magnitude                      : {gorkha_mag}
   In deduplicated catalogue      : {'YES' if gorkha['in_merged'] else 'NO'}
   In declustered catalogue       : {'YES' if gorkha['in_declustered'] else 'NO'}
   Retained as a mainshock        : {'YES' if gorkha['is_mainshock'] else 'NO'}
   In Mc-screened catalogue       : {'YES' if gorkha['in_screened'] else 'NO'}
   Removed as a duplicate         : {gorkha_dup}
   Removed as an aftershock       : {gorkha_as}

Notes
-----
Local NSC magnitudes (ML, or rows with no magnitude type) were converted to
Mw with the Scordilis (2006) body-wave scaling, the relation Rajaure & Paudel
(2018) applied to Nepal local magnitudes. For catalog homogenization, ML at
M >= 4.0 is carried onto the Mw scale by that relation rather than left as a
separate local scale; magnitudes already reported as Mw were not changed.
Declustering uses the Gardner–Knopoff windows of Van Stiphout et al. (2011),
with a 1000-day cap on the M>=6.5 time window (GK 1974 graphical saturation)
and a spatial scale reduction if removal exceeds 70%. Completeness is the
Maximum Curvature (MAXC) estimate per period, reported to two decimal places.
Clustering and KDE use only the seismogenic box (27.5–30.5°N, 80.0–88.5°E),
so the Gangetic foreland is not part of the density field. DBSCAN eps is the
kneedle of the 5-distance haversine curve, clipped to 15–25 km. HDBSCAN uses
min_cluster_size=5 and min_samples=5. The Monte Carlo null draws N points
from a broad Gaussian KDE of the real epicentres (numpy choice on the
flattened intensity grid, then uniform placement inside the chosen 0.5° cell).
Each draw is scored with the analysis KDE. Exact cell p-values are the
fraction of the 1000 simulations at least as low as the data, and gaps are
the interior cells that remain after Benjamini–Hochberg FDR at 0.05. The
outer 0.5° of the seismogenic box is excluded from that test. Areas are
spherical rectangles. Gutenberg–Richter panels state Mc to two decimal places
in both the legend and the figure text. Web map tiles are not used.
"""
    path = OUTPUT_DIR / "summary_report.txt"
    path.write_text(text, encoding="utf-8")
    print(f"[report] wrote {path}")
    print(text)
    return path


def main() -> dict:
    ensure_output_dirs()
    gaps_path = OUTPUT_DIR / "gap_cells.csv"
    try:
        gaps = pd.read_csv(gaps_path) if gaps_path.exists() else pd.DataFrame()
        if gaps.empty and len(gaps.columns) == 0:
            gaps = pd.DataFrame()
    except pd.errors.EmptyDataError:
        gaps = pd.DataFrame()
    western = cells_in_western_gap(gaps)
    plot_validation(gaps, western, FIGURE_DIR / "validation_western_gap.png")

    recovered = bool(len(western) > 0)
    areas = chance_metrics()
    n_gaps = int(len(gaps))
    hit_rate = (100.0 * len(western) / n_gaps) if n_gaps else None
    stats = update_stats(
        n_western_gap_cells=int(len(western)),
        western_nepal_gap_recovered=recovered,
        observed_gap_hit_rate_percent=hit_rate,
        **areas,
    )
    write_report(stats, gaps, western)
    return stats


if __name__ == "__main__":
    main()
