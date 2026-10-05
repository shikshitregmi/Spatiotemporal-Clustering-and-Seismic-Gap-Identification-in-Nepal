"""Publication-ready maps and Gutenberg–Richter figures (DPI = 300)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import FIG_DPI, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.utils.geo import load_countries_clip, load_nepal_outline, load_thrust_traces


def apply_style() -> None:
    plt.rcParams.update(
        {
            "savefig.dpi": FIG_DPI,
            "figure.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
        }
    )


def savefig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


_BASEMAP_OK: bool | None = None


def _draw_nepal_outline(ax) -> None:
    """Filled Natural Earth land + Nepal — never fetch web map tiles."""
    ax.set_facecolor("#c5d6e6")
    countries = load_countries_clip()
    if countries is not None and len(countries):
        countries.plot(
            ax=ax,
            facecolor="#f6f0e6",
            edgecolor="#c4b8a8",
            linewidth=0.4,
            zorder=0,
        )
    nepal = load_nepal_outline()
    if nepal is not None and len(nepal):
        nepal.plot(
            ax=ax,
            facecolor="#ead9c4",
            edgecolor="#5a3d2b",
            linewidth=1.05,
            zorder=1,
        )


def _add_basemap(ax) -> None:
    """
    Local cartographic backdrop only.

    OpenStreetMap / Carto / Esri web tiles are not used: OSM volunteer
    servers block unidentified clients (HTTP 404 "Access blocked"), which
    previously burned into temporal_analysis_panels.png and
    validation_western_gap.png.
    """
    _draw_nepal_outline(ax)


def prepare_map_ax(ax, title: str | None = None):
    # Draw land first, then lock the Nepal window so Natural Earth polygons
    # cannot expand the axes (that previously left empty OSM-tile regions).
    _add_basemap(ax)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    mean_lat = 0.5 * (LAT_MIN + LAT_MAX)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)), adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#4a4a4a")
    ax.grid(True, ls=":", lw=0.5, color="#8a8a8a", alpha=0.35, zorder=1.4)
    ax.set_axisbelow(False)
    if title:
        ax.set_title(title, pad=8)
    return ax


def overlay_thrusts(ax, linewidth: float = 1.6) -> None:
    """Draw MFT (solid dark red) and MBT (dashed dark orange)."""
    traces = load_thrust_traces()
    for _, row in traces.iterrows():
        name = str(row.get("name", "")).lower()
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        is_mft = "frontal" in name or "mft" in name
        color = "#8b1e1e" if is_mft else "#c45c26"
        ls = "-" if is_mft else "--"
        try:
            xs, ys = geom.xy
            ax.plot(xs, ys, color=color, lw=linewidth, ls=ls, zorder=6, solid_capstyle="round")
        except Exception:
            # MultiLineString
            for part in getattr(geom, "geoms", []):
                xs, ys = part.xy
                ax.plot(xs, ys, color=color, lw=linewidth, ls=ls, zorder=6)

    handles = [
        Line2D([0], [0], color="#8b1e1e", lw=1.6, label="MFT"),
        Line2D([0], [0], color="#c45c26", lw=1.6, ls="--", label="MBT"),
    ]
    return handles


def plot_cluster_map(
    df: pd.DataFrame,
    label_col: str,
    path: Path,
    title: str,
) -> Path:
    """Scatter map of cluster labels (noise = light grey)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    prepare_map_ax(ax, title)
    thrust_handles = overlay_thrusts(ax)

    labels = df[label_col].to_numpy()
    unique = [lab for lab in np.unique(labels) if lab != -1]
    noise = df[labels == -1]
    if len(noise):
        ax.scatter(
            noise["Longitude"],
            noise["Latitude"],
            s=8,
            c="#b0b0b0",
            alpha=0.55,
            linewidths=0,
            zorder=3,
            label="Noise",
        )

    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    for i, lab in enumerate(unique):
        sub = df[labels == lab]
        ax.scatter(
            sub["Longitude"],
            sub["Latitude"],
            s=18,
            color=cmap(i),
            alpha=0.85,
            linewidths=0.2,
            edgecolors="k",
            zorder=4,
            label=f"Cluster {lab}" if len(unique) <= 12 else None,
        )

    if len(unique) <= 12:
        ax.legend(
            handles=list(ax.get_legend_handles_labels()[0]) + list(thrust_handles),
            loc="lower left",
            framealpha=0.92,
            ncol=1,
        )
    else:
        ax.legend(handles=thrust_handles, loc="lower left", framealpha=0.92)

    return savefig(fig, path)


def plot_gr_figures(period_mags: dict[str, np.ndarray], mc_values: dict[str, float], figure_dir: Path) -> list[Path]:
    """Write one compact GR figure per period (no tall combined panel)."""
    apply_style()
    from src.config import PERIOD_LABELS
    from src.utils.completeness import gutenberg_richter

    written: list[Path] = []
    for key, mag in period_mags.items():
        centres, counts, cumulative = gutenberg_richter(mag)
        mc = mc_values.get(key, np.nan)
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        if centres.size:
            ax.bar(
                centres,
                counts,
                width=0.08,
                color="#7eb6d9",
                edgecolor="#1f4e79",
                linewidth=0.4,
                label="Non-cumulative",
                zorder=2,
            )
            ax.plot(
                centres,
                cumulative,
                color="#c0392b",
                marker="o",
                ms=4,
                lw=1.4,
                label="N(≥M)",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Magnitude (Mw)")
        ax.set_ylabel("Count")
        mc_txt = f"Mc = {float(mc):.2f}" if np.isfinite(mc) else "Mc = n/a"
        ax.set_title(f"Gutenberg–Richter  ·  {PERIOD_LABELS.get(key, key)}  ·  {mc_txt}")
        if np.isfinite(mc):
            ax.axvline(mc, color="#1a7f37", ls="--", lw=1.5, label=mc_txt)
            ax.text(
                0.98,
                0.62,
                mc_txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#1a7f37",
                fontsize=10,
                fontweight="bold",
            )
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#dddddd")
        fig.tight_layout()
        written.append(savefig(fig, figure_dir / f"mc_{key}.png"))
    return written
