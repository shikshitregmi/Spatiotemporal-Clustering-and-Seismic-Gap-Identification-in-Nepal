"""
Publication flowchart for the Nepal seismic-gap pipeline.

Drawn with matplotlib patches only (rounded boxes and arrows).
Saves ``outputs/figures/methodology_flowchart.png`` at 300 dpi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURE_DIR, ensure_output_dirs  # noqa: E402
from src.utils.plotting import apply_style, savefig  # noqa: E402

STAGES = [
    "Data Ingestion (USGS + Kaggle)",
    "Deduplication",
    "Gardner-Knopoff Declustering",
    "MAXC Completeness Screening",
    "DBSCAN / HDBSCAN Clustering",
    "Monte Carlo Tested KDE Gaps",
]


def _box(ax, x, y, w, h, facecolor: str = "#1f4e79"):
    patch = FancyBboxPatch(
        (x - w / 2.0, y - h / 2.0),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor="#0d1b2a",
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>,head_length=8,head_width=5",
            mutation_scale=1.0,
            linewidth=1.6,
            color="#222222",
            zorder=4,
            shrinkA=0.0,
            shrinkB=0.0,
        )
    )


def main() -> Path:
    ensure_output_dirs()
    out = FIGURE_DIR / "methodology_flowchart.png"
    if out.exists():
        out.unlink()

    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 10.6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.975,
        "Methodology",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111111",
        zorder=5,
    )

    box_w, box_h = 0.72, 0.090
    x = 0.50
    ys = [0.875 - i * 0.140 for i in range(len(STAGES))]
    for i, label in enumerate(STAGES):
        _box(ax, x, ys[i], box_w, box_h)
        ax.text(
            x,
            ys[i],
            label,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        if i < len(STAGES) - 1:
            _arrow(
                ax,
                x,
                ys[i] - box_h / 2.0 - 0.008,
                x,
                ys[i + 1] + box_h / 2.0 + 0.010,
            )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02)
    return savefig(fig, out)


if __name__ == "__main__":
    main()
