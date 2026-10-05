"""
Run the Nepal seismic-gap pipeline end-to-end (Phases 1–7).

Usage (from the project root):
    python run_pipeline.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PHASES = [
    SRC / "01_ingest_preprocess.py",
    SRC / "02_decluster_completeness.py",
    SRC / "03_spatial_clustering.py",
    SRC / "04_kde_gaps.py",
    SRC / "05_temporal_analysis.py",
    SRC / "06_validate_report.py",
    SRC / "07_methodology_flowchart.py",
]


def _ensure_imports() -> None:
    """Install requirements if core scientific packages are missing."""
    missing: list[str] = []
    for mod in (
        "pandas",
        "numpy",
        "sklearn",
        "geopandas",
        "shapely",
        "matplotlib",
        "seaborn",
        "scipy",
        "kagglehub",
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    extra_checks = {
        "hdbscan": "hdbscan",
    }
    for mod, pkg in extra_checks.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        import obspy  # noqa: F401
    except ImportError:
        print("[setup] obspy is optional and not installed; using local Gardner–Knopoff")

    if not missing:
        return

    print(f"[setup] missing packages: {missing}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--default-timeout=120", *missing]
    )


def _run_phase(path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"Running {path.name}")
    print("=" * 72)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        raise RuntimeError(f"{path.name} has no main()")
    mod.main()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.config import configure_stdio, ensure_output_dirs

    configure_stdio()
    _ensure_imports()

    ensure_output_dirs()
    for phase in PHASES:
        _run_phase(phase)
    print("\nPipeline complete. See outputs/summary_report.txt")


if __name__ == "__main__":
    main()
