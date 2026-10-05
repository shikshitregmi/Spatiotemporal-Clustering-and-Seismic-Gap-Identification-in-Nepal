# Spatiotemporal Clustering and Seismic Gap Identification in Nepal Using Density-Based Machine Learning (1990–2026)

Reproducible Python pipeline for a catalogue study of Nepal Himalayan seismicity. It merges public earthquake catalogues, removes aftershocks, estimates completeness, clusters epicentres, and tests candidate seismic gaps against an inhomogeneous Monte Carlo null.

Clustering, kernel density estimation, and the gap test use the seismogenic belt **27.5–30.5°N, 80.0–88.5°E**. The Gangetic foreland south of the Himalayan front is not part of that analysis.

## Setup

Python 3.10 or newer is assumed.

```bash
pip install -r requirements.txt
```

KDE evaluation uses PyTorch and will run on a CUDA GPU when one is available. The same calculations run on CPU if CUDA is not present.

The Kaggle download needs a Kaggle account configured for `kagglehub` (an API token in `~/.kaggle/kaggle.json`, which is not part of this repository).

## Data that is not in this repository

Earthquake catalogues and generated products are intentionally omitted.

| File | Role |
| --- | --- |
| `query.csv` | USGS ComCat extract for the Nepal region. Place this file in the project root before running the pipeline. |
| Kaggle dataset `amansinghnp/nepal-earthquake-seismicity-dataset-1990-2026` | Downloaded at runtime. The National Seismological Centre file and the historical USGS dump are used. The pre-merged master file is skipped so USGS events are not counted twice. |
| `outputs/` | Catalogues, figures, cached map files, and `summary_report.txt`. Created when the pipeline runs. |

## Run

From the project root:

```bash
python run_pipeline.py
```

The script runs these stages in order:

1. **Ingest.** Standardise the USGS and Kaggle tables and convert magnitudes to Mw (Scordilis, 2006).
2. **Deduplicate.** Drop matches within ±90 seconds and 50 km. USGS records are kept ahead of the other sources.
3. **Declustering.** Gardner–Knopoff (1974) space–time windows, as parameterised by Van Stiphout et al. (2011), with a 1000-day cap on the large-magnitude time window.
4. **Completeness.** Maximum Curvature magnitude of completeness for 1990–2000, 2000–2015, and 2015–2026. Each period keeps events at or above its own Mc.
5. **Clustering.** DBSCAN with a haversine metric, and HDBSCAN (`min_cluster_size=5`, `min_samples=5`) on the same distances. Both use only events inside the seismogenic belt.
6. **Gaps.** Gaussian KDE on a 0.5° grid. A broad KDE of the real epicentres is the sampling surface for 1000 Monte Carlo catalogues. Exact one-sided p-values are corrected with Benjamini–Hochberg FDR at 0.05. Cells in the outer 0.5° of the seismogenic box are not tested.
7. **Reporting.** Comparison with the Western Nepal Seismic Gap reference box, a check that the 25 April 2015 Mw 7.8 Gorkha mainshock remains in the catalogue, and `outputs/summary_report.txt`. A methodology flowchart is written to `outputs/figures/`.

## Layout

```
run_pipeline.py          entry point
src/config.py            paths, windows, and analysis constants
src/01_…07_*.py          one script per stage
src/utils/               magnitudes, declustering, KDE, maps
query.csv                local USGS extract (not tracked)
outputs/                 results (not tracked)
```

Individual stages can be run after their inputs exist, for example `python src/03_spatial_clustering.py`.
