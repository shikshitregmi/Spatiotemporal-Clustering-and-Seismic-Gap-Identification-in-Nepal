
# Role and Objective
You are an expert Data Scientist, Seismologist, and Python Developer. Your objective is to build a complete, reproducible Python pipeline for a research paper titled: "Spatiotemporal Clustering and Seismic Gap Identification in Nepal Using Density-Based Machine Learning (1990–2026)". 

# Environment & Hardware
- I am working on a machine with an RTX 4060 Laptop GPU (8GB VRAM), CUDA-enabled via PyTorch.
- While clustering algorithms (DBSCAN/HDBSCAN) are primarily CPU-bound, you should utilize GPU acceleration where beneficial (e.g., data manipulation with cuDF/cupy if compatible, or PyTorch for distance matrix calculations/KDE weightings) to optimize performance.
- All code must be written in Python.

# Data Sources Available
1. `query.csv`: Raw earthquake data queried directly from the USGS ComCat API for the Nepal region (approx. 1990-2026). This file is located in the root project directory.
2. **Kaggle Dataset:** Instead of local files, the code must programmatically download the Kaggle dataset "Nepal Earthquake Seismicity Dataset (1990-2026)" using the `kagglehub` library.

# Project Structure & Execution Plan
Please build the project modularly. Create separate Python scripts for each phase, or a well-organized Jupyter Notebook pipeline. I recommend a `src/` directory for scripts and an `outputs/` directory for results.

Execute the following phases step-by-step, writing the code, running it, and verifying the outputs:

## Phase 1: Project Setup & Dependencies
1. Create a `requirements.txt` file containing necessary libraries: `pandas`, `numpy`, `scikit-learn`, `hdbscan`, `geopandas`, `shapely`, `matplotlib`, `seaborn`, `contextily` (for basemaps), `obspy` (for seismological utilities), `torch` (for GPU tasks), `scipy`, and `kagglehub[pandas-datasets]`.
2. Instruct me to run `pip install -r requirements.txt` if any imports fail.

## Phase 2: Data Ingestion, Merging, & Preprocessing
1. **Load USGS Data:** Load `query.csv` into a Pandas DataFrame.
2. **Download & Load Kaggle Data:** Use the `kagglehub` library to programmatically download and load the dataset. Use the following logic as a base:
   ```python
   import kagglehub
   from kagglehub import KaggleDatasetAdapter

   # You need to determine the correct file_path(s) within the dataset 
   # (e.g., the historical or processed CSV files). List the files if necessary.
   file_path = "" # Cursor, please determine the correct file paths

   kaggle_df = kagglehub.load_dataset(
     KaggleDatasetAdapter.PANDAS,
     "amansinghnp/nepal-earthquake-seismicity-dataset-1990-2026",
     file_path,
   )
   ```
3. Standardize column names across both USGS and Kaggle data (Timestamp, Latitude, Longitude, Magnitude, Depth).
4. **Deduplication:** Write a function to deduplicate events across the USGS and Kaggle catalogs using a time-proximity rule (±60–120 seconds) and spatial-proximity rule (~50 km using Haversine distance).
5. **Magnitude Standardization:** Ensure all magnitudes are on a unified scale (e.g., Mw). If conversion formulas are needed in the Kaggle dataset, apply them.
6. Save the merged catalog as `outputs/merged_catalog.csv`.

## Phase 3: Declustering & Completeness Screening
1. **Declustering:** Implement the Gardner-Knopoff (1974) declustering algorithm (or use `obspy`'s built-in declustering capabilities) to remove aftershocks. Print and save the event counts before and after declustering.
2. **Magnitude of Completeness (Mc):** 
   - Divide the declustered catalog into 3 sub-periods: 1990–2000, 2000–2015, 2015–2026.
   - Calculate Mc for each period using the Maximum Curvature (MAXC) method.
   - Truncate each sub-catalog to include only events >= its specific Mc threshold.
   - Save the three screened sub-catalogs and a figure showing the Gutenberg-Richter plots with Mc lines (`outputs/figures/mc_plots.png`).

## Phase 4: Spatial Clustering (DBSCAN & HDBSCAN)
1. **DBSCAN:** 
   - Use `scikit-learn`'s DBSCAN with the `haversine` metric. 
   - Generate a k-distance plot to programmatically identify the "elbow" for the `eps` parameter. Save as `outputs/figures/k_distance_plot.png`.
   - Run DBSCAN using the identified `eps` and a suitable `min_samples`.
2. **HDBSCAN:**
   - Run `hdbscan` as a robustness check on the same spatial data.
3. Export cluster labels back to the main dataframe. 
4. Generate a map showing the DBSCAN clusters (`outputs/figures/dbscan_clusters.png`) and HDBSCAN clusters (`outputs/figures/hdbscan_clusters.png`), using `contextily` to add a geographical basemap.

## Phase 5: Gap Definition & KDE Mapping
1. **Spatial Grid:** Create a regular spatial grid over Nepal (0.25° × 0.25° cells).
2. **KDE Surfaces:** Use `scipy.stats.gaussian_kde` to compute event density across the grid. (If the data size is massive, attempt to utilize PyTorch tensors for the KDE matrix multiplications to leverage the RTX 4060 GPU).
3. **Gap Identification:** 
   - Identify "candidate gap cells" as those in the bottom 10% of density.
   - Filter these to only include cells spatially bounded by higher-density neighbors (to avoid empty borders).
4. Generate a high-quality map overlaying the KDE heatmap, the candidate gap polygons, and the approximate trace of the Main Frontal Thrust (MFT) / Main Boundary Thrust (MBT). Save as `outputs/figures/kde_gaps_map.png`.

## Phase 6: Temporal Analysis
1. Repeat the KDE and Gap Definition procedure (Phase 5) for the 3 sub-periods (1990–2000, 2000–2015, 2015–2026).
2. Generate a 1x3 panel figure showing the decade-by-decade maps to visualize if the gaps are stable, emerging, or disappearing. Save as `outputs/figures/temporal_analysis_panels.png`.

## Phase 7: Validation & Final Outputs
1. **Validation:** Overlay the computationally identified gaps on a map with the known "Western Nepal Seismic Gap" coordinates (approx. longitude 79°E to 82°E, along the MFT).
2. **Summary Generation:** Write a script to generate a `outputs/summary_report.txt` file that prints:
   - Total events before/after deduplication.
   - Total events before/after declustering.
   - Mc values for each sub-period.
   - Optimal DBSCAN `eps` value.
   - Coordinates of the identified candidate gap cells.
3. Ensure all figures are publication-ready (clear labels, legends, colorbars, high DPI=300).

# Execution Rules for Cursor:
1. Do not ask me for permission to run code. Execute it and check the outputs yourself.
2. If you encounter a missing library, attempt to install it via terminal programmatically.
3. If a specific algorithm (like Gardner-Knopoff) is complex to write from scratch, utilize established libraries like `obspy` or well-tested community implementations.
4. Comment your code thoroughly, as this will eventually be reviewed for a peer-reviewed journal (NPRC).
5. Once complete, provide a brief summary of the results: what the Mc values were, what eps was chosen, and whether the Western Nepal gap was successfully recovered.