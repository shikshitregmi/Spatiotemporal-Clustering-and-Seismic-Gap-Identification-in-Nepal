"""
Kernel density estimation on a lon/lat grid.

``scipy.stats.gaussian_kde`` supplies Scott/Silverman bandwidths. When a
CUDA device is visible to PyTorch the *evaluation* of the Gaussian kernel
on the grid is performed with batched tensor matmuls on the GPU (RTX 4060).
For the Nepal catalogue (10³ events, ~10³ grid cells) both backends are
fast; the GPU path is retained so the pipeline uses the available hardware.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from src.config import (
    GRID_RES_DEG,
    KDE_BW_FACTOR_FULL,
    KDE_BW_FACTOR_SPARSE,
    KDE_SPARSE_N,
    MC_INTENSITY_BW_FACTOR,
    MC_N_ITER,
    MC_SEED,
    SEIS_LAT_MAX,
    SEIS_LAT_MIN,
    SEIS_LON_MAX,
    SEIS_LON_MIN,
)


def study_grid(
    res: float = GRID_RES_DEG,
    lon_min: float = SEIS_LON_MIN,
    lon_max: float = SEIS_LON_MAX,
    lat_min: float = SEIS_LAT_MIN,
    lat_max: float = SEIS_LAT_MAX,
):
    """Return 1-D lon/lat centres and a meshgrid inside the seismogenic box."""
    lon = np.arange(lon_min + res / 2.0, lon_max, res)
    lat = np.arange(lat_min + res / 2.0, lat_max, res)
    lon_m, lat_m = np.meshgrid(lon, lat)
    return lon, lat, lon_m, lat_m


def _torch_device():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def resolve_bw_factor(n: int, bw_factor: float | None) -> float:
    """Scott multiplier used for a catalogue of size ``n``."""
    if bw_factor is not None:
        return float(bw_factor)
    if n < KDE_SPARSE_N:
        return float(KDE_BW_FACTOR_SPARSE)
    return float(KDE_BW_FACTOR_FULL)


def evaluate_kde(
    lon: np.ndarray,
    lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    bw_factor: float | None = None,
    quiet: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Evaluate a 2-D Gaussian KDE at grid coordinates.

    ``bw_factor`` multiplies Scott's rule. Decade sub-catalogues (n ≲ 150)
    use a broader kernel so the 10th-percentile gap threshold is not
    dominated by shot noise. A 0.04° Gaussian jitter is added when n < 80.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    n = int(lon.size)
    if n < 3:
        return np.zeros_like(grid_lon, dtype=float), "empty"

    bw_factor = resolve_bw_factor(n, bw_factor)

    if n < 80:
        # Small-n catalogues: tiny jitter stabilises the KDE covariance.
        rng = np.random.default_rng(42)
        lon = lon + rng.normal(0.0, 0.04, size=n)
        lat = lat + rng.normal(0.0, 0.04, size=n)
        if not quiet:
            print(f"[kde] n={n} < 80; applied 0.04° jitter and bw_factor={bw_factor}")
    elif not quiet:
        print(f"[kde] n={n}; Scott bandwidth × {bw_factor}")

    points = np.vstack([lon, lat])
    kde = gaussian_kde(points)
    kde.set_bandwidth(bw_method=float(kde.factor) * float(bw_factor))
    grid = np.vstack([grid_lon.ravel(), grid_lat.ravel()])

    device = _torch_device()
    if device is None:
        dens = kde(grid).reshape(grid_lon.shape)
        return dens, "scipy"

    try:
        import torch

        # Match scipy.stats.gaussian_kde: covariance = data_cov * factor²
        dataset = torch.as_tensor(points, dtype=torch.float64, device=device)
        grid_t = torch.as_tensor(grid, dtype=torch.float64, device=device)
        cov = torch.as_tensor(np.asarray(kde.covariance), dtype=torch.float64, device=device)
        inv_cov = torch.linalg.inv(cov)
        d, n = dataset.shape
        norm = torch.sqrt((2.0 * torch.pi) ** d * torch.linalg.det(cov)) * n

        # (M, N, d) diffs — catalogue is small enough to broadcast fully.
        diff = grid_t.T.unsqueeze(1) - dataset.T.unsqueeze(0)  # (M, N, 2)
        # Mahalanobis: sum((diff @ inv_cov) * diff, dim=-1)
        maha = torch.einsum("mni,ij,mnj->mn", diff, inv_cov, diff)
        dens = torch.exp(-0.5 * maha).sum(dim=1) / norm
        out = dens.detach().cpu().numpy().reshape(grid_lon.shape)
        backend = "torch-cuda" if device.type == "cuda" else "torch-cpu"
        return out, backend
    except Exception as exc:  # noqa: BLE001
        print(f"[kde] PyTorch evaluation failed ({exc}); falling back to scipy")
        dens = kde(grid).reshape(grid_lon.shape)
        return dens, "scipy"


def _mc_torch_batch(points: np.ndarray, grid: np.ndarray, bw_factor: float, device) -> np.ndarray:
    """
    Gaussian KDE for a batch of catalogues.

    ``points`` is (B, 2, N), ``grid`` is (2, M). The covariance is Scott's
    rule times ``bw_factor``, matching ``evaluate_kde``.
    """
    import torch

    pts = torch.as_tensor(points, dtype=torch.float64, device=device)
    grd = torch.as_tensor(grid, dtype=torch.float64, device=device)
    _batch, dim, n = pts.shape
    scott = float(n) ** (-1.0 / (dim + 4))
    factor = scott * float(bw_factor)
    mean = pts.mean(dim=-1, keepdim=True)
    xc = pts - mean
    data_cov = torch.matmul(xc, xc.transpose(-1, -2)) / (n - 1)
    cov = data_cov * (factor**2)
    inv_cov = torch.linalg.inv(cov)
    det = torch.linalg.det(cov).clamp_min(1e-30)
    norm = torch.sqrt((2.0 * torch.pi) ** dim * det) * n

    # (B, M, N, 2)
    diff = grd.T[None, :, None, :] - pts.permute(0, 2, 1)[:, None, :, :]
    transformed = torch.einsum("bmni,bij->bmnj", diff, inv_cov)
    maha = (transformed * diff).sum(dim=-1)
    dens = torch.exp(-0.5 * maha).sum(dim=-1) / norm[:, None]
    return dens.detach().cpu().numpy()


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg rejection mask for a 1-D array of p-values.

    The largest rank k with p(k) <= (k/m) * alpha is the cutoff. Every
    hypothesis with p <= p(k) is rejected. An empty input returns an empty
    boolean array.
    """
    p = np.asarray(p_values, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    if p.size == 0:
        return reject
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = ranked.size
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = np.isfinite(ranked) & (ranked <= thresh)
    if not np.any(passed):
        return reject
    cutoff = float(ranked[np.max(np.where(passed))])
    reject = np.isfinite(p) & (p <= cutoff)
    return reject


def monte_carlo_null(
    real_lon: np.ndarray,
    real_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    bw_factor: float,
    n_iter: int = MC_N_ITER,
    seed: int = MC_SEED,
    intensity_bw_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Cell-wise null from an inhomogeneous Poisson sample of the real arc.

    A broad Gaussian KDE of the observed epicentres supplies a spatial
    intensity on the analysis grid. Each iteration draws ``N`` cell indices
    with ``numpy.random.Generator.choice`` weighted by that flattened
    intensity, then places one point uniformly inside the chosen cell.
    The draw is scored with the same analysis KDE used on the real catalogue.

    Returns ``(intensity, null_stack, backend)``. ``null_stack`` has shape
    ``(n_iter, *grid)``.
    """
    grid_lon = np.asarray(grid_lon, dtype=float)
    grid_lat = np.asarray(grid_lat, dtype=float)
    shape = grid_lon.shape
    real_lon = np.asarray(real_lon, dtype=float)
    real_lat = np.asarray(real_lat, dtype=float)
    ok = np.isfinite(real_lon) & np.isfinite(real_lat)
    real_lon, real_lat = real_lon[ok], real_lat[ok]
    n_points = int(real_lon.size)
    if n_points < 3 or n_iter < 1 or grid_lon.size < 2:
        empty = np.zeros((max(n_iter, 1), *shape), dtype=float)
        return np.zeros(shape, dtype=float), empty, "empty"

    bw_factor = float(bw_factor)
    if intensity_bw_factor is None:
        intensity_bw_factor = max(float(MC_INTENSITY_BW_FACTOR), bw_factor * 1.5)
    intensity, intensity_backend = evaluate_kde(
        real_lon,
        real_lat,
        grid_lon,
        grid_lat,
        bw_factor=float(intensity_bw_factor),
        quiet=True,
    )
    weights = np.clip(np.asarray(intensity, dtype=float).ravel(), 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(weights.size, dtype=float)
    weights = weights / weights.sum()
    # Generator.choice rejects weights that drift off 1 by a few ulps.
    weights[-1] = max(0.0, 1.0 - float(weights[:-1].sum()))
    weights = weights / weights.sum()

    dlon = float(grid_lon[0, 1] - grid_lon[0, 0]) if grid_lon.shape[1] > 1 else GRID_RES_DEG
    dlat = float(grid_lat[1, 0] - grid_lat[0, 0]) if grid_lat.shape[0] > 1 else GRID_RES_DEG
    lon_c = grid_lon.ravel()
    lat_c = grid_lat.ravel()
    n_cells = int(weights.size)
    rng = np.random.default_rng(seed)
    lon_s = np.empty((n_iter, n_points), dtype=float)
    lat_s = np.empty((n_iter, n_points), dtype=float)
    for i in range(n_iter):
        # Weighted draw from the flattened broad-KDE intensity grid.
        idx = rng.choice(n_cells, size=n_points, replace=True, p=weights)
        lon_s[i] = lon_c[idx] + rng.uniform(-dlon / 2.0, dlon / 2.0, size=n_points)
        lat_s[i] = lat_c[idx] + rng.uniform(-dlat / 2.0, dlat / 2.0, size=n_points)
    print(
        f"[kde] inhomogeneous null: broad bw × {intensity_bw_factor:.2f} "
        f"({intensity_backend}), N={n_points}, cells={n_cells}"
    )

    grid = np.vstack([grid_lon.ravel(), grid_lat.ravel()])
    stack = np.empty((n_iter, grid.shape[1]), dtype=np.float64)

    device = _torch_device()
    backend = "scipy"
    batch = 20
    used_torch = False
    if device is not None:
        try:
            for start in range(0, n_iter, batch):
                stop = min(start + batch, n_iter)
                pts = np.stack([lon_s[start:stop], lat_s[start:stop]], axis=1)
                stack[start:stop] = _mc_torch_batch(pts, grid, bw_factor, device)
                if start % 200 == 0:
                    print(f"[kde] Monte Carlo {stop}/{n_iter} ({device.type})")
            backend = "torch-cuda" if device.type == "cuda" else "torch-cpu"
            used_torch = True
        except Exception as exc:  # noqa: BLE001
            print(f"[kde] batched Monte Carlo failed ({exc}); using evaluate_kde")
            used_torch = False

    if not used_torch:
        for i in range(n_iter):
            dens, backend = evaluate_kde(
                lon_s[i], lat_s[i], grid_lon, grid_lat, bw_factor=bw_factor, quiet=True
            )
            stack[i] = dens.ravel()
            if i % 200 == 0:
                print(f"[kde] Monte Carlo {i + 1}/{n_iter} (scipy)")
        backend = "scipy"

    # Confirm the batched kernel matches the catalogue KDE on one draw.
    if used_torch and n_points >= 80:
        ref, ref_backend = evaluate_kde(
            lon_s[0], lat_s[0], grid_lon, grid_lat, bw_factor=bw_factor, quiet=True
        )
        rel = np.max(np.abs(stack[0] - ref.ravel()) / np.maximum(np.abs(ref.ravel()), 1e-12))
        print(f"[kde] Monte Carlo kernel check vs {ref_backend}: max relative error {rel:.3e}")
        if rel > 1e-4:
            raise RuntimeError(
                f"Monte Carlo KDE does not match evaluate_kde (relative error {rel:.3e})"
            )

    null = stack.reshape((n_iter, *shape))
    return intensity, null, backend
