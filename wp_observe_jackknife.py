import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Any

import numpy as np
from Corrfunc.theory.DDrppi import DDrppi


@dataclass
class ObserveRpPiPairCounts:
    pair_counts: Dict[Tuple[int, int], np.ndarray]
    total_counts: np.ndarray
    involved_counts: np.ndarray
    rp_bin_edges: np.ndarray
    pi_edges: np.ndarray
    n_subboxes: int
    n_points_total: int
    n_points_subbox: np.ndarray
    n_rp_bins: int
    n_pi_bins: int


@dataclass
class ObserveRpPiCrossPairCounts:
    pair_counts: np.ndarray
    total_counts: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray
    n_subboxes: int
    n_points_1_total: int
    n_points_2_total: int
    n_points_1_subbox: np.ndarray
    n_points_2_subbox: np.ndarray


def _split_xyz(sample_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (N,3) coordinates to x,y,z arrays."""
    sample_xyz = np.asarray(sample_xyz)
    if sample_xyz.ndim != 2 or sample_xyz.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")
    return sample_xyz[:, 0], sample_xyz[:, 1], sample_xyz[:, 2]


def _validate_dpi_and_pimax(dpi: float, pimax: float) -> Tuple[float, int]:
    """Validate Corrfunc DDrppi pi-binning: fixed unit bins with integer pimax."""
    if not np.isclose(float(dpi), 1.0):
        raise ValueError("Corrfunc.theory.DDrppi does not take `dpi`; internal pi-bin width is fixed to 1. Use dpi=1.")
    if float(pimax) <= 0:
        raise ValueError("pimax must be > 0")

    n_pi = int(round(float(pimax)))
    if not np.isclose(float(pimax), float(n_pi)):
        raise ValueError("With DDrppi, pimax must be an integer because pi-bin width is fixed to 1.")

    return 1.0, n_pi


def _parse_bounds(
    sample_xyz: np.ndarray,
    random_xyz: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
) -> np.ndarray:
    """Build bounds for subbox partition; default uses data+random min/max."""
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (3, 2):
            raise ValueError("bounds must have shape ((xmin,xmax),(ymin,ymax),(zmin,zmax))")
        return b

    all_xyz = np.vstack((np.asarray(sample_xyz), np.asarray(random_xyz)))
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    b = np.column_stack((mins, maxs))

    eps = 1.0e-12
    for i in range(3):
        if b[i, 1] <= b[i, 0]:
            b[i, 1] = b[i, 0] + eps
    return b


def _assign_subbox_ids_observe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds_xyz: np.ndarray,
    ndiv: int,
) -> np.ndarray:
    """Assign points to ndiv^3 subboxes using explicit bounds."""
    mins = bounds_xyz[:, 0]
    maxs = bounds_xyz[:, 1]
    size = np.maximum(maxs - mins, 1.0e-12)

    fx = (x - mins[0]) / size[0]
    fy = (y - mins[1]) / size[1]
    fz = (z - mins[2]) / size[2]

    ix = np.minimum(np.maximum((fx * ndiv).astype(np.int32), 0), ndiv - 1)
    iy = np.minimum(np.maximum((fy * ndiv).astype(np.int32), 0), ndiv - 1)
    iz = np.minimum(np.maximum((fz * ndiv).astype(np.int32), 0), ndiv - 1)

    return ix + ndiv * (iy + ndiv * iz)


def _xi_landy_szalay_from_counts(
    dd_counts: np.ndarray,
    dr_counts: np.ndarray,
    rr_counts: np.ndarray,
    n_data: int,
    n_rand: int,
) -> np.ndarray:
    """Landy-Szalay xi = (DD - 2DR + RR)/RR with pair-count normalization."""
    if n_data < 2 or n_rand < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)

    dd_norm = dd_counts / (n_data * (n_data - 1.0))
    dr_norm = dr_counts / (n_data * n_rand)
    rr_norm = rr_counts / (n_rand * (n_rand - 1.0))

    xi = np.zeros_like(dd_counts, dtype=np.float64)
    valid = rr_norm > 0
    xi[valid] = (dd_norm[valid] - 2.0 * dr_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return xi


def _wp_from_xi_rppi_flat(xi_flat: np.ndarray, n_rp_bins: int, n_pi_bins: int, dpi: float) -> np.ndarray:
    xi2d = xi_flat.reshape(n_rp_bins, n_pi_bins)
    return 2.0 * np.sum(xi2d * dpi, axis=1)


def precompute_observe_rppi_pair_counts(
    sample_xyz: np.ndarray,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
) -> ObserveRpPiPairCounts:
    """Precompute DD(i,i), DD(i,j) for non-periodic (rp,pi) counting."""
    x, y, z = _split_xyz(sample_xyz)
    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)

    sub_ids = _assign_subbox_ids_observe(x, y, z, bounds_xyz=bounds_xyz, ndiv=ndiv)
    n_sub = ndiv ** 3
    indices: List[np.ndarray] = [np.where(sub_ids == i)[0] for i in range(n_sub)]
    n_points_subbox = np.array([idx.size for idx in indices], dtype=np.int64)

    n_rp = len(rp_bins) - 1
    n_flat = n_rp * n_pi

    pair_counts: Dict[Tuple[int, int], np.ndarray] = {}
    total_counts = np.zeros(n_flat, dtype=np.float64)
    involved_counts = np.zeros((n_sub, n_flat), dtype=np.float64)

    for i, j in combinations_with_replacement(range(n_sub), 2):
        idx_i = indices[i]
        idx_j = indices[j]
        if idx_i.size == 0 or idx_j.size == 0:
            counts = np.zeros(n_flat, dtype=np.float64)
        elif i == j:
            result = DDrppi(
                autocorr=1,
                nthreads=nthreads,
                pimax=pimax,
                binfile=rp_bins,
                X1=x[idx_i],
                Y1=y[idx_i],
                Z1=z[idx_i],
                periodic=False,
                output_rpavg=False,
                verbose=False,
            )
            counts = result["npairs"].astype(np.float64)
        else:
            result = DDrppi(
                autocorr=0,
                nthreads=nthreads,
                pimax=pimax,
                binfile=rp_bins,
                X1=x[idx_i],
                Y1=y[idx_i],
                Z1=z[idx_i],
                X2=x[idx_j],
                Y2=y[idx_j],
                Z2=z[idx_j],
                periodic=False,
                output_rpavg=False,
                verbose=False,
            )
            # autocorr=1 is ordered-pair convention, so cross terms need x2
            counts = 2.0 * result["npairs"].astype(np.float64)

        pair_counts[(i, j)] = counts
        total_counts += counts
        involved_counts[i] += counts
        if j != i:
            involved_counts[j] += counts

    pi_edges = np.arange(0.0, float(n_pi) + 1.0, 1.0, dtype=np.float64)

    return ObserveRpPiPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        involved_counts=involved_counts,
        rp_bin_edges=np.asarray(rp_bins, dtype=np.float64),
        pi_edges=pi_edges,
        n_subboxes=n_sub,
        n_points_total=x.size,
        n_points_subbox=n_points_subbox,
        n_rp_bins=n_rp,
        n_pi_bins=n_pi,
    )


def precompute_observe_rppi_cross_pair_counts(
    sample_xyz_1: np.ndarray,
    sample_xyz_2: np.ndarray,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
) -> ObserveRpPiCrossPairCounts:
    """Precompute DR(i,j) for non-periodic (rp,pi) counting."""
    x1, y1, z1 = _split_xyz(sample_xyz_1)
    x2, y2, z2 = _split_xyz(sample_xyz_2)
    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)

    sub_ids_1 = _assign_subbox_ids_observe(x1, y1, z1, bounds_xyz=bounds_xyz, ndiv=ndiv)
    sub_ids_2 = _assign_subbox_ids_observe(x2, y2, z2, bounds_xyz=bounds_xyz, ndiv=ndiv)

    n_sub = ndiv ** 3
    idxs_1: List[np.ndarray] = [np.where(sub_ids_1 == i)[0] for i in range(n_sub)]
    idxs_2: List[np.ndarray] = [np.where(sub_ids_2 == i)[0] for i in range(n_sub)]

    n_points_1_sub = np.array([idx.size for idx in idxs_1], dtype=np.int64)
    n_points_2_sub = np.array([idx.size for idx in idxs_2], dtype=np.int64)

    n_rp = len(rp_bins) - 1
    n_flat = n_rp * n_pi

    pair_counts = np.zeros((n_sub, n_sub, n_flat), dtype=np.float64)

    for i in range(n_sub):
        idx_i = idxs_1[i]
        if idx_i.size == 0:
            continue
        for j in range(n_sub):
            idx_j = idxs_2[j]
            if idx_j.size == 0:
                continue
            result = DDrppi(
                autocorr=0,
                nthreads=nthreads,
                pimax=pimax,
                binfile=rp_bins,
                X1=x1[idx_i],
                Y1=y1[idx_i],
                Z1=z1[idx_i],
                X2=x2[idx_j],
                Y2=y2[idx_j],
                Z2=z2[idx_j],
                periodic=False,
                output_rpavg=False,
                verbose=False,
            )
            pair_counts[i, j] = result["npairs"].astype(np.float64)

    row_sums = pair_counts.sum(axis=1)
    col_sums = pair_counts.sum(axis=0)
    total_counts = pair_counts.sum(axis=(0, 1))

    return ObserveRpPiCrossPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        row_sums=row_sums,
        col_sums=col_sums,
        n_subboxes=n_sub,
        n_points_1_total=x1.size,
        n_points_2_total=x2.size,
        n_points_1_subbox=n_points_1_sub,
        n_points_2_subbox=n_points_2_sub,
    )


def corrfunc_wp_observe_jackknife(
    sample_xyz: np.ndarray,
    random_xyz: np.ndarray,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    ndiv: int = 4,
    nthreads: int = 8,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
    estimator: str = "landy-szalay",
) -> Dict[str, Any]:
    """
    Jackknife wp(rp) for observed (non-periodic) samples.

    Note:
      - Only Landy-Szalay estimator is supported for observed-data mode.
      - Natural mode is intentionally not supported.
      - Corrfunc DDrppi has fixed pi-bin width of 1, so this wrapper requires dpi=1.
    """
    if estimator.lower() != "landy-szalay":
        raise ValueError("For observed data, only estimator='landy-szalay' is supported.")

    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    dpi, n_pi_fixed = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)
    pimax = float(n_pi_fixed)
    bounds_xyz = _parse_bounds(sample_xyz=sample_xyz, random_xyz=random_xyz, bounds=bounds)

    t0 = time.perf_counter()
    dd_precomp = precompute_observe_rppi_pair_counts(
        sample_xyz=sample_xyz,
        rp_bins=rp_bins,
        pimax=pimax,
        dpi=dpi,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    rr_precomp = precompute_observe_rppi_pair_counts(
        sample_xyz=random_xyz,
        rp_bins=rp_bins,
        pimax=pimax,
        dpi=dpi,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    dr_precomp = precompute_observe_rppi_cross_pair_counts(
        sample_xyz_1=sample_xyz,
        sample_xyz_2=random_xyz,
        rp_bins=rp_bins,
        pimax=pimax,
        dpi=dpi,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    t1 = time.perf_counter()

    if not (dd_precomp.n_subboxes == rr_precomp.n_subboxes == dr_precomp.n_subboxes):
        raise RuntimeError("subbox number mismatch between DD, RR and DR")

    n_sub = dd_precomp.n_subboxes

    xi_full = _xi_landy_szalay_from_counts(
        dd_counts=dd_precomp.total_counts,
        dr_counts=dr_precomp.total_counts,
        rr_counts=rr_precomp.total_counts,
        n_data=dd_precomp.n_points_total,
        n_rand=rr_precomp.n_points_total,
    )

    xi_jack = np.zeros((n_sub, dd_precomp.total_counts.size), dtype=np.float64)
    for k in range(n_sub):
        dd_loo = dd_precomp.total_counts - dd_precomp.involved_counts[k]
        rr_loo = rr_precomp.total_counts - rr_precomp.involved_counts[k]

        dr_involving_k = dr_precomp.row_sums[k] + dr_precomp.col_sums[k] - dr_precomp.pair_counts[k, k]
        dr_loo = dr_precomp.total_counts - dr_involving_k

        n_d_loo = dd_precomp.n_points_total - dd_precomp.n_points_subbox[k]
        n_r_loo = rr_precomp.n_points_total - rr_precomp.n_points_subbox[k]
        xi_jack[k] = _xi_landy_szalay_from_counts(
            dd_counts=dd_loo,
            dr_counts=dr_loo,
            rr_counts=rr_loo,
            n_data=n_d_loo,
            n_rand=n_r_loo,
        )

    xi_jack_mean = xi_jack.mean(axis=0)
    diff_xi = xi_jack - xi_jack_mean
    cov_xi = (n_sub - 1.0) / n_sub * (diff_xi.T @ diff_xi)

    wp_full = _wp_from_xi_rppi_flat(xi_full, dd_precomp.n_rp_bins, dd_precomp.n_pi_bins, dpi)
    wp_jack = np.array([
        _wp_from_xi_rppi_flat(xi_i, dd_precomp.n_rp_bins, dd_precomp.n_pi_bins, dpi)
        for xi_i in xi_jack
    ])
    wp_jack_mean = wp_jack.mean(axis=0)
    diff_wp = wp_jack - wp_jack_mean
    cov_wp = (n_sub - 1.0) / n_sub * (diff_wp.T @ diff_wp)
    wp_err = np.sqrt(np.diag(cov_wp))

    t2 = time.perf_counter()

    timing = {
        "pair_precompute_s": t1 - t0,
        "jackknife_s": t2 - t1,
        "total_s": t2 - t0,
    }

    return {
        "wp_mean": wp_full,
        "wp_full": wp_full,
        "wp_jack_mean": wp_jack_mean,
        "wp_err": wp_err,
        "cov_wp": cov_wp,
        "wp_jack": wp_jack,
        "xi_full_rppi": xi_full,
        "xi_jack_mean_rppi": xi_jack_mean,
        "cov_xi_rppi": cov_xi,
        "xi_jack_rppi": xi_jack,
        "estimator": "landy-szalay",
        "rr_mode": "loo_randoms",
        "dd_total": dd_precomp.total_counts.copy(),
        "rr_full": rr_precomp.total_counts.copy(),
        "dr_full": dr_precomp.total_counts.copy(),
        "n_points_subbox_data": dd_precomp.n_points_subbox.copy(),
        "n_points_subbox_rand": rr_precomp.n_points_subbox.copy(),
        "rp_bin_edges": rp_bins.copy(),
        "rp_bin_centers": np.sqrt(rp_bins[:-1] * rp_bins[1:]),
        "pi_edges": dd_precomp.pi_edges.copy(),
        "pimax": float(pimax),
        "dpi": float(dpi),
        "n_subboxes": n_sub,
        "bounds_xyz": bounds_xyz.copy(),
        "timing": timing,
    }
