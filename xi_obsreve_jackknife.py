import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Any

import numpy as np
from Corrfunc.theory.DD import DD


@dataclass
class ObserveSubboxPairCounts:
    pair_counts: Dict[Tuple[int, int], np.ndarray]
    total_counts: np.ndarray
    involved_counts: np.ndarray
    bin_edges: np.ndarray
    n_subboxes: int
    n_points_total: int
    n_points_subbox: np.ndarray


@dataclass
class ObserveSubboxCrossPairCounts:
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


def _parse_bounds(
    sample_xyz: np.ndarray,
    random_xyz: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Build bounds used for cubic subbox partition.
    If bounds is None, use combined min/max from data + randoms.
    """
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (3, 2):
            raise ValueError("bounds must have shape ((xmin,xmax),(ymin,ymax),(zmin,zmax))")
        return b

    all_xyz = np.vstack((np.asarray(sample_xyz), np.asarray(random_xyz)))
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    b = np.column_stack((mins, maxs))

    # avoid zero width in pathological axis
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
    """Assign points to regular ndiv^3 grid using explicit coordinate bounds."""
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


def precompute_observe_subbox_pair_counts(
    sample_xyz: np.ndarray,
    bin_edges: np.ndarray,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
) -> ObserveSubboxPairCounts:
    """Precompute all subbox-level pair counts DD(i,i), DD(i,j) for non-periodic data."""
    x, y, z = _split_xyz(sample_xyz)

    sub_ids = _assign_subbox_ids_observe(x, y, z, bounds_xyz=bounds_xyz, ndiv=ndiv)
    n_sub = ndiv ** 3
    indices: List[np.ndarray] = [np.where(sub_ids == i)[0] for i in range(n_sub)]
    n_points_subbox = np.array([idx.size for idx in indices], dtype=np.int64)

    nbin = len(bin_edges) - 1
    pair_counts: Dict[Tuple[int, int], np.ndarray] = {}
    total_counts = np.zeros(nbin, dtype=np.float64)
    involved_counts = np.zeros((n_sub, nbin), dtype=np.float64)

    for i, j in combinations_with_replacement(range(n_sub), 2):
        idx_i = indices[i]
        idx_j = indices[j]
        if idx_i.size == 0 or idx_j.size == 0:
            counts = np.zeros(nbin, dtype=np.float64)
        elif i == j:
            result = DD(
                autocorr=1,
                nthreads=nthreads,
                binfile=bin_edges,
                X1=x[idx_i],
                Y1=y[idx_i],
                Z1=z[idx_i],
                periodic=False,
                output_ravg=False,
                verbose=False,
            )
            counts = result["npairs"].astype(np.float64)
        else:
            result = DD(
                autocorr=0,
                nthreads=nthreads,
                binfile=bin_edges,
                X1=x[idx_i],
                Y1=y[idx_i],
                Z1=z[idx_i],
                X2=x[idx_j],
                Y2=y[idx_j],
                Z2=z[idx_j],
                periodic=False,
                output_ravg=False,
                verbose=False,
            )
            # autocorr=1 is ordered-pair convention, so cross terms need x2
            counts = 2.0 * result["npairs"].astype(np.float64)

        pair_counts[(i, j)] = counts
        total_counts += counts
        involved_counts[i] += counts
        if j != i:
            involved_counts[j] += counts

    return ObserveSubboxPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        involved_counts=involved_counts,
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        n_subboxes=n_sub,
        n_points_total=x.size,
        n_points_subbox=n_points_subbox,
    )


def precompute_observe_subbox_cross_pair_counts(
    sample_xyz_1: np.ndarray,
    sample_xyz_2: np.ndarray,
    bin_edges: np.ndarray,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
) -> ObserveSubboxCrossPairCounts:
    """Precompute all cross subbox-level pair counts DR(i,j) for non-periodic data."""
    x1, y1, z1 = _split_xyz(sample_xyz_1)
    x2, y2, z2 = _split_xyz(sample_xyz_2)

    sub_ids_1 = _assign_subbox_ids_observe(x1, y1, z1, bounds_xyz=bounds_xyz, ndiv=ndiv)
    sub_ids_2 = _assign_subbox_ids_observe(x2, y2, z2, bounds_xyz=bounds_xyz, ndiv=ndiv)

    n_sub = ndiv ** 3
    idxs_1: List[np.ndarray] = [np.where(sub_ids_1 == i)[0] for i in range(n_sub)]
    idxs_2: List[np.ndarray] = [np.where(sub_ids_2 == i)[0] for i in range(n_sub)]

    n_points_1_sub = np.array([idx.size for idx in idxs_1], dtype=np.int64)
    n_points_2_sub = np.array([idx.size for idx in idxs_2], dtype=np.int64)

    nbin = len(bin_edges) - 1
    pair_counts = np.zeros((n_sub, n_sub, nbin), dtype=np.float64)

    for i in range(n_sub):
        idx_i = idxs_1[i]
        if idx_i.size == 0:
            continue
        for j in range(n_sub):
            idx_j = idxs_2[j]
            if idx_j.size == 0:
                continue
            result = DD(
                autocorr=0,
                nthreads=nthreads,
                binfile=bin_edges,
                X1=x1[idx_i],
                Y1=y1[idx_i],
                Z1=z1[idx_i],
                X2=x2[idx_j],
                Y2=y2[idx_j],
                Z2=z2[idx_j],
                periodic=False,
                output_ravg=False,
                verbose=False,
            )
            pair_counts[i, j] = result["npairs"].astype(np.float64)

    row_sums = pair_counts.sum(axis=1)
    col_sums = pair_counts.sum(axis=0)
    total_counts = pair_counts.sum(axis=(0, 1))

    return ObserveSubboxCrossPairCounts(
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


def corrfunc_xi_obsreve_jackknife(
    sample_xyz: np.ndarray,
    random_xyz: np.ndarray,
    rbins: np.ndarray,
    ndiv: int = 4,
    nthreads: int = 8,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
    estimator: str = "landy-szalay",
) -> Dict[str, Any]:
    """
    Jackknife xi(r) for observed (non-periodic) samples.

    Note:
      - Only Landy-Szalay estimator is supported for observed-data mode.
      - Natural mode is intentionally not supported.
    """
    if estimator.lower() != "landy-szalay":
        raise ValueError("For observed data, only estimator='landy-szalay' is supported.")

    rbins = np.asarray(rbins, dtype=np.float64)
    bounds_xyz = _parse_bounds(sample_xyz=sample_xyz, random_xyz=random_xyz, bounds=bounds)

    t0 = time.perf_counter()
    dd_precomp = precompute_observe_subbox_pair_counts(
        sample_xyz=sample_xyz,
        bin_edges=rbins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    rr_precomp = precompute_observe_subbox_pair_counts(
        sample_xyz=random_xyz,
        bin_edges=rbins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    dr_precomp = precompute_observe_subbox_cross_pair_counts(
        sample_xyz_1=sample_xyz,
        sample_xyz_2=random_xyz,
        bin_edges=rbins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
    )
    t1 = time.perf_counter()

    if not (dd_precomp.n_subboxes == rr_precomp.n_subboxes == dr_precomp.n_subboxes):
        raise RuntimeError("subbox number mismatch between DD, RR and DR")

    n_sub = dd_precomp.n_subboxes
    nbin = len(rbins) - 1

    xi_full = _xi_landy_szalay_from_counts(
        dd_counts=dd_precomp.total_counts,
        dr_counts=dr_precomp.total_counts,
        rr_counts=rr_precomp.total_counts,
        n_data=dd_precomp.n_points_total,
        n_rand=rr_precomp.n_points_total,
    )

    xi_jack = np.zeros((n_sub, nbin), dtype=np.float64)
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
    diff = xi_jack - xi_jack_mean
    cov = (n_sub - 1.0) / n_sub * (diff.T @ diff)
    xi_err = np.sqrt(np.diag(cov))

    t2 = time.perf_counter()

    timing = {
        "pair_precompute_s": t1 - t0,
        "jackknife_s": t2 - t1,
        "total_s": t2 - t0,
    }

    return {
        "xi_mean": xi_full,
        "xi_full": xi_full,
        "xi_jack_mean": xi_jack_mean,
        "xi_err": xi_err,
        "cov": cov,
        "xi_jack": xi_jack,
        "estimator": "landy-szalay",
        "rr_mode": "loo_randoms",
        "dd_total": dd_precomp.total_counts.copy(),
        "rr_full": rr_precomp.total_counts.copy(),
        "dr_full": dr_precomp.total_counts.copy(),
        "n_points_subbox_data": dd_precomp.n_points_subbox.copy(),
        "n_points_subbox_rand": rr_precomp.n_points_subbox.copy(),
        "r_bin_edges": rbins.copy(),
        "r_bin_centers": np.sqrt(rbins[:-1] * rbins[1:]),
        "n_subboxes": n_sub,
        "bounds_xyz": bounds_xyz.copy(),
        "timing": timing,
    }
