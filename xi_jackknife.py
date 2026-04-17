import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Any

import numpy as np
from Corrfunc.theory.DD import DD


@dataclass
class SubboxPairCounts:
    pair_counts: Dict[Tuple[int, int], np.ndarray]
    total_counts: np.ndarray
    involved_counts: np.ndarray
    bin_edges: np.ndarray
    n_subboxes: int
    n_points_total: int
    n_points_subbox: np.ndarray


@dataclass
class SubboxCrossPairCounts:
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


def _assign_subbox_ids(x: np.ndarray, y: np.ndarray, z: np.ndarray, boxsize: float, ndiv: int) -> np.ndarray:
    """Assign each point to one subbox in an ndiv^3 regular grid."""
    cell = boxsize / ndiv
    ix = np.minimum((x / cell).astype(np.int32), ndiv - 1)
    iy = np.minimum((y / cell).astype(np.int32), ndiv - 1)
    iz = np.minimum((z / cell).astype(np.int32), ndiv - 1)
    return ix + ndiv * (iy + ndiv * iz)


def _analytic_rr_counts(bin_edges: np.ndarray, n_points: int, boxsize: float) -> np.ndarray:
    """Expected random-random unique pair counts for periodic cube and radial bins."""
    shell_vol = (4.0 * np.pi / 3.0) * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    volume = boxsize ** 3
    # Corrfunc DD(autocorr=1) counts ordered pairs in npairs, so no 1/2 factor here.
    rr = n_points * (n_points - 1) * shell_vol / volume
    return rr


def _xi_from_dd_rr(dd_counts: np.ndarray, rr_counts: np.ndarray) -> np.ndarray:
    """Compute xi = DD/RR - 1 with safe zero handling."""
    xi = np.zeros_like(dd_counts, dtype=np.float64)
    valid = rr_counts > 0
    xi[valid] = dd_counts[valid] / rr_counts[valid] - 1.0
    return xi


def _xi_natural_from_counts(
    dd_counts: np.ndarray,
    rr_counts: np.ndarray,
    n_data: int,
    n_rand: int = None,
) -> np.ndarray:
    """
    Natural estimator with pair-count normalization.

    If n_rand is None, assumes rr_counts already corresponds to n_data
    (e.g., analytic RR with the same N): xi = DD/RR - 1.
    Otherwise: xi = (DD/RR) * [NR(NR-1)/ND(ND-1)] - 1.
    """
    if n_data < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)
    if n_rand is None:
        return _xi_from_dd_rr(dd_counts, rr_counts)
    if n_rand < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)

    xi = np.zeros_like(dd_counts, dtype=np.float64)
    valid = rr_counts > 0
    norm = (n_rand * (n_rand - 1.0)) / (n_data * (n_data - 1.0))
    xi[valid] = norm * dd_counts[valid] / rr_counts[valid] - 1.0
    return xi


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


def precompute_subbox_pair_counts(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    boxsize: float,
    bin_edges: np.ndarray,
    ndiv: int = 4,
    nthreads: int = 8,
) -> SubboxPairCounts:
    """
    Precompute all subbox-level pair counts DD(i,i) and DD(i,j), i<=j.
    """
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same shape")

    sub_ids = _assign_subbox_ids(x, y, z, boxsize=boxsize, ndiv=ndiv)
    n_sub = ndiv ** 3
    indices: List[np.ndarray] = [np.where(sub_ids == i)[0] for i in range(n_sub)]
    n_points_subbox = np.array([idx.size for idx in indices], dtype=np.int64)

    nbins = len(bin_edges) - 1
    total_counts = np.zeros(nbins, dtype=np.float64)
    involved_counts = np.zeros((n_sub, nbins), dtype=np.float64)
    pair_counts: Dict[Tuple[int, int], np.ndarray] = {}

    for i, j in combinations_with_replacement(range(n_sub), 2):
        idx_i = indices[i]
        idx_j = indices[j]
        if idx_i.size == 0 or idx_j.size == 0:
            counts = np.zeros(nbins, dtype=np.float64)
        elif i == j:
            result = DD(
                autocorr=1,
                nthreads=nthreads,
                binfile=bin_edges,
                X1=x[idx_i],
                Y1=y[idx_i],
                Z1=z[idx_i],
                periodic=True,
                boxsize=boxsize,
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
                periodic=True,
                boxsize=boxsize,
                output_ravg=False,
                verbose=False,
            )
            # full autocorr=1 DD uses ordered pairs, so cross-subbox terms need x2
            # to account for both (i,j) and (j,i) contributions.
            counts = 2.0 * result["npairs"].astype(np.float64)

        pair_counts[(i, j)] = counts
        total_counts += counts
        involved_counts[i] += counts
        if j != i:
            involved_counts[j] += counts

    return SubboxPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        involved_counts=involved_counts,
        bin_edges=bin_edges,
        n_subboxes=n_sub,
        n_points_total=x.size,
        n_points_subbox=n_points_subbox,
    )


def precompute_subbox_cross_pair_counts(
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    boxsize: float,
    bin_edges: np.ndarray,
    ndiv: int = 4,
    nthreads: int = 8,
) -> SubboxCrossPairCounts:
    """
    Precompute all cross subbox-level pair counts DR(i,j), for all i,j.
    """
    if not (x1.shape == y1.shape == z1.shape):
        raise ValueError("x1, y1, z1 must have the same shape")
    if not (x2.shape == y2.shape == z2.shape):
        raise ValueError("x2, y2, z2 must have the same shape")

    sub_ids_1 = _assign_subbox_ids(x1, y1, z1, boxsize=boxsize, ndiv=ndiv)
    sub_ids_2 = _assign_subbox_ids(x2, y2, z2, boxsize=boxsize, ndiv=ndiv)
    n_sub = ndiv ** 3
    idxs_1: List[np.ndarray] = [np.where(sub_ids_1 == i)[0] for i in range(n_sub)]
    idxs_2: List[np.ndarray] = [np.where(sub_ids_2 == i)[0] for i in range(n_sub)]

    n_points_1_sub = np.array([idx.size for idx in idxs_1], dtype=np.int64)
    n_points_2_sub = np.array([idx.size for idx in idxs_2], dtype=np.int64)

    nbins = len(bin_edges) - 1
    pair_counts = np.zeros((n_sub, n_sub, nbins), dtype=np.float64)

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
                periodic=True,
                boxsize=boxsize,
                output_ravg=False,
                verbose=False,
            )
            pair_counts[i, j] = result["npairs"].astype(np.float64)

    row_sums = pair_counts.sum(axis=1)
    col_sums = pair_counts.sum(axis=0)
    total_counts = pair_counts.sum(axis=(0, 1))

    return SubboxCrossPairCounts(
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


def jackknife_loo_counts(precomp: SubboxPairCounts, leave_out_subbox: int) -> np.ndarray:
    """Construct leave-one-out DD by subtraction: DD_all - DD_involving_k."""
    if leave_out_subbox < 0 or leave_out_subbox >= precomp.n_subboxes:
        raise ValueError("leave_out_subbox is out of range")
    return precomp.total_counts - precomp.involved_counts[leave_out_subbox]


def jackknife_xi_and_error(
    precomp: SubboxPairCounts, boxsize: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute xi for each leave-one-out sample and JK error using analytic RR.
    """
    n_sub = precomp.n_subboxes
    nbins = len(precomp.bin_edges) - 1
    xis = np.zeros((n_sub, nbins), dtype=np.float64)

    for k in range(n_sub):
        dd_loo = jackknife_loo_counts(precomp, k)
        n_loo = precomp.n_points_total - precomp.n_points_subbox[k]
        rr_loo = _analytic_rr_counts(precomp.bin_edges, n_loo, boxsize)
        xis[k] = _xi_from_dd_rr(dd_loo, rr_loo)

    xi_mean = xis.mean(axis=0)
    diff = xis - xi_mean
    cov = (n_sub - 1.0) / n_sub * (diff.T @ diff)
    err = np.sqrt(np.diag(cov))
    return xi_mean, err, xis, cov


def corrfunc_xi_jackknife(
    sample_xyz: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    ndiv: int = 4,
    nthreads: int = 8,
    random_xyz: np.ndarray = None,
    estimator: str = "natural",
    natural_rr_mode: str = "analytic",
) -> Dict[str, Any]:
    """
    High-level callable API.

    Parameters
    ----------
    sample_xyz : array-like, shape (N, 3)
        Input points.
    rbins : array-like, shape (nbins+1,)
        Radial bin edges.
    boxsize : float
        Periodic box size.
    ndiv : int
        Number of subboxes per dimension.
    nthreads : int
        Corrfunc thread count.
    random_xyz : array-like, shape (Nr, 3), optional
        Random points; required for `estimator='landy-szalay'`.
    estimator : {'natural', 'landy-szalay'}
        Correlation-function estimator for full xi and each LOO xi.
    natural_rr_mode : {'analytic', 'random'}
        Only used when estimator='natural'.
        'analytic': fast RR approximation for each LOO.
        'random': use LOO random catalogs for RR (geometry-matched, slower, more accurate at large scales).

    Returns
    -------
    dict
        `xi_mean` is set to full-sample xi (recommended central value).
        `xi_jack_mean` is the mean across leave-one-out xi (diagnostic only).
        JK is used to estimate covariance/error.
    """
    rbins = np.asarray(rbins, dtype=np.float64)
    estimator = estimator.lower()
    natural_rr_mode = natural_rr_mode.lower()
    if estimator not in ("natural", "landy-szalay"):
        raise ValueError("estimator must be 'natural' or 'landy-szalay'")
    if natural_rr_mode not in ("analytic", "random"):
        raise ValueError("natural_rr_mode must be 'analytic' or 'random'")

    x, y, z = _split_xyz(sample_xyz)

    t0 = time.perf_counter()
    precomp = precompute_subbox_pair_counts(
        x=x,
        y=y,
        z=z,
        boxsize=boxsize,
        bin_edges=rbins,
        ndiv=ndiv,
        nthreads=nthreads,
    )
    t1 = time.perf_counter()

    if estimator == "natural":
        if natural_rr_mode == "analytic":
            xi_jack_mean, xi_err, xi_jack, cov = jackknife_xi_and_error(precomp, boxsize=boxsize)
            rr_full = _analytic_rr_counts(rbins, precomp.n_points_total, boxsize)
            xi_full = _xi_natural_from_counts(
                dd_counts=precomp.total_counts,
                rr_counts=rr_full,
                n_data=precomp.n_points_total,
                n_rand=None,
            )
            rr_mode = "analytic"
        else:
            if random_xyz is None:
                raise ValueError("random_xyz is required when estimator='natural' and natural_rr_mode='random'")
            xr, yr, zr = _split_xyz(random_xyz)
            rr_precomp = precompute_subbox_pair_counts(
                x=xr,
                y=yr,
                z=zr,
                boxsize=boxsize,
                bin_edges=rbins,
                ndiv=ndiv,
                nthreads=nthreads,
            )
            n_sub = precomp.n_subboxes
            nbins = len(rbins) - 1
            xi_jack = np.zeros((n_sub, nbins), dtype=np.float64)
            for k in range(n_sub):
                dd_loo = precomp.total_counts - precomp.involved_counts[k]
                rr_loo = rr_precomp.total_counts - rr_precomp.involved_counts[k]
                n_d_loo = precomp.n_points_total - precomp.n_points_subbox[k]
                n_r_loo = rr_precomp.n_points_total - rr_precomp.n_points_subbox[k]
                xi_jack[k] = _xi_natural_from_counts(
                    dd_counts=dd_loo,
                    rr_counts=rr_loo,
                    n_data=n_d_loo,
                    n_rand=n_r_loo,
                )

            xi_jack_mean = xi_jack.mean(axis=0)
            diff = xi_jack - xi_jack_mean
            cov = (n_sub - 1.0) / n_sub * (diff.T @ diff)
            xi_err = np.sqrt(np.diag(cov))
            rr_full = rr_precomp.total_counts.copy()
            xi_full = _xi_natural_from_counts(
                dd_counts=precomp.total_counts,
                rr_counts=rr_full,
                n_data=precomp.n_points_total,
                n_rand=rr_precomp.n_points_total,
            )
            rr_mode = "loo_randoms"
        dr_full = None
    else:
        if random_xyz is None:
            raise ValueError("random_xyz is required when estimator='landy-szalay'")
        xr, yr, zr = _split_xyz(random_xyz)
        rr_precomp = precompute_subbox_pair_counts(
            x=xr,
            y=yr,
            z=zr,
            boxsize=boxsize,
            bin_edges=rbins,
            ndiv=ndiv,
            nthreads=nthreads,
        )
        dr_precomp = precompute_subbox_cross_pair_counts(
            x1=x,
            y1=y,
            z1=z,
            x2=xr,
            y2=yr,
            z2=zr,
            boxsize=boxsize,
            bin_edges=rbins,
            ndiv=ndiv,
            nthreads=nthreads,
        )

        if rr_precomp.n_subboxes != precomp.n_subboxes or dr_precomp.n_subboxes != precomp.n_subboxes:
            raise RuntimeError("subbox number mismatch between DD, RR, and DR precompute")

        n_sub = precomp.n_subboxes
        nbins = len(rbins) - 1
        xi_jack = np.zeros((n_sub, nbins), dtype=np.float64)
        for k in range(n_sub):
            dd_loo = precomp.total_counts - precomp.involved_counts[k]
            rr_loo = rr_precomp.total_counts - rr_precomp.involved_counts[k]
            dr_involving_k = dr_precomp.row_sums[k] + dr_precomp.col_sums[k] - dr_precomp.pair_counts[k, k]
            dr_loo = dr_precomp.total_counts - dr_involving_k

            n_d_loo = precomp.n_points_total - precomp.n_points_subbox[k]
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
        rr_full = rr_precomp.total_counts.copy()
        dr_full = dr_precomp.total_counts.copy()
        xi_full = _xi_landy_szalay_from_counts(
            dd_counts=precomp.total_counts,
            dr_counts=dr_full,
            rr_counts=rr_full,
            n_data=precomp.n_points_total,
            n_rand=rr_precomp.n_points_total,
        )
        rr_mode = "loo_randoms"
    t2 = time.perf_counter()

    rcent = np.sqrt(rbins[:-1] * rbins[1:])
    timing = {
        "pair_precompute_s": t1 - t0,
        "jackknife_s": t2 - t1,
        "total_s": t2 - t0,
    }

    return {
        # Keep `xi_mean` as the recommended central estimate (full-sample xi).
        "xi_mean": xi_full,
        "xi_full": xi_full,
        "xi_jack_mean": xi_jack_mean,
        "xi_err": xi_err,
        "cov": cov,
        "xi_jack": xi_jack,
        "estimator": estimator,
        "rr_mode": rr_mode,
        "dd_total": precomp.total_counts.copy(),
        "rr_full": rr_full,
        "dr_full": dr_full,
        "n_points_subbox": precomp.n_points_subbox.copy(),
        "r_bin_edges": rbins.copy(),
        "r_bin_centers": rcent,
        "n_subboxes": precomp.n_subboxes,
        "timing": timing,
    }


def benchmark_random_points(
    n_points: int = 50000,
    boxsize: float = 1000.0,
    ndiv: int = 4,
    nthreads: int = 8,
    rmin: float = 0.5,
    rmax: float = 50.0,
    nbins: int = 20,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, boxsize, n_points).astype(np.float64)
    y = rng.uniform(0.0, boxsize, n_points).astype(np.float64)
    z = rng.uniform(0.0, boxsize, n_points).astype(np.float64)
    bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)

    sample_xyz = np.column_stack((x, y, z))
    result = corrfunc_xi_jackknife(
        sample_xyz=sample_xyz,
        rbins=bin_edges,
        boxsize=boxsize,
        ndiv=ndiv,
        nthreads=nthreads,
    )

    print("=== Corrfunc subbox jackknife benchmark ===")
    print(f"N points                : {n_points}")
    print(f"Boxsize                 : {boxsize}")
    print(f"Subbox grid             : {ndiv}^3 = {result['n_subboxes']}")
    print(f"Non-empty subboxes      : {(result['n_points_subbox'] > 0).sum()}")
    print(f"Radial bins             : {nbins} ({rmin} -> {rmax})")
    print(f"Pair precompute time    : {result['timing']['pair_precompute_s']:.3f} s")
    print(f"JK aggregate+xi time    : {result['timing']['jackknife_s']:.3f} s")
    print(f"Total time              : {result['timing']['total_s']:.3f} s")
    print("Example xi_full[:5]     :", np.array2string(result["xi_full"][:5], precision=4))
    print("Example xi_jack_mean[:5]:", np.array2string(result["xi_jack_mean"][:5], precision=4))
    print("Example xi_err[:5]      :", np.array2string(result["xi_err"][:5], precision=4))


if __name__ == "__main__":
    benchmark_random_points(
        n_points=50000,
        boxsize=1000.0,
        ndiv=4,
        nthreads=8,
        rmin=0.5,
        rmax=50.0,
        nbins=20,
        seed=42,
    )
