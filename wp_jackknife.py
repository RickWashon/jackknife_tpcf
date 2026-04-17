import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Any

import numpy as np
from Corrfunc.theory.DDrppi import DDrppi


@dataclass
class SubboxRpPiPairCounts:
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
class SubboxRpPiCrossPairCounts:
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


def _flatten_rppi_npairs(result: np.ndarray) -> np.ndarray:
    return result["npairs"].astype(np.float64)


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


def _analytic_rr_rppi_counts(
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    n_points: int,
    boxsize: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Expected random-random ordered pair counts in (rp, pi) bins for periodic cube."""
    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)
    pi_edges = np.arange(0.0, float(n_pi) + 1.0, 1.0, dtype=np.float64)

    dr2 = rp_bins[1:] ** 2 - rp_bins[:-1] ** 2
    # DDrppi output is flattened over (rp, pi), so analytic RR must have shape (n_rp, n_pi)
    # before flattening, not just (n_rp, 1).
    shell_vol_rp = np.pi * dr2 * (2.0 * dpi)
    shell_vol = np.repeat(shell_vol_rp[:, None], n_pi, axis=1)

    volume = boxsize ** 3
    rr = n_points * (n_points - 1) * shell_vol / volume
    return rr.reshape(-1), pi_edges, n_pi


def _xi_natural_from_counts(
    dd_counts: np.ndarray,
    rr_counts: np.ndarray,
    n_data: int,
    n_rand: int = None,
) -> np.ndarray:
    """Natural estimator with pair-count normalization."""
    if n_data < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)
    if n_rand is None:
        xi = np.zeros_like(dd_counts, dtype=np.float64)
        valid = rr_counts > 0
        xi[valid] = dd_counts[valid] / rr_counts[valid] - 1.0
        return xi
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


def _wp_from_xi_rppi_flat(xi_flat: np.ndarray, n_rp_bins: int, n_pi_bins: int, dpi: float) -> np.ndarray:
    xi2d = xi_flat.reshape(n_rp_bins, n_pi_bins)
    return 2.0 * np.sum(xi2d * dpi, axis=1)


def precompute_subbox_rppi_pair_counts(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    boxsize: float,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    ndiv: int = 4,
    nthreads: int = 8,
) -> SubboxRpPiPairCounts:
    """Precompute all subbox-level pair counts DD(i,i) and DD(i,j), i<=j in (rp,pi)."""
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same shape")

    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)

    sub_ids = _assign_subbox_ids(x, y, z, boxsize=boxsize, ndiv=ndiv)
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
                periodic=True,
                boxsize=boxsize,
                output_rpavg=False,
                verbose=False,
            )
            counts = _flatten_rppi_npairs(result)
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
                periodic=True,
                boxsize=boxsize,
                output_rpavg=False,
                verbose=False,
            )
            # full autocorr=1 DD uses ordered pairs, so cross-subbox terms need x2
            counts = 2.0 * _flatten_rppi_npairs(result)

        pair_counts[(i, j)] = counts
        total_counts += counts
        involved_counts[i] += counts
        if j != i:
            involved_counts[j] += counts

    pi_edges = np.arange(0.0, float(n_pi) + 1.0, 1.0, dtype=np.float64)

    return SubboxRpPiPairCounts(
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


def precompute_subbox_rppi_cross_pair_counts(
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    boxsize: float,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    ndiv: int = 4,
    nthreads: int = 8,
) -> SubboxRpPiCrossPairCounts:
    """Precompute all cross subbox-level pair counts DR(i,j), for all i,j in (rp,pi)."""
    if not (x1.shape == y1.shape == z1.shape):
        raise ValueError("x1, y1, z1 must have the same shape")
    if not (x2.shape == y2.shape == z2.shape):
        raise ValueError("x2, y2, z2 must have the same shape")

    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)

    sub_ids_1 = _assign_subbox_ids(x1, y1, z1, boxsize=boxsize, ndiv=ndiv)
    sub_ids_2 = _assign_subbox_ids(x2, y2, z2, boxsize=boxsize, ndiv=ndiv)
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
                periodic=True,
                boxsize=boxsize,
                output_rpavg=False,
                verbose=False,
            )
            pair_counts[i, j] = _flatten_rppi_npairs(result)

    row_sums = pair_counts.sum(axis=1)
    col_sums = pair_counts.sum(axis=0)
    total_counts = pair_counts.sum(axis=(0, 1))

    return SubboxRpPiCrossPairCounts(
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


def corrfunc_wp_jackknife(
    sample_xyz: np.ndarray,
    rp_bins: np.ndarray,
    pimax: float,
    dpi: float,
    boxsize: float,
    ndiv: int = 4,
    nthreads: int = 8,
    random_xyz: np.ndarray = None,
    estimator: str = "natural",
    natural_rr_mode: str = "analytic",
) -> Dict[str, Any]:
    """
    Jackknife wp(rp) in cubic periodic box.

    estimator:
      - 'natural'      : DD/RR - 1
      - 'landy-szalay' : (DD - 2DR + RR)/RR

    Notes on dpi:
      Corrfunc DDrppi has fixed pi-bin width of 1. This wrapper requires dpi=1.
    """
    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    estimator = estimator.lower()
    natural_rr_mode = natural_rr_mode.lower()
    dpi, n_pi_fixed = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)
    pimax = float(n_pi_fixed)

    if estimator not in ("natural", "landy-szalay"):
        raise ValueError("estimator must be 'natural' or 'landy-szalay'")
    if natural_rr_mode not in ("analytic", "random"):
        raise ValueError("natural_rr_mode must be 'analytic' or 'random'")

    x, y, z = _split_xyz(sample_xyz)

    t0 = time.perf_counter()
    dd_precomp = precompute_subbox_rppi_pair_counts(
        x=x,
        y=y,
        z=z,
        boxsize=boxsize,
        rp_bins=rp_bins,
        pimax=pimax,
        dpi=dpi,
        ndiv=ndiv,
        nthreads=nthreads,
    )
    t1 = time.perf_counter()

    n_sub = dd_precomp.n_subboxes
    n_flat = dd_precomp.total_counts.size

    if estimator == "natural":
        if natural_rr_mode == "analytic":
            rr_full, pi_edges, n_pi = _analytic_rr_rppi_counts(
                rp_bins=rp_bins,
                pimax=pimax,
                dpi=dpi,
                n_points=dd_precomp.n_points_total,
                boxsize=boxsize,
            )
            xi_full = _xi_natural_from_counts(
                dd_counts=dd_precomp.total_counts,
                rr_counts=rr_full,
                n_data=dd_precomp.n_points_total,
                n_rand=None,
            )

            xi_jack = np.zeros((n_sub, n_flat), dtype=np.float64)
            for k in range(n_sub):
                dd_loo = dd_precomp.total_counts - dd_precomp.involved_counts[k]
                n_d_loo = dd_precomp.n_points_total - dd_precomp.n_points_subbox[k]
                rr_loo, _, _ = _analytic_rr_rppi_counts(
                    rp_bins=rp_bins,
                    pimax=pimax,
                    dpi=dpi,
                    n_points=n_d_loo,
                    boxsize=boxsize,
                )
                xi_jack[k] = _xi_natural_from_counts(
                    dd_counts=dd_loo,
                    rr_counts=rr_loo,
                    n_data=n_d_loo,
                    n_rand=None,
                )

            rr_mode = "analytic"
            dr_full = None
        else:
            if random_xyz is None:
                raise ValueError("random_xyz is required when estimator='natural' and natural_rr_mode='random'")
            xr, yr, zr = _split_xyz(random_xyz)
            rr_precomp = precompute_subbox_rppi_pair_counts(
                x=xr,
                y=yr,
                z=zr,
                boxsize=boxsize,
                rp_bins=rp_bins,
                pimax=pimax,
                dpi=dpi,
                ndiv=ndiv,
                nthreads=nthreads,
            )
            rr_full = rr_precomp.total_counts.copy()
            xi_full = _xi_natural_from_counts(
                dd_counts=dd_precomp.total_counts,
                rr_counts=rr_full,
                n_data=dd_precomp.n_points_total,
                n_rand=rr_precomp.n_points_total,
            )

            xi_jack = np.zeros((n_sub, n_flat), dtype=np.float64)
            for k in range(n_sub):
                dd_loo = dd_precomp.total_counts - dd_precomp.involved_counts[k]
                rr_loo = rr_precomp.total_counts - rr_precomp.involved_counts[k]
                n_d_loo = dd_precomp.n_points_total - dd_precomp.n_points_subbox[k]
                n_r_loo = rr_precomp.n_points_total - rr_precomp.n_points_subbox[k]
                xi_jack[k] = _xi_natural_from_counts(
                    dd_counts=dd_loo,
                    rr_counts=rr_loo,
                    n_data=n_d_loo,
                    n_rand=n_r_loo,
                )

            rr_mode = "loo_randoms"
            dr_full = None
            pi_edges = dd_precomp.pi_edges
            n_pi = dd_precomp.n_pi_bins
    else:
        if random_xyz is None:
            raise ValueError("random_xyz is required when estimator='landy-szalay'")

        xr, yr, zr = _split_xyz(random_xyz)
        rr_precomp = precompute_subbox_rppi_pair_counts(
            x=xr,
            y=yr,
            z=zr,
            boxsize=boxsize,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            ndiv=ndiv,
            nthreads=nthreads,
        )
        dr_precomp = precompute_subbox_rppi_cross_pair_counts(
            x1=x,
            y1=y,
            z1=z,
            x2=xr,
            y2=yr,
            z2=zr,
            boxsize=boxsize,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            ndiv=ndiv,
            nthreads=nthreads,
        )

        rr_full = rr_precomp.total_counts.copy()
        dr_full = dr_precomp.total_counts.copy()
        xi_full = _xi_landy_szalay_from_counts(
            dd_counts=dd_precomp.total_counts,
            dr_counts=dr_full,
            rr_counts=rr_full,
            n_data=dd_precomp.n_points_total,
            n_rand=rr_precomp.n_points_total,
        )

        xi_jack = np.zeros((n_sub, n_flat), dtype=np.float64)
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

        rr_mode = "loo_randoms"
        pi_edges = dd_precomp.pi_edges
        n_pi = dd_precomp.n_pi_bins

    t2 = time.perf_counter()

    xi_jack_mean = xi_jack.mean(axis=0)
    diff_xi = xi_jack - xi_jack_mean
    cov_xi = (n_sub - 1.0) / n_sub * (diff_xi.T @ diff_xi)

    wp_full = _wp_from_xi_rppi_flat(xi_full, dd_precomp.n_rp_bins, n_pi, dpi)
    wp_jack = np.array([_wp_from_xi_rppi_flat(xi_i, dd_precomp.n_rp_bins, n_pi, dpi) for xi_i in xi_jack])
    wp_jack_mean = wp_jack.mean(axis=0)
    diff_wp = wp_jack - wp_jack_mean
    cov_wp = (n_sub - 1.0) / n_sub * (diff_wp.T @ diff_wp)
    wp_err = np.sqrt(np.diag(cov_wp))

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
        "estimator": estimator,
        "rr_mode": rr_mode,
        "dd_total": dd_precomp.total_counts.copy(),
        "rr_full": rr_full,
        "dr_full": dr_full,
        "n_points_subbox": dd_precomp.n_points_subbox.copy(),
        "rp_bin_edges": rp_bins.copy(),
        "rp_bin_centers": np.sqrt(rp_bins[:-1] * rp_bins[1:]),
        "pi_edges": pi_edges.copy(),
        "pimax": float(pimax),
        "dpi": float(dpi),
        "n_subboxes": n_sub,
        "timing": timing,
    }


# backward-compatible alias
wp_jackknife = corrfunc_wp_jackknife
