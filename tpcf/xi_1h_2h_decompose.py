from typing import Any, Dict, Optional, Tuple

import numpy as np
from Corrfunc.theory.DD import DD
try:
    from utils.weighted_dd import weighted_dd_1h2h_auto, weighted_dd_1h2h_cross
except ImportError:
    from utils.weighted_dd import weighted_dd_1h2h_auto, weighted_dd_1h2h_cross


def _split_xyz(sample_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(sample_xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")
    return arr[:, 0], arr[:, 1], arr[:, 2], arr


def _validate_host_ids(host_halo_id: np.ndarray, n_points: int, name: str) -> np.ndarray:
    ids = np.asarray(host_halo_id)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if ids.size != n_points:
        raise ValueError(f"{name} must have length equal to the number of sample points")
    return ids


def _group_indices(host_halo_id: np.ndarray) -> Dict[Any, np.ndarray]:
    unique_ids, inverse = np.unique(host_halo_id, return_inverse=True)
    return {halo_id: np.where(inverse == i)[0] for i, halo_id in enumerate(unique_ids)}


def _generate_random_xyz(n_random: int, boxsize: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, float(boxsize), size=(int(n_random), 3)).astype(np.float64)


def _count_dd(
    rbins: np.ndarray,
    nthreads: int,
    boxsize: float,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
    x2: Optional[np.ndarray] = None,
    y2: Optional[np.ndarray] = None,
    z2: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x2 is None:
        return DD(
            autocorr=1,
            nthreads=int(nthreads),
            binfile=rbins,
            X1=x1,
            Y1=y1,
            Z1=z1,
            periodic=True,
            boxsize=float(boxsize),
            output_ravg=False,
            verbose=False,
        )["npairs"].astype(np.float64)

    return DD(
        autocorr=0,
        nthreads=int(nthreads),
        binfile=rbins,
        X1=x1,
        Y1=y1,
        Z1=z1,
        X2=x2,
        Y2=y2,
        Z2=z2,
        periodic=True,
        boxsize=float(boxsize),
        output_ravg=False,
        verbose=False,
    )["npairs"].astype(np.float64)


def _analytic_rr_auto(rbins: np.ndarray, n_points: int, boxsize: float) -> np.ndarray:
    shell_vol = (4.0 * np.pi / 3.0) * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    return n_points * (n_points - 1.0) * shell_vol / (float(boxsize) ** 3)


def _analytic_rr_cross(rbins: np.ndarray, n1: int, n2: int, boxsize: float) -> np.ndarray:
    shell_vol = (4.0 * np.pi / 3.0) * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    return n1 * n2 * shell_vol / (float(boxsize) ** 3)


def _xi_natural(dd: np.ndarray, rr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dd, dtype=np.float64)
    valid = rr > 0
    out[valid] = dd[valid] / rr[valid] - 1.0
    return out


def _xi_ls_auto(dd: np.ndarray, dr: np.ndarray, rr: np.ndarray, nd: int, nr: int) -> np.ndarray:
    if nd < 2 or nr < 2:
        return np.zeros_like(dd, dtype=np.float64)
    dd_norm = dd / (nd * (nd - 1.0))
    dr_norm = dr / (nd * nr)
    rr_norm = rr / (nr * (nr - 1.0))
    out = np.zeros_like(dd, dtype=np.float64)
    valid = rr_norm > 0
    out[valid] = (dd_norm[valid] - 2.0 * dr_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return out


def _xi_ls_cross(
    d1d2: np.ndarray,
    d1r2: np.ndarray,
    d2r1: np.ndarray,
    r1r2: np.ndarray,
    n1: int,
    n2: int,
    nr1: int,
    nr2: int,
) -> np.ndarray:
    out = np.zeros_like(d1d2, dtype=np.float64)
    if min(n1, n2, nr1, nr2) < 1:
        return out

    dd_norm = d1d2 / (n1 * n2)
    d1r2_norm = d1r2 / (n1 * nr2)
    d2r1_norm = d2r1 / (n2 * nr1)
    rr_norm = r1r2 / (nr1 * nr2)
    valid = rr_norm > 0
    out[valid] = (dd_norm[valid] - d1r2_norm[valid] - d2r1_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return out


def _prepare_randoms(
    estimator: str,
    sample1: np.ndarray,
    sample2: Optional[np.ndarray],
    boxsize: float,
    random_xyz: Optional[np.ndarray],
    random2_xyz: Optional[np.ndarray],
    n_random: Optional[int],
    random_seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    if estimator == "natural":
        return random_xyz, random2_xyz, False

    if random_xyz is None:
        if n_random is None:
            base = sample1.shape[0] if sample2 is None else max(sample1.shape[0], sample2.shape[0])
            n_random = base
        random_xyz = _generate_random_xyz(n_random=n_random, boxsize=boxsize, seed=random_seed)

    if sample2 is not None and random2_xyz is None:
        random2_xyz = np.asarray(random_xyz, dtype=np.float64)

    return np.asarray(random_xyz, dtype=np.float64), None if random2_xyz is None else np.asarray(random2_xyz, dtype=np.float64), True


def _one_halo_auto_counts(
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
    sample_xyz: np.ndarray,
    host_halo_id: np.ndarray,
) -> np.ndarray:
    """
    Slow fallback implementation for 1-halo auto DD counts.

    This function loops over host-halo groups in Python and calls Corrfunc DD
    repeatedly for each halo. It is kept for validation/debug and compatibility.
    Production runs should prefer `use_c_weighted_dd=True`, which computes
    `dd_total/dd_1h/dd_2h` in one C/OpenMP pass.
    """
    x, y, z, _ = _split_xyz(sample_xyz)
    counts = np.zeros(len(rbins) - 1, dtype=np.float64)
    groups = _group_indices(host_halo_id)

    for idx in groups.values():
        if idx.size < 2:
            continue
        counts += _count_dd(rbins, nthreads, boxsize, x[idx], y[idx], z[idx])

    return counts


def _one_halo_cross_counts(
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
    sample1_xyz: np.ndarray,
    sample2_xyz: np.ndarray,
    host1: np.ndarray,
    host2: np.ndarray,
) -> np.ndarray:
    """
    Slow fallback implementation for 1-halo cross DD counts.

    This function intersects host IDs between two samples, then repeatedly calls
    Corrfunc DD on per-halo sub-samples. It is intended for correctness checks
    and fallback behavior, not for high-performance production workloads.
    """
    x1, y1, z1, _ = _split_xyz(sample1_xyz)
    x2, y2, z2, _ = _split_xyz(sample2_xyz)
    counts = np.zeros(len(rbins) - 1, dtype=np.float64)

    groups1 = _group_indices(host1)
    groups2 = _group_indices(host2)

    for halo_id in groups1.keys() & groups2.keys():
        idx1 = groups1[halo_id]
        idx2 = groups2[halo_id]
        if idx1.size == 0 or idx2.size == 0:
            continue
        counts += _count_dd(rbins, nthreads, boxsize, x1[idx1], y1[idx1], z1[idx1], x2[idx2], y2[idx2], z2[idx2])

    return counts


def _auto_decomp(
    sample_xyz: np.ndarray,
    host_halo_id: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
    estimator: str,
    random_xyz: Optional[np.ndarray],
    use_c_weighted_dd: bool,
    approx_cell_size: Optional[float],
    refine_factor: int,
    max_cells_per_dim: int,
    use_float32: bool,
) -> Dict[str, Any]:
    """
    Auto-correlation decomposition backend.

    Fast path:
    - `use_c_weighted_dd=True`: use shared C/OpenMP backend to compute
      `dd_total`, `dd_1h`, `dd_2h` in one pass.

    Slow fallback path:
    - `use_c_weighted_dd=False`: uses Corrfunc DD + Python halo loops
      (`_one_halo_auto_counts`), intended mainly for debug/validation.
    """
    x, y, z, arr = _split_xyz(sample_xyz)
    if use_c_weighted_dd:
        c_counts = weighted_dd_1h2h_auto(
            sample_xyz=arr,
            host_halo_id=host_halo_id,
            rbins=rbins,
            boxsize=boxsize,
            nthreads=nthreads,
            approx_cell_size=approx_cell_size,
            refine_factor=refine_factor,
            max_cells_per_dim=max_cells_per_dim,
            use_float32=use_float32,
        )
        dd_total = c_counts["dd_total"]
        dd_1h = c_counts["dd_1h"]
        dd_2h = c_counts["dd_2h"]
    else:
        dd_total = _count_dd(rbins, nthreads, boxsize, x, y, z)
        dd_1h = _one_halo_auto_counts(rbins, boxsize, nthreads, arr, host_halo_id)
        dd_2h = dd_total - dd_1h

    result: Dict[str, Any] = {
        "dd_total": dd_total,
        "dd_1h": dd_1h,
        "dd_2h": dd_2h,
    }

    if estimator == "natural":
        rr = _analytic_rr_auto(rbins, arr.shape[0], boxsize)
        result["rr"] = rr
        result["xi_overall"] = _xi_natural(dd_total, rr)
        result["xi_1h"] = _xi_natural(dd_1h, rr)
        result["xi_2h"] = _xi_natural(dd_2h, rr)
        return result

    xr, yr, zr, rand = _split_xyz(random_xyz)
    dr = _count_dd(rbins, nthreads, boxsize, x, y, z, xr, yr, zr)
    rr = _count_dd(rbins, nthreads, boxsize, xr, yr, zr)
    result["dr"] = dr
    result["rr"] = rr
    result["xi_overall"] = _xi_ls_auto(dd_total, dr, rr, arr.shape[0], rand.shape[0])
    result["xi_1h"] = _xi_ls_auto(dd_1h, dr, rr, arr.shape[0], rand.shape[0])
    result["xi_2h"] = _xi_ls_auto(dd_2h, dr, rr, arr.shape[0], rand.shape[0])
    return result


def _cross_decomp(
    sample1_xyz: np.ndarray,
    sample2_xyz: np.ndarray,
    host1: np.ndarray,
    host2: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int,
    estimator: str,
    random1_xyz: Optional[np.ndarray],
    random2_xyz: Optional[np.ndarray],
    use_c_weighted_dd: bool,
    approx_cell_size: Optional[float],
    refine_factor: int,
    max_cells_per_dim: int,
    use_float32: bool,
) -> Dict[str, Any]:
    """
    Cross-correlation decomposition backend.

    Current implementation uses Corrfunc DD for total cross counts plus Python
    halo-group loops for 1-halo cross counts (`_one_halo_cross_counts`).
    This is a correctness-oriented path and can be significantly slower than
    the auto fast path backed by the C/OpenMP weighted-DD kernel.
    """
    x1, y1, z1, arr1 = _split_xyz(sample1_xyz)
    x2, y2, z2, arr2 = _split_xyz(sample2_xyz)

    if use_c_weighted_dd:
        c_counts = weighted_dd_1h2h_cross(
            sample1_xyz=arr1,
            host1_halo_id=host1,
            sample2_xyz=arr2,
            host2_halo_id=host2,
            rbins=rbins,
            boxsize=boxsize,
            nthreads=nthreads,
            approx_cell_size=approx_cell_size,
            refine_factor=refine_factor,
            max_cells_per_dim=max_cells_per_dim,
            use_float32=use_float32,
        )
        dd_total = c_counts["dd_total"]
        dd_1h = c_counts["dd_1h"]
        dd_2h = c_counts["dd_2h"]
    else:
        dd_total = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, x2, y2, z2)
        dd_1h = _one_halo_cross_counts(rbins, boxsize, nthreads, arr1, arr2, host1, host2)
        dd_2h = dd_total - dd_1h

    result: Dict[str, Any] = {
        "dd_total": dd_total,
        "dd_1h": dd_1h,
        "dd_2h": dd_2h,
    }

    if estimator == "natural":
        rr = _analytic_rr_cross(rbins, arr1.shape[0], arr2.shape[0], boxsize)
        result["rr"] = rr
        result["xi_overall"] = _xi_natural(dd_total, rr)
        result["xi_1h"] = _xi_natural(dd_1h, rr)
        result["xi_2h"] = _xi_natural(dd_2h, rr)
        return result

    xr1, yr1, zr1, rand1 = _split_xyz(random1_xyz)
    xr2, yr2, zr2, rand2 = _split_xyz(random2_xyz)
    d1r2 = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, xr2, yr2, zr2)
    d2r1 = _count_dd(rbins, nthreads, boxsize, x2, y2, z2, xr1, yr1, zr1)
    r1r2 = _count_dd(rbins, nthreads, boxsize, xr1, yr1, zr1, xr2, yr2, zr2)
    result["d1r2"] = d1r2
    result["d2r1"] = d2r1
    result["r1r2"] = r1r2
    result["xi_overall"] = _xi_ls_cross(dd_total, d1r2, d2r1, r1r2, arr1.shape[0], arr2.shape[0], rand1.shape[0], rand2.shape[0])
    result["xi_1h"] = _xi_ls_cross(dd_1h, d1r2, d2r1, r1r2, arr1.shape[0], arr2.shape[0], rand1.shape[0], rand2.shape[0])
    result["xi_2h"] = _xi_ls_cross(dd_2h, d1r2, d2r1, r1r2, arr1.shape[0], arr2.shape[0], rand1.shape[0], rand2.shape[0])
    return result


def corrfunc_xi_1h_2h_decompose(
    sample_xyz: np.ndarray,
    host_halo_id: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int = 1,
    sample2_xyz: np.ndarray = None,
    sample2_host_halo_id: np.ndarray = None,
    do_auto: bool = True,
    do_cross: bool = False,
    estimator: str = "natural",
    random_xyz: np.ndarray = None,
    random2_xyz: np.ndarray = None,
    n_random: int = None,
    random_seed: int = 42,
    use_c_weighted_dd: bool = True,
    approx_cell_size: float = None,
    refine_factor: int = 2,
    max_cells_per_dim: int = 100,
    use_float32: bool = False,
) -> Dict[str, Any]:
    """
    Decompose periodic-box xi into one-halo and two-halo terms using host halo IDs.

    Parameters
    ----------
    sample_xyz : ndarray of shape (N1, 3)
        Primary sample positions [x, y, z].
    host_halo_id : ndarray of shape (N1,)
        Host halo ID for each point in `sample_xyz`.
    rbins : 1D ndarray
        Radial separation bin edges.
    boxsize : float
        Side length of the periodic cubic box.
    nthreads : int, default=1
        Corrfunc thread count.
    sample2_xyz : ndarray of shape (N2, 3), optional
        Secondary sample positions. If None, only the sample1 auto decomposition is computed.
    sample2_host_halo_id : ndarray of shape (N2,), optional
        Host halo IDs for `sample2_xyz`. Required when `sample2_xyz` is provided.
    do_auto : bool, default=True
        In two-sample mode, compute auto decompositions for sample1 and sample2.
    do_cross : bool, default=False
        In two-sample mode, compute cross decomposition between sample1 and sample2.
    estimator : {'natural', 'landy-szalay'}, default='natural'
        Estimator used for overall, 1-halo, and 2-halo xi outputs.
    random_xyz : ndarray of shape (Nr1, 3), optional
        Randoms for sample1 in LS mode. Auto-generated if omitted.
    random2_xyz : ndarray of shape (Nr2, 3), optional
        Randoms for sample2 in LS two-sample mode. Defaults to `random_xyz` if omitted.
    n_random : int, optional
        Number of auto-generated random points in LS mode.
    random_seed : int, default=42
        Seed used when generating random catalogs.
    use_c_weighted_dd : bool, default=True
        If True, auto decomposition uses the local C/OpenMP weighted DD backend
        to compute `dd_total`, `dd_1h`, and `dd_2h` in one pass. This affects
        single-sample mode and the two auto branches (`11` and `22`).
    approx_cell_size : float, optional
        Approximate cell size used by the C weighted-DD backend. If None, defaults
        to `boxsize/10`, similar to Halotools' default.
    refine_factor : int, default=2
        Multiplies cell divisions to refine the grid, similar in spirit to Corrfunc's
        bin refine factors.
    max_cells_per_dim : int, default=100
        Upper limit on the number of grid cells per dimension for the C backend.
    use_float32 : bool, default=False
        If True, evaluate pair distances in float32 precision inside the local C backend
        (experimental speed/precision tradeoff). Bin edges and outputs remain float64.

    Returns
    -------
    dict
        Single-sample mode returns:
        - `xi_overall`, `xi_1h`, `xi_2h`
        - `dd_total`, `dd_1h`, `dd_2h`
        - random-count terms required by the estimator

        Two-sample mode returns keyed outputs:
        - auto sample1: `xi_11_overall`, `xi_11_1h`, `xi_11_2h`
        - auto sample2: `xi_22_overall`, `xi_22_1h`, `xi_22_2h`
        - cross: `xi_12_overall`, `xi_12_1h`, `xi_12_2h`

    Notes
    -----
    - The decomposition is defined by whether a pair has the same host halo ID (`1h`)
      or different host halo IDs (`2h`).
    - In auto branches, `DD_total`, `DD_1h`, and `DD_2h` can be obtained in one pass
      with the local C/OpenMP backend (`use_c_weighted_dd=True`).
    - In cross branches, `DD_2h` is obtained by subtracting `DD_1h` from `DD_total`.
    """
    estimator = str(estimator).lower()
    if estimator not in ("natural", "landy-szalay"):
        raise ValueError("estimator must be 'natural' or 'landy-szalay'")

    rbins = np.asarray(rbins, dtype=np.float64)
    if rbins.ndim != 1 or rbins.size < 2:
        raise ValueError("rbins must be a 1D array with at least 2 edges")

    _, _, _, arr1 = _split_xyz(sample_xyz)
    host1 = _validate_host_ids(host_halo_id, arr1.shape[0], "host_halo_id")

    sample2 = None
    host2 = None
    if sample2_xyz is not None:
        _, _, _, sample2 = _split_xyz(sample2_xyz)
        if sample2_host_halo_id is None:
            raise ValueError("sample2_host_halo_id is required when sample2_xyz is provided")
        host2 = _validate_host_ids(sample2_host_halo_id, sample2.shape[0], "sample2_host_halo_id")
        if not (bool(do_auto) or bool(do_cross)):
            raise ValueError("At least one of do_auto/do_cross must be True when sample2_xyz is provided")

    random_xyz, random2_xyz, random_auto_generated = _prepare_randoms(
        estimator=estimator,
        sample1=arr1,
        sample2=sample2,
        boxsize=float(boxsize),
        random_xyz=random_xyz,
        random2_xyz=random2_xyz,
        n_random=n_random,
        random_seed=int(random_seed),
    )

    if sample2 is None:
        result = _auto_decomp(
            arr1,
            host1,
            rbins,
            boxsize,
            nthreads,
            estimator,
            random_xyz,
            use_c_weighted_dd,
            approx_cell_size,
            refine_factor,
            max_cells_per_dim,
            use_float32,
        )
        result.update(
            {
                "mode": "single",
                "estimator": estimator,
                "r_bin_edges": rbins.copy(),
                "r_bin_centers": np.sqrt(rbins[:-1] * rbins[1:]),
                "n_points": int(arr1.shape[0]),
                "random_auto_generated": bool(random_auto_generated),
                "use_c_weighted_dd": bool(use_c_weighted_dd),
                "approx_cell_size": None if approx_cell_size is None else float(approx_cell_size),
                "refine_factor": int(refine_factor),
                "max_cells_per_dim": int(max_cells_per_dim),
                "use_float32": bool(use_float32),
            }
        )
        return result

    out: Dict[str, Any] = {
        "mode": "two_sample",
        "estimator": estimator,
        "do_auto": bool(do_auto),
        "do_cross": bool(do_cross),
        "r_bin_edges": rbins.copy(),
        "r_bin_centers": np.sqrt(rbins[:-1] * rbins[1:]),
        "n1": int(arr1.shape[0]),
        "n2": int(sample2.shape[0]),
        "random_auto_generated": bool(random_auto_generated),
        "use_c_weighted_dd": bool(use_c_weighted_dd),
        "approx_cell_size": None if approx_cell_size is None else float(approx_cell_size),
        "refine_factor": int(refine_factor),
        "max_cells_per_dim": int(max_cells_per_dim),
        "use_float32": bool(use_float32),
    }

    if do_auto:
        auto11 = _auto_decomp(
            arr1,
            host1,
            rbins,
            boxsize,
            nthreads,
            estimator,
            random_xyz,
            use_c_weighted_dd,
            approx_cell_size,
            refine_factor,
            max_cells_per_dim,
            use_float32,
        )
        out["xi_11_overall"] = auto11["xi_overall"]
        out["xi_11_1h"] = auto11["xi_1h"]
        out["xi_11_2h"] = auto11["xi_2h"]
        out["dd_11_total"] = auto11["dd_total"]
        out["dd_11_1h"] = auto11["dd_1h"]
        out["dd_11_2h"] = auto11["dd_2h"]

        auto22 = _auto_decomp(
            sample2,
            host2,
            rbins,
            boxsize,
            nthreads,
            estimator,
            random2_xyz,
            use_c_weighted_dd,
            approx_cell_size,
            refine_factor,
            max_cells_per_dim,
            use_float32,
        )
        out["xi_22_overall"] = auto22["xi_overall"]
        out["xi_22_1h"] = auto22["xi_1h"]
        out["xi_22_2h"] = auto22["xi_2h"]
        out["dd_22_total"] = auto22["dd_total"]
        out["dd_22_1h"] = auto22["dd_1h"]
        out["dd_22_2h"] = auto22["dd_2h"]

    if do_cross:
        cross12 = _cross_decomp(
            arr1,
            sample2,
            host1,
            host2,
            rbins,
            boxsize,
            nthreads,
            estimator,
            random_xyz,
            random2_xyz,
            use_c_weighted_dd,
            approx_cell_size,
            refine_factor,
            max_cells_per_dim,
            use_float32,
        )
        out["xi_12_overall"] = cross12["xi_overall"]
        out["xi_12_1h"] = cross12["xi_1h"]
        out["xi_12_2h"] = cross12["xi_2h"]
        out["dd_12_total"] = cross12["dd_total"]
        out["dd_12_1h"] = cross12["dd_1h"]
        out["dd_12_2h"] = cross12["dd_2h"]

    return out


xi_1h_2h_decompose = corrfunc_xi_1h_2h_decompose
