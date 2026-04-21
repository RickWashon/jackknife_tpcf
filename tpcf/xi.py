from typing import Any, Dict, Optional, Tuple

import numpy as np
from Corrfunc.theory.DD import DD
from Corrfunc.theory.xi import xi as corrfunc_xi_theory
from utils.weighted_dd import weighted_dd_1h2h_auto
from utils.dd import dd_auto


def _split_xyz(sample_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(sample_xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")
    return arr[:, 0], arr[:, 1], arr[:, 2], arr


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


def _count_dd_weighted_total_auto(
    rbins: np.ndarray,
    nthreads: int,
    boxsize: float,
    sample_xyz: np.ndarray,
    approx_cell_size: Optional[float],
    refine_factor: int,
    max_cells_per_dim: int,
    use_float32: bool,
) -> np.ndarray:
    # dd_total is independent of host labels; unique labels make 1h path trivial.
    n = int(sample_xyz.shape[0])
    host = np.arange(n, dtype=np.int64)
    res = weighted_dd_1h2h_auto(
        sample_xyz=sample_xyz,
        host_halo_id=host,
        rbins=rbins,
        boxsize=float(boxsize),
        nthreads=int(nthreads),
        approx_cell_size=approx_cell_size,
        refine_factor=int(refine_factor),
        max_cells_per_dim=int(max_cells_per_dim),
        use_float32=bool(use_float32),
    )
    return np.asarray(res["dd_total"], dtype=np.float64)

def _count_dd_plain_auto(
    rbins: np.ndarray,
    nthreads: int,
    boxsize: float,
    sample_xyz: np.ndarray,
    approx_cell_size: Optional[float],
    refine_factor: int,
    max_cells_per_dim: int,
    use_float32: bool,
) -> np.ndarray:
    res = dd_auto(
        sample_xyz=sample_xyz,
        rbins=rbins,
        boxsize=float(boxsize),
        nthreads=int(nthreads),
        approx_cell_size=approx_cell_size,
        refine_factor=int(refine_factor),
        max_cells_per_dim=int(max_cells_per_dim),
        use_float32=bool(use_float32),
    )
    return np.asarray(res["dd_counts"], dtype=np.float64)


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
    if n1 < 1 or n2 < 1 or nr1 < 1 or nr2 < 1:
        return np.zeros_like(d1d2, dtype=np.float64)
    dd_norm = d1d2 / (n1 * n2)
    d1r2_norm = d1r2 / (n1 * nr2)
    d2r1_norm = d2r1 / (n2 * nr1)
    rr_norm = r1r2 / (nr1 * nr2)
    out = np.zeros_like(d1d2, dtype=np.float64)
    valid = rr_norm > 0
    out[valid] = (dd_norm[valid] - d1r2_norm[valid] - d2r1_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return out


def _prepare_randoms(
    estimator: str,
    do_auto: bool,
    do_cross: bool,
    sample1: np.ndarray,
    sample2: Optional[np.ndarray],
    boxsize: float,
    random_xyz: Optional[np.ndarray],
    random2_xyz: Optional[np.ndarray],
    n_random: Optional[int],
    random_seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    need_random = estimator == "landy-szalay"
    if not need_random:
        return random_xyz, random2_xyz, False

    if random_xyz is None:
        if n_random is None:
            base = sample1.shape[0] if sample2 is None else max(sample1.shape[0], sample2.shape[0])
            n_random = base
        random_xyz = _generate_random_xyz(n_random=n_random, boxsize=boxsize, seed=random_seed)

    if sample2 is not None and (do_auto or do_cross):
        if random2_xyz is None:
            random2_xyz = np.asarray(random_xyz, dtype=np.float64)

    return np.asarray(random_xyz, dtype=np.float64), None if random2_xyz is None else np.asarray(random2_xyz, dtype=np.float64), True


def corrfunc_xi(
    sample_xyz: np.ndarray,
    rbins: np.ndarray,
    boxsize: float,
    nthreads: int = 8,
    sample2_xyz: np.ndarray = None,
    do_auto: bool = True,
    do_cross: bool = False,
    estimator: str = "natural",
    random_xyz: np.ndarray = None,
    random2_xyz: np.ndarray = None,
    n_random: int = None,
    random_seed: int = 42,
    output_ravg: bool = False,
    dd_backend: str = "corrfunc",
    weighted_approx_cell_size: Optional[float] = None,
    weighted_refine_factor: int = 2,
    weighted_max_cells_per_dim: int = 100,
    weighted_use_float32: bool = False,
) -> Dict[str, Any]:
    """
    Full-sample xi(r) in a periodic cubic box.

    Parameters
    ----------
    sample_xyz : ndarray of shape (N1, 3)
        Primary sample positions [x, y, z].
    rbins : 1D ndarray of shape (Nr+1,)
        Radial bin edges. Must be increasing and in the same length unit as coordinates.
    boxsize : float
        Side length of the periodic cube.
    nthreads : int, default=8
        Number of OpenMP threads passed to Corrfunc.
    sample2_xyz : ndarray of shape (N2, 3), optional
        Secondary sample. If None, single-sample mode is used.
    do_auto : bool, default=True
        In two-sample mode, compute auto correlations for sample1 and sample2.
    do_cross : bool, default=False
        In two-sample mode, compute cross correlation between sample1 and sample2.
    estimator : {'natural', 'landy-szalay'}, default='natural'
        Estimator applied consistently to requested outputs (auto/cross):
        - 'natural': DD/RR - 1
        - 'landy-szalay': (DD - 2DR + RR)/RR for auto,
          and (D1D2 - D1R2 - D2R1 + R1R2)/R1R2 for cross.
    random_xyz : ndarray of shape (Nr1, 3), optional
        Random catalog for sample1 in LS mode.
        If omitted and LS is requested, randoms are auto-generated uniformly in box.
    random2_xyz : ndarray of shape (Nr2, 3), optional
        Random catalog for sample2 in LS two-sample mode.
        If omitted, defaults to `random_xyz` (same random set used for both samples).
    n_random : int, optional
        Number of auto-generated random points (used only when randoms are omitted in LS mode).
        Default behavior: use sample size in single-sample mode, or max(N1, N2) in two-sample mode.
    random_seed : int, default=42
        Seed for random auto-generation.
    output_ravg : bool, default=False
        Passed to Corrfunc.theory.xi in the single-sample natural fast path.
    dd_backend : {'corrfunc', 'weighted', 'plain'}, default='corrfunc'
        Backend for single-sample natural DD counting:
        - 'corrfunc': use Corrfunc.theory.xi fast path
        - 'weighted': use local weighted_dd counter for DD_total, then analytic RR
    weighted_approx_cell_size : float, optional
        Cell size passed to weighted DD backend when `dd_backend='weighted'`.
    weighted_refine_factor : int, default=2
        Grid refine factor for weighted DD backend.
    weighted_max_cells_per_dim : int, default=100
        Max cells per axis for weighted DD backend.
    weighted_use_float32 : bool, default=False
        If True, evaluate distances in mixed float32 mode in weighted DD backend.

    Returns
    -------
    dict
        Single-sample natural mode:
        - 'xi', 'r_bin_edges', 'r_bin_centers', 'mode', 'estimator', 'n_points'
        - optionally 'ravg', 'dd_counts' (Corrfunc dependent)

        Single-sample LS mode:
        - 'xi', 'dd', 'dr', 'rr', 'n_random', 'random_auto_generated', plus bin metadata

        Two-sample mode:
        - always: 'mode', 'estimator', 'do_auto', 'do_cross', 'n1', 'n2', bins
        - auto terms (if requested): 'xi_11', 'xi_22' (+ DD/DR/RR components for LS)
        - cross term (if requested): 'xi_12'
          (+ 'dd_12' and RR/DR components depending on estimator)

    Notes
    -----
    - In single-sample natural mode, this wrapper directly calls Corrfunc.theory.xi.
    - In LS mode, random catalogs are required; this function can auto-generate them.
    - In two-sample mode, at least one of `do_auto` or `do_cross` must be True.
    """
    estimator = str(estimator).lower()
    if estimator not in ("natural", "landy-szalay"):
        raise ValueError("estimator must be 'natural' or 'landy-szalay'")
    dd_backend = str(dd_backend).lower()
    if dd_backend not in ("corrfunc", "weighted", "plain"):
        raise ValueError("dd_backend must be 'corrfunc', 'weighted', or 'plain'")

    rbins = np.asarray(rbins, dtype=np.float64)
    if rbins.ndim != 1 or rbins.size < 2:
        raise ValueError("rbins must be a 1D array with at least 2 edges")

    x1, y1, z1, arr1 = _split_xyz(sample_xyz)
    rcent = np.sqrt(rbins[:-1] * rbins[1:])

    # single-sample fast natural path via Corrfunc.theory.xi
    if sample2_xyz is None and estimator == "natural":
        if dd_backend == "corrfunc":
            res = corrfunc_xi_theory(
                boxsize=float(boxsize),
                nthreads=int(nthreads),
                binfile=rbins,
                X=x1,
                Y=y1,
                Z=z1,
                output_ravg=bool(output_ravg),
                verbose=False,
            )
            out = {
                "xi": np.asarray(res["xi"], dtype=np.float64),
                "r_bin_edges": rbins.copy(),
                "r_bin_centers": rcent,
                "mode": "single",
                "estimator": "natural",
                "n_points": int(arr1.shape[0]),
                "dd_backend": "corrfunc",
            }
            if "ravg" in res.dtype.names:
                out["ravg"] = np.asarray(res["ravg"], dtype=np.float64)
            if "npairs" in res.dtype.names:
                out["dd_counts"] = np.asarray(res["npairs"], dtype=np.float64)
            return out
        else:
            if dd_backend == "weighted":
                dd_counts = _count_dd_weighted_total_auto(
                    rbins=rbins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                    sample_xyz=arr1,
                    approx_cell_size=weighted_approx_cell_size,
                    refine_factor=weighted_refine_factor,
                    max_cells_per_dim=weighted_max_cells_per_dim,
                    use_float32=weighted_use_float32,
                )
            else:
                dd_counts = _count_dd_plain_auto(
                    rbins=rbins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                    sample_xyz=arr1,
                    approx_cell_size=weighted_approx_cell_size,
                    refine_factor=weighted_refine_factor,
                    max_cells_per_dim=weighted_max_cells_per_dim,
                    use_float32=weighted_use_float32,
                )
            rr = _analytic_rr_auto(rbins, arr1.shape[0], boxsize)
            return {
                "xi": _xi_natural(dd_counts, rr),
                "dd_counts": dd_counts,
                "rr": rr,
                "r_bin_edges": rbins.copy(),
                "r_bin_centers": rcent,
                "mode": "single",
                "estimator": "natural",
                "n_points": int(arr1.shape[0]),
                "dd_backend": dd_backend,
                "weighted_use_float32": bool(weighted_use_float32),
                "weighted_approx_cell_size": None if weighted_approx_cell_size is None else float(weighted_approx_cell_size),
                "weighted_refine_factor": int(weighted_refine_factor),
                "weighted_max_cells_per_dim": int(weighted_max_cells_per_dim),
            }

    sample2 = None
    if sample2_xyz is not None:
        _, _, _, sample2 = _split_xyz(sample2_xyz)
        if not (bool(do_auto) or bool(do_cross)):
            raise ValueError("At least one of do_auto/do_cross must be True when sample2_xyz is provided")

    random_xyz, random2_xyz, random_auto_generated = _prepare_randoms(
        estimator=estimator,
        do_auto=bool(do_auto),
        do_cross=bool(do_cross),
        sample1=arr1,
        sample2=sample2,
        boxsize=float(boxsize),
        random_xyz=random_xyz,
        random2_xyz=random2_xyz,
        n_random=n_random,
        random_seed=int(random_seed),
    )

    # single-sample LS path
    if sample2 is None:
        xr, yr, zr, rand1 = _split_xyz(random_xyz)
        dd = _count_dd(rbins, nthreads, boxsize, x1, y1, z1)
        dr = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, xr, yr, zr)
        rr = _count_dd(rbins, nthreads, boxsize, xr, yr, zr)
        xi = _xi_ls_auto(dd, dr, rr, arr1.shape[0], rand1.shape[0])
        return {
            "mode": "single",
            "estimator": "landy-szalay",
            "xi": xi,
            "dd": dd,
            "dr": dr,
            "rr": rr,
            "r_bin_edges": rbins.copy(),
            "r_bin_centers": rcent,
            "n_points": int(arr1.shape[0]),
            "n_random": int(rand1.shape[0]),
            "random_auto_generated": bool(random_auto_generated),
        }

    # two-sample mode
    x2, y2, z2, arr2 = _split_xyz(sample2)
    out: Dict[str, Any] = {
        "mode": "two_sample",
        "estimator": estimator,
        "do_auto": bool(do_auto),
        "do_cross": bool(do_cross),
        "r_bin_edges": rbins.copy(),
        "r_bin_centers": rcent,
        "n1": int(arr1.shape[0]),
        "n2": int(arr2.shape[0]),
        "random_auto_generated": bool(random_auto_generated),
    }

    rand1 = None
    rand2 = None
    if estimator == "landy-szalay":
        _, _, _, rand1 = _split_xyz(random_xyz)
        _, _, _, rand2 = _split_xyz(random2_xyz)
        out["n_random1"] = int(rand1.shape[0])
        out["n_random2"] = int(rand2.shape[0])

    if do_auto:
        if estimator == "natural":
            res1 = corrfunc_xi_theory(
                boxsize=float(boxsize),
                nthreads=int(nthreads),
                binfile=rbins,
                X=x1,
                Y=y1,
                Z=z1,
                output_ravg=False,
                verbose=False,
            )
            res2 = corrfunc_xi_theory(
                boxsize=float(boxsize),
                nthreads=int(nthreads),
                binfile=rbins,
                X=x2,
                Y=y2,
                Z=z2,
                output_ravg=False,
                verbose=False,
            )
            out["xi_11"] = np.asarray(res1["xi"], dtype=np.float64)
            out["xi_22"] = np.asarray(res2["xi"], dtype=np.float64)
        else:
            xr1, yr1, zr1, _ = _split_xyz(rand1)
            xr2, yr2, zr2, _ = _split_xyz(rand2)

            dd11 = _count_dd(rbins, nthreads, boxsize, x1, y1, z1)
            dr11 = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, xr1, yr1, zr1)
            rr11 = _count_dd(rbins, nthreads, boxsize, xr1, yr1, zr1)
            out["xi_11"] = _xi_ls_auto(dd11, dr11, rr11, arr1.shape[0], rand1.shape[0])
            out["dd_11"] = dd11
            out["dr_11"] = dr11
            out["rr_11"] = rr11

            dd22 = _count_dd(rbins, nthreads, boxsize, x2, y2, z2)
            dr22 = _count_dd(rbins, nthreads, boxsize, x2, y2, z2, xr2, yr2, zr2)
            rr22 = _count_dd(rbins, nthreads, boxsize, xr2, yr2, zr2)
            out["xi_22"] = _xi_ls_auto(dd22, dr22, rr22, arr2.shape[0], rand2.shape[0])
            out["dd_22"] = dd22
            out["dr_22"] = dr22
            out["rr_22"] = rr22

    if do_cross:
        dd12 = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, x2, y2, z2)
        out["dd_12"] = dd12

        if estimator == "natural":
            rr12 = _analytic_rr_cross(rbins, arr1.shape[0], arr2.shape[0], boxsize)
            out["rr_12"] = rr12
            out["xi_12"] = _xi_natural(dd12, rr12)
        else:
            xr1, yr1, zr1, _ = _split_xyz(rand1)
            xr2, yr2, zr2, _ = _split_xyz(rand2)
            d1r2 = _count_dd(rbins, nthreads, boxsize, x1, y1, z1, xr2, yr2, zr2)
            d2r1 = _count_dd(rbins, nthreads, boxsize, x2, y2, z2, xr1, yr1, zr1)
            r1r2 = _count_dd(rbins, nthreads, boxsize, xr1, yr1, zr1, xr2, yr2, zr2)
            out["d1r2"] = d1r2
            out["d2r1"] = d2r1
            out["r1r2"] = r1r2
            out["xi_12"] = _xi_ls_cross(
                d1d2=dd12,
                d1r2=d1r2,
                d2r1=d2r1,
                r1r2=r1r2,
                n1=arr1.shape[0],
                n2=arr2.shape[0],
                nr1=rand1.shape[0],
                nr2=rand2.shape[0],
            )

    return out


xi = corrfunc_xi
