from typing import Any, Dict, Optional, Tuple

import numpy as np
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.theory.wp import wp as corrfunc_wp_theory


def _split_xyz(sample_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(sample_xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sample_xyz must have shape (N, 3)")
    return arr[:, 0], arr[:, 1], arr[:, 2], arr


def _validate_dpi_and_pimax(dpi: float, pimax: float) -> Tuple[float, int]:
    if not np.isclose(float(dpi), 1.0):
        raise ValueError("Corrfunc.theory.DDrppi uses fixed pi-bin width 1. Use dpi=1.")
    if float(pimax) <= 0:
        raise ValueError("pimax must be > 0")
    n_pi = int(round(float(pimax)))
    if not np.isclose(float(pimax), float(n_pi)):
        raise ValueError("pimax must be integer-valued for DDrppi fixed pi-bin width")
    return 1.0, n_pi


def _generate_random_xyz(n_random: int, boxsize: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, float(boxsize), size=(int(n_random), 3)).astype(np.float64)


def _count_rppi(
    rp_bins: np.ndarray,
    pimax: float,
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
        return DDrppi(
            autocorr=1,
            nthreads=int(nthreads),
            pimax=float(pimax),
            binfile=rp_bins,
            X1=x1,
            Y1=y1,
            Z1=z1,
            periodic=True,
            boxsize=float(boxsize),
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)

    return DDrppi(
        autocorr=0,
        nthreads=int(nthreads),
        pimax=float(pimax),
        binfile=rp_bins,
        X1=x1,
        Y1=y1,
        Z1=z1,
        X2=x2,
        Y2=y2,
        Z2=z2,
        periodic=True,
        boxsize=float(boxsize),
        output_rpavg=False,
        verbose=False,
    )["npairs"].astype(np.float64)


def _analytic_rr_rppi_auto(rp_bins: np.ndarray, n_points: int, boxsize: float, n_pi: int) -> np.ndarray:
    dr2 = rp_bins[1:] ** 2 - rp_bins[:-1] ** 2
    shell_vol_per_pi = np.pi * dr2 * 2.0  # 2*dpi with dpi=1
    shell_vol = np.repeat(shell_vol_per_pi[:, None], n_pi, axis=1)
    rr = n_points * (n_points - 1.0) * shell_vol / (float(boxsize) ** 3)
    return rr.reshape(-1)


def _analytic_rr_rppi_cross(rp_bins: np.ndarray, n1: int, n2: int, boxsize: float, n_pi: int) -> np.ndarray:
    dr2 = rp_bins[1:] ** 2 - rp_bins[:-1] ** 2
    shell_vol_per_pi = np.pi * dr2 * 2.0
    shell_vol = np.repeat(shell_vol_per_pi[:, None], n_pi, axis=1)
    rr = n1 * n2 * shell_vol / (float(boxsize) ** 3)
    return rr.reshape(-1)


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
    dd_norm = d1d2 / (n1 * n2)
    d1r2_norm = d1r2 / (n1 * nr2)
    d2r1_norm = d2r1 / (n2 * nr1)
    rr_norm = r1r2 / (nr1 * nr2)
    out = np.zeros_like(d1d2, dtype=np.float64)
    valid = rr_norm > 0
    out[valid] = (dd_norm[valid] - d1r2_norm[valid] - d2r1_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return out


def _wp_from_xi_flat(xi_flat: np.ndarray, n_rp: int, n_pi: int, dpi: float) -> np.ndarray:
    xi2d = xi_flat.reshape(n_rp, n_pi)
    return 2.0 * np.sum(xi2d * dpi, axis=1)


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


def corrfunc_wp(
    sample_xyz: np.ndarray,
    rp_bins: np.ndarray,
    pimax: float,
    boxsize: float,
    dpi: float = 1.0,
    nthreads: int = 8,
    sample2_xyz: np.ndarray = None,
    do_auto: bool = True,
    do_cross: bool = False,
    estimator: str = "natural",
    random_xyz: np.ndarray = None,
    random2_xyz: np.ndarray = None,
    n_random: int = None,
    random_seed: int = 42,
    output_rpavg: bool = False,
) -> Dict[str, Any]:
    """
    Full-sample projected correlation wp(rp) in a periodic cubic box.

    Parameters
    ----------
    sample_xyz : ndarray of shape (N1, 3)
        Primary sample positions [x, y, z].
    rp_bins : 1D ndarray of shape (Nrp+1,)
        Projected separation bin edges.
    pimax : float
        Maximum line-of-sight separation used in rp-pi counting.
        Must be integer-valued for Corrfunc DDrppi fixed pi binning.
    boxsize : float
        Side length of the periodic cube.
    dpi : float, default=1.0
        Pi-bin width argument. Corrfunc DDrppi requires dpi=1 in this wrapper.
    nthreads : int, default=8
        Number of OpenMP threads passed to Corrfunc.
    sample2_xyz : ndarray of shape (N2, 3), optional
        Secondary sample. If None, single-sample mode is used.
    do_auto : bool, default=True
        In two-sample mode, compute auto projected correlations for sample1 and sample2.
    do_cross : bool, default=False
        In two-sample mode, compute cross projected correlation wp_12.
    estimator : {'natural', 'landy-szalay'}, default='natural'
        Estimator in rp-pi prior to projection:
        - 'natural': xi = DD/RR - 1
        - 'landy-szalay': LS form for auto and cross
    random_xyz : ndarray of shape (Nr1, 3), optional
        Random catalog for sample1 in LS mode.
    random2_xyz : ndarray of shape (Nr2, 3), optional
        Random catalog for sample2 in LS two-sample mode.
        If omitted, defaults to `random_xyz`.
    n_random : int, optional
        Number of auto-generated random points when randoms are omitted in LS mode.
    random_seed : int, default=42
        Seed for random auto-generation.
    output_rpavg : bool, default=False
        Passed to Corrfunc.theory.wp in single-sample natural fast path.

    Returns
    -------
    dict
        Single-sample natural mode:
        - 'wp', bins, 'pimax', 'dpi', 'mode', 'estimator', 'n_points'
        - optionally 'rpavg' (Corrfunc dependent)

        Single-sample LS mode:
        - 'wp', 'xi_rppi', 'dd_rppi', 'dr_rppi', 'rr_rppi', random metadata, bins

        Two-sample mode:
        - always: metadata and bin descriptors
        - auto terms (if requested): 'wp_11', 'wp_22'
          (+ xi/DD/DR/RR components for LS)
        - cross term (if requested): 'wp_12'
          (+ xi/DD/DR/RR components in rp-pi as available)

    Notes
    -----
    - In single-sample natural mode, this wrapper directly calls Corrfunc.theory.wp.
    - Projection convention is wp = 2 * sum_{pi>0} xi(rp,pi) * dpi.
    - In two-sample mode, at least one of `do_auto` or `do_cross` must be True.
    """
    estimator = str(estimator).lower()
    if estimator not in ("natural", "landy-szalay"):
        raise ValueError("estimator must be 'natural' or 'landy-szalay'")

    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    if rp_bins.ndim != 1 or rp_bins.size < 2:
        raise ValueError("rp_bins must be a 1D array with at least 2 edges")

    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)
    pimax = float(n_pi)

    x1, y1, z1, arr1 = _split_xyz(sample_xyz)
    rp_cent = np.sqrt(rp_bins[:-1] * rp_bins[1:])

    # single-sample fast natural path via Corrfunc.theory.wp
    if sample2_xyz is None and estimator == "natural":
        res = corrfunc_wp_theory(
            boxsize=float(boxsize),
            pimax=float(pimax),
            nthreads=int(nthreads),
            binfile=rp_bins,
            X=x1,
            Y=y1,
            Z=z1,
            output_rpavg=bool(output_rpavg),
            verbose=False,
        )
        out = {
            "wp": np.asarray(res["wp"], dtype=np.float64),
            "rp_bin_edges": rp_bins.copy(),
            "rp_bin_centers": rp_cent,
            "pimax": float(pimax),
            "dpi": float(dpi),
            "mode": "single",
            "estimator": "natural",
            "n_points": int(arr1.shape[0]),
        }
        if "rpavg" in res.dtype.names:
            out["rpavg"] = np.asarray(res["rpavg"], dtype=np.float64)
        return out

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
        dd = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1)
        dr = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1, xr, yr, zr)
        rr = _count_rppi(rp_bins, pimax, nthreads, boxsize, xr, yr, zr)
        xi = _xi_ls_auto(dd, dr, rr, arr1.shape[0], rand1.shape[0])
        wp = _wp_from_xi_flat(xi, rp_bins.size - 1, n_pi, dpi)
        return {
            "mode": "single",
            "estimator": "landy-szalay",
            "wp": wp,
            "xi_rppi": xi,
            "dd_rppi": dd,
            "dr_rppi": dr,
            "rr_rppi": rr,
            "rp_bin_edges": rp_bins.copy(),
            "rp_bin_centers": rp_cent,
            "pi_edges": np.arange(0.0, float(n_pi) + 1.0, 1.0),
            "pimax": float(pimax),
            "dpi": float(dpi),
            "n_points": int(arr1.shape[0]),
            "n_random": int(rand1.shape[0]),
            "random_auto_generated": bool(random_auto_generated),
        }

    x2, y2, z2, arr2 = _split_xyz(sample2)
    out: Dict[str, Any] = {
        "mode": "two_sample",
        "estimator": estimator,
        "do_auto": bool(do_auto),
        "do_cross": bool(do_cross),
        "rp_bin_edges": rp_bins.copy(),
        "rp_bin_centers": rp_cent,
        "pi_edges": np.arange(0.0, float(n_pi) + 1.0, 1.0),
        "pimax": float(pimax),
        "dpi": float(dpi),
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
            res1 = corrfunc_wp_theory(
                boxsize=float(boxsize),
                pimax=float(pimax),
                nthreads=int(nthreads),
                binfile=rp_bins,
                X=x1,
                Y=y1,
                Z=z1,
                output_rpavg=False,
                verbose=False,
            )
            res2 = corrfunc_wp_theory(
                boxsize=float(boxsize),
                pimax=float(pimax),
                nthreads=int(nthreads),
                binfile=rp_bins,
                X=x2,
                Y=y2,
                Z=z2,
                output_rpavg=False,
                verbose=False,
            )
            out["wp_11"] = np.asarray(res1["wp"], dtype=np.float64)
            out["wp_22"] = np.asarray(res2["wp"], dtype=np.float64)
        else:
            xr1, yr1, zr1, _ = _split_xyz(rand1)
            xr2, yr2, zr2, _ = _split_xyz(rand2)

            dd11 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1)
            dr11 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1, xr1, yr1, zr1)
            rr11 = _count_rppi(rp_bins, pimax, nthreads, boxsize, xr1, yr1, zr1)
            xi11 = _xi_ls_auto(dd11, dr11, rr11, arr1.shape[0], rand1.shape[0])
            out["wp_11"] = _wp_from_xi_flat(xi11, rp_bins.size - 1, n_pi, dpi)
            out["xi_11_rppi"] = xi11
            out["dd_11_rppi"] = dd11
            out["dr_11_rppi"] = dr11
            out["rr_11_rppi"] = rr11

            dd22 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x2, y2, z2)
            dr22 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x2, y2, z2, xr2, yr2, zr2)
            rr22 = _count_rppi(rp_bins, pimax, nthreads, boxsize, xr2, yr2, zr2)
            xi22 = _xi_ls_auto(dd22, dr22, rr22, arr2.shape[0], rand2.shape[0])
            out["wp_22"] = _wp_from_xi_flat(xi22, rp_bins.size - 1, n_pi, dpi)
            out["xi_22_rppi"] = xi22
            out["dd_22_rppi"] = dd22
            out["dr_22_rppi"] = dr22
            out["rr_22_rppi"] = rr22

    if do_cross:
        dd12 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1, x2, y2, z2)
        out["dd_12_rppi"] = dd12

        if estimator == "natural":
            rr12 = _analytic_rr_rppi_cross(rp_bins, arr1.shape[0], arr2.shape[0], boxsize, n_pi)
            xi12 = _xi_natural(dd12, rr12)
            out["rr_12_rppi"] = rr12
            out["xi_12_rppi"] = xi12
            out["wp_12"] = _wp_from_xi_flat(xi12, rp_bins.size - 1, n_pi, dpi)
        else:
            xr1, yr1, zr1, _ = _split_xyz(rand1)
            xr2, yr2, zr2, _ = _split_xyz(rand2)
            d1r2 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x1, y1, z1, xr2, yr2, zr2)
            d2r1 = _count_rppi(rp_bins, pimax, nthreads, boxsize, x2, y2, z2, xr1, yr1, zr1)
            r1r2 = _count_rppi(rp_bins, pimax, nthreads, boxsize, xr1, yr1, zr1, xr2, yr2, zr2)
            xi12 = _xi_ls_cross(
                d1d2=dd12,
                d1r2=d1r2,
                d2r1=d2r1,
                r1r2=r1r2,
                n1=arr1.shape[0],
                n2=arr2.shape[0],
                nr1=rand1.shape[0],
                nr2=rand2.shape[0],
            )
            out["d1r2_rppi"] = d1r2
            out["d2r1_rppi"] = d2r1
            out["r1r2_rppi"] = r1r2
            out["xi_12_rppi"] = xi12
            out["wp_12"] = _wp_from_xi_flat(xi12, rp_bins.size - 1, n_pi, dpi)

    return out


wp = corrfunc_wp
