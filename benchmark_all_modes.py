import argparse
import time
from typing import Dict, Any, Callable, List, Tuple

import numpy as np

from xi_jackknife import corrfunc_xi_jackknife
from wp_jackknife import corrfunc_wp_jackknife
from xi_obsreve_jackknife import corrfunc_xi_obsreve_jackknife
from wp_observe_jackknife import corrfunc_wp_observe_jackknife


def _build_data(n_points: int, boxsize: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, boxsize, size=(n_points, 3)).astype(np.float64)
    randoms = rng.uniform(0.0, boxsize, size=(n_points, 3)).astype(np.float64)
    return xyz, randoms


def _case_builders(
    xyz: np.ndarray,
    randoms: np.ndarray,
    rbins: np.ndarray,
    rp_bins: np.ndarray,
    boxsize: float,
    ndiv: int,
    nthreads: int,
    pimax: int,
    dpi: float,
) -> Dict[str, Callable[[], Dict[str, Any]]]:
    return {
        "xi_box_natural_analytic": lambda: corrfunc_xi_jackknife(
            sample_xyz=xyz,
            rbins=rbins,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            estimator="natural",
            natural_rr_mode="analytic",
        ),
        "xi_box_natural_random": lambda: corrfunc_xi_jackknife(
            sample_xyz=xyz,
            rbins=rbins,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            random_xyz=randoms,
            estimator="natural",
            natural_rr_mode="random",
        ),
        "xi_box_ls": lambda: corrfunc_xi_jackknife(
            sample_xyz=xyz,
            rbins=rbins,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            random_xyz=randoms,
            estimator="landy-szalay",
        ),
        "wp_box_natural_analytic": lambda: corrfunc_wp_jackknife(
            sample_xyz=xyz,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            random_xyz=randoms,
            estimator="natural",
            natural_rr_mode="analytic",
        ),
        "wp_box_natural_random": lambda: corrfunc_wp_jackknife(
            sample_xyz=xyz,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            random_xyz=randoms,
            estimator="natural",
            natural_rr_mode="random",
        ),
        "wp_box_ls": lambda: corrfunc_wp_jackknife(
            sample_xyz=xyz,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            boxsize=boxsize,
            ndiv=ndiv,
            nthreads=nthreads,
            random_xyz=randoms,
            estimator="landy-szalay",
        ),
        "xi_obs_ls": lambda: corrfunc_xi_obsreve_jackknife(
            sample_xyz=xyz,
            random_xyz=randoms,
            rbins=rbins,
            ndiv=ndiv,
            nthreads=nthreads,
            estimator="landy-szalay",
        ),
        "wp_obs_ls": lambda: corrfunc_wp_observe_jackknife(
            sample_xyz=xyz,
            random_xyz=randoms,
            rp_bins=rp_bins,
            pimax=pimax,
            dpi=dpi,
            ndiv=ndiv,
            nthreads=nthreads,
            estimator="landy-szalay",
        ),
    }


def _print_single(name: str, res: Dict[str, Any], wall: float) -> None:
    metric = "xi" if name.startswith("xi_") else "wp"
    full_key = f"{metric}_full"
    err_key = f"{metric}_err"
    arr = np.asarray(res[full_key])
    err = np.asarray(res[err_key])
    print(f"{name}: total_s={res['timing']['total_s']:.3f}, wall={wall:.3f}")
    print(f"{full_key}[:5]:", np.array2string(arr[:5], precision=4))
    print(f"{err_key}[:5]: ", np.array2string(err[:5], precision=4))


def _run_case(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = fn()
    wall = time.perf_counter() - t0
    metric = "xi" if name.startswith("xi_") else "wp"
    arr = np.asarray(res[f"{metric}_full"])
    err = np.asarray(res[f"{metric}_err"])
    print(
        f"{name}: total_s={res['timing']['total_s']:.3f}, wall={wall:.3f}, "
        f"mean={arr.mean():.5f}, mean|.|={np.mean(np.abs(arr)):.5f}, mean_err={err.mean():.5f}"
    )
    return res


def _compare_three(label: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, ea: np.ndarray, eb: np.ndarray, ec: np.ndarray) -> None:
    print(f"\n=== Comparisons: {label} ===")
    print("mean abs diff (analytic vs random):", float(np.mean(np.abs(a - b))))
    print("mean abs diff (analytic vs LS):    ", float(np.mean(np.abs(a - c))))
    print("mean abs diff (random vs LS):      ", float(np.mean(np.abs(b - c))))
    print("mean err analytic/random/LS:       ", float(np.mean(ea)), float(np.mean(eb)), float(np.mean(ec)))


def _run_all(cases: Dict[str, Callable[[], Dict[str, Any]]]) -> None:
    order: List[str] = [
        "xi_box_natural_analytic",
        "xi_box_natural_random",
        "xi_box_ls",
        "wp_box_natural_analytic",
        "wp_box_natural_random",
        "wp_box_ls",
        "xi_obs_ls",
        "wp_obs_ls",
    ]

    results: Dict[str, Dict[str, Any]] = {}
    print("=== Multi-option benchmark ===")
    for name in order:
        results[name] = _run_case(name, cases[name])

    _compare_three(
        "periodic xi",
        results["xi_box_natural_analytic"]["xi_full"],
        results["xi_box_natural_random"]["xi_full"],
        results["xi_box_ls"]["xi_full"],
        results["xi_box_natural_analytic"]["xi_err"],
        results["xi_box_natural_random"]["xi_err"],
        results["xi_box_ls"]["xi_err"],
    )

    _compare_three(
        "periodic wp",
        results["wp_box_natural_analytic"]["wp_full"],
        results["wp_box_natural_random"]["wp_full"],
        results["wp_box_ls"]["wp_full"],
        results["wp_box_natural_analytic"]["wp_err"],
        results["wp_box_natural_random"]["wp_err"],
        results["wp_box_ls"]["wp_err"],
    )

    print("\n=== Constraint checks (observe natural should fail) ===")
    try:
        cases_bad = cases.copy()
        _ = corrfunc_xi_obsreve_jackknife(
            sample_xyz=xyz_global,
            random_xyz=randoms_global,
            rbins=rbins_global,
            estimator="natural",
        )
        print("xi_obs natural: UNEXPECTED PASS")
    except Exception as e:
        print("xi_obs natural:", type(e).__name__, str(e)[:120])

    try:
        _ = corrfunc_wp_observe_jackknife(
            sample_xyz=xyz_global,
            random_xyz=randoms_global,
            rp_bins=rp_bins_global,
            pimax=pimax_global,
            dpi=dpi_global,
            estimator="natural",
        )
        print("wp_obs natural: UNEXPECTED PASS")
    except Exception as e:
        print("wp_obs natural:", type(e).__name__, str(e)[:120])


def _build_one_case(
    function_name: str,
    estimator: str,
    natural_rr_mode: str,
    xyz: np.ndarray,
    randoms: np.ndarray,
    rbins: np.ndarray,
    rp_bins: np.ndarray,
    boxsize: float,
    ndiv: int,
    nthreads: int,
    pimax: int,
    dpi: float,
) -> Tuple[str, Callable[[], Dict[str, Any]]]:
    estimator = estimator.lower()
    natural_rr_mode = natural_rr_mode.lower()

    if function_name == "xi_box":
        name = f"xi_box_{'ls' if estimator=='landy-szalay' else 'natural_' + natural_rr_mode}"
        if estimator == "landy-szalay":
            fn = lambda: corrfunc_xi_jackknife(
                sample_xyz=xyz, rbins=rbins, boxsize=boxsize, ndiv=ndiv, nthreads=nthreads,
                random_xyz=randoms, estimator="landy-szalay"
            )
        else:
            fn = lambda: corrfunc_xi_jackknife(
                sample_xyz=xyz, rbins=rbins, boxsize=boxsize, ndiv=ndiv, nthreads=nthreads,
                random_xyz=(randoms if natural_rr_mode == "random" else None),
                estimator="natural", natural_rr_mode=natural_rr_mode
            )
        return name, fn

    if function_name == "wp_box":
        name = f"wp_box_{'ls' if estimator=='landy-szalay' else 'natural_' + natural_rr_mode}"
        if estimator == "landy-szalay":
            fn = lambda: corrfunc_wp_jackknife(
                sample_xyz=xyz, rp_bins=rp_bins, pimax=pimax, dpi=dpi, boxsize=boxsize,
                ndiv=ndiv, nthreads=nthreads, random_xyz=randoms, estimator="landy-szalay"
            )
        else:
            fn = lambda: corrfunc_wp_jackknife(
                sample_xyz=xyz, rp_bins=rp_bins, pimax=pimax, dpi=dpi, boxsize=boxsize,
                ndiv=ndiv, nthreads=nthreads,
                random_xyz=(randoms if natural_rr_mode == "random" else randoms),
                estimator="natural", natural_rr_mode=natural_rr_mode
            )
        return name, fn

    if function_name == "xi_obs":
        if estimator != "landy-szalay":
            raise ValueError("xi_obs supports only estimator='landy-szalay'")
        return "xi_obs_ls", lambda: corrfunc_xi_obsreve_jackknife(
            sample_xyz=xyz, random_xyz=randoms, rbins=rbins, ndiv=ndiv, nthreads=nthreads, estimator="landy-szalay"
        )

    if function_name == "wp_obs":
        if estimator != "landy-szalay":
            raise ValueError("wp_obs supports only estimator='landy-szalay'")
        return "wp_obs_ls", lambda: corrfunc_wp_observe_jackknife(
            sample_xyz=xyz, random_xyz=randoms, rp_bins=rp_bins, pimax=pimax, dpi=dpi,
            ndiv=ndiv, nthreads=nthreads, estimator="landy-szalay"
        )

    raise ValueError("function must be one of: xi_box, wp_box, xi_obs, wp_obs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all/single jackknife modes.")
    parser.add_argument("--mode", choices=["all", "one"], default="all")
    parser.add_argument("--function", choices=["xi_box", "wp_box", "xi_obs", "wp_obs"], default="xi_box")
    parser.add_argument("--estimator", choices=["natural", "landy-szalay"], default="natural")
    parser.add_argument("--natural-rr-mode", choices=["analytic", "random"], default="analytic")
    parser.add_argument("--n-points", type=int, default=50000)
    parser.add_argument("--boxsize", type=float, default=1000.0)
    parser.add_argument("--ndiv", type=int, default=3)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--pimax", type=int, default=10)
    parser.add_argument("--dpi", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260417)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    xyz, randoms = _build_data(args.n_points, args.boxsize, args.seed)
    rbins = np.logspace(np.log10(0.5), np.log10(50.0), 21)
    rp_bins = np.logspace(np.log10(0.5), np.log10(30.0), 16)

    global xyz_global, randoms_global, rbins_global, rp_bins_global, pimax_global, dpi_global
    xyz_global = xyz
    randoms_global = randoms
    rbins_global = rbins
    rp_bins_global = rp_bins
    pimax_global = args.pimax
    dpi_global = args.dpi

    print("=== Random-data benchmark ===")
    print(f"N={args.n_points}, boxsize={args.boxsize}, ndiv={args.ndiv}, nthreads={args.nthreads}")

    cases = _case_builders(
        xyz=xyz,
        randoms=randoms,
        rbins=rbins,
        rp_bins=rp_bins,
        boxsize=args.boxsize,
        ndiv=args.ndiv,
        nthreads=args.nthreads,
        pimax=args.pimax,
        dpi=args.dpi,
    )

    if args.mode == "all":
        _run_all(cases)
    else:
        name, fn = _build_one_case(
            function_name=args.function,
            estimator=args.estimator,
            natural_rr_mode=args.natural_rr_mode,
            xyz=xyz,
            randoms=randoms,
            rbins=rbins,
            rp_bins=rp_bins,
            boxsize=args.boxsize,
            ndiv=args.ndiv,
            nthreads=args.nthreads,
            pimax=args.pimax,
            dpi=args.dpi,
        )
        t0 = time.perf_counter()
        res = fn()
        wall = time.perf_counter() - t0
        print("=== Single case ===")
        _print_single(name, res, wall)
