import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Any
from numbers import Integral

import numpy as np
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.theory.DD import DD


@dataclass
class ObserveSubboxPairCounts:
    pair_counts: Dict[Tuple[int, int], np.ndarray]
    total_counts: np.ndarray
    involved_counts: np.ndarray
    s_bin_edges: np.ndarray
    n_subboxes: int
    n_points_total: int
    n_points_subbox: np.ndarray
    n_s_bins: int
    n_mu_bins: int


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


def _resolve_samples(
    coord_system: str,
    sample_radecz: np.ndarray = None,
    random_radecz: np.ndarray = None,
    sample_xyz: np.ndarray = None,
    random_xyz: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    coord = coord_system.lower()
    if coord not in ("radecz", "xyz"):
        raise ValueError("coord_system must be 'radecz' or 'xyz'")

    if coord == "radecz":
        sample = sample_radecz if sample_radecz is not None else sample_xyz
        randoms = random_radecz if random_radecz is not None else random_xyz
    else:
        sample = sample_xyz if sample_xyz is not None else sample_radecz
        randoms = random_xyz if random_xyz is not None else random_radecz

    if sample is None or randoms is None:
        raise ValueError("Missing sample/random inputs for the selected coord_system")

    sample = np.asarray(sample, dtype=np.float64)
    randoms = np.asarray(randoms, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[1] != 3:
        raise ValueError("sample input must have shape (N, 3)")
    if randoms.ndim != 2 or randoms.shape[1] != 3:
        raise ValueError("random input must have shape (N, 3)")
    return sample, randoms






def _coerce_h0_km_s_mpc(h0: Any) -> float:
    h0v = float(h0)
    if h0v <= 0:
        raise ValueError("H0 must be > 0")
    if h0v < 2.0:
        raise ValueError(
            "H0 looks like little-h (e.g. 0.677). "
            "Please pass H0 in km/s/Mpc (e.g. 67.7)."
        )
    return h0v


def _coerce_astropy_cosmology_obj(cosmo_like: Any) -> Any:
    """Build/validate an astropy cosmology object with .comoving_distance."""
    if cosmo_like is None:
        raise ValueError("astropy cosmology input is None")

    if hasattr(cosmo_like, "comoving_distance"):
        return cosmo_like

    if isinstance(cosmo_like, str):
        import astropy.cosmology as acosmo

        if not hasattr(acosmo, cosmo_like):
            raise ValueError(f"Unknown astropy cosmology preset: {cosmo_like}")
        obj = getattr(acosmo, cosmo_like)
        if not hasattr(obj, "comoving_distance"):
            raise ValueError(f"astropy preset {cosmo_like} is not a cosmology instance")
        return obj

    if isinstance(cosmo_like, dict):
        from astropy.cosmology import FlatLambdaCDM, LambdaCDM

        d = dict(cosmo_like)
        if "H0" not in d:
            if "h" in d:
                d["H0"] = 100.0 * float(d.pop("h"))
            else:
                raise ValueError("cosmology dict must contain H0 (or h) and Om0")

        if "Om0" not in d:
            raise ValueError("cosmology dict must contain Om0")

        H0 = _coerce_h0_km_s_mpc(d["H0"])
        Om0 = float(d["Om0"])

        if "Ode0" in d:
            return LambdaCDM(H0=H0, Om0=Om0, Ode0=float(d["Ode0"]))
        return FlatLambdaCDM(H0=H0, Om0=Om0)

    raise TypeError(
        "astropy cosmology input must be an astropy cosmology object, "
        "a preset string (e.g. 'Planck18'), or a dict {'H0':67.7, 'Om0':0.31}."
    )


def _resolve_radecz_cosmology_config(
    cosmology: Any,
    use_astropy_comoving: bool,
    astropy_cosmology: Any,
) -> Tuple[int, Any]:
    """
    Returns:
      corrfunc_cosmology_index: int in {1,2}
      astropy_cosmology_obj: cosmology object or None
    """
    if use_astropy_comoving:
        source = astropy_cosmology if astropy_cosmology is not None else cosmology
        return 1, _coerce_astropy_cosmology_obj(source)

    if isinstance(cosmology, Integral):
        cidx = int(cosmology)
        if cidx not in (1, 2):
            raise ValueError("For Corrfunc mocks, cosmology must be 1 or 2")
        return cidx, None

    raise ValueError(
        "When use_astropy_comoving=False in radecz mode, cosmology must be Corrfunc index 1 or 2. "
        "To use arbitrary cosmology dict/object, set use_astropy_comoving=True."
    )


def _maybe_convert_radecz_to_comoving(
    sample_pos: np.ndarray,
    random_pos: np.ndarray,
    coord_system: str,
    is_comoving_dist: bool,
    use_astropy_comoving: bool,
    astropy_cosmology: Any,
    input_is_cz: bool,
    speed_of_light_kms: float = 299792.458,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Optionally convert input [RA, DEC, z/cz] to [RA, DEC, comoving_distance(Mpc)]
    with astropy cosmology, then force is_comoving_dist=True for Corrfunc.mocks.
    """
    if coord_system.lower() != "radecz":
        return sample_pos, random_pos, bool(is_comoving_dist)

    if not use_astropy_comoving:
        return sample_pos, random_pos, bool(is_comoving_dist)

    if astropy_cosmology is None:
        raise ValueError("astropy_cosmology is required when use_astropy_comoving=True")

    c_kms = float(speed_of_light_kms)

    def _to_comoving_mpc(col3: np.ndarray) -> np.ndarray:
        z = col3 / c_kms if input_is_cz else col3
        dist = astropy_cosmology.comoving_distance(z)
        if hasattr(dist, "to_value"):
            try:
                return np.asarray(dist.to_value("Mpc"), dtype=np.float64)
            except Exception:
                return np.asarray(dist.to_value(), dtype=np.float64)
        return np.asarray(dist, dtype=np.float64)

    s = np.asarray(sample_pos, dtype=np.float64).copy()
    r = np.asarray(random_pos, dtype=np.float64).copy()
    s[:, 2] = _to_comoving_mpc(s[:, 2])
    r[:, 2] = _to_comoving_mpc(r[:, 2])
    return s, r, True


def _parse_bounds(
    sample_pos: np.ndarray,
    random_pos: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
) -> np.ndarray:
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (3, 2):
            raise ValueError("bounds must have shape ((x0,x1),(y0,y1),(z0,z1))")
        return b

    all_pos = np.vstack((sample_pos, random_pos))
    mins = np.min(all_pos, axis=0)
    maxs = np.max(all_pos, axis=0)
    b = np.column_stack((mins, maxs))

    eps = 1.0e-12
    for i in range(3):
        if b[i, 1] <= b[i, 0]:
            b[i, 1] = b[i, 0] + eps
    return b


def _assign_subbox_ids(pos: np.ndarray, bounds_xyz: np.ndarray, ndiv: int) -> np.ndarray:
    mins = bounds_xyz[:, 0]
    maxs = bounds_xyz[:, 1]
    size = np.maximum(maxs - mins, 1.0e-12)

    frac = (pos - mins[None, :]) / size[None, :]
    idx = np.minimum(np.maximum((frac * ndiv).astype(np.int32), 0), ndiv - 1)
    return idx[:, 0] + ndiv * (idx[:, 1] + ndiv * idx[:, 2])


def _xi_landy_szalay_from_counts(
    dd_counts: np.ndarray,
    dr_counts: np.ndarray,
    rr_counts: np.ndarray,
    n_data: int,
    n_rand: int,
) -> np.ndarray:
    if n_data < 2 or n_rand < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)

    dd_norm = dd_counts / (n_data * (n_data - 1.0))
    dr_norm = dr_counts / (n_data * n_rand)
    rr_norm = rr_counts / (n_rand * (n_rand - 1.0))

    xi = np.zeros_like(dd_counts, dtype=np.float64)
    valid = rr_norm > 0
    xi[valid] = (dd_norm[valid] - 2.0 * dr_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return xi


def _xi_s_from_smu(xi_smu_flat: np.ndarray, n_s_bins: int, n_mu_bins: int) -> np.ndarray:
    xi2d = xi_smu_flat.reshape(n_s_bins, n_mu_bins)
    return xi2d.mean(axis=1)


def _pair_counts_autocorr(
    p1: np.ndarray,
    p2: np.ndarray,
    s_bins: np.ndarray,
    nthreads: int,
    coord_system: str,
    cosmology: int,
    is_comoving_dist: bool,
    mu_max: float,
    nmu_bins: int,
    autocorr: bool,
) -> np.ndarray:
    coord = coord_system.lower()
    if coord == "radecz":
        if autocorr:
            res = DDsmu_mocks(
                autocorr=1,
                cosmology=cosmology,
                nthreads=nthreads,
                mu_max=mu_max,
                nmu_bins=nmu_bins,
                binfile=s_bins,
                RA1=p1[:, 0],
                DEC1=p1[:, 1],
                CZ1=p1[:, 2],
                is_comoving_dist=is_comoving_dist,
                output_savg=False,
                verbose=False,
            )
            return res["npairs"].astype(np.float64)
        res = DDsmu_mocks(
            autocorr=0,
            cosmology=cosmology,
            nthreads=nthreads,
            mu_max=mu_max,
            nmu_bins=nmu_bins,
            binfile=s_bins,
            RA1=p1[:, 0],
            DEC1=p1[:, 1],
            CZ1=p1[:, 2],
            RA2=p2[:, 0],
            DEC2=p2[:, 1],
            CZ2=p2[:, 2],
            is_comoving_dist=is_comoving_dist,
            output_savg=False,
            verbose=False,
        )
        return res["npairs"].astype(np.float64)

    if autocorr:
        res = DD(
            autocorr=1,
            nthreads=nthreads,
            binfile=s_bins,
            X1=p1[:, 0],
            Y1=p1[:, 1],
            Z1=p1[:, 2],
            periodic=False,
            output_ravg=False,
            verbose=False,
        )
        return res["npairs"].astype(np.float64)
    res = DD(
        autocorr=0,
        nthreads=nthreads,
        binfile=s_bins,
        X1=p1[:, 0],
        Y1=p1[:, 1],
        Z1=p1[:, 2],
        X2=p2[:, 0],
        Y2=p2[:, 1],
        Z2=p2[:, 2],
        periodic=False,
        output_ravg=False,
        verbose=False,
    )
    return res["npairs"].astype(np.float64)


def _precompute_pair(
    sample_pos: np.ndarray,
    s_bins: np.ndarray,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
    coord_system: str,
    cosmology: int,
    is_comoving_dist: bool,
    mu_max: float,
    nmu_bins: int,
) -> ObserveSubboxPairCounts:
    sub_ids = _assign_subbox_ids(sample_pos, bounds_xyz=bounds_xyz, ndiv=ndiv)
    n_sub = ndiv ** 3
    idxs: List[np.ndarray] = [np.where(sub_ids == i)[0] for i in range(n_sub)]
    n_points_subbox = np.array([idx.size for idx in idxs], dtype=np.int64)

    n_s = len(s_bins) - 1
    n_mu = nmu_bins if coord_system == "radecz" else 1
    n_flat = n_s * n_mu

    pair_counts: Dict[Tuple[int, int], np.ndarray] = {}
    total_counts = np.zeros(n_flat, dtype=np.float64)
    involved_counts = np.zeros((n_sub, n_flat), dtype=np.float64)

    for i, j in combinations_with_replacement(range(n_sub), 2):
        iidx = idxs[i]
        jidx = idxs[j]
        if iidx.size == 0 or jidx.size == 0:
            counts = np.zeros(n_flat, dtype=np.float64)
        elif i == j:
            counts = _pair_counts_autocorr(
                sample_pos[iidx], sample_pos[iidx], s_bins, nthreads,
                coord_system, cosmology, is_comoving_dist, mu_max, nmu_bins, autocorr=True
            )
        else:
            counts = _pair_counts_autocorr(
                sample_pos[iidx], sample_pos[jidx], s_bins, nthreads,
                coord_system, cosmology, is_comoving_dist, mu_max, nmu_bins, autocorr=False
            )
            counts = 2.0 * counts

        pair_counts[(i, j)] = counts
        total_counts += counts
        involved_counts[i] += counts
        if j != i:
            involved_counts[j] += counts

    return ObserveSubboxPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        involved_counts=involved_counts,
        s_bin_edges=np.asarray(s_bins, dtype=np.float64),
        n_subboxes=n_sub,
        n_points_total=sample_pos.shape[0],
        n_points_subbox=n_points_subbox,
        n_s_bins=n_s,
        n_mu_bins=n_mu,
    )


def _precompute_cross(
    sample_pos_1: np.ndarray,
    sample_pos_2: np.ndarray,
    s_bins: np.ndarray,
    ndiv: int,
    nthreads: int,
    bounds_xyz: np.ndarray,
    coord_system: str,
    cosmology: int,
    is_comoving_dist: bool,
    mu_max: float,
    nmu_bins: int,
) -> ObserveSubboxCrossPairCounts:
    sub1 = _assign_subbox_ids(sample_pos_1, bounds_xyz=bounds_xyz, ndiv=ndiv)
    sub2 = _assign_subbox_ids(sample_pos_2, bounds_xyz=bounds_xyz, ndiv=ndiv)

    n_sub = ndiv ** 3
    idx1: List[np.ndarray] = [np.where(sub1 == i)[0] for i in range(n_sub)]
    idx2: List[np.ndarray] = [np.where(sub2 == i)[0] for i in range(n_sub)]

    n_points_1_sub = np.array([x.size for x in idx1], dtype=np.int64)
    n_points_2_sub = np.array([x.size for x in idx2], dtype=np.int64)

    n_s = len(s_bins) - 1
    n_mu = nmu_bins if coord_system == "radecz" else 1
    n_flat = n_s * n_mu

    pair_counts = np.zeros((n_sub, n_sub, n_flat), dtype=np.float64)

    for i in range(n_sub):
        iidx = idx1[i]
        if iidx.size == 0:
            continue
        for j in range(n_sub):
            jidx = idx2[j]
            if jidx.size == 0:
                continue
            pair_counts[i, j] = _pair_counts_autocorr(
                sample_pos_1[iidx], sample_pos_2[jidx], s_bins, nthreads,
                coord_system, cosmology, is_comoving_dist, mu_max, nmu_bins, autocorr=False
            )

    row_sums = pair_counts.sum(axis=1)
    col_sums = pair_counts.sum(axis=0)
    total_counts = pair_counts.sum(axis=(0, 1))

    return ObserveSubboxCrossPairCounts(
        pair_counts=pair_counts,
        total_counts=total_counts,
        row_sums=row_sums,
        col_sums=col_sums,
        n_subboxes=n_sub,
        n_points_1_total=sample_pos_1.shape[0],
        n_points_2_total=sample_pos_2.shape[0],
        n_points_1_subbox=n_points_1_sub,
        n_points_2_subbox=n_points_2_sub,
    )


def corrfunc_xi_obsreve_jackknife(
    sample_radecz: np.ndarray = None,
    random_radecz: np.ndarray = None,
    s_bins: np.ndarray = None,
    ndiv: int = 4,
    nthreads: int = 8,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
    estimator: str = "landy-szalay",
    coord_system: str = "radecz",
    sample_xyz: np.ndarray = None,
    random_xyz: np.ndarray = None,
    cosmology: Any = 1,
    is_comoving_dist: bool = False,
    mu_max: float = 1.0,
    nmu_bins: int = 20,
    use_astropy_comoving: bool = False,
    astropy_cosmology: Any = None,
    input_is_cz: bool = True,
    speed_of_light_kms: float = 299792.458,
) -> Dict[str, Any]:
    """
    Jackknife estimator of two-point correlation xi for observed/non-periodic catalogs.

    Parameters
    ----------
    sample_radecz : ndarray of shape (N, 3), optional
        Data catalog in observed coordinates. Columns are [RA(deg), DEC(deg), col3].
        col3 can be z, cz(km/s), or comoving distance, depending on
        `use_astropy_comoving`, `input_is_cz`, and `is_comoving_dist`.
        Used when `coord_system='radecz'`.
    random_radecz : ndarray of shape (Nr, 3), optional
        Random catalog in the same coordinate convention as `sample_radecz`.
    s_bins : 1D ndarray of shape (Ns+1,)
        Radial separation bin edges for xi(s). Must be increasing.
    ndiv : int, default=4
        Number of jackknife splits per axis. Total subboxes = ndiv^3.
    nthreads : int, default=8
        Number of OpenMP threads passed to Corrfunc.
    bounds : tuple of 3 tuples, optional
        Manual bounding box used for subbox assignment:
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        If None, bounds are inferred from combined sample+random coordinates.
    estimator : {'landy-szalay'}, default='landy-szalay'
        Observe mode only supports Landy-Szalay.
    coord_system : {'radecz', 'xyz'}, default='radecz'
        'radecz' uses Corrfunc.mocks.DDsmu_mocks (non-periodic observe mode).
        'xyz' uses Corrfunc.theory.DD(periodic=False).
    sample_xyz : ndarray of shape (N, 3), optional
        Data catalog in Cartesian coordinates [x, y, z].
        Used when `coord_system='xyz'`.
    random_xyz : ndarray of shape (Nr, 3), optional
        Random catalog in Cartesian coordinates [x, y, z].
    cosmology : int or str or dict or astropy cosmology object, default=1
        Cosmology control in `coord_system='radecz'`:
        1) If `use_astropy_comoving=False`: must be Corrfunc mocks index {1, 2}.
           - 1: LasDamas (Omega_m=0.25, Omega_L=0.75)
           - 2: Planck-like (Omega_m=0.302, Omega_L=0.698)
        2) If `use_astropy_comoving=True`: can be astropy cosmology object,
           preset name string (e.g. 'Planck18'), or dict like
           {'H0': 67.7, 'Om0': 0.31}. Used only when `astropy_cosmology` is None.
    is_comoving_dist : bool, default=False
        Passed to Corrfunc only in `radecz` mode when
        `use_astropy_comoving=False`. If True, col3 is already comoving distance.
    mu_max : float, default=1.0
        Maximum |mu| for DDsmu_mocks in `radecz` mode. Must be in (0, 1].
    nmu_bins : int, default=20
        Number of mu bins for DDsmu_mocks in `radecz` mode. Must be >= 1.
    use_astropy_comoving : bool, default=False
        If True in `radecz` mode, convert col3 to comoving distance with astropy,
        then force Corrfunc input as comoving distance.
    astropy_cosmology : str or dict or astropy cosmology object, optional
        Explicit cosmology source for astropy conversion. If None and
        `use_astropy_comoving=True`, fallback to `cosmology`.
    input_is_cz : bool, default=True
        Only used when `use_astropy_comoving=True` and `coord_system='radecz'`.
        True: col3 is cz (km/s), converted to z by z=cz/c.
        False: col3 is treated as redshift z directly.
    speed_of_light_kms : float, default=299792.458
        Speed of light used for cz->z conversion.

    Returns
    -------
    result : dict
        Main keys:
        - xi_full / xi_mean : ndarray, shape (Ns,)
        - xi_jack : ndarray, shape (n_subboxes, Ns)
        - xi_err : ndarray, shape (Ns,)
        - cov : ndarray, shape (Ns, Ns)
        - s_bin_edges, s_bin_centers
        - timing: pair_precompute_s, jackknife_s, total_s
        - cosmology diagnostics: cosmology, corrfunc_cosmology,
          is_comoving_dist, use_astropy_comoving, astropy_cosmology_name

    Notes
    -----
    - This function is for non-periodic/observed catalogs.
    - For `coord_system='xyz'`, mu-binning is not used and xi is computed directly in s-bins.
    """
    if estimator.lower() != "landy-szalay":
        raise ValueError("For observed data, only estimator='landy-szalay' is supported.")
    if s_bins is None:
        raise ValueError("s_bins is required")

    coord_system = coord_system.lower()
    if coord_system not in ("radecz", "xyz"):
        raise ValueError("coord_system must be 'radecz' or 'xyz'")

    corrfunc_cosmology = 1
    astropy_cosmology_obj = None
    if coord_system == "radecz":
        corrfunc_cosmology, astropy_cosmology_obj = _resolve_radecz_cosmology_config(
            cosmology=cosmology,
            use_astropy_comoving=use_astropy_comoving,
            astropy_cosmology=astropy_cosmology,
        )

    if coord_system == "radecz":
        if not (0.0 < float(mu_max) <= 1.0):
            raise ValueError("mu_max must be in (0, 1] for radecz mode")
        if int(nmu_bins) < 1:
            raise ValueError("nmu_bins must be >= 1 for radecz mode")
        nmu_bins = int(nmu_bins)
    else:
        # xyz mode computes xi(s) directly; no mu-binning
        nmu_bins = 1

    sample_pos, random_pos = _resolve_samples(
        coord_system=coord_system,
        sample_radecz=sample_radecz,
        random_radecz=random_radecz,
        sample_xyz=sample_xyz,
        random_xyz=random_xyz,
    )

    sample_pos, random_pos, is_comoving_dist_eff = _maybe_convert_radecz_to_comoving(
        sample_pos=sample_pos,
        random_pos=random_pos,
        coord_system=coord_system,
        is_comoving_dist=is_comoving_dist,
        use_astropy_comoving=use_astropy_comoving,
        astropy_cosmology=astropy_cosmology,
        input_is_cz=input_is_cz,
        speed_of_light_kms=speed_of_light_kms,
    )

    is_comoving_dist = is_comoving_dist_eff

    s_bins = np.asarray(s_bins, dtype=np.float64)
    bounds_xyz = _parse_bounds(sample_pos=sample_pos, random_pos=random_pos, bounds=bounds)

    t0 = time.perf_counter()
    dd_precomp = _precompute_pair(
        sample_pos=sample_pos,
        s_bins=s_bins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
        coord_system=coord_system,
        cosmology=corrfunc_cosmology,
        is_comoving_dist=is_comoving_dist,
        mu_max=mu_max,
        nmu_bins=nmu_bins,
    )
    rr_precomp = _precompute_pair(
        sample_pos=random_pos,
        s_bins=s_bins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
        coord_system=coord_system,
        cosmology=corrfunc_cosmology,
        is_comoving_dist=is_comoving_dist,
        mu_max=mu_max,
        nmu_bins=nmu_bins,
    )
    dr_precomp = _precompute_cross(
        sample_pos_1=sample_pos,
        sample_pos_2=random_pos,
        s_bins=s_bins,
        ndiv=ndiv,
        nthreads=nthreads,
        bounds_xyz=bounds_xyz,
        coord_system=coord_system,
        cosmology=corrfunc_cosmology,
        is_comoving_dist=is_comoving_dist,
        mu_max=mu_max,
        nmu_bins=nmu_bins,
    )
    t1 = time.perf_counter()

    if not (dd_precomp.n_subboxes == rr_precomp.n_subboxes == dr_precomp.n_subboxes):
        raise RuntimeError("subbox number mismatch between DD, RR and DR")

    n_sub = dd_precomp.n_subboxes
    xi_full_flat = _xi_landy_szalay_from_counts(
        dd_counts=dd_precomp.total_counts,
        dr_counts=dr_precomp.total_counts,
        rr_counts=rr_precomp.total_counts,
        n_data=dd_precomp.n_points_total,
        n_rand=rr_precomp.n_points_total,
    )

    xi_jack_flat = np.zeros((n_sub, dd_precomp.total_counts.size), dtype=np.float64)
    for k in range(n_sub):
        dd_loo = dd_precomp.total_counts - dd_precomp.involved_counts[k]
        rr_loo = rr_precomp.total_counts - rr_precomp.involved_counts[k]
        dr_involving_k = dr_precomp.row_sums[k] + dr_precomp.col_sums[k] - dr_precomp.pair_counts[k, k]
        dr_loo = dr_precomp.total_counts - dr_involving_k

        n_d_loo = dd_precomp.n_points_total - dd_precomp.n_points_subbox[k]
        n_r_loo = rr_precomp.n_points_total - rr_precomp.n_points_subbox[k]
        xi_jack_flat[k] = _xi_landy_szalay_from_counts(dd_loo, dr_loo, rr_loo, n_d_loo, n_r_loo)

    if coord_system == "radecz":
        xi_full = _xi_s_from_smu(xi_full_flat, dd_precomp.n_s_bins, dd_precomp.n_mu_bins)
        xi_jack = np.array([
            _xi_s_from_smu(x, dd_precomp.n_s_bins, dd_precomp.n_mu_bins) for x in xi_jack_flat
        ])
        xi_full_smu = xi_full_flat
        xi_jack_smu = xi_jack_flat
    else:
        xi_full = xi_full_flat
        xi_jack = xi_jack_flat
        xi_full_smu = None
        xi_jack_smu = None

    xi_jack_mean = xi_jack.mean(axis=0)
    diff = xi_jack - xi_jack_mean
    cov = (n_sub - 1.0) / n_sub * (diff.T @ diff)
    xi_err = np.sqrt(np.diag(cov))

    t2 = time.perf_counter()

    return {
        "xi_mean": xi_full,
        "xi_full": xi_full,
        "xi_jack_mean": xi_jack_mean,
        "xi_err": xi_err,
        "cov": cov,
        "xi_jack": xi_jack,
        "xi_full_smu": xi_full_smu,
        "xi_jack_smu": xi_jack_smu,
        "coord_system": coord_system,
        "estimator": "landy-szalay",
        "rr_mode": "loo_randoms",
        "dd_total": dd_precomp.total_counts.copy(),
        "rr_full": rr_precomp.total_counts.copy(),
        "dr_full": dr_precomp.total_counts.copy(),
        "n_points_subbox_data": dd_precomp.n_points_subbox.copy(),
        "n_points_subbox_rand": rr_precomp.n_points_subbox.copy(),
        "s_bin_edges": s_bins.copy(),
        "s_bin_centers": np.sqrt(s_bins[:-1] * s_bins[1:]),
        "nmu_bins": dd_precomp.n_mu_bins,
        "mu_max": float(mu_max),
        "n_subboxes": n_sub,
        "bounds_coords": bounds_xyz.copy(),
        "cosmology": cosmology,
        "corrfunc_cosmology": int(corrfunc_cosmology),
        "is_comoving_dist": bool(is_comoving_dist_eff),
        "use_astropy_comoving": bool(use_astropy_comoving),
        "input_is_cz": bool(input_is_cz),
        "astropy_cosmology_name": None if astropy_cosmology_obj is None else getattr(astropy_cosmology_obj, "name", astropy_cosmology_obj.__class__.__name__),
        "timing": {
            "pair_precompute_s": t1 - t0,
            "jackknife_s": t2 - t1,
            "total_s": t2 - t0,
        },
    }
