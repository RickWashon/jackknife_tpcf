from numbers import Integral
from typing import Any, Dict, Tuple

import numpy as np
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi


def _resolve_samples(
    coord_system: str,
    sample_radecz: np.ndarray = None,
    random_radecz: np.ndarray = None,
    sample_xyz: np.ndarray = None,
    random_xyz: np.ndarray = None,
):
    coord = coord_system.lower()
    if coord not in ("radecz", "xyz"):
        raise ValueError("coord_system must be 'radecz' or 'xyz'")

    if coord == "radecz":
        s = sample_radecz if sample_radecz is not None else sample_xyz
        r = random_radecz if random_radecz is not None else random_xyz
    else:
        s = sample_xyz if sample_xyz is not None else sample_radecz
        r = random_xyz if random_xyz is not None else random_radecz

    if s is None or r is None:
        raise ValueError("Missing sample/random inputs for selected coord_system")

    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("sample input must have shape (N,3)")
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("random input must have shape (Nr,3)")
    return s, r


def _coerce_h0_km_s_mpc(h0: Any) -> float:
    h0 = float(h0)
    if h0 <= 0:
        raise ValueError("H0 must be > 0")
    if h0 < 2.0:
        raise ValueError("H0 looks like little-h (e.g. 0.677). Use H0 in km/s/Mpc (e.g. 67.7).")
    return h0


def _coerce_astropy_cosmology_obj(cosmo_like: Any) -> Any:
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
    raise TypeError("astropy cosmology input must be astropy object, preset string, or dict")


def _resolve_radecz_cosmology_config(cosmology: Any, use_astropy_comoving: bool, astropy_cosmology: Any):
    if use_astropy_comoving:
        source = astropy_cosmology if astropy_cosmology is not None else cosmology
        return 1, _coerce_astropy_cosmology_obj(source)
    if isinstance(cosmology, Integral):
        cidx = int(cosmology)
        if cidx not in (1, 2):
            raise ValueError("For Corrfunc mocks, cosmology must be 1 or 2")
        return cidx, None
    raise ValueError("With use_astropy_comoving=False in radecz mode, cosmology must be Corrfunc index 1 or 2")


def _maybe_convert_radecz_to_comoving(
    sample_pos: np.ndarray,
    random_pos: np.ndarray,
    coord_system: str,
    is_comoving_dist: bool,
    use_astropy_comoving: bool,
    astropy_cosmology: Any,
    input_is_cz: bool,
    speed_of_light_kms: float,
):
    if coord_system.lower() != "radecz":
        return sample_pos, random_pos, bool(is_comoving_dist)
    if not use_astropy_comoving:
        return sample_pos, random_pos, bool(is_comoving_dist)
    if astropy_cosmology is None:
        raise ValueError("astropy_cosmology is required when use_astropy_comoving=True")

    c_kms = float(speed_of_light_kms)

    def to_comoving(col3):
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
    s[:, 2] = to_comoving(s[:, 2])
    r[:, 2] = to_comoving(r[:, 2])
    return s, r, True


def _validate_dpi_and_pimax(dpi: float, pimax: float) -> Tuple[float, int]:
    if not np.isclose(float(dpi), 1.0):
        raise ValueError("Corrfunc DDrppi uses fixed pi-bin width 1. Use dpi=1.")
    if float(pimax) <= 0:
        raise ValueError("pimax must be > 0")
    n_pi = int(round(float(pimax)))
    if not np.isclose(float(pimax), float(n_pi)):
        raise ValueError("pimax must be integer-valued for DDrppi fixed pi-bin width")
    return 1.0, n_pi


def _xi_landy_szalay_from_counts(dd_counts, dr_counts, rr_counts, n_data: int, n_rand: int):
    if n_data < 2 or n_rand < 2:
        return np.zeros_like(dd_counts, dtype=np.float64)
    dd_norm = dd_counts / (n_data * (n_data - 1.0))
    dr_norm = dr_counts / (n_data * n_rand)
    rr_norm = rr_counts / (n_rand * (n_rand - 1.0))
    xi = np.zeros_like(dd_counts, dtype=np.float64)
    valid = rr_norm > 0
    xi[valid] = (dd_norm[valid] - 2.0 * dr_norm[valid] + rr_norm[valid]) / rr_norm[valid]
    return xi


def _wp_from_xi_flat(xi_flat: np.ndarray, n_rp: int, n_pi: int, dpi: float) -> np.ndarray:
    return 2.0 * np.sum(xi_flat.reshape(n_rp, n_pi) * dpi, axis=1)


def corrfunc_wp_observe(
    sample_radecz: np.ndarray = None,
    random_radecz: np.ndarray = None,
    rp_bins: np.ndarray = None,
    pimax: float = 40,
    dpi: float = 1,
    nthreads: int = 8,
    estimator: str = "landy-szalay",
    coord_system: str = "radecz",
    sample_xyz: np.ndarray = None,
    random_xyz: np.ndarray = None,
    cosmology: Any = 1,
    is_comoving_dist: bool = False,
    use_astropy_comoving: bool = False,
    astropy_cosmology: Any = None,
    input_is_cz: bool = True,
    speed_of_light_kms: float = 299792.458,
) -> Dict[str, Any]:
    """
    Full-sample observed/non-periodic wp(rp) (LS-only).

    Parameters
    ----------
    sample_radecz : ndarray of shape (N, 3), optional
        Observed coordinates [RA(deg), DEC(deg), col3] for data.
    random_radecz : ndarray of shape (Nr, 3), optional
        Random catalog in matching coordinate convention.
    rp_bins : 1D ndarray
        Projected separation bin edges.
    pimax : float, default=40
        Maximum LOS separation. Must be integer-valued for DDrppi fixed bins.
    dpi : float, default=1
        Pi-bin width; this wrapper requires dpi=1.
    nthreads : int, default=8
        Corrfunc thread count.
    estimator : {'landy-szalay'}, default='landy-szalay'
        Only LS is supported in observed mode.
    coord_system : {'radecz', 'xyz'}, default='radecz'
        - 'radecz': uses Corrfunc.mocks.DDrppi_mocks
        - 'xyz': uses Corrfunc.theory.DDrppi(periodic=False)
    sample_xyz, random_xyz : ndarray, optional
        Cartesian non-periodic inputs for coord_system='xyz'.
    cosmology : int or str or dict or astropy cosmology object, default=1
        radecz mode cosmology control, same behavior as xi_observe.
    is_comoving_dist : bool, default=False
        Passed through when not using astropy conversion.
    use_astropy_comoving : bool, default=False
        If True in radecz mode, convert col3 to comoving distance with astropy.
    astropy_cosmology : object/str/dict, optional
        Astropy cosmology source.
    input_is_cz : bool, default=True
        True for cz(km/s), False for z input in col3.
    speed_of_light_kms : float, default=299792.458
        Speed of light for cz->z conversion.

    Returns
    -------
    dict
        - 'wp': projected correlation
        - 'xi_rppi': flattened xi(rp,pi)
        - 'dd_rppi', 'dr_rppi', 'rr_rppi'
        - 'rp_bin_edges', 'rp_bin_centers', 'pi_edges', 'pimax', 'dpi'
        - coordinate/cosmology metadata

    Notes
    -----
    - This function is non-jackknife; it computes full-sample wp only.
    - LS estimator only.
    """
    if estimator.lower() != "landy-szalay":
        raise ValueError("For observed data, only estimator='landy-szalay' is supported")
    if rp_bins is None:
        raise ValueError("rp_bins is required")

    coord_system = coord_system.lower()
    if coord_system not in ("radecz", "xyz"):
        raise ValueError("coord_system must be 'radecz' or 'xyz'")

    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    dpi, n_pi = _validate_dpi_and_pimax(dpi=dpi, pimax=pimax)
    pimax = float(n_pi)

    corrfunc_cosmology = 1
    astropy_cosmology_obj = None
    if coord_system == "radecz":
        corrfunc_cosmology, astropy_cosmology_obj = _resolve_radecz_cosmology_config(
            cosmology=cosmology,
            use_astropy_comoving=use_astropy_comoving,
            astropy_cosmology=astropy_cosmology,
        )

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
        astropy_cosmology=astropy_cosmology_obj,
        input_is_cz=input_is_cz,
        speed_of_light_kms=speed_of_light_kms,
    )

    if coord_system == "radecz":
        dd = DDrppi_mocks(
            autocorr=1,
            cosmology=corrfunc_cosmology,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            RA1=sample_pos[:, 0],
            DEC1=sample_pos[:, 1],
            CZ1=sample_pos[:, 2],
            is_comoving_dist=bool(is_comoving_dist_eff),
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)
        rr = DDrppi_mocks(
            autocorr=1,
            cosmology=corrfunc_cosmology,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            RA1=random_pos[:, 0],
            DEC1=random_pos[:, 1],
            CZ1=random_pos[:, 2],
            is_comoving_dist=bool(is_comoving_dist_eff),
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)
        dr = DDrppi_mocks(
            autocorr=0,
            cosmology=corrfunc_cosmology,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            RA1=sample_pos[:, 0],
            DEC1=sample_pos[:, 1],
            CZ1=sample_pos[:, 2],
            RA2=random_pos[:, 0],
            DEC2=random_pos[:, 1],
            CZ2=random_pos[:, 2],
            is_comoving_dist=bool(is_comoving_dist_eff),
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)
    else:
        dd = DDrppi(
            autocorr=1,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            X1=sample_pos[:, 0],
            Y1=sample_pos[:, 1],
            Z1=sample_pos[:, 2],
            periodic=False,
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)
        rr = DDrppi(
            autocorr=1,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            X1=random_pos[:, 0],
            Y1=random_pos[:, 1],
            Z1=random_pos[:, 2],
            periodic=False,
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)
        dr = DDrppi(
            autocorr=0,
            nthreads=int(nthreads),
            pimax=pimax,
            binfile=rp_bins,
            X1=sample_pos[:, 0],
            Y1=sample_pos[:, 1],
            Z1=sample_pos[:, 2],
            X2=random_pos[:, 0],
            Y2=random_pos[:, 1],
            Z2=random_pos[:, 2],
            periodic=False,
            output_rpavg=False,
            verbose=False,
        )["npairs"].astype(np.float64)

    xi_rppi = _xi_landy_szalay_from_counts(dd, dr, rr, sample_pos.shape[0], random_pos.shape[0])
    wp = _wp_from_xi_flat(xi_rppi, rp_bins.size - 1, n_pi, dpi)

    return {
        "wp": wp,
        "xi_rppi": xi_rppi,
        "dd_rppi": dd,
        "dr_rppi": dr,
        "rr_rppi": rr,
        "rp_bin_edges": rp_bins.copy(),
        "rp_bin_centers": np.sqrt(rp_bins[:-1] * rp_bins[1:]),
        "pi_edges": np.arange(0.0, float(n_pi) + 1.0, 1.0),
        "pimax": float(pimax),
        "dpi": float(dpi),
        "coord_system": coord_system,
        "estimator": "landy-szalay",
        "cosmology": cosmology,
        "corrfunc_cosmology": int(corrfunc_cosmology),
        "is_comoving_dist": bool(is_comoving_dist_eff),
        "use_astropy_comoving": bool(use_astropy_comoving),
        "astropy_cosmology_name": None
        if astropy_cosmology_obj is None
        else getattr(astropy_cosmology_obj, "name", astropy_cosmology_obj.__class__.__name__),
    }


# convenience alias
wp_observe = corrfunc_wp_observe
