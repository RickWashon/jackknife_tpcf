# jackknife_corrfunc_tools

Corrfunc-based clustering tools for periodic-box simulations and observed catalogs.

This repository currently provides:

- `jackknife/`: jackknife covariance / error estimation
- `tpcf/`: full-sample (non-jackknife) xi/wp and 1h/2h decomposition
- `utils/`: low-level C/OpenMP pair counters used by `tpcf`

---

## Requirements

- Python 3.8+
- `numpy`
- `Corrfunc`
- `halotools` (only needed when you want direct consistency checks against Halotools)
- `astropy` (observe-mode cosmology conversion)

```bash
pip install numpy Corrfunc halotools astropy
```

---

## Repository Layout

### Jackknife

- `jackknife/xi_jackknife.py`
- `jackknife/wp_jackknife.py`
- `jackknife/xi_obsreve_jackknife.py`
- `jackknife/wp_observe_jackknife.py`
- `jackknife/benchmark_all_modes.py`

### Full-sample TPCF

- `tpcf/xi.py`
- `tpcf/wp.py`
- `tpcf/xi_observe.py`
- `tpcf/wp_observe.py`
- `tpcf/xi_1h_2h_decompose.py`

### Low-level Counters

- `utils/dd_counter.c`, `utils/dd.py`: fast total DD for periodic box
- `utils/weighted_dd_counter.c`, `utils/weighted_dd.py`: DD total/1h/2h (auto + cross)

---

## 1) Jackknife APIs

### Periodic box `xi`

```python
from jackknife.xi_jackknife import corrfunc_xi_jackknife

res = corrfunc_xi_jackknife(
    sample_xyz=xyz,
    rbins=rbins,
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    estimator="natural",          # "natural" | "landy-szalay"
    natural_rr_mode="analytic",   # "analytic" | "random"
    random_xyz=None,
)
```

### Periodic box `wp`

```python
from jackknife.wp_jackknife import corrfunc_wp_jackknife

res = corrfunc_wp_jackknife(
    sample_xyz=xyz,
    rp_bins=rp_bins,
    pimax=40,
    dpi=1,
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    estimator="natural",          # "natural" | "landy-szalay"
    natural_rr_mode="analytic",   # "analytic" | "random"
    random_xyz=None,
)
```

### Observed-data jackknife (`LS` only)

- `jackknife/xi_obsreve_jackknife.py`
- `jackknife/wp_observe_jackknife.py`

Supported coordinate styles:

- `coord_system="radecz"`:
  input columns are `[RA(deg), DEC(deg), col3]`
- `coord_system="xyz"`:
  non-periodic Cartesian `[x, y, z]`

`radecz` supports two distance workflows:

1. Corrfunc native cosmology (`cosmology in {1,2}`)
2. Astropy comoving conversion (`use_astropy_comoving=True`)

For Astropy mode, `astropy_cosmology` can be:

- cosmology object (e.g. `Planck18`)
- preset string (e.g. `"Planck18"`)
- dict (e.g. `{"H0": 67.7, "Om0": 0.31}`)

`H0` is in km/s/Mpc (e.g. `67.7`), not little-`h`.

---

## 2) Full-sample TPCF APIs (`tpcf/`)

## 2.1 `tpcf/xi.py`

Main function: `corrfunc_xi(...)`

- `estimator="natural"` or `"landy-szalay"`
- one-sample and two-sample mode
- in two-sample mode:
  - `do_auto=True` computes `xi_11` and `xi_22`
  - `do_cross=True` computes `xi_12`

Example:

```python
from tpcf.xi import corrfunc_xi

res = corrfunc_xi(
    sample_xyz=xyz1,
    rbins=rbins,
    boxsize=1000.0,
    estimator="landy-szalay",
    sample2_xyz=xyz2,
    do_auto=True,
    do_cross=True,
    random_xyz=None,
    random2_xyz=None,
)
```

## 2.2 `tpcf/wp.py`

Main function: `corrfunc_wp(...)`

- same two-sample logic as `xi.py`
- requires `dpi=1`, integer `pimax`

```python
from tpcf.wp import corrfunc_wp

res = corrfunc_wp(
    sample_xyz=xyz1,
    rp_bins=rp_bins,
    pimax=40,
    dpi=1,
    boxsize=1000.0,
    estimator="landy-szalay",
    sample2_xyz=xyz2,
    do_auto=True,
    do_cross=True,
)
```

## 2.3 `tpcf/xi_observe.py` and `tpcf/wp_observe.py`

Observed (non-jackknife) full-sample xi/wp.

- LS-only
- supports both `radecz` and non-periodic `xyz`
- same cosmology handling as observe jackknife APIs

---

## 3) `xi_1h_2h_decompose` (important)

File:

- `tpcf/xi_1h_2h_decompose.py`

Main function:

- `corrfunc_xi_1h_2h_decompose(...)`

What it computes:

- overall xi
- 1-halo xi
- 2-halo xi
- and corresponding DD components (`dd_total`, `dd_1h`, `dd_2h`)

Supported modes:

- single-sample auto decomposition
- two-sample mode:
  - auto for sample1/sample2 (`11`, `22`)
  - cross decomposition (`12`)

Estimator:

- `natural` and `landy-szalay`

Fast backend:

- `use_c_weighted_dd=True` uses C/OpenMP backend from `utils/weighted_dd_counter.c`
- now includes both auto and cross fast paths

Example:

```python
from tpcf.xi_1h_2h_decompose import corrfunc_xi_1h_2h_decompose

res = corrfunc_xi_1h_2h_decompose(
    sample_xyz=xyz1,
    host_halo_id=host1,
    rbins=rbins,
    boxsize=200.0,
    nthreads=16,
    sample2_xyz=xyz2,
    sample2_host_halo_id=host2,
    do_auto=True,
    do_cross=True,
    estimator="natural",
    use_c_weighted_dd=True,
)
```

---

## 4) Performance Notes (current status)

### 4.1 `xi_1h_2h_decompose` vs Halotools

In our current tests (same periodic-box setup), the C/OpenMP backend is much faster than Halotools.

Typical speedup:

- around `~10x` or higher, depending on `N`, `rbins`, `nthreads`, and mode (`auto/cross`).

Consistency with Halotools:

- cross 1h/2h xi agrees to machine precision in tested cases
- auto branches are also consistent within expected numerical tolerance

### 4.2 `utils/dd_counter` vs Corrfunc DD

Even after multiple optimizations, our custom DD counter is still slower than Corrfunc.

Current practical gap:

- roughly `3-4x` slower than `Corrfunc.theory.DD` in typical periodic-box tests.

So for pure total DD throughput, Corrfunc remains the speed reference.

---

## 5) Notes

- `__pycache__` is Python cache and should be ignored by git.
- If you want reproducible benchmark comparison, fix:
  - random seed
  - `rbins`
  - `boxsize`
  - `nthreads`
  - estimator and random catalogs
