# jackknife_corrfunc_tools

Corrfunc-based jackknife utilities for correlation statistics in periodic simulation boxes and observed (non-periodic) catalogs.

## Modules

- `xi_jackknife.py`: `xi(r)` jackknife for cubic periodic box
- `wp_jackknife.py`: `wp(rp)` jackknife for cubic periodic box
- `xi_obsreve_jackknife.py`: observed-data `xi` jackknife (LS-only, dual coord mode)
- `wp_observe_jackknife.py`: observed-data `wp` jackknife (LS-only, dual coord mode)
- `benchmark_all_modes.py`: benchmark runner for all modes or a single function/mode

## Common Jackknife Strategy

All modules follow the same structure:

1. Split data volume into `ndiv x ndiv x ndiv` subboxes.
2. Precompute subbox pair counts once (`DD(i,i)`, `DD(i,j)`, and when needed `RR`, `DR`).
3. Build leave-one-out (LOO) jackknife counts by subtraction.
4. Compute statistic per LOO sample, then covariance/error.

## Requirements

- Python 3.8+
- `numpy`
- `Corrfunc`
- `astropy` (only needed when using astropy-based `z/cz -> comoving distance` conversion)

```bash
pip install numpy Corrfunc astropy
```

## 1) Periodic Box: xi(r)

File: `xi_jackknife.py`

```python
from xi_jackknife import corrfunc_xi_jackknife

res = corrfunc_xi_jackknife(
    sample_xyz,              # (N,3)
    rbins,
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    random_xyz=None,         # required for LS or natural_rr_mode="random"
    estimator="natural",     # "natural" | "landy-szalay"
    natural_rr_mode="analytic"  # "analytic" | "random"
)
```

## 2) Periodic Box: wp(rp)

File: `wp_jackknife.py`

```python
from wp_jackknife import corrfunc_wp_jackknife

res = corrfunc_wp_jackknife(
    sample_xyz,              # (N,3)
    rp_bins,
    pimax=40,
    dpi=1,                   # Corrfunc DDrppi fixed pi-bin width
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    random_xyz=None,
    estimator="natural",     # "natural" | "landy-szalay"
    natural_rr_mode="analytic"  # "analytic" | "random"
)
```

Notes:

- `wp_jackknife` alias is kept for backward compatibility.
- `DDrppi` does not accept custom `dpi`; wrapper enforces `dpi=1` and integer `pimax`.
- For comparison to Corrfunc full-sample results, use `wp_full` as central value (`wp_jack_mean` is diagnostic).

## 3) Observed Data: xi (LS-only, dual coord mode)

File: `xi_obsreve_jackknife.py`

Main function: `corrfunc_xi_obsreve_jackknife(...)`

### Coordinate modes

1. `coord_system="radecz"`
- Uses `Corrfunc.mocks.DDsmu_mocks`
- Input columns: `[RA, DEC, col3]`, where col3 can be `z`, `cz`, or comoving distance

2. `coord_system="xyz"`
- Uses `Corrfunc.theory.DD` with `periodic=False`
- Inputs: `sample_xyz`, `random_xyz`

### Cosmology / distance options in `radecz`

- If `use_astropy_comoving=False`:
  - `cosmology` must be Corrfunc index `1` or `2`
  - `is_comoving_dist` is passed directly to Corrfunc

- If `use_astropy_comoving=True`:
  - col3 is converted to comoving distance with astropy, then internally uses `is_comoving_dist=True`
  - `astropy_cosmology` supports:
    - astropy cosmology object (e.g. `Planck18`)
    - preset string (e.g. `"Planck18"`)
    - dict (e.g. `{"H0":67.7,"Om0":0.31}` or `{"h":0.677,"Om0":0.31}`)
  - if `astropy_cosmology=None`, fallback source is `cosmology`

H0 note:

- `H0` must be in `km/s/Mpc` (e.g. `67.7`), not little-h (`0.677`).

### Example

```python
from astropy.cosmology import Planck18
from xi_obsreve_jackknife import corrfunc_xi_obsreve_jackknife

res_rdz = corrfunc_xi_obsreve_jackknife(
    sample_radecz=data_radecz,
    random_radecz=rand_radecz,
    s_bins=s_bins,
    coord_system="radecz",
    estimator="landy-szalay",
    use_astropy_comoving=True,
    astropy_cosmology=Planck18,
    input_is_cz=True,
    mu_max=1.0,
    nmu_bins=20,
)
```

## 4) Observed Data: wp(rp) (LS-only, dual coord mode)

File: `wp_observe_jackknife.py`

Main function: `corrfunc_wp_observe_jackknife(...)`

### Coordinate modes

1. `coord_system="radecz"`
- Uses `Corrfunc.mocks.DDrppi_mocks`
- Inputs: `sample_radecz`, `random_radecz`

2. `coord_system="xyz"`
- Uses `Corrfunc.theory.DDrppi` with `periodic=False`
- Inputs: `sample_xyz`, `random_xyz`

### Constraints

- LS-only (`estimator="landy-szalay"`)
- `dpi=1`, integer `pimax`
- Cosmology handling is the same as `xi_obsreve_jackknife`

### Example

```python
from astropy.cosmology import FlatLambdaCDM
from wp_observe_jackknife import corrfunc_wp_observe_jackknife

mycosmo = FlatLambdaCDM(H0=67.7, Om0=0.31)

res_wp_rdz = corrfunc_wp_observe_jackknife(
    sample_radecz=data_radecz,
    random_radecz=rand_radecz,
    rp_bins=rp_bins,
    pimax=40,
    dpi=1,
    coord_system="radecz",
    estimator="landy-szalay",
    use_astropy_comoving=True,
    astropy_cosmology=mycosmo,
    input_is_cz=False,
)
```

## Benchmark

Use `benchmark_all_modes.py` for full or single-mode tests:

```bash
# run all configured modes
python benchmark_all_modes.py --mode all

# run one function/mode
python benchmark_all_modes.py --mode one --function xi_box --estimator natural --natural-rr-mode analytic
python benchmark_all_modes.py --mode one --function wp_box --estimator landy-szalay
python benchmark_all_modes.py --mode one --function xi_obs --estimator landy-szalay
python benchmark_all_modes.py --mode one --function wp_obs --estimator landy-szalay
```

## Docstring Note

The four main callable APIs now include detailed Corrfunc-style docstrings in code
(type/shape/options/units for each parameter plus return-key summaries):

- `corrfunc_xi_jackknife`
- `corrfunc_wp_jackknife`
- `corrfunc_xi_obsreve_jackknife`
- `corrfunc_wp_observe_jackknife`

## Naming Note

Filename `xi_obsreve_jackknife.py` keeps existing spelling for compatibility.
