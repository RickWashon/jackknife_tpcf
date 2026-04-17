# jackknife_corrfunc_tools

Corrfunc-based tools for:

- `jackknife/`: jackknife covariance/error estimation
- `tpcf/`: full-sample (non-jackknife) xi/wp calculations

## Directory Layout

- `jackknife/xi_jackknife.py`
- `jackknife/wp_jackknife.py`
- `jackknife/xi_obsreve_jackknife.py`
- `jackknife/wp_observe_jackknife.py`
- `jackknife/benchmark_all_modes.py`
- `tpcf/xi.py`
- `tpcf/wp.py`
- `tpcf/xi_observe.py`
- `tpcf/wp_observe.py`

## Requirements

- Python 3.8+
- `numpy`
- `Corrfunc`
- `astropy` (only needed for observe-mode astropy cosmology conversion)

```bash
pip install numpy Corrfunc astropy
```

## A) Jackknife APIs

### Periodic box xi

```python
from jackknife.xi_jackknife import corrfunc_xi_jackknife

res = corrfunc_xi_jackknife(
    sample_xyz,
    rbins,
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    estimator="natural",          # "natural" | "landy-szalay"
    natural_rr_mode="analytic",   # "analytic" | "random"
    random_xyz=None,
)
```

### Periodic box wp

```python
from jackknife.wp_jackknife import corrfunc_wp_jackknife

res = corrfunc_wp_jackknife(
    sample_xyz,
    rp_bins,
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

### Observed-data jackknife

- `jackknife/xi_obsreve_jackknife.py`
- `jackknife/wp_observe_jackknife.py`

These two files are for jackknife error estimation on observed / non-periodic data.
Both are LS-only and support two input styles:

- `coord_system='radecz'`
  Uses Corrfunc `mocks` functions.
  Input columns are `[RA(deg), DEC(deg), col3]`.
  `col3` can be one of:
  `z` : redshift
  `cz` : recession velocity in km/s
  comoving distance : if `is_comoving_dist=True`
- `coord_system='xyz'`
  Uses non-periodic Corrfunc `theory` functions with `periodic=False`.
  Input columns are `[x, y, z]`.

For `radecz`, there are two distance workflows:

- Corrfunc native mode:
  set `use_astropy_comoving=False`
  `cosmology` must be Corrfunc index `1` or `2`
  `is_comoving_dist=False` means col3 is interpreted as redshift-like input by Corrfunc
  `is_comoving_dist=True` means col3 is already comoving distance
- Astropy conversion mode:
  set `use_astropy_comoving=True`
  the code first converts col3 to comoving distance, then internally uses `is_comoving_dist=True`
  `astropy_cosmology` can be:
  an astropy cosmology object, for example `Planck18`
  a preset string, for example `"Planck18"`
  a dict, for example `{"H0": 67.7, "Om0": 0.31}` or `{"h": 0.677, "Om0": 0.31}`
  if `astropy_cosmology=None`, the code falls back to `cosmology`

How to choose `input_is_cz`:

- `input_is_cz=True`:
  col3 is `cz` in km/s, and the code converts to redshift by `z = cz / c`
- `input_is_cz=False`:
  col3 is already `z`

H0 note:

- `H0` must be in `km/s/Mpc` (for example `67.7`), not little-h (`0.677`)

Minimal examples:

```python
from astropy.cosmology import Planck18
from jackknife.xi_obsreve_jackknife import corrfunc_xi_obsreve_jackknife
from jackknife.wp_observe_jackknife import corrfunc_wp_observe_jackknife

# true observed coordinates, third column is z
res_xi = corrfunc_xi_obsreve_jackknife(
    sample_radecz=data_radecz,
    random_radecz=rand_radecz,
    s_bins=s_bins,
    coord_system="radecz",
    estimator="landy-szalay",
    use_astropy_comoving=True,
    astropy_cosmology=Planck18,
    input_is_cz=False,
)

# non-periodic Cartesian catalog
res_wp = corrfunc_wp_observe_jackknife(
    sample_xyz=data_xyz,
    random_xyz=rand_xyz,
    rp_bins=rp_bins,
    pimax=40,
    dpi=1,
    coord_system="xyz",
    estimator="landy-szalay",
)
```

## B) Full-Sample TPCF APIs (`tpcf/`)

## 1) `tpcf/xi.py`

Main function: `corrfunc_xi(...)`

- Supports `estimator='natural'` and `estimator='landy-szalay'`
- Supports one-sample and two-sample modes
- Two-sample mode supports:
  - `do_auto=True` -> compute `xi_11` and `xi_22`
  - `do_cross=True` -> compute `xi_12`
- Defaults: `do_auto=True`, `do_cross=False`

Random handling (LS mode):

- can pass only `random_xyz`
- can pass both `random_xyz` and `random2_xyz`
- can pass none -> auto-generate randoms uniformly in the same box

Single-sample + natural:

- directly calls `Corrfunc.theory.xi`.

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
    random_xyz=None,      # auto-generate if None in LS mode
    random2_xyz=None,     # defaults to random_xyz
)
```

## 2) `tpcf/wp.py`

Main function: `corrfunc_wp(...)`

- Same mode logic as `tpcf/xi.py` (`natural` / `landy-szalay`, `do_auto`, `do_cross`)
- Defaults: `do_auto=True`, `do_cross=False`
- Random handling same as xi
- Requires `dpi=1` and integer-valued `pimax`

Single-sample + natural:

- directly calls `Corrfunc.theory.wp`.

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

## 3) `tpcf/xi_observe.py`

Main function: `corrfunc_xi_observe(...)`

- Full-sample observed xi (non-jackknife)
- LS-only
- same input convention as `jackknife/xi_obsreve_jackknife.py`

How to use:

- `coord_system='radecz'`
  pass `sample_radecz` and `random_radecz`
  each row is `[RA, DEC, col3]`
  choose one of:
  `use_astropy_comoving=False` and `cosmology in {1,2}`
  or `use_astropy_comoving=True` with astropy cosmology support
- `coord_system='xyz'`
  pass `sample_xyz` and `random_xyz`
  each row is `[x, y, z]`
  no periodic boundary is assumed

Returned values:

- `xi`
  final 1D xi(s)
- `xi_smu`
  only in `radecz` mode
  flattened xi(s, mu) before averaging over mu
- `dd`, `dr`, `rr`
  raw pair counts used by LS

Example:

```python
from tpcf.xi_observe import corrfunc_xi_observe

res = corrfunc_xi_observe(
    sample_radecz=data_radecz,
    random_radecz=rand_radecz,
    s_bins=s_bins,
    coord_system="radecz",
    estimator="landy-szalay",
    use_astropy_comoving=True,
    astropy_cosmology={"H0": 67.7, "Om0": 0.31},
    input_is_cz=True,
)
```

## 4) `tpcf/wp_observe.py`

Main function: `corrfunc_wp_observe(...)`

- Full-sample observed wp (non-jackknife)
- LS-only
- same input convention as `jackknife/wp_observe_jackknife.py`

How to use:

- `coord_system='radecz'`
  pass `sample_radecz` and `random_radecz`
  each row is `[RA, DEC, col3]`
  the function computes DD / DR / RR in `(rp, pi)` and then projects to `wp(rp)`
- `coord_system='xyz'`
  pass `sample_xyz` and `random_xyz`
  each row is `[x, y, z]`
  non-periodic geometry

Important constraints:

- `estimator` must be `"landy-szalay"`
- `dpi=1`
- `pimax` must be integer-valued

Returned values:

- `wp`
  final projected correlation
- `xi_rppi`
  flattened xi(rp, pi) before projection
- `dd_rppi`, `dr_rppi`, `rr_rppi`
  pair counts in rp-pi bins

Example:

```python
from tpcf.wp_observe import corrfunc_wp_observe

res = corrfunc_wp_observe(
    sample_xyz=data_xyz,
    random_xyz=rand_xyz,
    rp_bins=rp_bins,
    pimax=40,
    dpi=1,
    coord_system="xyz",
    estimator="landy-szalay",
)
```

## Benchmark / Consistency Check

A direct natural-estimator consistency check (`N=50000`) between:

- `tpcf/xi.py` vs `Corrfunc.theory.xi`
- `tpcf/wp.py` vs `Corrfunc.theory.wp`

has been written to:

- `check.log`

Current result summary:

- xi: identical (`max_abs_diff = 0`, `allclose=True`)
- wp: identical (`max_abs_diff = 0`, `allclose=True`)
- wrapper timing is slightly slower (expected Python wrapper overhead).

## Docstring Note

All main callable APIs in `jackknife/` and `tpcf/` now include detailed docstrings:

- parameter type/shape/options/units
- return-key summaries
- mode-specific notes and constraints

## Naming Note

Filename `xi_obsreve_jackknife.py` keeps existing spelling for compatibility.
