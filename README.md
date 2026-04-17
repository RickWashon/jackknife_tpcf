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

Both are LS-only and support:

- `coord_system='radecz'` (Corrfunc.mocks)
- `coord_system='xyz'` (non-periodic Corrfunc.theory)
- astropy-based arbitrary cosmology conversion

H0 note:

- `H0` must be in `km/s/Mpc` (e.g. `67.7`), not little-h (`0.677`).

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
- `coord_system='radecz'` or `'xyz'`
- Supports astropy cosmology conversion options consistent with jackknife observe APIs

## 4) `tpcf/wp_observe.py`

Main function: `corrfunc_wp_observe(...)`

- Full-sample observed wp (non-jackknife)
- LS-only
- `coord_system='radecz'` or `'xyz'`
- Supports astropy cosmology conversion options consistent with jackknife observe APIs

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
