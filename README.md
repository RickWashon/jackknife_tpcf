# jackknife_corrfunc_tools

Corrfunc-based jackknife utilities for correlation statistics in both periodic simulation boxes and observed (non-periodic) catalogs.

Current modules:

- `xi_jackknife.py`: `xi(r)` jackknife for cubic periodic box
- `wp_jackknife.py`: `wp(rp)` jackknife for cubic periodic box
- `xi_obsreve_jackknife.py`: observed-data `xi(r)` jackknife (LS only)
- `wp_observe_jackknife.py`: observed-data `wp(rp)` jackknife (LS only)
- `benchmark_all_modes.py`: benchmark runner for all modes or a single function/mode

## Common Jackknife Strategy

All modules use the same core workflow:

1. Split volume into `ndiv x ndiv x ndiv` subboxes.
2. Precompute subbox pair counts once (`DD(i,i)`, `DD(i,j)`, and when needed `RR`, `DR`).
3. Build leave-one-out (LOO) jackknife counts by subtraction.
4. Compute statistic per LOO sample and jackknife covariance/error.

This avoids recomputing full pair counts for every jackknife realization.

## Requirements

- Python 3.8+
- `numpy`
- `Corrfunc`

Example install:

```bash
pip install numpy Corrfunc
```

## 1) Periodic Box: xi(r)

File: `xi_jackknife.py`

Main API:

```python
from xi_jackknife import corrfunc_xi_jackknife

res = corrfunc_xi_jackknife(
    sample_xyz,              # (N,3)
    rbins,                   # radial edges
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    random_xyz=None,         # required for LS or natural_rr_mode="random"
    estimator="natural",     # "natural" | "landy-szalay"
    natural_rr_mode="analytic"  # "analytic" | "random"
)
```

Returns include:

- `xi_full` (same as `xi_mean`), `xi_jack_mean`, `xi_jack`
- `xi_err`, `cov`
- `dd_total`, `rr_full`, `dr_full`
- `timing`, bin info, subbox info

## 2) Periodic Box: wp(rp)

File: `wp_jackknife.py`

Main API:

```python
from wp_jackknife import corrfunc_wp_jackknife

res = corrfunc_wp_jackknife(
    sample_xyz,              # (N,3)
    rp_bins,                 # rp edges
    pimax=40,
    dpi=1,                   # must be 1 for Corrfunc DDrppi
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    random_xyz=None,
    estimator="natural",     # "natural" | "landy-szalay"
    natural_rr_mode="analytic"  # "analytic" | "random"
)
```

Returns include:

- `wp_full` (same as `wp_mean`), `wp_jack_mean`, `wp_jack`
- `wp_err`, `cov_wp`
- Also `xi(rp,pi)` products: `xi_full_rppi`, `xi_jack_rppi`, `cov_xi_rppi`
- Pair-count totals, timing, bin/subbox metadata

Notes:

- `wp_jackknife` alias is kept for backward compatibility.
- Corrfunc `DDrppi` does **not** accept custom `dpi`; wrapper enforces `dpi=1` and integer `pimax`.
- If comparing against Corrfunc full-sample output, use `wp_full` as central value.  
  `wp_jack_mean` is jackknife-LOO mean (diagnostic) and can sit systematically above/below `wp_full`.

## 3) Observed Data: xi(r) (LS only)

File: `xi_obsreve_jackknife.py`

Main API:

```python
from xi_obsreve_jackknife import corrfunc_xi_obsreve_jackknife

res = corrfunc_xi_obsreve_jackknife(
    sample_xyz,              # data (N,3)
    random_xyz,              # random (Nr,3)
    rbins,
    ndiv=4,
    nthreads=8,
    bounds=None,             # optional ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    estimator="landy-szalay"
)
```

Important:

- Observed mode supports **only** `estimator="landy-szalay"`.
- Natural estimator is intentionally disabled.
- Non-periodic pair counting (`periodic=False`) is used.

## 4) Observed Data: wp(rp) (LS only)

File: `wp_observe_jackknife.py`

Main API:

```python
from wp_observe_jackknife import corrfunc_wp_observe_jackknife

res = corrfunc_wp_observe_jackknife(
    sample_xyz,
    random_xyz,
    rp_bins,
    pimax=40,
    dpi=1,                   # must be 1
    ndiv=4,
    nthreads=8,
    bounds=None,
    estimator="landy-szalay"
)
```

Important:

- Observed mode supports **only** LS.
- Non-periodic counting is used.
- `dpi=1` + integer `pimax` are required (Corrfunc `DDrppi` behavior).

## Naming Notes

- Function names follow `corrfunc_*_jackknife` style for consistency.
- Existing filename `xi_obsreve_jackknife.py` is kept as-is (spelling preserved for compatibility).

## Quick Guidance

- Fastest periodic run: `xi`/`wp` with `estimator="natural"` + `natural_rr_mode="analytic"`.
- More robust large-scale behavior: use `estimator="landy-szalay"` with random catalog.
- For observed data, always use LS modules.

## Benchmark Script

Use the benchmark helper to run all modes or only one target function:

```bash
# all available modes
python benchmark_all_modes.py --mode all

# single target function/mode
python benchmark_all_modes.py --mode one --function xi_box --estimator natural --natural-rr-mode analytic
python benchmark_all_modes.py --mode one --function wp_box --estimator landy-szalay
python benchmark_all_modes.py --mode one --function xi_obs --estimator landy-szalay
python benchmark_all_modes.py --mode one --function wp_obs --estimator landy-szalay
```
