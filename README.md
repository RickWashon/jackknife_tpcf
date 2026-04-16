# corrfunc_subbox_jackknife

`corrfunc_subbox_jackknife.py` computes the 2-point correlation function (TPCF) and jackknife uncertainties in a periodic simulation box using Corrfunc pair counts.

The implementation partitions the box into an `ndiv × ndiv × ndiv` regular grid of subboxes, precomputes subbox-level pair counts once, and then builds leave-one-out (LOO) jackknife samples by subtraction.

## Features

- Fast jackknife workflow built on Corrfunc `DD`
- Supports:
  - **Natural estimator**
    - analytic RR (`natural_rr_mode="analytic"`)
    - random-catalog RR (`natural_rr_mode="random"`)
  - **Landy–Szalay estimator** (`estimator="landy-szalay"`)
- Returns full-sample `xi`, jackknife realizations, covariance, errors, and timing

## Requirements

- Python 3.9+
- `numpy`
- `Corrfunc`

Install dependencies (example):

```bash
pip install numpy Corrfunc
```

## Main API

```python
from corrfunc_subbox_jackknife import corrfunc_subbox_jackknife
```

```python
result = corrfunc_subbox_jackknife(
    sample_xyz,        # shape (N, 3)
    rbins,             # shape (nbins+1,)
    boxsize=1000.0,
    ndiv=4,
    nthreads=8,
    random_xyz=None,   # required for landy-szalay or natural_rr_mode="random"
    estimator="natural",
    natural_rr_mode="analytic",
)
```

### Parameters

- `sample_xyz`: data points, shape `(N, 3)`
- `rbins`: radial bin edges
- `boxsize`: periodic cube size
- `ndiv`: subboxes per axis (`n_subboxes = ndiv**3`)
- `nthreads`: Corrfunc thread count
- `random_xyz`: random catalog, shape `(Nr, 3)` when required
- `estimator`: `"natural"` or `"landy-szalay"`
- `natural_rr_mode`: `"analytic"` or `"random"` (used only with `estimator="natural"`)

### Returns

`dict` with keys:

- `xi_mean` / `xi_full`: full-sample correlation estimate (recommended central value)
- `xi_jack_mean`: mean of jackknife LOO estimates (diagnostic)
- `xi_jack`: array of jackknife realizations, shape `(n_subboxes, nbins)`
- `xi_err`: jackknife standard error per bin
- `cov`: jackknife covariance matrix
- `dd_total`, `rr_full`, `dr_full`: full-sample pair-count totals
- `n_points_subbox`, `n_subboxes`
- `r_bin_edges`, `r_bin_centers`
- `estimator`, `rr_mode`
- `timing`

## Estimator notes

- **Natural + analytic RR** is fastest.
- **Natural + random RR** is slower but uses geometry-matched randoms in each LOO sample.
- **Landy–Szalay** requires `random_xyz` and computes DD, DR, RR consistently for each LOO sample.

## Benchmark script

Running the module directly executes a random-point benchmark:

```bash
python corrfunc_subbox_jackknife.py
```

This prints runtime and example output slices (`xi_full`, `xi_jack_mean`, `xi_err`).
