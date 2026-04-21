#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static inline int64_t wrap_index(const int64_t i, const int64_t n) {
    int64_t x = i % n;
    if (x < 0) {
        x += n;
    }
    return x;
}

static inline int64_t flatten3(const int64_t ix, const int64_t iy, const int64_t iz, const int64_t nside) {
    return ix + nside * (iy + nside * iz);
}

static inline void unflatten3(const int64_t c, const int64_t nside, int64_t *ix, int64_t *iy, int64_t *iz) {
    *ix = c % nside;
    *iy = (c / nside) % nside;
    *iz = c / (nside * nside);
}

static inline int64_t coord_to_cell(const double x, const double inv_cellsize, const int64_t nside) {
    int64_t ix = (int64_t)floor(x * inv_cellsize);
    if (ix < 0) {
        ix = 0;
    } else if (ix >= nside) {
        ix = nside - 1;
    }
    return ix;
}

static inline double axis_min_sep_periodic(
    const int64_t i1,
    const int64_t i2,
    const int64_t nside,
    const double cellsize
) {
    int64_t d = i1 - i2;
    if (d < 0) {
        d = -d;
    }
    if (d > (nside - d)) {
        d = nside - d;
    }
    if (d <= 1) {
        return 0.0;
    }
    return ((double)(d - 1)) * cellsize;
}

static inline double axis_max_sep_periodic(
    const int64_t i1,
    const int64_t i2,
    const int64_t nside,
    const double cellsize,
    const double half_box
) {
    int64_t d = i1 - i2;
    if (d < 0) {
        d = -d;
    }
    if (d > (nside - d)) {
        d = nside - d;
    }
    {
        double out = ((double)(d + 1)) * cellsize;
        if (out > half_box) {
            out = half_box;
        }
        return out;
    }
}

static inline __attribute__((always_inline)) int64_t locate_bin_lut(
    const double r2,
    const double *rbins2,
    const int64_t nbins,
    const double r2_min,
    const double r2_max,
    const int32_t *lut,
    const int64_t lut_n,
    const double inv_lut_step
) {
    int64_t idx;
    int64_t b;

    if (r2 < r2_min || r2 >= r2_max) {
        return -1;
    }

    idx = (int64_t)((r2 - r2_min) * inv_lut_step);
    if (idx < 0) {
        idx = 0;
    } else if (idx >= lut_n) {
        idx = lut_n - 1;
    }

    b = (int64_t)lut[idx];
    if (b < 0) {
        b = 0;
    } else if (b >= nbins) {
        b = nbins - 1;
    }

    /* Fast local correction around LUT guess; full while fallback remains
       for numerical corner cases near bin boundaries. */
    if (r2 < rbins2[b]) {
        b--;
    } else if (r2 >= rbins2[b + 1]) {
        b++;
    }
    while (b > 0 && r2 < rbins2[b]) {
        b--;
    }
    while (b < nbins - 1 && r2 >= rbins2[b + 1]) {
        b++;
    }

    if (r2 >= rbins2[b] && r2 < rbins2[b + 1]) {
        return b;
    }
    return -1;
}

static inline __attribute__((always_inline)) double r2_eval(const double dx, const double dy, const double dz, const int use_float32) {
    if (use_float32) {
        const float fx = (float)dx;
        const float fy = (float)dy;
        const float fz = (float)dz;
        const double dfx = (double)fx;
        const double dfy = (double)fy;
        const double dfz = (double)fz;
        return dfx * dfx + dfy * dfy + dfz * dfz;
    }
    return dx * dx + dy * dy + dz * dz;
}

int weighted_dd_1h2h_auto(
    const double *x,
    const double *y,
    const double *z,
    const int64_t *label,
    const int64_t npts,
    const double *rbins,
    const int64_t nbins,
    const double boxsize,
    const int nthreads,
    const double approx_cell_size,
    const int refine_factor,
    const int max_cells_per_dim,
    const int use_float32,
    uint64_t *dd_total,
    uint64_t *dd_1h,
    uint64_t *dd_2h
) {
    int64_t i;

    if (x == NULL || y == NULL || z == NULL || label == NULL || rbins == NULL || dd_total == NULL || dd_1h == NULL || dd_2h == NULL) {
        return -1;
    }
    if (npts < 0 || nbins <= 0 || boxsize <= 0.0) {
        return -2;
    }

    for (i = 0; i < nbins; i++) {
        dd_total[i] = 0ULL;
        dd_1h[i] = 0ULL;
        dd_2h[i] = 0ULL;
    }
    if (npts < 2) {
        return 0;
    }

    double *rbins2 = (double *)malloc((size_t)(nbins + 1) * sizeof(double));
    if (rbins2 == NULL) {
        return -3;
    }
    for (i = 0; i < nbins + 1; i++) {
        rbins2[i] = rbins[i] * rbins[i];
    }

    const double r2_min = rbins2[0];
    const double r2_max = rbins2[nbins];
    const double rmax = rbins[nbins];
    const double half_box = 0.5 * boxsize;

    const int64_t lut_n = (nbins <= 64) ? 4096 : 65536;
    int32_t *bin_lut = (int32_t *)malloc((size_t)lut_n * sizeof(int32_t));
    if (bin_lut == NULL) {
        free(rbins2);
        return -7;
    }
    {
        const double r2_range = r2_max - r2_min;
        const double lut_step = r2_range / (double)lut_n;
        int64_t cur = 0;
        int64_t t;
        for (t = 0; t < lut_n; t++) {
            const double probe = r2_min + ((double)t + 0.5) * lut_step;
            while (cur < nbins - 1 && probe >= rbins2[cur + 1]) {
                cur++;
            }
            bin_lut[t] = (int32_t)cur;
        }
    }
    const double inv_lut_step = (double)lut_n / (r2_max - r2_min);

    double requested_cell = approx_cell_size;
    if (requested_cell <= 0.0) {
        requested_cell = boxsize / 10.0;
    }
    if (requested_cell <= 0.0) {
        requested_cell = rmax;
    }

    int rf = refine_factor;
    if (rf < 1) {
        rf = 1;
    }

    int mcpd = max_cells_per_dim;
    if (mcpd < 1) {
        mcpd = 1;
    }

    int64_t nside = (int64_t)floor(boxsize / requested_cell);
    if (nside < 1) {
        nside = 1;
    }
    nside *= (int64_t)rf;
    if (nside < 1) {
        nside = 1;
    }
    if (nside > (int64_t)mcpd) {
        nside = (int64_t)mcpd;
    }

    {
        const double target_cells = 8.0 * (double)npts;
        const double max_side_f = floor(cbrt(target_cells));
        int64_t max_side = (int64_t)(max_side_f > 1.0 ? max_side_f : 1.0);
        if (nside > max_side) {
            nside = max_side;
        }
    }

    const int64_t ncells = nside * nside * nside;
    const double cellsize = boxsize / (double)nside;
    const double inv_cellsize = 1.0 / cellsize;
    const int64_t reach = (int64_t)ceil(rmax / cellsize);

    int64_t *cell_count = (int64_t *)calloc((size_t)ncells, sizeof(int64_t));
    int64_t *cell_offset = (int64_t *)malloc((size_t)(ncells + 1) * sizeof(int64_t));
    int64_t *cell_cursor = (int64_t *)malloc((size_t)ncells * sizeof(int64_t));
    int64_t *point_cell = (int64_t *)malloc((size_t)npts * sizeof(int64_t));
    int64_t *cell_points = (int64_t *)malloc((size_t)npts * sizeof(int64_t));
    uint8_t *nonempty_flag = (uint8_t *)calloc((size_t)ncells, sizeof(uint8_t));

    if (cell_count == NULL || cell_offset == NULL || cell_cursor == NULL || point_cell == NULL || cell_points == NULL || nonempty_flag == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        return -4;
    }

    for (i = 0; i < npts; i++) {
        const int64_t ix = coord_to_cell(x[i], inv_cellsize, nside);
        const int64_t iy = coord_to_cell(y[i], inv_cellsize, nside);
        const int64_t iz = coord_to_cell(z[i], inv_cellsize, nside);
        const int64_t c = flatten3(ix, iy, iz, nside);
        point_cell[i] = c;
        cell_count[c] += 1;
    }

    cell_offset[0] = 0;
    for (i = 0; i < ncells; i++) {
        cell_offset[i + 1] = cell_offset[i] + cell_count[i];
        cell_cursor[i] = cell_offset[i];
        if (cell_count[i] > 0) {
            nonempty_flag[i] = 1;
        }
    }

    for (i = 0; i < npts; i++) {
        const int64_t c = point_cell[i];
        const int64_t pos = cell_cursor[c];
        cell_points[pos] = i;
        cell_cursor[c] = pos + 1;
    }

    int64_t n_nonempty = 0;
    for (i = 0; i < ncells; i++) {
        if (cell_count[i] > 0) {
            n_nonempty += 1;
        }
    }

    int64_t *nonempty_cells = (int64_t *)malloc((size_t)n_nonempty * sizeof(int64_t));
    int32_t *nonempty_ix = (int32_t *)malloc((size_t)n_nonempty * sizeof(int32_t));
    int32_t *nonempty_iy = (int32_t *)malloc((size_t)n_nonempty * sizeof(int32_t));
    int32_t *nonempty_iz = (int32_t *)malloc((size_t)n_nonempty * sizeof(int32_t));
    if (nonempty_cells == NULL || nonempty_ix == NULL || nonempty_iy == NULL || nonempty_iz == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        return -6;
    }

    {
        int64_t k = 0;
        for (i = 0; i < ncells; i++) {
            if (cell_count[i] > 0) {
                int64_t ix, iy, iz;
                nonempty_cells[k] = i;
                unflatten3(i, nside, &ix, &iy, &iz);
                nonempty_ix[k] = (int32_t)ix;
                nonempty_iy[k] = (int32_t)iy;
                nonempty_iz[k] = (int32_t)iz;
                k++;
            }
        }
    }

    const int64_t max_nbr = (2 * reach + 1) * (2 * reach + 1) * (2 * reach + 1);
    int64_t *nbr_counts = (int64_t *)calloc((size_t)n_nonempty, sizeof(int64_t));
    int64_t *nbr_offsets = (int64_t *)malloc((size_t)(n_nonempty + 1) * sizeof(int64_t));
    int32_t *nbr_cells;
    double *nbr_shiftx;
    double *nbr_shifty;
    double *nbr_shiftz;
    if (nbr_counts == NULL || nbr_offsets == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        return -8;
    }

    int nth = (nthreads > 0) ? nthreads : 1;
#ifdef _OPENMP
    if (nth > omp_get_max_threads()) {
        nth = omp_get_max_threads();
    }
#else
    nth = 1;
#endif

int bad_nbr_count = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32) num_threads(nth) reduction(|:bad_nbr_count)
#endif
    for (int64_t idx1 = 0; idx1 < n_nonempty; idx1++) {
        const int64_t c1 = nonempty_cells[idx1];
        const int64_t ix1 = (int64_t)nonempty_ix[idx1];
        const int64_t iy1 = (int64_t)nonempty_iy[idx1];
        const int64_t iz1 = (int64_t)nonempty_iz[idx1];
        int64_t cnt = 0;
        for (int64_t dx_cell = -reach; dx_cell <= reach; dx_cell++) {
            for (int64_t dy_cell = -reach; dy_cell <= reach; dy_cell++) {
                for (int64_t dz_cell = -reach; dz_cell <= reach; dz_cell++) {
                    const int64_t ix2 = wrap_index(ix1 + dx_cell, nside);
                    const int64_t iy2 = wrap_index(iy1 + dy_cell, nside);
                    const int64_t iz2 = wrap_index(iz1 + dz_cell, nside);
                    const int64_t c2 = flatten3(ix2, iy2, iz2, nside);
                    if (!nonempty_flag[c2]) {
                        continue;
                    }
                    if (c2 < c1) {
                        continue;
                    }
                    {
                        const double min_dx = axis_min_sep_periodic(ix1, ix2, nside, cellsize);
                        const double min_dy = axis_min_sep_periodic(iy1, iy2, nside, cellsize);
                        const double min_dz = axis_min_sep_periodic(iz1, iz2, nside, cellsize);
                        const double min_r2 = min_dx * min_dx + min_dy * min_dy + min_dz * min_dz;
                        const double max_dx = axis_max_sep_periodic(ix1, ix2, nside, cellsize, half_box);
                        const double max_dy = axis_max_sep_periodic(iy1, iy2, nside, cellsize, half_box);
                        const double max_dz = axis_max_sep_periodic(iz1, iz2, nside, cellsize, half_box);
                        const double max_r2 = max_dx * max_dx + max_dy * max_dy + max_dz * max_dz;
                        if (min_r2 >= r2_max) {
                            continue;
                        }
                        if (max_r2 < r2_min) {
                            continue;
                        }
                    }
                    cnt++;
                }
            }
        }
        nbr_counts[idx1] = cnt;
        if (cnt > max_nbr) {
            bad_nbr_count = 1;
        }
    }
    if (bad_nbr_count) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        return -9;
    }

    nbr_offsets[0] = 0;
    for (int64_t idx1 = 0; idx1 < n_nonempty; idx1++) {
        nbr_offsets[idx1 + 1] = nbr_offsets[idx1] + nbr_counts[idx1];
    }
    nbr_cells = (int32_t *)malloc((size_t)nbr_offsets[n_nonempty] * sizeof(int32_t));
    nbr_shiftx = (double *)malloc((size_t)nbr_offsets[n_nonempty] * sizeof(double));
    nbr_shifty = (double *)malloc((size_t)nbr_offsets[n_nonempty] * sizeof(double));
    nbr_shiftz = (double *)malloc((size_t)nbr_offsets[n_nonempty] * sizeof(double));
    if (nbr_cells == NULL || nbr_shiftx == NULL || nbr_shifty == NULL || nbr_shiftz == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        free(nbr_shiftx);
        free(nbr_shifty);
        free(nbr_shiftz);
        return -10;
    }

int bad_nbr_fill = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32) num_threads(nth) reduction(|:bad_nbr_fill)
#endif
    for (int64_t idx1 = 0; idx1 < n_nonempty; idx1++) {
        const int64_t c1 = nonempty_cells[idx1];
        const int64_t ix1 = (int64_t)nonempty_ix[idx1];
        const int64_t iy1 = (int64_t)nonempty_iy[idx1];
        const int64_t iz1 = (int64_t)nonempty_iz[idx1];
        int64_t pos = nbr_offsets[idx1];
        const int64_t end = nbr_offsets[idx1 + 1];
        for (int64_t dx_cell = -reach; dx_cell <= reach; dx_cell++) {
            for (int64_t dy_cell = -reach; dy_cell <= reach; dy_cell++) {
                for (int64_t dz_cell = -reach; dz_cell <= reach; dz_cell++) {
                    const int64_t ix2 = wrap_index(ix1 + dx_cell, nside);
                    const int64_t iy2 = wrap_index(iy1 + dy_cell, nside);
                    const int64_t iz2 = wrap_index(iz1 + dz_cell, nside);
                    const int64_t c2 = flatten3(ix2, iy2, iz2, nside);
                    if (!nonempty_flag[c2]) {
                        continue;
                    }
                    if (c2 < c1) {
                        continue;
                    }
                    {
                        const double min_dx = axis_min_sep_periodic(ix1, ix2, nside, cellsize);
                        const double min_dy = axis_min_sep_periodic(iy1, iy2, nside, cellsize);
                        const double min_dz = axis_min_sep_periodic(iz1, iz2, nside, cellsize);
                        const double min_r2 = min_dx * min_dx + min_dy * min_dy + min_dz * min_dz;
                        const double max_dx = axis_max_sep_periodic(ix1, ix2, nside, cellsize, half_box);
                        const double max_dy = axis_max_sep_periodic(iy1, iy2, nside, cellsize, half_box);
                        const double max_dz = axis_max_sep_periodic(iz1, iz2, nside, cellsize, half_box);
                        const double max_r2 = max_dx * max_dx + max_dy * max_dy + max_dz * max_dz;
                        if (min_r2 >= r2_max) {
                            continue;
                        }
                        if (max_r2 < r2_min) {
                            continue;
                        }
                    }
                    if (pos >= end) {
                        bad_nbr_fill = 1;
                        continue;
                    }
                    nbr_cells[pos] = (int32_t)c2;
                    {
                        int64_t dix = ix2 - ix1;
                        int64_t diy = iy2 - iy1;
                        int64_t diz = iz2 - iz1;
                        double sx = 0.0;
                        double sy = 0.0;
                        double sz = 0.0;
                        if (dix > nside / 2) sx = -boxsize;
                        else if (dix < -(nside / 2)) sx = boxsize;
                        if (diy > nside / 2) sy = -boxsize;
                        else if (diy < -(nside / 2)) sy = boxsize;
                        if (diz > nside / 2) sz = -boxsize;
                        else if (diz < -(nside / 2)) sz = boxsize;
                        nbr_shiftx[pos] = sx;
                        nbr_shifty[pos] = sy;
                        nbr_shiftz[pos] = sz;
                    }
                    pos++;
                }
            }
        }
        if (pos != end) {
            bad_nbr_fill = 1;
        }
    }
    if (bad_nbr_fill) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        free(nbr_cells);
        free(nbr_shiftx);
        free(nbr_shifty);
        free(nbr_shiftz);
        return -12;
    }

    /* Build contiguous cell-sorted views for faster pair loops. */
    double *x_sorted = (double *)malloc((size_t)npts * sizeof(double));
    double *y_sorted = (double *)malloc((size_t)npts * sizeof(double));
    double *z_sorted = (double *)malloc((size_t)npts * sizeof(double));
    int64_t *label_sorted = (int64_t *)malloc((size_t)npts * sizeof(int64_t));
    if (x_sorted == NULL || y_sorted == NULL || z_sorted == NULL || label_sorted == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        free(nbr_cells);
        free(nbr_shiftx);
        free(nbr_shifty);
        free(nbr_shiftz);
        free(x_sorted);
        free(y_sorted);
        free(z_sorted);
        free(label_sorted);
        return -11;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth)
#endif
    for (i = 0; i < npts; i++) {
        const int64_t p = cell_points[i];
        x_sorted[i] = x[p];
        y_sorted[i] = y[p];
        z_sorted[i] = z[p];
        label_sorted[i] = label[p];
    }

    uint64_t *acc_total = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));
    uint64_t *acc_1h = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));
    uint64_t *acc_2h = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));

    int use_task_split = 0;
    int64_t ntasks = n_nonempty;
    int64_t *task_idx1 = NULL;
    int64_t *task_k0 = NULL;
    int64_t *task_k1 = NULL;

    if (acc_total == NULL || acc_1h == NULL || acc_2h == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count);
        free(cell_offset);
        free(cell_cursor);
        free(point_cell);
        free(cell_points);
        free(nonempty_flag);
        free(nonempty_cells);
        free(nonempty_ix);
        free(nonempty_iy);
        free(nonempty_iz);
        free(nbr_counts);
        free(nbr_offsets);
        free(nbr_cells);
        free(nbr_shiftx);
        free(nbr_shifty);
        free(nbr_shiftz);
        free(x_sorted);
        free(y_sorted);
        free(z_sorted);
        free(label_sorted);
        free(task_idx1);
        free(task_k0);
        free(task_k1);
        free(acc_total);
        free(acc_1h);
        free(acc_2h);
        return -5;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(nth)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint64_t *local_total = acc_total + (size_t)tid * (size_t)nbins;
        uint64_t *local_1h = acc_1h + (size_t)tid * (size_t)nbins;
        uint64_t *local_2h = acc_2h + (size_t)tid * (size_t)nbins;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (int64_t idx1 = 0; idx1 < n_nonempty; idx1++) {
            int64_t kk_begin = nbr_offsets[idx1];
            int64_t kk_end = nbr_offsets[idx1 + 1];
            const int64_t c1 = nonempty_cells[idx1];
            const int64_t s1 = cell_offset[c1];
            const int64_t e1 = cell_offset[c1 + 1];
            if (s1 >= e1) {
                continue;
            }

            for (int64_t kk = kk_begin; kk < kk_end; kk++) {
                const int64_t c2 = (int64_t)nbr_cells[kk];
                const int64_t s2 = cell_offset[c2];
                const int64_t e2 = cell_offset[c2 + 1];

                if (c2 == c1) {
                    for (int64_t a = s1; a < e1 - 1; a++) {
                        const double xi = x_sorted[a];
                        const double yi = y_sorted[a];
                        const double zi = z_sorted[a];
                        const int64_t li = label_sorted[a];

                        int64_t bidx = a + 1;
#if defined(__AVX2__)
                        {
                            const __m256d vxi = _mm256_set1_pd(xi);
                            const __m256d vyi = _mm256_set1_pd(yi);
                            const __m256d vzi = _mm256_set1_pd(zi);
                            const __m256d vhalf = _mm256_set1_pd(half_box);
                            const __m256d vnhalf = _mm256_set1_pd(-half_box);
                            const __m256d vbox = _mm256_set1_pd(boxsize);
                            for (; bidx + 3 < e1; bidx += 4) {
                                __m256d vx = _mm256_loadu_pd(x_sorted + bidx);
                                __m256d vy = _mm256_loadu_pd(y_sorted + bidx);
                                __m256d vz = _mm256_loadu_pd(z_sorted + bidx);
                                double r2buf[4];

                                __m256d vdx = _mm256_sub_pd(vx, vxi);
                                __m256d vdy = _mm256_sub_pd(vy, vyi);
                                __m256d vdz = _mm256_sub_pd(vz, vzi);

                                __m256d m_gt = _mm256_cmp_pd(vdx, vhalf, _CMP_GT_OQ);
                                __m256d m_lt = _mm256_cmp_pd(vdx, vnhalf, _CMP_LT_OQ);
                                vdx = _mm256_blendv_pd(vdx, _mm256_sub_pd(vdx, vbox), m_gt);
                                vdx = _mm256_blendv_pd(vdx, _mm256_add_pd(vdx, vbox), m_lt);

                                m_gt = _mm256_cmp_pd(vdy, vhalf, _CMP_GT_OQ);
                                m_lt = _mm256_cmp_pd(vdy, vnhalf, _CMP_LT_OQ);
                                vdy = _mm256_blendv_pd(vdy, _mm256_sub_pd(vdy, vbox), m_gt);
                                vdy = _mm256_blendv_pd(vdy, _mm256_add_pd(vdy, vbox), m_lt);

                                m_gt = _mm256_cmp_pd(vdz, vhalf, _CMP_GT_OQ);
                                m_lt = _mm256_cmp_pd(vdz, vnhalf, _CMP_LT_OQ);
                                vdz = _mm256_blendv_pd(vdz, _mm256_sub_pd(vdz, vbox), m_gt);
                                vdz = _mm256_blendv_pd(vdz, _mm256_add_pd(vdz, vbox), m_lt);

#if defined(__FMA__)
                                __m256d vr2 = _mm256_fmadd_pd(vdx, vdx, _mm256_mul_pd(vdy, vdy));
                                vr2 = _mm256_fmadd_pd(vdz, vdz, vr2);
#else
                                __m256d vr2 = _mm256_add_pd(_mm256_mul_pd(vdx, vdx), _mm256_mul_pd(vdy, vdy));
                                vr2 = _mm256_add_pd(vr2, _mm256_mul_pd(vdz, vdz));
#endif
                                _mm256_storeu_pd(r2buf, vr2);

                                for (int lane = 0; lane < 4; lane++) {
                                    const int64_t jpos = bidx + (int64_t)lane;
                                    const int64_t b = locate_bin_lut(
                                        r2buf[lane],
                                        rbins2,
                                        nbins,
                                        r2_min,
                                        r2_max,
                                        bin_lut,
                                        lut_n,
                                        inv_lut_step
                                    );
                                    if (b < 0) {
                                        continue;
                                    }
                                    local_total[b] += 2ULL;
                                    if (label_sorted[jpos] == li) {
                                        local_1h[b] += 2ULL;
                                    } else {
                                        local_2h[b] += 2ULL;
                                    }
                                }
                            }
                        }
#endif
                        for (; bidx < e1; bidx++) {
                            double dx = x_sorted[bidx] - xi;
                            double dy = y_sorted[bidx] - yi;
                            double dz = z_sorted[bidx] - zi;

                            if (dx > half_box) dx -= boxsize;
                            if (dx < -half_box) dx += boxsize;
                            if (dy > half_box) dy -= boxsize;
                            if (dy < -half_box) dy += boxsize;
                            if (dz > half_box) dz -= boxsize;
                            if (dz < -half_box) dz += boxsize;

                            int64_t b = locate_bin_lut(
                                r2_eval(dx, dy, dz, use_float32),
                                rbins2,
                                nbins,
                                r2_min,
                                r2_max,
                                bin_lut,
                                lut_n,
                                inv_lut_step
                            );
                            if (b < 0) {
                                continue;
                            }

                            local_total[b] += 2ULL;
                            if (label_sorted[bidx] == li) {
                                local_1h[b] += 2ULL;
                            } else {
                                local_2h[b] += 2ULL;
                            }
                        }
                    }
                } else {
                    const double sx = nbr_shiftx[kk];
                    const double sy = nbr_shifty[kk];
                    const double sz = nbr_shiftz[kk];
                    for (int64_t a = s1; a < e1; a++) {
                        const double xi = x_sorted[a];
                        const double yi = y_sorted[a];
                        const double zi = z_sorted[a];
                        const int64_t li = label_sorted[a];

                        int64_t bidx = s2;
#if defined(__AVX2__)
                        {
                            const __m256d vxi = _mm256_set1_pd(xi);
                            const __m256d vyi = _mm256_set1_pd(yi);
                            const __m256d vzi = _mm256_set1_pd(zi);
                            const __m256d vsx = _mm256_set1_pd(sx);
                            const __m256d vsy = _mm256_set1_pd(sy);
                            const __m256d vsz = _mm256_set1_pd(sz);
                            for (; bidx + 3 < e2; bidx += 4) {
                                __m256d vx = _mm256_loadu_pd(x_sorted + bidx);
                                __m256d vy = _mm256_loadu_pd(y_sorted + bidx);
                                __m256d vz = _mm256_loadu_pd(z_sorted + bidx);
                                double r2buf[4];

                                vx = _mm256_sub_pd(_mm256_add_pd(vx, vsx), vxi);
                                vy = _mm256_sub_pd(_mm256_add_pd(vy, vsy), vyi);
                                vz = _mm256_sub_pd(_mm256_add_pd(vz, vsz), vzi);

#if defined(__FMA__)
                                __m256d vr2 = _mm256_fmadd_pd(vx, vx, _mm256_mul_pd(vy, vy));
                                vr2 = _mm256_fmadd_pd(vz, vz, vr2);
#else
                                __m256d vr2 = _mm256_add_pd(_mm256_mul_pd(vx, vx), _mm256_mul_pd(vy, vy));
                                vr2 = _mm256_add_pd(vr2, _mm256_mul_pd(vz, vz));
#endif
                                _mm256_storeu_pd(r2buf, vr2);

                                for (int lane = 0; lane < 4; lane++) {
                                    const int64_t jpos = bidx + (int64_t)lane;
                                    const int64_t b = locate_bin_lut(
                                        r2buf[lane],
                                        rbins2,
                                        nbins,
                                        r2_min,
                                        r2_max,
                                        bin_lut,
                                        lut_n,
                                        inv_lut_step
                                    );
                                    if (b < 0) {
                                        continue;
                                    }
                                    local_total[b] += 2ULL;
                                    if (label_sorted[jpos] == li) {
                                        local_1h[b] += 2ULL;
                                    } else {
                                        local_2h[b] += 2ULL;
                                    }
                                }
                            }
                        }
#endif
                        for (; bidx < e2; bidx++) {
                            double dx = (x_sorted[bidx] + sx) - xi;
                            double dy = (y_sorted[bidx] + sy) - yi;
                            double dz = (z_sorted[bidx] + sz) - zi;

                            int64_t b = locate_bin_lut(
                                r2_eval(dx, dy, dz, use_float32),
                                rbins2,
                                nbins,
                                r2_min,
                                r2_max,
                                bin_lut,
                                lut_n,
                                inv_lut_step
                            );
                            if (b < 0) {
                                continue;
                            }

                            local_total[b] += 2ULL;
                            if (label_sorted[bidx] == li) {
                                local_1h[b] += 2ULL;
                            } else {
                                local_2h[b] += 2ULL;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int t = 0; t < nth; t++) {
        const uint64_t *lt = acc_total + (size_t)t * (size_t)nbins;
        const uint64_t *l1 = acc_1h + (size_t)t * (size_t)nbins;
        const uint64_t *l2 = acc_2h + (size_t)t * (size_t)nbins;
        for (i = 0; i < nbins; i++) {
            dd_total[i] += lt[i];
            dd_1h[i] += l1[i];
            dd_2h[i] += l2[i];
        }
    }

    free(rbins2);
    free(bin_lut);
    free(cell_count);
    free(cell_offset);
    free(cell_cursor);
    free(point_cell);
    free(cell_points);
    free(nonempty_flag);
    free(nonempty_cells);
    free(nonempty_ix);
    free(nonempty_iy);
    free(nonempty_iz);
    free(nbr_counts);
    free(nbr_offsets);
    free(nbr_cells);
    free(nbr_shiftx);
    free(nbr_shifty);
    free(nbr_shiftz);
    free(x_sorted);
    free(y_sorted);
    free(z_sorted);
    free(label_sorted);
    free(task_idx1);
    free(task_k0);
    free(task_k1);
    free(acc_total);
    free(acc_1h);
    free(acc_2h);
    return 0;
}

int dd_auto_no_weight(
    const double *x,
    const double *y,
    const double *z,
    const int64_t npts,
    const double *rbins,
    const int64_t nbins,
    const double boxsize,
    const int nthreads,
    const double approx_cell_size,
    const int refine_factor,
    const int max_cells_per_dim,
    const int use_float32,
    uint64_t *dd_total
) {
    int64_t i;
    int64_t *label = NULL;
    uint64_t *dd_1h = NULL;
    uint64_t *dd_2h = NULL;
    int rc;

    if (x == NULL || y == NULL || z == NULL || rbins == NULL || dd_total == NULL) {
        return -1;
    }
    if (npts < 0 || nbins <= 0 || boxsize <= 0.0) {
        return -2;
    }
    if (npts < 2) {
        for (i = 0; i < nbins; i++) {
            dd_total[i] = 0ULL;
        }
        return 0;
    }

    label = (int64_t *)malloc((size_t)npts * sizeof(int64_t));
    dd_1h = (uint64_t *)calloc((size_t)nbins, sizeof(uint64_t));
    dd_2h = (uint64_t *)calloc((size_t)nbins, sizeof(uint64_t));
    if (label == NULL || dd_1h == NULL || dd_2h == NULL) {
        free(label);
        free(dd_1h);
        free(dd_2h);
        return -3;
    }

    for (i = 0; i < npts; i++) {
        label[i] = i;
    }

    rc = weighted_dd_1h2h_auto(
        x,
        y,
        z,
        label,
        npts,
        rbins,
        nbins,
        boxsize,
        nthreads,
        approx_cell_size,
        refine_factor,
        max_cells_per_dim,
        use_float32,
        dd_total,
        dd_1h,
        dd_2h
    );

    free(label);
    free(dd_1h);
    free(dd_2h);
    return rc;
}

int weighted_dd_1h2h_cross(
    const double *x1,
    const double *y1,
    const double *z1,
    const int64_t *label1,
    const int64_t npts1,
    const double *x2,
    const double *y2,
    const double *z2,
    const int64_t *label2,
    const int64_t npts2,
    const double *rbins,
    const int64_t nbins,
    const double boxsize,
    const int nthreads,
    const double approx_cell_size,
    const int refine_factor,
    const int max_cells_per_dim,
    const int use_float32,
    uint64_t *dd_total,
    uint64_t *dd_1h,
    uint64_t *dd_2h
) {
    int64_t i;

    if (x1 == NULL || y1 == NULL || z1 == NULL || label1 == NULL ||
        x2 == NULL || y2 == NULL || z2 == NULL || label2 == NULL ||
        rbins == NULL || dd_total == NULL || dd_1h == NULL || dd_2h == NULL) {
        return -1;
    }
    if (npts1 < 0 || npts2 < 0 || nbins <= 0 || boxsize <= 0.0) {
        return -2;
    }

    for (i = 0; i < nbins; i++) {
        dd_total[i] = 0ULL;
        dd_1h[i] = 0ULL;
        dd_2h[i] = 0ULL;
    }
    if (npts1 < 1 || npts2 < 1) {
        return 0;
    }

    double *rbins2 = (double *)malloc((size_t)(nbins + 1) * sizeof(double));
    if (rbins2 == NULL) {
        return -3;
    }
    for (i = 0; i < nbins + 1; i++) {
        rbins2[i] = rbins[i] * rbins[i];
    }

    const double r2_min = rbins2[0];
    const double r2_max = rbins2[nbins];
    const double rmax = rbins[nbins];
    const double half_box = 0.5 * boxsize;

    const int64_t lut_n = (nbins <= 64) ? 4096 : 65536;
    int32_t *bin_lut = (int32_t *)malloc((size_t)lut_n * sizeof(int32_t));
    if (bin_lut == NULL) {
        free(rbins2);
        return -7;
    }
    {
        const double r2_range = r2_max - r2_min;
        const double lut_step = r2_range / (double)lut_n;
        int64_t cur = 0;
        int64_t t;
        for (t = 0; t < lut_n; t++) {
            const double probe = r2_min + ((double)t + 0.5) * lut_step;
            while (cur < nbins - 1 && probe >= rbins2[cur + 1]) {
                cur++;
            }
            bin_lut[t] = (int32_t)cur;
        }
    }
    const double inv_lut_step = (double)lut_n / (r2_max - r2_min);

    double requested_cell = approx_cell_size;
    if (requested_cell <= 0.0) {
        requested_cell = boxsize / 10.0;
    }
    if (requested_cell <= 0.0) {
        requested_cell = rmax;
    }

    int rf = refine_factor;
    if (rf < 1) {
        rf = 1;
    }

    int mcpd = max_cells_per_dim;
    if (mcpd < 1) {
        mcpd = 1;
    }

    int64_t nside = (int64_t)floor(boxsize / requested_cell);
    if (nside < 1) {
        nside = 1;
    }
    nside *= (int64_t)rf;
    if (nside < 1) {
        nside = 1;
    }
    if (nside > (int64_t)mcpd) {
        nside = (int64_t)mcpd;
    }

    {
        const double target_cells = 8.0 * (double)(npts1 + npts2);
        const double max_side_f = floor(cbrt(target_cells));
        int64_t max_side = (int64_t)(max_side_f > 1.0 ? max_side_f : 1.0);
        if (nside > max_side) {
            nside = max_side;
        }
    }

    const int64_t ncells = nside * nside * nside;
    const double cellsize = boxsize / (double)nside;
    const double inv_cellsize = 1.0 / cellsize;
    const int64_t reach = (int64_t)ceil(rmax / cellsize);

    int64_t *cell_count1 = (int64_t *)calloc((size_t)ncells, sizeof(int64_t));
    int64_t *cell_count2 = (int64_t *)calloc((size_t)ncells, sizeof(int64_t));
    int64_t *cell_offset1 = (int64_t *)malloc((size_t)(ncells + 1) * sizeof(int64_t));
    int64_t *cell_offset2 = (int64_t *)malloc((size_t)(ncells + 1) * sizeof(int64_t));
    int64_t *cell_cursor1 = (int64_t *)malloc((size_t)ncells * sizeof(int64_t));
    int64_t *cell_cursor2 = (int64_t *)malloc((size_t)ncells * sizeof(int64_t));
    int64_t *point_cell1 = (int64_t *)malloc((size_t)npts1 * sizeof(int64_t));
    int64_t *point_cell2 = (int64_t *)malloc((size_t)npts2 * sizeof(int64_t));
    int64_t *cell_points1 = (int64_t *)malloc((size_t)npts1 * sizeof(int64_t));
    int64_t *cell_points2 = (int64_t *)malloc((size_t)npts2 * sizeof(int64_t));
    uint8_t *nonempty2 = (uint8_t *)calloc((size_t)ncells, sizeof(uint8_t));

    if (cell_count1 == NULL || cell_count2 == NULL ||
        cell_offset1 == NULL || cell_offset2 == NULL ||
        cell_cursor1 == NULL || cell_cursor2 == NULL ||
        point_cell1 == NULL || point_cell2 == NULL ||
        cell_points1 == NULL || cell_points2 == NULL ||
        nonempty2 == NULL) {
        free(rbins2);
        free(bin_lut);
        free(cell_count1); free(cell_count2);
        free(cell_offset1); free(cell_offset2);
        free(cell_cursor1); free(cell_cursor2);
        free(point_cell1); free(point_cell2);
        free(cell_points1); free(cell_points2);
        free(nonempty2);
        return -4;
    }

    for (i = 0; i < npts1; i++) {
        const int64_t ix = coord_to_cell(x1[i], inv_cellsize, nside);
        const int64_t iy = coord_to_cell(y1[i], inv_cellsize, nside);
        const int64_t iz = coord_to_cell(z1[i], inv_cellsize, nside);
        const int64_t c = flatten3(ix, iy, iz, nside);
        point_cell1[i] = c;
        cell_count1[c] += 1;
    }
    for (i = 0; i < npts2; i++) {
        const int64_t ix = coord_to_cell(x2[i], inv_cellsize, nside);
        const int64_t iy = coord_to_cell(y2[i], inv_cellsize, nside);
        const int64_t iz = coord_to_cell(z2[i], inv_cellsize, nside);
        const int64_t c = flatten3(ix, iy, iz, nside);
        point_cell2[i] = c;
        cell_count2[c] += 1;
    }

    cell_offset1[0] = 0;
    cell_offset2[0] = 0;
    for (i = 0; i < ncells; i++) {
        cell_offset1[i + 1] = cell_offset1[i] + cell_count1[i];
        cell_offset2[i + 1] = cell_offset2[i] + cell_count2[i];
        cell_cursor1[i] = cell_offset1[i];
        cell_cursor2[i] = cell_offset2[i];
        if (cell_count2[i] > 0) {
            nonempty2[i] = 1;
        }
    }

    for (i = 0; i < npts1; i++) {
        const int64_t c = point_cell1[i];
        const int64_t pos = cell_cursor1[c];
        cell_points1[pos] = i;
        cell_cursor1[c] = pos + 1;
    }
    for (i = 0; i < npts2; i++) {
        const int64_t c = point_cell2[i];
        const int64_t pos = cell_cursor2[c];
        cell_points2[pos] = i;
        cell_cursor2[c] = pos + 1;
    }

    int64_t n_nonempty1 = 0;
    for (i = 0; i < ncells; i++) {
        if (cell_count1[i] > 0) {
            n_nonempty1 += 1;
        }
    }

    int64_t *nonempty_cells1 = (int64_t *)malloc((size_t)n_nonempty1 * sizeof(int64_t));
    int32_t *nonempty1_ix = (int32_t *)malloc((size_t)n_nonempty1 * sizeof(int32_t));
    int32_t *nonempty1_iy = (int32_t *)malloc((size_t)n_nonempty1 * sizeof(int32_t));
    int32_t *nonempty1_iz = (int32_t *)malloc((size_t)n_nonempty1 * sizeof(int32_t));
    if (nonempty_cells1 == NULL || nonempty1_ix == NULL || nonempty1_iy == NULL || nonempty1_iz == NULL) {
        free(rbins2); free(bin_lut);
        free(cell_count1); free(cell_count2);
        free(cell_offset1); free(cell_offset2);
        free(cell_cursor1); free(cell_cursor2);
        free(point_cell1); free(point_cell2);
        free(cell_points1); free(cell_points2);
        free(nonempty2);
        free(nonempty_cells1); free(nonempty1_ix); free(nonempty1_iy); free(nonempty1_iz);
        return -6;
    }

    {
        int64_t k = 0;
        for (i = 0; i < ncells; i++) {
            if (cell_count1[i] > 0) {
                int64_t ix, iy, iz;
                nonempty_cells1[k] = i;
                unflatten3(i, nside, &ix, &iy, &iz);
                nonempty1_ix[k] = (int32_t)ix;
                nonempty1_iy[k] = (int32_t)iy;
                nonempty1_iz[k] = (int32_t)iz;
                k++;
            }
        }
    }

    double *x1_sorted = (double *)malloc((size_t)npts1 * sizeof(double));
    double *y1_sorted = (double *)malloc((size_t)npts1 * sizeof(double));
    double *z1_sorted = (double *)malloc((size_t)npts1 * sizeof(double));
    int64_t *label1_sorted = (int64_t *)malloc((size_t)npts1 * sizeof(int64_t));
    double *x2_sorted = (double *)malloc((size_t)npts2 * sizeof(double));
    double *y2_sorted = (double *)malloc((size_t)npts2 * sizeof(double));
    double *z2_sorted = (double *)malloc((size_t)npts2 * sizeof(double));
    int64_t *label2_sorted = (int64_t *)malloc((size_t)npts2 * sizeof(int64_t));
    if (x1_sorted == NULL || y1_sorted == NULL || z1_sorted == NULL || label1_sorted == NULL ||
        x2_sorted == NULL || y2_sorted == NULL || z2_sorted == NULL || label2_sorted == NULL) {
        free(rbins2); free(bin_lut);
        free(cell_count1); free(cell_count2);
        free(cell_offset1); free(cell_offset2);
        free(cell_cursor1); free(cell_cursor2);
        free(point_cell1); free(point_cell2);
        free(cell_points1); free(cell_points2);
        free(nonempty2);
        free(nonempty_cells1); free(nonempty1_ix); free(nonempty1_iy); free(nonempty1_iz);
        free(x1_sorted); free(y1_sorted); free(z1_sorted); free(label1_sorted);
        free(x2_sorted); free(y2_sorted); free(z2_sorted); free(label2_sorted);
        return -11;
    }

#ifdef _OPENMP
    int nth = (nthreads > 0) ? nthreads : 1;
    if (nth > omp_get_max_threads()) {
        nth = omp_get_max_threads();
    }
#else
    int nth = 1;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth)
#endif
    for (i = 0; i < npts1; i++) {
        const int64_t p = cell_points1[i];
        x1_sorted[i] = x1[p];
        y1_sorted[i] = y1[p];
        z1_sorted[i] = z1[p];
        label1_sorted[i] = label1[p];
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth)
#endif
    for (i = 0; i < npts2; i++) {
        const int64_t p = cell_points2[i];
        x2_sorted[i] = x2[p];
        y2_sorted[i] = y2[p];
        z2_sorted[i] = z2[p];
        label2_sorted[i] = label2[p];
    }

    uint64_t *acc_total = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));
    uint64_t *acc_1h = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));
    uint64_t *acc_2h = (uint64_t *)calloc((size_t)nth * (size_t)nbins, sizeof(uint64_t));
    if (acc_total == NULL || acc_1h == NULL || acc_2h == NULL) {
        free(rbins2); free(bin_lut);
        free(cell_count1); free(cell_count2);
        free(cell_offset1); free(cell_offset2);
        free(cell_cursor1); free(cell_cursor2);
        free(point_cell1); free(point_cell2);
        free(cell_points1); free(cell_points2);
        free(nonempty2);
        free(nonempty_cells1); free(nonempty1_ix); free(nonempty1_iy); free(nonempty1_iz);
        free(x1_sorted); free(y1_sorted); free(z1_sorted); free(label1_sorted);
        free(x2_sorted); free(y2_sorted); free(z2_sorted); free(label2_sorted);
        free(acc_total); free(acc_1h); free(acc_2h);
        return -5;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(nth)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint64_t *local_total = acc_total + (size_t)tid * (size_t)nbins;
        uint64_t *local_1h = acc_1h + (size_t)tid * (size_t)nbins;
        uint64_t *local_2h = acc_2h + (size_t)tid * (size_t)nbins;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (int64_t idx1 = 0; idx1 < n_nonempty1; idx1++) {
            const int64_t c1 = nonempty_cells1[idx1];
            const int64_t ix1 = (int64_t)nonempty1_ix[idx1];
            const int64_t iy1 = (int64_t)nonempty1_iy[idx1];
            const int64_t iz1 = (int64_t)nonempty1_iz[idx1];
            const int64_t s1 = cell_offset1[c1];
            const int64_t e1 = cell_offset1[c1 + 1];
            if (s1 >= e1) {
                continue;
            }

            for (int64_t dx_cell = -reach; dx_cell <= reach; dx_cell++) {
                for (int64_t dy_cell = -reach; dy_cell <= reach; dy_cell++) {
                    for (int64_t dz_cell = -reach; dz_cell <= reach; dz_cell++) {
                        const int64_t ix2 = wrap_index(ix1 + dx_cell, nside);
                        const int64_t iy2 = wrap_index(iy1 + dy_cell, nside);
                        const int64_t iz2 = wrap_index(iz1 + dz_cell, nside);
                        const int64_t c2 = flatten3(ix2, iy2, iz2, nside);
                        if (!nonempty2[c2]) {
                            continue;
                        }

                        {
                            const double min_dx = axis_min_sep_periodic(ix1, ix2, nside, cellsize);
                            const double min_dy = axis_min_sep_periodic(iy1, iy2, nside, cellsize);
                            const double min_dz = axis_min_sep_periodic(iz1, iz2, nside, cellsize);
                            const double min_r2 = min_dx * min_dx + min_dy * min_dy + min_dz * min_dz;
                            const double max_dx = axis_max_sep_periodic(ix1, ix2, nside, cellsize, half_box);
                            const double max_dy = axis_max_sep_periodic(iy1, iy2, nside, cellsize, half_box);
                            const double max_dz = axis_max_sep_periodic(iz1, iz2, nside, cellsize, half_box);
                            const double max_r2 = max_dx * max_dx + max_dy * max_dy + max_dz * max_dz;
                            if (min_r2 >= r2_max || max_r2 < r2_min) {
                                continue;
                            }
                        }

                        const int64_t s2 = cell_offset2[c2];
                        const int64_t e2 = cell_offset2[c2 + 1];
                        if (s2 >= e2) {
                            continue;
                        }

                        int64_t dix = ix2 - ix1;
                        int64_t diy = iy2 - iy1;
                        int64_t diz = iz2 - iz1;
                        double sx = 0.0, sy = 0.0, sz = 0.0;
                        if (dix > nside / 2) sx = -boxsize;
                        else if (dix < -(nside / 2)) sx = boxsize;
                        if (diy > nside / 2) sy = -boxsize;
                        else if (diy < -(nside / 2)) sy = boxsize;
                        if (diz > nside / 2) sz = -boxsize;
                        else if (diz < -(nside / 2)) sz = boxsize;

                        for (int64_t a = s1; a < e1; a++) {
                            const double xi = x1_sorted[a];
                            const double yi = y1_sorted[a];
                            const double zi = z1_sorted[a];
                            const int64_t li = label1_sorted[a];

                            for (int64_t bidx = s2; bidx < e2; bidx++) {
                                const double dx = (x2_sorted[bidx] + sx) - xi;
                                const double dy = (y2_sorted[bidx] + sy) - yi;
                                const double dz = (z2_sorted[bidx] + sz) - zi;
                                const int64_t b = locate_bin_lut(
                                    r2_eval(dx, dy, dz, use_float32),
                                    rbins2,
                                    nbins,
                                    r2_min,
                                    r2_max,
                                    bin_lut,
                                    lut_n,
                                    inv_lut_step
                                );
                                if (b < 0) {
                                    continue;
                                }
                                local_total[b] += 1ULL;
                                if (label2_sorted[bidx] == li) {
                                    local_1h[b] += 1ULL;
                                } else {
                                    local_2h[b] += 1ULL;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int t = 0; t < nth; t++) {
        const uint64_t *lt = acc_total + (size_t)t * (size_t)nbins;
        const uint64_t *l1 = acc_1h + (size_t)t * (size_t)nbins;
        const uint64_t *l2 = acc_2h + (size_t)t * (size_t)nbins;
        for (i = 0; i < nbins; i++) {
            dd_total[i] += lt[i];
            dd_1h[i] += l1[i];
            dd_2h[i] += l2[i];
        }
    }

    free(rbins2); free(bin_lut);
    free(cell_count1); free(cell_count2);
    free(cell_offset1); free(cell_offset2);
    free(cell_cursor1); free(cell_cursor2);
    free(point_cell1); free(point_cell2);
    free(cell_points1); free(cell_points2);
    free(nonempty2);
    free(nonempty_cells1); free(nonempty1_ix); free(nonempty1_iy); free(nonempty1_iz);
    free(x1_sorted); free(y1_sorted); free(z1_sorted); free(label1_sorted);
    free(x2_sorted); free(y2_sorted); free(z2_sorted); free(label2_sorted);
    free(acc_total); free(acc_1h); free(acc_2h);

    return 0;
}
