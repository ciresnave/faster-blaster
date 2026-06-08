/**
 * @file test_judge_gelsd_complex_regression.c
 * @brief Compare judge CGELSD/ZGELSD results against direct reference exports.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../include/faster-blaster/backend_ids.h"
#include "../include/faster-blaster/backend_plugin.h"
#include "../include/faster-blaster/judge.h"
#include "../include/benchmark_types.h"
#include "../src/backends/reference.h"
#include "../src/judge/judge_corpus.h"
#include "../src/judge/judge_op_ids.h"
#include "../src/judge/judge_profile.h"
#include "../src/judge/judge_types.h"

#ifndef FB_REFERENCE_DLL_DIR
#define FB_REFERENCE_DLL_DIR "../faster-blaster-reference/build-extended"
#endif

extern void fb_judge_register_oracle(const fb_backend_vtable_t *vtable);
extern void fb_judge_register_backend(uint32_t backend_id,
                                      const fb_backend_vtable_t *vtable);
typedef struct {
    fb_judge_case_result_t residual;
    fb_judge_case_result_t orthogonality;
    double kappa_estimate;
} fb_judge_solve_result_t;

extern fb_judge_status_t fb_judge_run_solve_case(
    const fb_backend_vtable_t *oracle,
    const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc,
    fb_judge_solve_result_t *result_out,
    uint64_t *ns_out);

typedef void (*fb_cgelsd_fortran_probe_fn_t)(int *m, int *n, int *nrhs,
                                             fb_complex_float_t *a, int *lda,
                                             fb_complex_float_t *b, int *ldb,
                                             float *s, float *rcond, int *rank,
                                             fb_complex_float_t *work,
                                             int *lwork, float *rwork,
                                             int *iwork, int *info);

typedef void (*fb_zgelsd_fortran_probe_fn_t)(int *m, int *n, int *nrhs,
                                             fb_complex_double_t *a, int *lda,
                                             fb_complex_double_t *b, int *ldb,
                                             double *s, double *rcond, int *rank,
                                             fb_complex_double_t *work,
                                             int *lwork, double *rwork,
                                             int *iwork, int *info);

static void *fb_fn_to_symbol(fb_generic_fn fn)
{
    void *symbol = NULL;
    size_t copy_size = sizeof(symbol) < sizeof(fn) ? sizeof(symbol) : sizeof(fn);
    memcpy(&symbol, &fn, copy_size);
    return symbol;
}

static fb_lib_handle_t load_reference_library(void)
{
#if defined(_WIN32)
    const char *ref_names[] = {
        "faster_blaster_reference.dll",
        "libfaster_blaster_reference.dll",
        NULL
    };
#elif defined(__APPLE__)
    const char *ref_names[] = { "libfaster_blaster_reference.dylib", NULL };
#else
    const char *ref_names[] = { "libfaster_blaster_reference.so", NULL };
#endif
    const char *ref_paths[] = {
        FB_REFERENCE_DLL_DIR,
        ".",
        NULL
    };

    return fb_plugin_load_library(ref_names, ref_paths);
}

typedef struct {
    double scale;
    double ssq;
} fb_scaled_sumsq_t;

static void fb_scaled_sumsq_init(fb_scaled_sumsq_t *acc)
{
    acc->scale = 0.0;
    acc->ssq = 1.0;
}

static void fb_scaled_sumsq_add(fb_scaled_sumsq_t *acc, double absx)
{
    if (absx <= 0.0) {
        return;
    }
    if (acc->scale < absx) {
        double ratio = (acc->scale > 0.0) ? (acc->scale / absx) : 0.0;
        acc->ssq = 1.0 + acc->ssq * ratio * ratio;
        acc->scale = absx;
    } else {
        double ratio = absx / acc->scale;
        acc->ssq += ratio * ratio;
    }
}

static double fb_scaled_sumsq_norm(const fb_scaled_sumsq_t *acc)
{
    return (acc->scale > 0.0) ? (acc->scale * sqrt(acc->ssq)) : 0.0;
}

static double fb_safe_positive_mul_add(double a, double b, double c)
{
    double product;

    if (a <= 0.0 || b <= 0.0) {
        return c;
    }

    {
        int exp_a = 0;
        int exp_b = 0;
        double mant_a = frexp(a, &exp_a);
        double mant_b = frexp(b, &exp_b);
        int product_exp = exp_a + exp_b;
        double product_mant = mant_a * mant_b;

        if (product_mant == 0.0) {
            product = 0.0;
        } else {
            if (fabs(product_mant) >= 1.0) {
                product_mant *= 0.5;
                ++product_exp;
            }
            if (product_exp > DBL_MAX_EXP) {
                product = INFINITY;
            } else if (product_exp < DBL_MIN_EXP - DBL_MANT_DIG) {
                product = 0.0;
            } else {
                product = ldexp(product_mant, product_exp);
            }
        }
    }

    if (isinf(product)) {
        return INFINITY;
    }

    {
        double max_term = fmax(product, c);
        if (max_term == 0.0) {
            return 0.0;
        }
        return max_term * (1.0 + fmin(product, c) / max_term);
    }
}

static double bwerr_cf32_direct(const fb_complex_float_t *A, int m, int n, int lda,
                                const fb_complex_float_t *b, int ldb,
                                const fb_complex_float_t *x, int ldx, int nrhs)
{
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < m; ++i) {
            double rij_re = (double)__real__(b[i * ldb + j]);
            double rij_im = (double)__imag__(b[i * ldb + j]);
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;

            for (int k = 0; k < n; ++k) {
                double ar = (double)__real__(A[i * lda + k]);
                double ai = (double)__imag__(A[i * lda + k]);
                double xr = (double)__real__(x[k * ldx + j]);
                double xi = (double)__imag__(x[k * ldx + j]);
                if (!isfinite(ar) || !isfinite(ai) || !isfinite(xr) || !isfinite(xi)) return NAN;
                rij_re -= ar * xr - ai * xi;
                rij_im -= ar * xi + ai * xr;
            }

            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            fb_scaled_sumsq_add(&r_acc, hypot(rij_re, rij_im));
            fb_scaled_sumsq_add(&b_acc,
                                hypot((double)__real__(b[i * ldb + j]),
                                      (double)__imag__(b[i * ldb + j])));
        }

        for (int i = 0; i < n; ++i) {
            double xr = (double)__real__(x[i * ldx + j]);
            double xi = (double)__imag__(x[i * ldx + j]);
            if (!isfinite(xr) || !isfinite(xi)) return NAN;
            fb_scaled_sumsq_add(&x_acc, hypot(xr, xi));
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            double ar = (double)__real__(A[i * lda + k]);
            double ai = (double)__imag__(A[i * lda + k]);
            if (!isfinite(ar) || !isfinite(ai)) return NAN;
            fb_scaled_sumsq_add(&a_acc, hypot(ar, ai));
        }
    }

    {
        double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                                fb_scaled_sumsq_norm(&x_acc),
                                                fb_scaled_sumsq_norm(&b_acc));
        if (denom < (double)FLT_EPSILON) {
            return (fb_scaled_sumsq_norm(&r_acc) < (double)FLT_EPSILON * (double)FLT_EPSILON) ? 0.0 : 1.0;
        }
        if (isinf(denom)) return 0.0;
        return fb_scaled_sumsq_norm(&r_acc) / denom;
    }
}

static double bwerr_cf64_direct(const fb_complex_double_t *A, int m, int n, int lda,
                                const fb_complex_double_t *b, int ldb,
                                const fb_complex_double_t *x, int ldx, int nrhs)
{
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < m; ++i) {
            double rij_re = __real__(b[i * ldb + j]);
            double rij_im = __imag__(b[i * ldb + j]);
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;

            for (int k = 0; k < n; ++k) {
                double ar = __real__(A[i * lda + k]);
                double ai = __imag__(A[i * lda + k]);
                double xr = __real__(x[k * ldx + j]);
                double xi = __imag__(x[k * ldx + j]);
                if (!isfinite(ar) || !isfinite(ai) || !isfinite(xr) || !isfinite(xi)) return NAN;
                rij_re -= ar * xr - ai * xi;
                rij_im -= ar * xi + ai * xr;
            }

            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            fb_scaled_sumsq_add(&r_acc, hypot(rij_re, rij_im));
            fb_scaled_sumsq_add(&b_acc,
                                hypot(__real__(b[i * ldb + j]),
                                      __imag__(b[i * ldb + j])));
        }

        for (int i = 0; i < n; ++i) {
            double xr = __real__(x[i * ldx + j]);
            double xi = __imag__(x[i * ldx + j]);
            if (!isfinite(xr) || !isfinite(xi)) return NAN;
            fb_scaled_sumsq_add(&x_acc, hypot(xr, xi));
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            double ar = __real__(A[i * lda + k]);
            double ai = __imag__(A[i * lda + k]);
            if (!isfinite(ar) || !isfinite(ai)) return NAN;
            fb_scaled_sumsq_add(&a_acc, hypot(ar, ai));
        }
    }

    {
        double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                                fb_scaled_sumsq_norm(&x_acc),
                                                fb_scaled_sumsq_norm(&b_acc));
        if (denom < DBL_EPSILON) {
            return (fb_scaled_sumsq_norm(&r_acc) < DBL_EPSILON * DBL_EPSILON) ? 0.0 : 1.0;
        }
        if (isinf(denom)) return 0.0;
        return fb_scaled_sumsq_norm(&r_acc) / denom;
    }
}

static size_t count_nonfinite_cf32_matrix(const fb_complex_float_t *x, int rows, int cols, int ldx)
{
    size_t count = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double xr = (double)__real__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            double xi = (double)__imag__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            if (!isfinite(xr) || !isfinite(xi)) {
                ++count;
            }
        }
    }

    return count;
}

static size_t count_nonfinite_cf64_matrix(const fb_complex_double_t *x, int rows, int cols, int ldx)
{
    size_t count = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double xr = __real__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            double xi = __imag__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            if (!isfinite(xr) || !isfinite(xi)) {
                ++count;
            }
        }
    }

    return count;
}

static double max_abs_cf32_matrix(const fb_complex_float_t *x, int rows, int cols, int ldx)
{
    double max_abs = 0.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double xr = (double)__real__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            double xi = (double)__imag__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            if (isfinite(xr) && isfinite(xi)) {
                double absx = hypot(xr, xi);
                if (absx > max_abs) {
                    max_abs = absx;
                }
            }
        }
    }

    return max_abs;
}

static double max_abs_cf64_matrix(const fb_complex_double_t *x, int rows, int cols, int ldx)
{
    double max_abs = 0.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double xr = __real__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            double xi = __imag__(x[(size_t)i * (size_t)ldx + (size_t)j]);
            if (isfinite(xr) && isfinite(xi)) {
                double absx = hypot(xr, xi);
                if (absx > max_abs) {
                    max_abs = absx;
                }
            }
        }
    }

    return max_abs;
}

static int query_size_from_cfloat(fb_complex_float_t query_size)
{
    float real_part = (float)__real__ query_size;
    return (real_part > 1.0f) ? (int)ceilf(real_part) : 1;
}

static int query_size_from_cdouble(fb_complex_double_t query_size)
{
    double real_part = (double)__real__ query_size;
    return (real_part > 1.0) ? (int)ceil(real_part) : 1;
}

static int query_size_from_float(float query_size)
{
    return (query_size > 1.0f) ? (int)ceilf(query_size) : 1;
}

static int query_size_from_double(double query_size)
{
    return (query_size > 1.0) ? (int)ceil(query_size) : 1;
}

static int direct_call_cgelsd(fb_cgelsd_fortran_probe_fn_t fn,
                              int m, int n, int nrhs,
                              const fb_complex_float_t *a_in, int lda_in,
                              const fb_complex_float_t *b_in, int ldb_in,
                              size_t b_bytes,
                              float *s_out, int *rank_out,
                              fb_complex_float_t *b_out)
{
    int m_ = m;
    int n_ = n;
    int nrhs_ = nrhs;
    int lda_col = (m > 0) ? m : 1;
    int solve_rows = (m > n) ? m : n;
    int stored_rows = (ldb_in > 0)
        ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_float_t)))
        : 0;
    int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
    int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
    int lwork = -1;
    int info = 0;
    float rcond = -1.0f;
    fb_complex_float_t work_query = 0.0f;
    float rwork_query[1] = {0.0f};
    int iwork_query[1] = {0};
    fb_complex_float_t *a_query = NULL;
    fb_complex_float_t *b_query = NULL;
    fb_complex_float_t *a_work = NULL;
    fb_complex_float_t *b_work = NULL;
    fb_complex_float_t *work = NULL;
    float *rwork = NULL;
    int *iwork = NULL;

    a_query = (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(*a_query));
    b_query = (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(*b_query));
    a_work = (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(*a_work));
    b_work = (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(*b_work));
    if (!a_query || !b_query || !a_work || !b_work) {
        free(a_query);
        free(b_query);
        free(a_work);
        free(b_work);
        return -101;
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fb_complex_float_t value = a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            a_query[(size_t)i + (size_t)j * (size_t)lda_col] = value;
            a_work[(size_t)i + (size_t)j * (size_t)lda_col] = value;
        }
    }
    for (int i = 0; i < row_copy_rows; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            fb_complex_float_t value = b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            b_query[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
            b_work[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
        }
    }

    fn(&m_, &n_, &nrhs_, a_query, &lda_col, b_query, &ldb_col,
       s_out, &rcond, rank_out, &work_query, &lwork,
       rwork_query, iwork_query, &info);
    if (info == 0) {
        int rwork_len = query_size_from_float(rwork_query[0]);
        int iwork_len = (iwork_query[0] > 0) ? iwork_query[0] : 1;

        lwork = query_size_from_cfloat(work_query);
        work = (fb_complex_float_t *)calloc((size_t)lwork, sizeof(*work));
        rwork = (float *)calloc((size_t)rwork_len, sizeof(*rwork));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(*iwork));
        if (!work || !rwork || !iwork) {
            info = -101;
        } else {
            fn(&m_, &n_, &nrhs_, a_work, &lda_col, b_work, &ldb_col,
               s_out, &rcond, rank_out, work, &lwork, rwork, iwork, &info);
        }
    }

    if (info == 0) {
        for (int i = 0; i < row_copy_rows; ++i) {
            for (int j = 0; j < nrhs; ++j) {
                b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                    b_work[(size_t)i + (size_t)j * (size_t)ldb_col];
            }
        }
    }

    free(a_query);
    free(b_query);
    free(a_work);
    free(b_work);
    free(work);
    free(rwork);
    free(iwork);
    return info;
}

static int direct_call_zgelsd(fb_zgelsd_fortran_probe_fn_t fn,
                              int m, int n, int nrhs,
                              const fb_complex_double_t *a_in, int lda_in,
                              const fb_complex_double_t *b_in, int ldb_in,
                              size_t b_bytes,
                              double *s_out, int *rank_out,
                              fb_complex_double_t *b_out)
{
    int m_ = m;
    int n_ = n;
    int nrhs_ = nrhs;
    int lda_col = (m > 0) ? m : 1;
    int solve_rows = (m > n) ? m : n;
    int stored_rows = (ldb_in > 0)
        ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_double_t)))
        : 0;
    int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
    int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
    int lwork = -1;
    int info = 0;
    double rcond = -1.0;
    fb_complex_double_t work_query = 0.0;
    double rwork_query[1] = {0.0};
    int iwork_query[1] = {0};
    fb_complex_double_t *a_query = NULL;
    fb_complex_double_t *b_query = NULL;
    fb_complex_double_t *a_work = NULL;
    fb_complex_double_t *b_work = NULL;
    fb_complex_double_t *work = NULL;
    double *rwork = NULL;
    int *iwork = NULL;

    a_query = (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(*a_query));
    b_query = (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(*b_query));
    a_work = (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(*a_work));
    b_work = (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(*b_work));
    if (!a_query || !b_query || !a_work || !b_work) {
        free(a_query);
        free(b_query);
        free(a_work);
        free(b_work);
        return -101;
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fb_complex_double_t value = a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            a_query[(size_t)i + (size_t)j * (size_t)lda_col] = value;
            a_work[(size_t)i + (size_t)j * (size_t)lda_col] = value;
        }
    }
    for (int i = 0; i < row_copy_rows; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            fb_complex_double_t value = b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            b_query[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
            b_work[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
        }
    }

    fn(&m_, &n_, &nrhs_, a_query, &lda_col, b_query, &ldb_col,
       s_out, &rcond, rank_out, &work_query, &lwork,
       rwork_query, iwork_query, &info);
    if (info == 0) {
        int rwork_len = query_size_from_double(rwork_query[0]);
        int iwork_len = (iwork_query[0] > 0) ? iwork_query[0] : 1;

        lwork = query_size_from_cdouble(work_query);
        work = (fb_complex_double_t *)calloc((size_t)lwork, sizeof(*work));
        rwork = (double *)calloc((size_t)rwork_len, sizeof(*rwork));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(*iwork));
        if (!work || !rwork || !iwork) {
            info = -101;
        } else {
            fn(&m_, &n_, &nrhs_, a_work, &lda_col, b_work, &ldb_col,
               s_out, &rcond, rank_out, work, &lwork, rwork, iwork, &info);
        }
    }

    if (info == 0) {
        for (int i = 0; i < row_copy_rows; ++i) {
            for (int j = 0; j < nrhs; ++j) {
                b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                    b_work[(size_t)i + (size_t)j * (size_t)ldb_col];
            }
        }
    }

    free(a_query);
    free(b_query);
    free(a_work);
    free(b_work);
    free(work);
    free(rwork);
    free(iwork);
    return info;
}

static int verify_direct_cgelsd(fb_cgelsd_fortran_probe_fn_t fn)
{
    const double threshold = 1.0e-4;
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_judge_status_t status = fb_corpus_generate(FB_OP_CGELSD, FB_SIZE_SMALL, FB_DTYPE_CF32, cases);
    int failures = 0;
    double worst_residual = 0.0;

    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_corpus_generate(CGELSD) failed: %d\n", (int)status);
        return 1;
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        const fb_corpus_case_t *tc = &cases[case_index];
        const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
        const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
        int minmn = (tc->m < tc->n) ? tc->m : tc->n;
        size_t b_bytes = tc->B_elems * sizeof(fb_complex_float_t);
        float *s = (float *)calloc((size_t)minmn, sizeof(float));
        fb_complex_float_t *x = (fb_complex_float_t *)calloc(tc->B_elems, sizeof(fb_complex_float_t));
        int rank = 0;
        int info;
        double residual;

        if (!s || !x) {
            fprintf(stderr, "[FAIL] CGELSD direct probe allocation failure\n");
            free(s);
            free(x);
            failures = 1;
            break;
        }

        info = direct_call_cgelsd(fn, tc->m, tc->n, tc->k,
                                  A0, tc->lda, b0, tc->ldb, b_bytes,
                                  s, &rank, x);
        residual = (info == 0)
            ? bwerr_cf32_direct(A0, tc->m, tc->n, tc->lda, b0, tc->ldb, x, tc->ldb, tc->k)
            : HUGE_VAL;

        if (isfinite(residual) && residual > worst_residual) {
            worst_residual = residual;
        }
        if (info != 0 || !isfinite(residual) || residual > threshold) {
            size_t x_nonfinite = count_nonfinite_cf32_matrix(x, tc->n, tc->k, tc->ldb);
            double x_maxabs = max_abs_cf32_matrix(x, tc->n, tc->k, tc->ldb);
            fprintf(stderr,
                "[FAIL] direct cgelsd case=%d category=%d info=%d rank=%d residual=%.6e threshold=%.6e x_nonfinite=%zu x_maxabs=%.6e\n",
                case_index, (int)tc->category, info, rank, residual, threshold,
                x_nonfinite, x_maxabs);
            ++failures;
        }

        free(s);
        free(x);
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        fb_corpus_case_free(&cases[case_index]);
    }

    printf("[INFO] direct cgelsd worst residual: %.6e\n", worst_residual);
    return failures != 0;
}

static int verify_direct_zgelsd(fb_zgelsd_fortran_probe_fn_t fn)
{
    const double threshold = 1.0e-10;
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_judge_status_t status = fb_corpus_generate(FB_OP_ZGELSD, FB_SIZE_SMALL, FB_DTYPE_CF64, cases);
    int failures = 0;
    double worst_residual = 0.0;

    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_corpus_generate(ZGELSD) failed: %d\n", (int)status);
        return 1;
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        const fb_corpus_case_t *tc = &cases[case_index];
        const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
        const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
        int minmn = (tc->m < tc->n) ? tc->m : tc->n;
        size_t b_bytes = tc->B_elems * sizeof(fb_complex_double_t);
        double *s = (double *)calloc((size_t)minmn, sizeof(double));
        fb_complex_double_t *x = (fb_complex_double_t *)calloc(tc->B_elems, sizeof(fb_complex_double_t));
        int rank = 0;
        int info;
        double residual;

        if (!s || !x) {
            fprintf(stderr, "[FAIL] ZGELSD direct probe allocation failure\n");
            free(s);
            free(x);
            failures = 1;
            break;
        }

        info = direct_call_zgelsd(fn, tc->m, tc->n, tc->k,
                                  A0, tc->lda, b0, tc->ldb, b_bytes,
                                  s, &rank, x);
        residual = (info == 0)
            ? bwerr_cf64_direct(A0, tc->m, tc->n, tc->lda, b0, tc->ldb, x, tc->ldb, tc->k)
            : HUGE_VAL;

        if (isfinite(residual) && residual > worst_residual) {
            worst_residual = residual;
        }
        if (info != 0 || !isfinite(residual) || residual > threshold) {
            size_t x_nonfinite = count_nonfinite_cf64_matrix(x, tc->n, tc->k, tc->ldb);
            double x_maxabs = max_abs_cf64_matrix(x, tc->n, tc->k, tc->ldb);
            fprintf(stderr,
                "[FAIL] direct zgelsd case=%d category=%d info=%d rank=%d residual=%.6e threshold=%.6e x_nonfinite=%zu x_maxabs=%.6e\n",
                case_index, (int)tc->category, info, rank, residual, threshold,
                x_nonfinite, x_maxabs);
            ++failures;
        }

        free(s);
        free(x);
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        fb_corpus_case_free(&cases[case_index]);
    }

    printf("[INFO] direct zgelsd worst residual: %.6e\n", worst_residual);
    return failures != 0;
}

static int verify_casewise_cgelsd(const fb_backend_vtable_t *vtable,
                                  fb_cgelsd_fortran_probe_fn_t fn)
{
    const double threshold = 1.0e-4;
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_judge_status_t status = fb_corpus_generate(FB_OP_CGELSD, FB_SIZE_SMALL, FB_DTYPE_CF32, cases);
    int failures = 0;

    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_corpus_generate(CGELSD casewise) failed: %d\n", (int)status);
        return 1;
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        const fb_corpus_case_t *tc = &cases[case_index];
        const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
        const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
        int minmn = (tc->m < tc->n) ? tc->m : tc->n;
        size_t b_bytes = tc->B_elems * sizeof(fb_complex_float_t);
        float *s = (float *)calloc((size_t)minmn, sizeof(float));
        fb_complex_float_t *x = (fb_complex_float_t *)calloc(tc->B_elems, sizeof(fb_complex_float_t));
        fb_judge_solve_result_t judge_result;
        int rank = 0;
        int info;
        double direct_residual;

        if (!s || !x) {
            fprintf(stderr, "[FAIL] CGELSD casewise allocation failure\n");
            free(s);
            free(x);
            failures = 1;
            break;
        }

        info = direct_call_cgelsd(fn, tc->m, tc->n, tc->k,
                                  A0, tc->lda, b0, tc->ldb, b_bytes,
                                  s, &rank, x);
        direct_residual = (info == 0)
            ? bwerr_cf32_direct(A0, tc->m, tc->n, tc->lda, b0, tc->ldb, x, tc->ldb, tc->k)
            : HUGE_VAL;

        memset(&judge_result, 0, sizeof(judge_result));
        status = fb_judge_run_solve_case(vtable, vtable, tc, &judge_result, NULL);
        if (status != FB_JUDGE_OK) {
            fprintf(stderr, "[FAIL] judge cgelsd case=%d status=%d\n", case_index, (int)status);
            ++failures;
        } else if (!isfinite(direct_residual)) {
            fprintf(stderr,
                    "[FAIL] direct cgelsd case=%d category=%d produced non-finite residual=%.6e\n",
                    case_index, (int)tc->category, direct_residual);
            ++failures;
        } else if (direct_residual <= threshold &&
                   (!isfinite(judge_result.residual.digits) ||
                    judge_result.residual.is_fatal ||
                judge_result.residual.relative_error > threshold ||
                judge_result.residual.digits < 1.0)) {
            fprintf(stderr,
                    "[FAIL] judge cgelsd case=%d category=%d direct=%.6e judge=%.6e digits=%.6f fatal=%d oracle_fatal=%d\n",
                    case_index, (int)tc->category, direct_residual,
                    judge_result.residual.relative_error,
                judge_result.residual.digits,
                    judge_result.residual.is_fatal ? 1 : 0,
                    judge_result.residual.is_oracle_fatal ? 1 : 0);
            ++failures;
        }

        free(s);
        free(x);
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        fb_corpus_case_free(&cases[case_index]);
    }

    return failures != 0;
}

static int verify_casewise_zgelsd(const fb_backend_vtable_t *vtable,
                                  fb_zgelsd_fortran_probe_fn_t fn)
{
    const double threshold = 1.0e-10;
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_judge_status_t status = fb_corpus_generate(FB_OP_ZGELSD, FB_SIZE_SMALL, FB_DTYPE_CF64, cases);
    int failures = 0;

    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_corpus_generate(ZGELSD casewise) failed: %d\n", (int)status);
        return 1;
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        const fb_corpus_case_t *tc = &cases[case_index];
        const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
        const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
        int minmn = (tc->m < tc->n) ? tc->m : tc->n;
        size_t b_bytes = tc->B_elems * sizeof(fb_complex_double_t);
        double *s = (double *)calloc((size_t)minmn, sizeof(double));
        fb_complex_double_t *x = (fb_complex_double_t *)calloc(tc->B_elems, sizeof(fb_complex_double_t));
        fb_judge_solve_result_t judge_result;
        int rank = 0;
        int info;
        double direct_residual;

        if (!s || !x) {
            fprintf(stderr, "[FAIL] ZGELSD casewise allocation failure\n");
            free(s);
            free(x);
            failures = 1;
            break;
        }

        info = direct_call_zgelsd(fn, tc->m, tc->n, tc->k,
                                  A0, tc->lda, b0, tc->ldb, b_bytes,
                                  s, &rank, x);
        direct_residual = (info == 0)
            ? bwerr_cf64_direct(A0, tc->m, tc->n, tc->lda, b0, tc->ldb, x, tc->ldb, tc->k)
            : HUGE_VAL;

        memset(&judge_result, 0, sizeof(judge_result));
        status = fb_judge_run_solve_case(vtable, vtable, tc, &judge_result, NULL);
        if (status != FB_JUDGE_OK) {
            fprintf(stderr, "[FAIL] judge zgelsd case=%d status=%d\n", case_index, (int)status);
            ++failures;
        } else if (!isfinite(direct_residual)) {
            fprintf(stderr,
                    "[FAIL] direct zgelsd case=%d category=%d produced non-finite residual=%.6e\n",
                    case_index, (int)tc->category, direct_residual);
            ++failures;
        } else if (direct_residual <= threshold &&
                   (!isfinite(judge_result.residual.digits) ||
                    judge_result.residual.is_fatal ||
                judge_result.residual.relative_error > threshold ||
                judge_result.residual.digits < 1.0)) {
            fprintf(stderr,
                    "[FAIL] judge zgelsd case=%d category=%d direct=%.6e judge=%.6e digits=%.6f fatal=%d oracle_fatal=%d\n",
                    case_index, (int)tc->category, direct_residual,
                    judge_result.residual.relative_error,
                judge_result.residual.digits,
                    judge_result.residual.is_fatal ? 1 : 0,
                    judge_result.residual.is_oracle_fatal ? 1 : 0);
            ++failures;
        }

        free(s);
        free(x);
    }

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        fb_corpus_case_free(&cases[case_index]);
    }

    return failures != 0;
}

static int verify_manual_accumulated_solve(const fb_backend_vtable_t *vtable,
                                           uint32_t op_id,
                                           fb_dtype_t dtype,
                                           const char *label,
                                           uint8_t min_digits)
{
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_metric_accum_t accum;
    fb_metric_profile_t profile;
    int case_map[FB_PROFILE_ACCUM_MAX_CASES];
    double relerr_map[FB_PROFILE_ACCUM_MAX_CASES];
    fb_judge_status_t status = fb_corpus_generate(op_id, FB_SIZE_SMALL, dtype, cases);
    int scored = 0;

    if (status != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_corpus_generate(%s manual) failed: %d\n", label, (int)status);
        return 1;
    }

    fb_metric_accum_init(&accum);
    memset(&profile, 0, sizeof(profile));
    memset(case_map, 0, sizeof(case_map));
    memset(relerr_map, 0, sizeof(relerr_map));

    for (int case_index = 0; case_index < FB_CORPUS_TOTAL_CASES; ++case_index) {
        fb_judge_solve_result_t judge_result;
        memset(&judge_result, 0, sizeof(judge_result));
        status = fb_judge_run_solve_case(vtable, vtable, &cases[case_index], &judge_result, NULL);
        if (status != FB_JUDGE_OK) {
            fprintf(stderr, "[FAIL] manual %s case=%d status=%d\n", label, case_index, (int)status);
            for (int i = case_index; i < FB_CORPUS_TOTAL_CASES; ++i) {
                fb_corpus_case_free(&cases[i]);
            }
            return 1;
        }
        if (!judge_result.residual.is_oracle_fatal && !isfinite(judge_result.residual.digits)) {
            fprintf(stderr,
                    "[FAIL] manual %s case=%d produced non-finite digits with relerr=%.6e\n",
                    label, case_index, judge_result.residual.relative_error);
            for (int i = case_index; i < FB_CORPUS_TOTAL_CASES; ++i) {
                fb_corpus_case_free(&cases[i]);
            }
            return 1;
        }
        if (!judge_result.residual.is_oracle_fatal) {
            if (accum.count < FB_PROFILE_ACCUM_MAX_CASES) {
                case_map[accum.count] = case_index;
                relerr_map[accum.count] = judge_result.residual.relative_error;
            }
            fb_metric_accum_add(&accum, &judge_result.residual);
            ++scored;
        }
        fb_corpus_case_free(&cases[case_index]);
    }

    if (scored == 0) {
        fprintf(stderr, "[FAIL] manual %s had no scored cases\n", label);
        return 1;
    }

    fb_metric_accum_finish(&accum, &profile);
    printf("[INFO] manual %s guaranteed digits: %u\n", label, profile.guaranteed_digits);
    if (profile.guaranteed_digits < min_digits) {
        fprintf(stderr,
                "[FAIL] manual %s guaranteed digits stayed below floor: %u\n",
                label, profile.guaranteed_digits);
        for (uint32_t i = 0; i < accum.count; ++i) {
            fprintf(stderr,
                    "[INFO] manual %s accum[%u]: case=%d relerr=%.6e digits=%.6f fatal=%d oracle_fatal=%d\n",
                    label, i, case_map[i], relerr_map[i], accum.digits[i],
                    accum.is_fatal[i] ? 1 : 0,
                    accum.is_oracle_fatal[i] ? 1 : 0);
        }
        return 1;
    }

    return 0;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fb_lib_handle_t ref_h = load_reference_library();
    fb_precision_profile_t cgelsd_profile;
    fb_precision_profile_t zgelsd_profile;
    const fb_backend_vtable_t *vtable;
    void *slot_cgelsd = NULL;
    void *slot_zgelsd = NULL;
    fb_cgelsd_fortran_probe_fn_t direct_cgelsd = NULL;
    fb_zgelsd_fortran_probe_fn_t direct_zgelsd = NULL;
    int failed = 0;

    if (!ref_h) {
        fprintf(stderr, "[FAIL] Could not load reference DLL from %s\n", FB_REFERENCE_DLL_DIR);
        return 1;
    }

    fb_reference_init(ref_h);
    vtable = fb_reference_backend();
    if (!vtable) {
        fprintf(stderr, "[FAIL] fb_reference_backend() returned NULL\n");
        return 1;
    }

    direct_cgelsd = (fb_cgelsd_fortran_probe_fn_t)fb_plugin_get_symbol(ref_h, "cgelsd_");
    direct_zgelsd = (fb_zgelsd_fortran_probe_fn_t)fb_plugin_get_symbol(ref_h, "zgelsd_");
    slot_cgelsd = fb_fn_to_symbol(vtable->ext_ops[FB_OP_CGELSD][FB_CONV_FORTRAN]);
    slot_zgelsd = fb_fn_to_symbol(vtable->ext_ops[FB_OP_ZGELSD][FB_CONV_FORTRAN]);

    if (!direct_cgelsd || !direct_zgelsd) {
        fprintf(stderr, "[FAIL] Direct cgelsd_/zgelsd_ exports not found\n");
        return 1;
    }

    printf("[INFO] cgelsd slot=%p direct=%p\n", slot_cgelsd, (void *)direct_cgelsd);
    printf("[INFO] zgelsd slot=%p direct=%p\n", slot_zgelsd, (void *)direct_zgelsd);

    if (slot_cgelsd != (void *)direct_cgelsd) {
        fprintf(stderr, "[FAIL] CGELSD Fortran slot does not bind cgelsd_ directly\n");
        failed = 1;
    }
    if (slot_zgelsd != (void *)direct_zgelsd) {
        fprintf(stderr, "[FAIL] ZGELSD Fortran slot does not bind zgelsd_ directly\n");
        failed = 1;
    }

    if (fb_judge_init(".") != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_judge_init() failed\n");
        return 1;
    }
    fb_judge_register_oracle(vtable);
    fb_judge_register_backend(FB_BACKEND_ID_REFERENCE, vtable);

    memset(&cgelsd_profile, 0, sizeof(cgelsd_profile));
    if (fb_judge_run(FB_OP_CGELSD, FB_BACKEND_ID_REFERENCE, 0,
                     FB_SIZE_SMALL, FB_DTYPE_CF32, false,
                     &cgelsd_profile) != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_judge_run(CGELSD) failed\n");
        failed = 1;
    }

    memset(&zgelsd_profile, 0, sizeof(zgelsd_profile));
    if (fb_judge_run(FB_OP_ZGELSD, FB_BACKEND_ID_REFERENCE, 0,
                     FB_SIZE_SMALL, FB_DTYPE_CF64, false,
                     &zgelsd_profile) != FB_JUDGE_OK) {
        fprintf(stderr, "[FAIL] fb_judge_run(ZGELSD) failed\n");
        failed = 1;
    }

    printf("[INFO] judge cgelsd guaranteed digits: %u\n", cgelsd_profile.residual.guaranteed_digits);
    printf("[INFO] judge zgelsd guaranteed digits: %u\n", zgelsd_profile.residual.guaranteed_digits);

    if (verify_direct_cgelsd(direct_cgelsd) != 0) {
        failed = 1;
    }
    if (verify_direct_zgelsd(direct_zgelsd) != 0) {
        failed = 1;
    }
    fprintf(stderr, "[INFO] starting casewise cgelsd probe\n");
    if (verify_casewise_cgelsd(vtable, direct_cgelsd) != 0) {
        failed = 1;
    }
    fprintf(stderr, "[INFO] starting casewise zgelsd probe\n");
    if (verify_casewise_zgelsd(vtable, direct_zgelsd) != 0) {
        failed = 1;
    }
    if (verify_manual_accumulated_solve(vtable, FB_OP_CGELSD, FB_DTYPE_CF32, "cgelsd", 4u) != 0) {
        failed = 1;
    }
    if (verify_manual_accumulated_solve(vtable, FB_OP_ZGELSD, FB_DTYPE_CF64, "zgelsd", 10u) != 0) {
        failed = 1;
    }

    if (cgelsd_profile.residual.guaranteed_digits < 4u) {
        fprintf(stderr,
                "[FAIL] Judge CGELSD residual digits stayed below floor despite direct export probe\n");
        failed = 1;
    }
    if (zgelsd_profile.residual.guaranteed_digits < 10u) {
        fprintf(stderr,
                "[FAIL] Judge ZGELSD residual digits stayed below floor despite direct export probe\n");
        failed = 1;
    }

    fb_judge_shutdown();

    if (!failed) {
        printf("[PASS] Judge CGELSD/ZGELSD match direct reference exports on the small corpus\n");
    }
    return failed ? 1 : 0;
}