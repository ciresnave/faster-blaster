/**
 * @file judge_direct.c
 * @brief FB_JUDGE_DIRECT archetype evaluator.
 *
 * Implements oracle-vs-candidate comparison for BLAS L1, L2, and L3 operations
 * with direct output comparison (no reconstruction needed).
 *
 * Phase 1 coverage:
 *   BLAS L1 real: SAXPY, DAXPY, SSCAL, DSCAL, SCOPY, DCOPY, SSWAP, DSWAP
 *                 SDOT, DDOT, SNRM2, DNRM2, SASUM, DASUM, ISAMAX, IDAMAX
 *   BLAS L2 real: SGEMV, DGEMV
 *   BLAS L3 real: SGEMM, DGEMM
 *
 * Complex variants (CAXPY, ZGEMM, etc.) return FB_JUDGE_ERR_NOT_IMPL; they
 * will be added in Phase 2.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_direct.h"
#include "judge_norm.h"
#include "judge_op_ids.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Constants
 * ========================================================================= */

/** Number of un-timed warmup calls before timing window. */
#define FB_JUDGE_WARMUP_RUNS   2
/** Number of timed calls; best (minimum) is recorded. */
#define FB_JUDGE_TIMING_RUNS   5
/** Small replicated batch count used to exercise batched runners. */
#define FB_JUDGE_BATCH_COUNT   2

/* =========================================================================
 * Private utilities
 * ========================================================================= */

/**
 * Allocate and deep-copy 'n' elements of 'elem_size' bytes from 'src'.
 * Returns NULL on allocation failure.
 */
static void *clone_buf(const void *src, size_t n, size_t elem_size)
{
    if (n == 0) return NULL;
    void *dst = malloc(n * elem_size);
    if (dst && src)
        memcpy(dst, src, n * elem_size);
    return dst;
}

static void *repeat_buf(const void *src, size_t n, size_t elem_size, int repeat)
{
    if (!src || n == 0 || elem_size == 0 || repeat <= 0) return NULL;

    size_t block_bytes = n * elem_size;
    size_t total_bytes = block_bytes * (size_t)repeat;
    unsigned char *dst = (unsigned char *)malloc(total_bytes);
    if (!dst) return NULL;

    for (int batch_index = 0; batch_index < repeat; batch_index++) {
        memcpy(dst + ((size_t)batch_index * block_bytes), src, block_bytes);
    }

    return dst;
}

static const void **make_const_batch_ptrs(const void *base, size_t stride_bytes,
                                          int batch_count)
{
    if (!base || stride_bytes == 0 || batch_count <= 0) return NULL;

    const unsigned char *bytes = (const unsigned char *)base;
    const void **ptrs = (const void **)malloc((size_t)batch_count * sizeof(*ptrs));
    if (!ptrs) return NULL;

    for (int batch_index = 0; batch_index < batch_count; batch_index++) {
        ptrs[batch_index] = bytes + ((size_t)batch_index * stride_bytes);
    }

    return ptrs;
}

static void **make_batch_ptrs(void *base, size_t stride_bytes, int batch_count)
{
    if (!base || stride_bytes == 0 || batch_count <= 0) return NULL;

    unsigned char *bytes = (unsigned char *)base;
    void **ptrs = (void **)malloc((size_t)batch_count * sizeof(*ptrs));
    if (!ptrs) return NULL;

    for (int batch_index = 0; batch_index < batch_count; batch_index++) {
        ptrs[batch_index] = bytes + ((size_t)batch_index * stride_bytes);
    }

    return ptrs;
}

/** Mark result as candidate fatal (NaN/Inf/crash). */
static void result_fatal(fb_judge_case_result_t *r)
{
    r->digits         = -1.0;
    r->relative_error = 1.0;
    r->is_fatal       = true;
    r->is_oracle_fatal = false;
}

/** Mark result as oracle fatal — caller should halt and fix reference. */
static void result_oracle_fatal(fb_judge_case_result_t *r)
{
    r->digits         = -1.0;
    r->relative_error = 1.0;
    r->is_fatal       = true;
    r->is_oracle_fatal = true;
}

/** Fill result from a pre-computed raw relative error. */
static void result_from_relerr(fb_judge_case_result_t *r, double relerr)
{
    r->relative_error = relerr;
    r->digits         = fb_judge_digits(relerr);
    r->is_fatal       = !isfinite(relerr) || (relerr >= 1.0 && !isfinite(r->digits));
    r->is_oracle_fatal = false;
}

/**
 * Compute relative error for a scalar result.
 * |cand - ref| / max(|ref|, eps * scale_hint)
 */
static double scalar_relerr_f32(float cand, float ref, double scale_hint)
{
    double delta = fabs((double)cand - (double)ref);
    double denom = fmax(fabs((double)ref), (double)FLT_EPSILON * fmax(scale_hint, 1.0));
    return delta / denom;
}

static double scalar_relerr_f64(double cand, double ref, double scale_hint)
{
    double delta = fabs(cand - ref);
    double denom = fmax(fabs(ref), DBL_EPSILON * fmax(scale_hint, 1.0));
    return delta / denom;
}

/* =========================================================================
 * Runner type
 * ========================================================================= */

/**
 * Internal per-op runner signature.
 * Responsible for allocating scratch, running oracle + candidate, timing,
 * and filling *res and *ns_out.
 */
typedef fb_judge_status_t (*fb_direct_runner_fn)(
    const fb_backend_vtable_t *oracle,
    const fb_backend_vtable_t *cand,
    const fb_corpus_case_t    *tc,
    fb_judge_case_result_t    *res,
    uint64_t                  *ns_out    /* nullable; receives best timing_run ns */
);

/* =========================================================================
 * BLAS L1 — Vector output (AXPY, SCAL, COPY, SWAP)
 *
 * Corpus layout for L1 real ops:
 *   tc->A       : x input vector  (A_elems = n)
 *   tc->B       : y initial value (B_elems = n)  [not used by SCAL/COPY]
 *   tc->n       : vector length
 *   tc->alpha   : scalar (double, cast to float for 's' variants)
 * ========================================================================= */

/* ---- SAXPY: y = alpha*x + y ------------------------------------------ */
static fb_judge_status_t run_saxpy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->saxpy || !cand->saxpy) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    float   alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A;

    /* Oracle run ---------------------------------------------------------- */
    float *y_oracle = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->saxpy(n, alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_F32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Candidate: single correctness run ---------------------------------- */
    float *y_cand = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->saxpy(n, alpha, x, 1, y_cand, 1);

    double scale = fb_norm_frob_f32(y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_F32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(y_cand);

    /* Timing run (data clobbering is intentional — latency only) --------- */
    if (ns_out) {
        float *y_time = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->saxpy(n, alpha, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->saxpy(n, alpha, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- DAXPY -------------------------------------------------------------- */
static fb_judge_status_t run_daxpy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->daxpy || !cand->daxpy) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    double   alpha = tc->alpha;
    const double *x = (const double *)tc->A;

    double *y_oracle = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->daxpy(n, alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_F64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *y_cand = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->daxpy(n, alpha, x, 1, y_cand, 1);

    double scale  = fb_norm_frob_f64((const double *)tc->B, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_F64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(y_cand);

    if (ns_out) {
        double *y_time = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->daxpy(n, alpha, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->daxpy(n, alpha, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- SSCAL: x = alpha*x  (in-place) ------------------------------------ */
static fb_judge_status_t run_sscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sscal || !cand->sscal) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    float   alpha = (float)tc->alpha;

    float *x_oracle = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_F32)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *x_cand = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sscal(n, alpha, x_cand, 1);

    double scale  = fb_norm_frob_f32(x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_F32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(x_cand);

    if (ns_out) {
        float *x_time = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->sscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->sscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(x_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- DSCAL -------------------------------------------------------------- */
static fb_judge_status_t run_dscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dscal || !cand->dscal) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    double  alpha = tc->alpha;

    double *x_oracle = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_F64)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *x_cand = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dscal(n, alpha, x_cand, 1);

    double scale  = fb_norm_frob_f64(x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_F64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(x_cand);

    if (ns_out) {
        double *x_time = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(x_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- SCOPY: y = x  (y is full output, independent of initial value) ----- */
static fb_judge_status_t run_scopy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->scopy || !cand->scopy) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const float *x = (const float *)tc->A;

    float *y_oracle = (float *)malloc((size_t)n * sizeof(float));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->scopy(n, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_F32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *y_cand = (float *)malloc((size_t)n * sizeof(float));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->scopy(n, x, 1, y_cand, 1);

    double scale  = fb_norm_frob_f32(x, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_F32, scale);
    result_from_relerr(res, relerr);
    free(y_cand);

    if (ns_out) {
        float *y_time = (float *)malloc((size_t)n * sizeof(float));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->scopy(n, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->scopy(n, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- DCOPY -------------------------------------------------------------- */
static fb_judge_status_t run_dcopy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dcopy || !cand->dcopy) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const double *x = (const double *)tc->A;

    double *y_oracle = (double *)malloc((size_t)n * sizeof(double));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dcopy(n, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_F64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *y_cand = (double *)malloc((size_t)n * sizeof(double));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dcopy(n, x, 1, y_cand, 1);

    double scale  = fb_norm_frob_f64(x, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_F64, scale);
    result_from_relerr(res, relerr);
    free(y_cand);

    if (ns_out) {
        double *y_time = (double *)malloc((size_t)n * sizeof(double));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dcopy(n, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dcopy(n, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- SSWAP: x <-> y  (both outputs) ------------------------------------ */
static fb_judge_status_t run_sswap(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sswap || !cand->sswap) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */

    /* Oracle: clone both x and y, swap them. */
    float *xo = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
    float *yo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sswap(n, xo, 1, yo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32) ||
        fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Candidate: similarly clone and swap. */
    float *xc = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
    float *yc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sswap(n, xc, 1, yc, 1);

    /* Compare combined: max error across both output vectors. */
    double sx = fb_norm_frob_f32(tc->A, (size_t)n);  /* scale = original vector norms */
    double sy = fb_norm_frob_f32(tc->B, (size_t)n);
    double errx = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, sx);
    double erry = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, sy);
    result_from_relerr(res, fmax(errx, erry));
    free(xc); free(yc);

    if (ns_out) {
        float *xt = (float *)clone_buf(tc->A, (size_t)n, sizeof(float));
        float *yt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->sswap(n, xt, 1, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->sswap(n, xt, 1, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
        free(xt); free(yt);
    }

    free(xo); free(yo);
    return FB_JUDGE_OK;
}

/* ---- DSWAP -------------------------------------------------------------- */
static fb_judge_status_t run_dswap(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dswap || !cand->dswap) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */

    double *xo = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
    double *yo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dswap(n, xo, 1, yo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64) ||
        fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *xc = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
    double *yc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dswap(n, xc, 1, yc, 1);

    double sx = fb_norm_frob_f64((const double *)tc->A, (size_t)n);
    double sy = fb_norm_frob_f64((const double *)tc->B, (size_t)n);
    result_from_relerr(res, fmax(
        fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, sx),
        fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, sy)));
    free(xc); free(yc);

    if (ns_out) {
        double *xt = (double *)clone_buf(tc->A, (size_t)n, sizeof(double));
        double *yt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dswap(n, xt, 1, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dswap(n, xt, 1, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
        free(xt); free(yt);
    }

    free(xo); free(yo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * BLAS L1 — Scalar output (DOT, NRM2, ASUM)
 *
 * Both oracle and candidate use the SAME read-only input buffers.
 * No cloning needed for correctness; timing reuses without re-init.
 * ========================================================================= */

/* ---- SDOT --------------------------------------------------------------- */
static fb_judge_status_t run_sdot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sdot || !cand->sdot) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const float *x = (const float *)tc->A;
    const float *y = (const float *)tc->B;

    float oracle_val = oracle->sdot(n, x, 1, y, 1);
    if (!isfinite((double)oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    float cand_val = cand->sdot(n, x, 1, y, 1);

    /* scale = geometric mean of input norms for DOT result scaling */
    double sx = fb_norm_frob_f32(x, (size_t)n);
    double sy = fb_norm_frob_f32(y, (size_t)n);
    double scale = sx * sy;  /* upper bound on |dot| by Cauchy-Schwarz */
    result_from_relerr(res, scalar_relerr_f32(cand_val, oracle_val, scale));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->sdot(n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sdot(n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- DDOT --------------------------------------------------------------- */
static fb_judge_status_t run_ddot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ddot || !cand->ddot) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const double *x = (const double *)tc->A;
    const double *y = (const double *)tc->B;

    double oracle_val = oracle->ddot(n, x, 1, y, 1);
    if (!isfinite(oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    double cand_val = cand->ddot(n, x, 1, y, 1);

    double sx = fb_norm_frob_f64(x, (size_t)n);
    double sy = fb_norm_frob_f64(y, (size_t)n);
    result_from_relerr(res, scalar_relerr_f64(cand_val, oracle_val, sx * sy));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->ddot(n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->ddot(n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SNRM2 -------------------------------------------------------------- */
static fb_judge_status_t run_snrm2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->snrm2 || !cand->snrm2) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const float *x = (const float *)tc->A;

    float oracle_val = oracle->snrm2(n, x, 1);
    if (!isfinite((double)oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    float cand_val = cand->snrm2(n, x, 1);
    result_from_relerr(res,
        scalar_relerr_f32(cand_val, oracle_val, (double)oracle_val));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->snrm2(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->snrm2(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- DNRM2 -------------------------------------------------------------- */
static fb_judge_status_t run_dnrm2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dnrm2 || !cand->dnrm2) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const double *x = (const double *)tc->A;

    double oracle_val = oracle->dnrm2(n, x, 1);
    if (!isfinite(oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    double cand_val = cand->dnrm2(n, x, 1);
    result_from_relerr(res, scalar_relerr_f64(cand_val, oracle_val, oracle_val));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->dnrm2(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dnrm2(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SASUM -------------------------------------------------------------- */
static fb_judge_status_t run_sasum(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sasum || !cand->sasum) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const float *x = (const float *)tc->A;

    float oracle_val = oracle->sasum(n, x, 1);
    if (!isfinite((double)oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    float cand_val = cand->sasum(n, x, 1);
    result_from_relerr(res,
        scalar_relerr_f32(cand_val, oracle_val, (double)oracle_val));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->sasum(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sasum(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- DASUM -------------------------------------------------------------- */
static fb_judge_status_t run_dasum(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dasum || !cand->dasum) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const double *x = (const double *)tc->A;

    double oracle_val = oracle->dasum(n, x, 1);
    if (!isfinite(oracle_val)) { result_oracle_fatal(res); return FB_JUDGE_OK; }

    double cand_val = cand->dasum(n, x, 1);
    result_from_relerr(res, scalar_relerr_f64(cand_val, oracle_val, oracle_val));

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->dasum(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dasum(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * BLAS L1 — Index output (ISAMAX, IDAMAX)
 *
 * Scoring:  exact index match → perfect (16 digits).
 *           Mismatch but |x[cand]| ≈ |x[oracle]| → partial credit.
 *           Mismatch with discernible difference → 0 digits.
 * ========================================================================= */

/* ---- ISAMAX ------------------------------------------------------------- */
static fb_judge_status_t run_isamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->isamax || !cand->isamax) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const float *x = (const float *)tc->A;

    int oracle_idx = oracle->isamax(n, x, 1);
    int cand_idx   = cand->isamax(n, x, 1);

    /* cblas_isamax in faster-blaster-reference returns 1-based index (1..n).
     * Guard: reject 0 (empty result) and anything > n. */
    if (oracle_idx < 1 || oracle_idx > n) {
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    if (cand_idx == oracle_idx) {
        /* Perfect match */
        res->digits         = 16.0;
        res->relative_error = 0.0;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    } else {
        /* Check whether the magnitudes are within tolerance (tie-breaking). */
        float abs_ref  = fabsf(x[oracle_idx - 1]);  /* 1-based → 0-based */
        float abs_cand = (cand_idx >= 1 && cand_idx <= n) ? fabsf(x[cand_idx - 1]) : 0.0f;
        double err = fabs((double)(abs_ref - abs_cand)) /
                     fmax((double)abs_ref, (double)FLT_EPSILON);
        result_from_relerr(res, err < (double)FLT_EPSILON * 4.0 ? 0.0 : err);
    }

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->isamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->isamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- IDAMAX ------------------------------------------------------------- */
static fb_judge_status_t run_idamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->idamax || !cand->idamax) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused) */
    const double *x = (const double *)tc->A;

    int oracle_idx = oracle->idamax(n, x, 1);
    int cand_idx   = cand->idamax(n, x, 1);

    /* 1-based index from reference cblas_idamax. */
    if (oracle_idx < 1 || oracle_idx > n) {
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    if (cand_idx == oracle_idx) {
        res->digits = 16.0; res->relative_error = 0.0;
        res->is_fatal = false; res->is_oracle_fatal = false;
    } else {
        double abs_ref  = fabs(x[oracle_idx - 1]);  /* 1-based → 0-based */
        double abs_cand = (cand_idx >= 1 && cand_idx <= n) ? fabs(x[cand_idx - 1]) : 0.0;
        double err = fabs(abs_ref - abs_cand) /
                     fmax(abs_ref, DBL_EPSILON);
        result_from_relerr(res, err < DBL_EPSILON * 4.0 ? 0.0 : err);
    }

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            (void)cand->idamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->idamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * BLAS L2 — GEMV (matrix-vector multiply)
 *
 * Corpus layout for GEMV:
 *   tc->A      : A matrix  (row-major, m rows × n cols, A_elems = m * lda)
 *   tc->B      : x vector  (B_elems = n for NO_TRANS)
 *   tc->C_init : y initial (C_elems = m)
 *   tc->m, tc->n, tc->lda, tc->alpha, tc->beta
 * ========================================================================= */

/* ---- SGEMV -------------------------------------------------------------- */
static fb_judge_status_t run_sgemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgemv || !cand->sgemv) return FB_JUDGE_ERR_NOT_IMPL;

    int      m    = (int)tc->m;
    int      n    = (int)tc->n;
    int      lda  = (int)tc->lda;
    float        alpha = (float)tc->alpha;
    float        beta  = (float)tc->beta;
    const float *A    = (const float *)tc->A;
    const float *x    = (const float *)tc->B;

    float *y_oracle = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                  alpha, A, lda, x, 1, beta, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)m, FB_DTYPE_F32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *y_cand = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                alpha, A, lda, x, 1, beta, y_cand, 1);

    double scale  = fb_norm_frob_f32(y_oracle, (size_t)m);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)m, FB_DTYPE_F32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)m, FB_DTYPE_F32)) res->is_fatal = true;
    free(y_cand);

    if (ns_out) {
        float *y_time = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->sgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                            alpha, A, lda, x, 1, beta, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->sgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                            alpha, A, lda, x, 1, beta, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- DGEMV -------------------------------------------------------------- */
static fb_judge_status_t run_dgemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgemv || !cand->dgemv) return FB_JUDGE_ERR_NOT_IMPL;

    int       m    = (int)tc->m;
    int       n    = (int)tc->n;
    int       lda  = (int)tc->lda;
    double        alpha = tc->alpha;
    double        beta  = tc->beta;
    const double *A    = (const double *)tc->A;
    const double *x    = (const double *)tc->B;

    double *y_oracle = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                  alpha, A, lda, x, 1, beta, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)m, FB_DTYPE_F64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *y_cand = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                alpha, A, lda, x, 1, beta, y_cand, 1);

    double scale  = fb_norm_frob_f64(y_oracle, (size_t)m);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)m, FB_DTYPE_F64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)m, FB_DTYPE_F64)) res->is_fatal = true;
    free(y_cand);

    if (ns_out) {
        double *y_time = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                            alpha, A, lda, x, 1, beta, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n,
                            alpha, A, lda, x, 1, beta, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(y_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(y_oracle);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * BLAS L3 — GEMM (matrix-matrix multiply)
 *
 * Corpus layout for GEMM:
 *   tc->A      : A matrix  (A_elems = m * lda)
 *   tc->B      : B matrix  (B_elems = k * ldb)
 *   tc->C_init : C initial (C_elems = m * ldc)
 *   tc->m, tc->n, tc->k, tc->lda, tc->ldb, tc->ldc, tc->alpha, tc->beta
 * ========================================================================= */

#define FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER(fn_name, field) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef int (*fn_t)(const char *, const int *, const int *, const int *); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    char identifier = 'A'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int oracle_val = oracle_fn(&identifier, &m, &n, &k); \
    int cand_val = cand_fn(&identifier, &m, &n, &k); \
    result_from_relerr(res, (oracle_val == cand_val) ? 0.0 : 1.0); \
    if (ns_out) { \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
            (void)cand_fn(&identifier, &m, &n, &k); \
        uint64_t best = UINT64_MAX; \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            uint64_t t0 = fb_judge_time_ns(); \
            (void)cand_fn(&identifier, &m, &n, &k); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER(fn_name, field) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef int (*fn_t)(int, int, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int identifier = 0; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int oracle_val = oracle_fn(identifier, m, n, k); \
    int cand_val = cand_fn(identifier, m, n, k); \
    result_from_relerr(res, (oracle_val == cand_val) ? 0.0 : 1.0); \
    if (ns_out) { \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
            (void)cand_fn(identifier, m, n, k); \
        uint64_t best = UINT64_MAX; \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            uint64_t t0 = fb_judge_time_ns(); \
            (void)cand_fn(identifier, m, n, k); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER(run_sgemm_pack_get_size, sgemm_pack_get_size)
FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER(run_dgemm_pack_get_size, dgemm_pack_get_size)
FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER(run_cgemm_pack_get_size, cgemm_pack_get_size)
FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER(run_zgemm_pack_get_size, zgemm_pack_get_size)
FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER(run_cblas_sgemm_pack_get_size, cblas_sgemm_pack_get_size)
FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER(run_cblas_dgemm_pack_get_size, cblas_dgemm_pack_get_size)
FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER(run_cblas_cgemm_pack_get_size, cblas_cgemm_pack_get_size)
FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER(run_cblas_zgemm_pack_get_size, cblas_zgemm_pack_get_size)

#undef FB_DEFINE_GEMM_PACK_GET_SIZE_RUNNER
#undef FB_DEFINE_CBLAS_GEMM_PACK_GET_SIZE_RUNNER

typedef void (*fb_sgemm_compute_fn_t)(const char *, const char *, const int *,
                                      const int *, const int *, const float *,
                                      const int *, const float *, const int *,
                                      const float *, float *, const int *);
typedef void (*fb_dgemm_compute_fn_t)(const char *, const char *, const int *,
                                      const int *, const int *, const double *,
                                      const int *, const double *, const int *,
                                      const double *, double *, const int *);
typedef void (*fb_cgemm_compute_fn_t)(const char *, const char *, const int *,
                                      const int *, const int *,
                                      const fb_complex_float_t *, const int *,
                                      const fb_complex_float_t *, const int *,
                                      const fb_complex_float_t *,
                                      fb_complex_float_t *, const int *);
typedef void (*fb_zgemm_compute_fn_t)(const char *, const char *, const int *,
                                      const int *, const int *,
                                      const fb_complex_double_t *, const int *,
                                      const fb_complex_double_t *, const int *,
                                      const fb_complex_double_t *,
                                      fb_complex_double_t *, const int *);

typedef void (*fb_cblas_sgemm_compute_fn_t)(int, int, int, int, int, int,
                                            const float *, int,
                                            const float *, int, float,
                                            float *, int);
typedef void (*fb_cblas_dgemm_compute_fn_t)(int, int, int, int, int, int,
                                            const double *, int,
                                            const double *, int, double,
                                            double *, int);
typedef void (*fb_cblas_cgemm_compute_fn_t)(int, int, int, int, int, int,
                                            const fb_complex_float_t *, int,
                                            const fb_complex_float_t *, int,
                                            fb_complex_float_t,
                                            fb_complex_float_t *, int);
typedef void (*fb_cblas_zgemm_compute_fn_t)(int, int, int, int, int, int,
                                            const fb_complex_double_t *, int,
                                            const fb_complex_double_t *, int,
                                            fb_complex_double_t,
                                            fb_complex_double_t *, int);

#define FB_DEFINE_GEMM_COMPUTE_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta = (scalar_type)tc->beta; \
    size_t C_sz = tc->C_elems; \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Co, &ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Cc, &ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn(Co, m, n, ldc))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM_COMPUTE_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta; __real__(beta) = (real_type)tc->beta; __imag__(beta) = 0; \
    size_t C_sz = (size_t)(m * ldc); \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Co, &ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Cc, &ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn((const real_type *)Co, C_sz))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int order = FB_LAYOUT_ROW_MAJOR; \
    int trans = FB_NO_TRANS; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta = (scalar_type)tc->beta; \
    size_t C_sz = tc->C_elems; \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Co, ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Cc, ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn(Co, m, n, ldc))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Ct, ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Ct, ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int order = FB_LAYOUT_ROW_MAJOR; \
    int trans = FB_NO_TRANS; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta; __real__(beta) = (real_type)tc->beta; __imag__(beta) = 0; \
    size_t C_sz = (size_t)(m * ldc); \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Co, ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Cc, ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn((const real_type *)Co, C_sz))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Ct, ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(order, trans, trans, m, n, k, A, lda, B, ldb, beta, Ct, ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

FB_DEFINE_GEMM_COMPUTE_RUNNER_REAL(run_sgemm_compute, sgemm_compute, fb_sgemm_compute_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_GEMM_COMPUTE_RUNNER_REAL(run_dgemm_compute, dgemm_compute, fb_dgemm_compute_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_GEMM_COMPUTE_RUNNER_COMPLEX(run_cgemm_compute, cgemm_compute, fb_cgemm_compute_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_GEMM_COMPUTE_RUNNER_COMPLEX(run_zgemm_compute, zgemm_compute, fb_zgemm_compute_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)
FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_REAL(run_cblas_sgemm_compute, cblas_sgemm_compute, fb_cblas_sgemm_compute_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_REAL(run_cblas_dgemm_compute, cblas_dgemm_compute, fb_cblas_dgemm_compute_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_COMPLEX(run_cblas_cgemm_compute, cblas_cgemm_compute, fb_cblas_cgemm_compute_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_COMPLEX(run_cblas_zgemm_compute, cblas_zgemm_compute, fb_cblas_zgemm_compute_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

typedef void *(*fb_gemm_ptr_getter_fn_t)(void);

#define FB_DEFINE_GEMM_PTR_RUNNER_REAL(fn_name, field, compute_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fb_gemm_ptr_getter_fn_t oracle_get = (fb_gemm_ptr_getter_fn_t)oracle->field; \
    fb_gemm_ptr_getter_fn_t cand_get = (fb_gemm_ptr_getter_fn_t)cand->field; \
    compute_type oracle_fn = (compute_type)oracle_get(); \
    if (!oracle_fn) { result_oracle_fatal(res); return FB_JUDGE_OK; } \
    compute_type cand_fn = (compute_type)cand_get(); \
    if (!cand_fn) { result_fatal(res); return FB_JUDGE_OK; } \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta = (scalar_type)tc->beta; \
    size_t C_sz = tc->C_elems; \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Co, &ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Cc, &ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn(Co, m, n, ldc))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM_PTR_RUNNER_COMPLEX(fn_name, field, compute_type, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fb_gemm_ptr_getter_fn_t oracle_get = (fb_gemm_ptr_getter_fn_t)oracle->field; \
    fb_gemm_ptr_getter_fn_t cand_get = (fb_gemm_ptr_getter_fn_t)cand->field; \
    compute_type oracle_fn = (compute_type)oracle_get(); \
    if (!oracle_fn) { result_oracle_fatal(res); return FB_JUDGE_OK; } \
    compute_type cand_fn = (compute_type)cand_get(); \
    if (!cand_fn) { result_fatal(res); return FB_JUDGE_OK; } \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    scalar_type beta; __real__(beta) = (real_type)tc->beta; __imag__(beta) = 0; \
    size_t C_sz = (size_t)(m * ldc); \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Co, &ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Cc, &ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn((const real_type *)Co, C_sz))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&trans, &trans, &m, &n, &k, A, &lda, B, &ldb, &beta, Ct, &ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

FB_DEFINE_GEMM_PTR_RUNNER_REAL(run_sgemm_ptr, sgemm_ptr, fb_sgemm_compute_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_GEMM_PTR_RUNNER_REAL(run_dgemm_ptr, dgemm_ptr, fb_dgemm_compute_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_GEMM_PTR_RUNNER_COMPLEX(run_cgemm_ptr, cgemm_ptr, fb_cgemm_compute_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_GEMM_PTR_RUNNER_COMPLEX(run_zgemm_ptr, zgemm_ptr, fb_zgemm_compute_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

typedef void *(*fb_mkl_jit_get_ptr_fn_t)(void *);
typedef void (*fb_mkl_jit_destroy_fn_t)(void *);
typedef int (*fb_mkl_jit_create_sgemm_fn_t)(void **, char, char, int, int, int,
                                            float, int, int, float, int);
typedef int (*fb_mkl_jit_create_dgemm_fn_t)(void **, char, char, int, int, int,
                                            double, int, int, double, int);
typedef int (*fb_mkl_jit_create_cgemm_fn_t)(void **, char, char, int, int, int,
                                            fb_complex_float_t, int, int,
                                            fb_complex_float_t, int);
typedef int (*fb_mkl_jit_create_zgemm_fn_t)(void **, char, char, int, int, int,
                                            fb_complex_double_t, int, int,
                                            fb_complex_double_t, int);
typedef void (*fb_mkl_jit_sgemm_kernel_fn_t)(void *, const float *, const float *, float *);
typedef void (*fb_mkl_jit_dgemm_kernel_fn_t)(void *, const double *, const double *, double *);
typedef void (*fb_mkl_jit_cgemm_kernel_fn_t)(void *, const fb_complex_float_t *, const fb_complex_float_t *, fb_complex_float_t *);
typedef void (*fb_mkl_jit_zgemm_kernel_fn_t)(void *, const fb_complex_double_t *, const fb_complex_double_t *, fb_complex_double_t *);

#define FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_REAL(fn_name, create_field, get_field, create_type, kernel_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->create_field || !oracle->get_field || !oracle->mkl_jit_destroy || \
        !cand->create_field || !cand->get_field || !cand->mkl_jit_destroy) return FB_JUDGE_ERR_NOT_IMPL; \
    create_type oracle_create = (create_type)oracle->create_field; \
    create_type cand_create = (create_type)cand->create_field; \
    fb_mkl_jit_get_ptr_fn_t oracle_get = (fb_mkl_jit_get_ptr_fn_t)oracle->get_field; \
    fb_mkl_jit_get_ptr_fn_t cand_get = (fb_mkl_jit_get_ptr_fn_t)cand->get_field; \
    fb_mkl_jit_destroy_fn_t oracle_destroy = (fb_mkl_jit_destroy_fn_t)oracle->mkl_jit_destroy; \
    fb_mkl_jit_destroy_fn_t cand_destroy = (fb_mkl_jit_destroy_fn_t)cand->mkl_jit_destroy; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    scalar_type beta = (scalar_type)tc->beta; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    size_t C_sz = tc->C_elems; \
    void *oracle_jitter = NULL; \
    void *cand_jitter = NULL; \
    if (oracle_create(&oracle_jitter, trans, trans, m, n, k, alpha, lda, ldb, beta, ldc) != 0 || !oracle_jitter) { \
        if (oracle_jitter) oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    kernel_type oracle_kernel = (kernel_type)oracle_get(oracle_jitter); \
    if (!oracle_kernel) { \
        oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (cand_create(&cand_jitter, trans, trans, m, n, k, alpha, lda, ldb, beta, ldc) != 0 || !cand_jitter) { \
        if (cand_jitter) cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    kernel_type cand_kernel = (kernel_type)cand_get(cand_jitter); \
    if (!cand_kernel) { \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_kernel(oracle_jitter, A, B, Co); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { \
        free(Co); \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { \
        free(Co); \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_kernel(cand_jitter, A, B, Cc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn(Co, m, n, ldc))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_kernel(cand_jitter, A, B, Ct); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_kernel(cand_jitter, A, B, Ct); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); \
    cand_destroy(cand_jitter); \
    oracle_destroy(oracle_jitter); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_COMPLEX(fn_name, create_field, get_field, create_type, kernel_type, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->create_field || !oracle->get_field || !oracle->mkl_jit_destroy || \
        !cand->create_field || !cand->get_field || !cand->mkl_jit_destroy) return FB_JUDGE_ERR_NOT_IMPL; \
    create_type oracle_create = (create_type)oracle->create_field; \
    create_type cand_create = (create_type)cand->create_field; \
    fb_mkl_jit_get_ptr_fn_t oracle_get = (fb_mkl_jit_get_ptr_fn_t)oracle->get_field; \
    fb_mkl_jit_get_ptr_fn_t cand_get = (fb_mkl_jit_get_ptr_fn_t)cand->get_field; \
    fb_mkl_jit_destroy_fn_t oracle_destroy = (fb_mkl_jit_destroy_fn_t)oracle->mkl_jit_destroy; \
    fb_mkl_jit_destroy_fn_t cand_destroy = (fb_mkl_jit_destroy_fn_t)cand->mkl_jit_destroy; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = 0; \
    scalar_type beta; __real__(beta) = (real_type)tc->beta; __imag__(beta) = 0; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    size_t C_sz = (size_t)(m * ldc); \
    void *oracle_jitter = NULL; \
    void *cand_jitter = NULL; \
    if (oracle_create(&oracle_jitter, trans, trans, m, n, k, alpha, lda, ldb, beta, ldc) != 0 || !oracle_jitter) { \
        if (oracle_jitter) oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    kernel_type oracle_kernel = (kernel_type)oracle_get(oracle_jitter); \
    if (!oracle_kernel) { \
        oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (cand_create(&cand_jitter, trans, trans, m, n, k, alpha, lda, ldb, beta, ldc) != 0 || !cand_jitter) { \
        if (cand_jitter) cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    kernel_type cand_kernel = (kernel_type)cand_get(cand_jitter); \
    if (!cand_kernel) { \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_kernel(oracle_jitter, A, B, Co); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { \
        free(Co); \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { \
        free(Co); \
        cand_destroy(cand_jitter); \
        oracle_destroy(oracle_jitter); \
        result_fatal(res); \
        return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_kernel(cand_jitter, A, B, Cc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, dtype_enum, norm_fn((const real_type *)Co, C_sz))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_kernel(cand_jitter, A, B, Ct); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_kernel(cand_jitter, A, B, Ct); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); \
    cand_destroy(cand_jitter); \
    oracle_destroy(oracle_jitter); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_REAL(run_mkl_jit_get_sgemm_ptr, mkl_jit_create_sgemm, mkl_jit_get_sgemm_ptr, fb_mkl_jit_create_sgemm_fn_t, fb_mkl_jit_sgemm_kernel_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_REAL(run_mkl_jit_get_dgemm_ptr, mkl_jit_create_dgemm, mkl_jit_get_dgemm_ptr, fb_mkl_jit_create_dgemm_fn_t, fb_mkl_jit_dgemm_kernel_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_COMPLEX(run_mkl_jit_get_cgemm_ptr, mkl_jit_create_cgemm, mkl_jit_get_cgemm_ptr, fb_mkl_jit_create_cgemm_fn_t, fb_mkl_jit_cgemm_kernel_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_COMPLEX(run_mkl_jit_get_zgemm_ptr, mkl_jit_create_zgemm, mkl_jit_get_zgemm_ptr, fb_mkl_jit_create_zgemm_fn_t, fb_mkl_jit_zgemm_kernel_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

static fb_judge_status_t run_mkl_jit_create_sgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    return run_mkl_jit_get_sgemm_ptr(oracle, cand, tc, res, ns_out);
}

static fb_judge_status_t run_mkl_jit_create_dgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    return run_mkl_jit_get_dgemm_ptr(oracle, cand, tc, res, ns_out);
}

static fb_judge_status_t run_mkl_jit_create_cgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    return run_mkl_jit_get_cgemm_ptr(oracle, cand, tc, res, ns_out);
}

static fb_judge_status_t run_mkl_jit_create_zgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    return run_mkl_jit_get_zgemm_ptr(oracle, cand, tc, res, ns_out);
}

static fb_judge_status_t run_mkl_jit_destroy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->mkl_jit_create_sgemm || !oracle->mkl_jit_get_sgemm_ptr ||
        !oracle->mkl_jit_destroy || !cand->mkl_jit_create_sgemm ||
        !cand->mkl_jit_get_sgemm_ptr || !cand->mkl_jit_destroy) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }

    fb_mkl_jit_create_sgemm_fn_t oracle_create =
        (fb_mkl_jit_create_sgemm_fn_t)oracle->mkl_jit_create_sgemm;
    fb_mkl_jit_create_sgemm_fn_t cand_create =
        (fb_mkl_jit_create_sgemm_fn_t)cand->mkl_jit_create_sgemm;
    fb_mkl_jit_get_ptr_fn_t oracle_get =
        (fb_mkl_jit_get_ptr_fn_t)oracle->mkl_jit_get_sgemm_ptr;
    fb_mkl_jit_get_ptr_fn_t cand_get =
        (fb_mkl_jit_get_ptr_fn_t)cand->mkl_jit_get_sgemm_ptr;
    fb_mkl_jit_destroy_fn_t oracle_destroy =
        (fb_mkl_jit_destroy_fn_t)oracle->mkl_jit_destroy;
    fb_mkl_jit_destroy_fn_t cand_destroy =
        (fb_mkl_jit_destroy_fn_t)cand->mkl_jit_destroy;
    char trans = 'N';
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha;
    float beta = (float)tc->beta;
    const float *A = (const float *)tc->A;
    const float *B = (const float *)tc->B;
    size_t C_sz = tc->C_elems;
    void *oracle_jitter = NULL;
    void *cand_jitter = NULL;

    oracle_destroy(NULL);
    cand_destroy(NULL);

    if (oracle_create(&oracle_jitter, trans, trans, m, n, k, alpha, lda, ldb,
                      beta, ldc) != 0 || !oracle_jitter) {
        if (oracle_jitter) oracle_destroy(oracle_jitter);
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }
    fb_mkl_jit_sgemm_kernel_fn_t oracle_kernel =
        (fb_mkl_jit_sgemm_kernel_fn_t)oracle_get(oracle_jitter);
    if (!oracle_kernel) {
        oracle_destroy(oracle_jitter);
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (cand_create(&cand_jitter, trans, trans, m, n, k, alpha, lda, ldb,
                    beta, ldc) != 0 || !cand_jitter) {
        if (cand_jitter) cand_destroy(cand_jitter);
        oracle_destroy(oracle_jitter);
        result_fatal(res);
        return FB_JUDGE_OK;
    }
    fb_mkl_jit_sgemm_kernel_fn_t cand_kernel =
        (fb_mkl_jit_sgemm_kernel_fn_t)cand_get(cand_jitter);
    if (!cand_kernel) {
        cand_destroy(cand_jitter);
        oracle_destroy(oracle_jitter);
        result_fatal(res);
        return FB_JUDGE_OK;
    }

    float *Co = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Co) {
        cand_destroy(cand_jitter);
        oracle_destroy(oracle_jitter);
        result_fatal(res);
        return FB_JUDGE_ERR_ALLOC;
    }
    oracle_kernel(oracle_jitter, A, B, Co);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F32)) {
        free(Co);
        cand_destroy(cand_jitter);
        oracle_destroy(oracle_jitter);
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *Cc = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Cc) {
        free(Co);
        cand_destroy(cand_jitter);
        oracle_destroy(oracle_jitter);
        result_fatal(res);
        return FB_JUDGE_ERR_ALLOC;
    }
    cand_kernel(cand_jitter, A, B, Cc);
    result_from_relerr(res,
        fb_judge_relerr_matrix(Cc, Co, m, n, ldc, ldc, FB_DTYPE_F32,
                               fb_matrix_norm_frob_f32(Co, m, n, ldc)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F32)) {
        res->is_fatal = true;
    }

    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            void *tmp = NULL;
            if (cand_create(&tmp, trans, trans, m, n, k, alpha, lda, ldb,
                            beta, ldc) == 0 && tmp) {
                cand_destroy(tmp);
            }
        }

        uint64_t best = UINT64_MAX;
        for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
            void *tmp = NULL;
            if (cand_create(&tmp, trans, trans, m, n, k, alpha, lda, ldb,
                            beta, ldc) != 0 || !tmp) {
                best = 0;
                break;
            }
            uint64_t t0 = fb_judge_time_ns();
            cand_destroy(tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = (best == UINT64_MAX) ? 0 : best;
    }

    free(Cc);
    free(Co);
    cand_destroy(cand_jitter);
    oracle_destroy(oracle_jitter);
    return FB_JUDGE_OK;
}

#undef FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_REAL
#undef FB_DEFINE_MKL_JIT_GET_GEMM_PTR_RUNNER_COMPLEX

#undef FB_DEFINE_GEMM_PTR_RUNNER_REAL
#undef FB_DEFINE_GEMM_PTR_RUNNER_COMPLEX

#undef FB_DEFINE_GEMM_COMPUTE_RUNNER_REAL
#undef FB_DEFINE_GEMM_COMPUTE_RUNNER_COMPLEX
#undef FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_REAL
#undef FB_DEFINE_CBLAS_GEMM_COMPUTE_RUNNER_COMPLEX

typedef void (*fb_sgemm_pack_fn_t)(const char *, const char *, const int *,
                                   const int *, const int *, const float *,
                                   const float *, const int *, float *);
typedef void (*fb_dgemm_pack_fn_t)(const char *, const char *, const int *,
                                   const int *, const int *, const double *,
                                   const double *, const int *, double *);
typedef void (*fb_cgemm_pack_fn_t)(const char *, const char *, const int *,
                                   const int *, const int *,
                                   const fb_complex_float_t *,
                                   const fb_complex_float_t *, const int *,
                                   fb_complex_float_t *);
typedef void (*fb_zgemm_pack_fn_t)(const char *, const char *, const int *,
                                   const int *, const int *,
                                   const fb_complex_double_t *,
                                   const fb_complex_double_t *, const int *,
                                   fb_complex_double_t *);

typedef void (*fb_cblas_sgemm_pack_fn_t)(int, int, int, int, int, int, float,
                                         const float *, int, float *);
typedef void (*fb_cblas_dgemm_pack_fn_t)(int, int, int, int, int, int, double,
                                         const double *, int, double *);
typedef void (*fb_cblas_cgemm_pack_fn_t)(int, int, int, int, int, int,
                                         fb_complex_float_t,
                                         const fb_complex_float_t *, int,
                                         fb_complex_float_t *);
typedef void (*fb_cblas_zgemm_pack_fn_t)(int, int, int, int, int, int,
                                         fb_complex_double_t,
                                         const fb_complex_double_t *, int,
                                         fb_complex_double_t *);

#define FB_DEFINE_GEMM_PACK_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    char identifier = 'A'; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k, pld = (int)tc->lda; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    const scalar_type *src = (const scalar_type *)tc->A; \
    size_t dest_count = (size_t)m * (size_t)k; \
    scalar_type *Do = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Do) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Do); \
    if (fb_judge_has_nan_inf(Do, dest_count, dtype_enum)) { free(Do); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Dc = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Dc) { free(Do); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dc); \
    result_from_relerr(res, fb_judge_relerr(Dc, Do, dest_count, dtype_enum, norm_fn(Do, dest_count))); \
    if (fb_judge_has_nan_inf(Dc, dest_count, dtype_enum)) res->is_fatal = true; \
    free(Dc); \
    if (ns_out) { \
        scalar_type *Dt = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
        if (Dt) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dt); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dt); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Dt); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Do); return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM_PACK_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    char identifier = 'A'; \
    char trans = 'N'; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k, pld = (int)tc->lda; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    const scalar_type *src = (const scalar_type *)tc->A; \
    size_t dest_count = (size_t)m * (size_t)k; \
    scalar_type *Do = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Do) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Do); \
    if (fb_judge_has_nan_inf(Do, dest_count, dtype_enum)) { free(Do); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Dc = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Dc) { free(Do); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dc); \
    result_from_relerr(res, fb_judge_relerr(Dc, Do, dest_count, dtype_enum, norm_fn((const real_type *)Do, dest_count))); \
    if (fb_judge_has_nan_inf(Dc, dest_count, dtype_enum)) res->is_fatal = true; \
    free(Dc); \
    if (ns_out) { \
        scalar_type *Dt = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
        if (Dt) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dt); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(&identifier, &trans, &m, &n, &k, &alpha, src, &pld, Dt); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Dt); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Do); return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int order = FB_LAYOUT_ROW_MAJOR; \
    int identifier = 'A'; \
    int trans = FB_NO_TRANS; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k, pld = (int)tc->lda; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    const scalar_type *src = (const scalar_type *)tc->A; \
    size_t dest_count = (size_t)m * (size_t)k; \
    scalar_type *Do = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Do) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(order, identifier, trans, m, n, k, alpha, src, pld, Do); \
    if (fb_judge_has_nan_inf(Do, dest_count, dtype_enum)) { free(Do); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Dc = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Dc) { free(Do); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dc); \
    result_from_relerr(res, fb_judge_relerr(Dc, Do, dest_count, dtype_enum, norm_fn(Do, dest_count))); \
    if (fb_judge_has_nan_inf(Dc, dest_count, dtype_enum)) res->is_fatal = true; \
    free(Dc); \
    if (ns_out) { \
        scalar_type *Dt = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
        if (Dt) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dt); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dt); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Dt); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Do); return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int order = FB_LAYOUT_ROW_MAJOR; \
    int identifier = 'A'; \
    int trans = FB_NO_TRANS; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k, pld = (int)tc->lda; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    const scalar_type *src = (const scalar_type *)tc->A; \
    size_t dest_count = (size_t)m * (size_t)k; \
    scalar_type *Do = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Do) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(order, identifier, trans, m, n, k, alpha, src, pld, Do); \
    if (fb_judge_has_nan_inf(Do, dest_count, dtype_enum)) { free(Do); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Dc = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
    if (!Dc) { free(Do); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dc); \
    result_from_relerr(res, fb_judge_relerr(Dc, Do, dest_count, dtype_enum, norm_fn((const real_type *)Do, dest_count))); \
    if (fb_judge_has_nan_inf(Dc, dest_count, dtype_enum)) res->is_fatal = true; \
    free(Dc); \
    if (ns_out) { \
        scalar_type *Dt = (scalar_type *)clone_buf(tc->C_init, dest_count, sizeof(scalar_type)); \
        if (Dt) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dt); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(order, identifier, trans, m, n, k, alpha, src, pld, Dt); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Dt); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Do); return FB_JUDGE_OK; \
}

FB_DEFINE_GEMM_PACK_RUNNER_REAL(run_sgemm_pack, sgemm_pack, fb_sgemm_pack_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_GEMM_PACK_RUNNER_REAL(run_dgemm_pack, dgemm_pack, fb_dgemm_pack_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_GEMM_PACK_RUNNER_COMPLEX(run_cgemm_pack, cgemm_pack, fb_cgemm_pack_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMM_PACK_RUNNER_COMPLEX(run_zgemm_pack, zgemm_pack, fb_zgemm_pack_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_REAL(run_cblas_sgemm_pack, cblas_sgemm_pack, fb_cblas_sgemm_pack_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_REAL(run_cblas_dgemm_pack, cblas_dgemm_pack, fb_cblas_dgemm_pack_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_COMPLEX(run_cblas_cgemm_pack, cblas_cgemm_pack, fb_cblas_cgemm_pack_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_COMPLEX(run_cblas_zgemm_pack, cblas_zgemm_pack, fb_cblas_zgemm_pack_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

#undef FB_DEFINE_GEMM_PACK_RUNNER_REAL
#undef FB_DEFINE_GEMM_PACK_RUNNER_COMPLEX
#undef FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_REAL
#undef FB_DEFINE_CBLAS_GEMM_PACK_RUNNER_COMPLEX

/* ---- SGEMM -------------------------------------------------------------- */
static fb_judge_status_t run_sgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgemm || !cand->sgemm) return FB_JUDGE_ERR_NOT_IMPL;

    int      m    = (int)tc->m;
    int      n    = (int)tc->n;
    int      k    = (int)tc->k;
    int      lda  = (int)tc->lda;
    int      ldb  = (int)tc->ldb;
    int      ldc  = (int)tc->ldc;
    float        alpha = (float)tc->alpha;
    float        beta  = (float)tc->beta;
    const float *A    = (const float *)tc->A;
    const float *B    = (const float *)tc->B;
    size_t       C_sz = tc->C_elems;

    float *C_oracle = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!C_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                  m, n, k, alpha, A, lda, B, ldb, beta, C_oracle, ldc);
    if (fb_judge_has_nan_inf(C_oracle, C_sz, FB_DTYPE_F32)) {
        free(C_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *C_cand = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!C_cand) { free(C_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                m, n, k, alpha, A, lda, B, ldb, beta, C_cand, ldc);

    /* Compare with leading-dimension-aware matrix norm */
    double relerr = fb_judge_relerr_matrix(
        C_cand, C_oracle, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_F32, fb_matrix_norm_frob_f32(C_oracle, (int)m, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(C_cand, C_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        float *C_time = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            m, n, k, alpha, A, lda, B, ldb, beta, C_time, ldc);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            m, n, k, alpha, A, lda, B, ldb, beta, C_time, ldc);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(C_oracle);
    return FB_JUDGE_OK;
}

/* ---- DGEMM -------------------------------------------------------------- */
static fb_judge_status_t run_dgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgemm || !cand->dgemm) return FB_JUDGE_ERR_NOT_IMPL;

    int       m    = (int)tc->m;
    int       n    = (int)tc->n;
    int       k    = (int)tc->k;
    int       lda  = (int)tc->lda;
    int       ldb  = (int)tc->ldb;
    int       ldc  = (int)tc->ldc;
    double        alpha = tc->alpha;
    double        beta  = tc->beta;
    const double *A    = (const double *)tc->A;
    const double *B    = (const double *)tc->B;
    size_t        C_sz = tc->C_elems;

    double *C_oracle = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!C_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                  m, n, k, alpha, A, lda, B, ldb, beta, C_oracle, ldc);
    if (fb_judge_has_nan_inf(C_oracle, C_sz, FB_DTYPE_F64)) {
        free(C_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *C_cand = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!C_cand) { free(C_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                m, n, k, alpha, A, lda, B, ldb, beta, C_cand, ldc);

    double relerr = fb_judge_relerr_matrix(
        C_cand, C_oracle, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_F64, fb_matrix_norm_frob_f64(C_oracle, (int)m, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(C_cand, C_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        double *C_time = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            m, n, k, alpha, A, lda, B, ldb, beta, C_time, ldc);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            m, n, k, alpha, A, lda, B, ldb, beta, C_time, ldc);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(C_oracle);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — Complex L1 vector-output runners
 * ========================================================================= */

/* ---- CAXPY: y = alpha*x + y (complex single) -------------------------- */
static fb_judge_status_t run_caxpy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->caxpy || !cand->caxpy) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m (tc->n is unused for L1) */
    fb_complex_float_t alpha;
    __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;

    fb_complex_float_t *y_oracle = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->caxpy(n, &alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *y_cand = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->caxpy(n, &alpha, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_float_t *y_time = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (y_time) {
          for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            cand->caxpy(n, &alpha, x, 1, y_time, 1);
          uint64_t best = UINT64_MAX;
          for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            cand->caxpy(n, &alpha, x, 1, y_time, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
              best = dt;
          }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- ZAXPY: y = alpha*x + y (complex double) -------------------------- */
static fb_judge_status_t run_zaxpy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zaxpy || !cand->zaxpy) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    fb_complex_double_t alpha;
    __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;

    fb_complex_double_t *y_oracle = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zaxpy(n, &alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *y_cand = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zaxpy(n, &alpha, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_double_t *y_time = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (y_time) {
          for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            cand->zaxpy(n, &alpha, x, 1, y_time, 1);
          uint64_t best = UINT64_MAX;
          for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            cand->zaxpy(n, &alpha, x, 1, y_time, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
              best = dt;
          }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- CSCAL: x = alpha*x (complex single) ------------------------------ */
static fb_judge_status_t run_cscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cscal || !cand->cscal) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    fb_complex_float_t alpha;
    __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;

    fb_complex_float_t *x_oracle = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cscal(n, &alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *x_cand = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cscal(n, &alpha, x_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_float_t *x_time = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
        if (x_time) {
          for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            cand->cscal(n, &alpha, x_time, 1);
          uint64_t best = UINT64_MAX;
          for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            cand->cscal(n, &alpha, x_time, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
              best = dt;
          }
            free(x_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- ZSCAL: x = alpha*x (complex double) ------------------------------ */
static fb_judge_status_t run_zscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zscal || !cand->zscal) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    fb_complex_double_t alpha;
    __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;

    fb_complex_double_t *x_oracle = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zscal(n, &alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *x_cand = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zscal(n, &alpha, x_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_double_t *x_time = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
        if (x_time) {
          for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
            cand->zscal(n, &alpha, x_time, 1);
          uint64_t best = UINT64_MAX;
          for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            cand->zscal(n, &alpha, x_time, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
              best = dt;
          }
            free(x_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- CSSCAL: x = alpha*x (real alpha, complex single x) --------------- */
static fb_judge_status_t run_csscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csscal || !cand->csscal) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    float alpha = (float)tc->alpha;

    fb_complex_float_t *x_oracle = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *x_cand = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csscal(n, alpha, x_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_float_t *x_time = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->csscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->csscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(x_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- ZDSCAL: x = alpha*x (real alpha, complex double x) --------------- */
static fb_judge_status_t run_zdscal(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zdscal || !cand->zdscal) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    double alpha = tc->alpha;

    fb_complex_double_t *x_oracle = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zdscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *x_cand = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zdscal(n, alpha, x_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_double_t *x_time = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zdscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zdscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(x_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(x_oracle);
    return FB_JUDGE_OK;
}

/* ---- CCOPY: y = x (complex single) ------------------------------------ */
static fb_judge_status_t run_ccopy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ccopy || !cand->ccopy) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;

    fb_complex_float_t *y_oracle = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ccopy(n, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *y_cand = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ccopy(n, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_float_t *y_time = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ccopy(n, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->ccopy(n, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

/* ---- ZCOPY: y = x (complex double) ------------------------------------ */
static fb_judge_status_t run_zcopy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zcopy || !cand->zcopy) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;

    fb_complex_double_t *y_oracle = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zcopy(n, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *y_cand = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zcopy(n, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_double_t *y_time = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zcopy(n, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zcopy(n, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Extended BLAS L1 — AXPBY and batched AXPY
 * ========================================================================= */

#define FB_DEFINE_AXPBY_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, scalar_type, scalar_type *, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    scalar_type beta = (scalar_type)tc->beta; \
    const scalar_type *x = (const scalar_type *)tc->A; \
    scalar_type *y_oracle = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(n, alpha, x, 1, beta, y_oracle, 1); \
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, dtype_enum)) { \
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(n, alpha, x, 1, beta, y_cand, 1); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, (size_t)n, dtype_enum, norm_fn(y_oracle, (size_t)n))); \
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x, 1, beta, y_time, 1); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x, 1, beta, y_time, 1); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_AXPBY_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, scalar_type, scalar_type *, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    const scalar_type *x = (const scalar_type *)tc->A; \
    scalar_type *y_oracle = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn(n, alpha, x, 1, beta, y_oracle, 1); \
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, dtype_enum)) { \
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn(n, alpha, x, 1, beta, y_cand, 1); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, (size_t)n, dtype_enum, norm_fn((const real_type *)y_oracle, (size_t)n))); \
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)clone_buf(tc->B, (size_t)n, sizeof(scalar_type)); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x, 1, beta, y_time, 1); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x, 1, beta, y_time, 1); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_AXPY_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, alpha, x_batch, 1, y_oracle, 1, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, alpha, x_batch, 1, y_cand, 1, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_AXPY_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, alpha, x_batch, 1, y_oracle, 1, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, alpha, x_batch, 1, y_cand, 1, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, alpha, x_batch, 1, (long long)x_stride, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, scalar_type, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    scalar_type alpha; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, alpha, x_batch, 1, (long long)x_stride, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, alpha, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_COPY_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, x_batch, 1, y_oracle, 1, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, x_batch, 1, y_cand, 1, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_COPY_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, x_batch, 1, y_oracle, 1, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, x_batch, 1, y_cand, 1, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, x_batch, 1, y_time, 1, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, x_batch, 1, (long long)x_stride, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, x_batch, 1, (long long)x_stride, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(int, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->m; \
    size_t x_stride = tc->A_elems; \
    size_t y_stride = tc->B_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->A, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!x_batch || !y_oracle) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(n, x_batch, 1, (long long)x_stride, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(x_batch); free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(x_batch); free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(n, x_batch, 1, (long long)x_stride, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->B, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand_fn(n, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(n, x_batch, 1, (long long)x_stride, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(y_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_AXPBY_RUNNER_REAL(run_saxpby, saxpby, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_AXPBY_RUNNER_REAL(run_daxpby, daxpby, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_AXPBY_RUNNER_COMPLEX(run_caxpby, caxpby, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_AXPBY_RUNNER_COMPLEX(run_zaxpby, zaxpby, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_AXPY_BATCH_RUNNER_REAL(run_saxpy_batch, saxpy_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_AXPY_BATCH_RUNNER_REAL(run_daxpy_batch, daxpy_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_AXPY_BATCH_RUNNER_COMPLEX(run_caxpy_batch, caxpy_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_AXPY_BATCH_RUNNER_COMPLEX(run_zaxpy_batch, zaxpy_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_REAL(run_saxpy_batch_strided, saxpy_batch_strided, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_REAL(run_daxpy_batch_strided, daxpy_batch_strided, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_COMPLEX(run_caxpy_batch_strided, caxpy_batch_strided, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_COMPLEX(run_zaxpy_batch_strided, zaxpy_batch_strided, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_COPY_BATCH_RUNNER_REAL(run_scopy_batch, scopy_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_COPY_BATCH_RUNNER_REAL(run_dcopy_batch, dcopy_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_COPY_BATCH_RUNNER_COMPLEX(run_ccopy_batch, ccopy_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_COPY_BATCH_RUNNER_COMPLEX(run_zcopy_batch, zcopy_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_REAL(run_scopy_batch_strided, scopy_batch_strided, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_REAL(run_dcopy_batch_strided, dcopy_batch_strided, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_COMPLEX(run_ccopy_batch_strided, ccopy_batch_strided, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_COMPLEX(run_zcopy_batch_strided, zcopy_batch_strided, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

#undef FB_DEFINE_AXPBY_RUNNER_REAL
#undef FB_DEFINE_AXPBY_RUNNER_COMPLEX
#undef FB_DEFINE_AXPY_BATCH_RUNNER_REAL
#undef FB_DEFINE_AXPY_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_REAL
#undef FB_DEFINE_AXPY_BATCH_STRIDED_RUNNER_COMPLEX
#undef FB_DEFINE_COPY_BATCH_RUNNER_REAL
#undef FB_DEFINE_COPY_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_REAL
#undef FB_DEFINE_COPY_BATCH_STRIDED_RUNNER_COMPLEX

/* ---- CSWAP: swap x,y (complex single) --------------------------------- */
static fb_judge_status_t run_cswap(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cswap || !cand->cswap) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */

    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cswap(n, xo, 1, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* Compare both x and y against oracle; use combined norm */
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cswap(n, xc, 1, yc, 1);
    double scale = fb_norm_frob_cf32((const float *)yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32,
                                fb_norm_frob_cf32((const float *)xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32) ||
        fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->cswap(n, xt, 1, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->cswap(n, xt, 1, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

/* ---- ZSWAP: swap x,y (complex double) --------------------------------- */
static fb_judge_status_t run_zswap(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zswap || !cand->zswap) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */

    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zswap(n, xo, 1, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zswap(n, xc, 1, yc, 1);
    double scale = fb_norm_frob_cf64((const double *)yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64,
                                fb_norm_frob_cf64((const double *)xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64) ||
        fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zswap(n, xt, 1, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zswap(n, xt, 1, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — Complex L1 scalar-result runners
 * ========================================================================= */

/* ---- CDOTU/CDOTC: complex dot products --------------------------------- */
static fb_judge_status_t run_cdotu(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cdotu || !cand->cdotu) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t r_oracle, r_cand;
    __real__(r_oracle) = 0.0f; __imag__(r_oracle) = 0.0f;
    oracle->cdotu(&r_oracle, n, x, 1, y, 1);
    __real__(r_cand) = 0.0f; __imag__(r_cand) = 0.0f;
    cand->cdotu(&r_cand, n, x, 1, y, 1);
    /* Compare as 2-element float array */
    double scale = fb_norm_frob_cf32((const float *)&r_oracle, 1);
    double relerr = fb_judge_relerr(&r_cand, &r_oracle, 1, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (ns_out) {
        fb_complex_float_t tmp;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->cdotu(&tmp, n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); cand->cdotu(&tmp, n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cdotc(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cdotc || !cand->cdotc) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t r_oracle, r_cand;
    __real__(r_oracle) = 0.0f; __imag__(r_oracle) = 0.0f;
    oracle->cdotc(&r_oracle, n, x, 1, y, 1);
    __real__(r_cand) = 0.0f; __imag__(r_cand) = 0.0f;
    cand->cdotc(&r_cand, n, x, 1, y, 1);
    double scale = fb_norm_frob_cf32((const float *)&r_oracle, 1);
    double relerr = fb_judge_relerr(&r_cand, &r_oracle, 1, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (ns_out) {
        fb_complex_float_t tmp;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->cdotc(&tmp, n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); cand->cdotc(&tmp, n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zdotu(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zdotu || !cand->zdotu) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t r_oracle, r_cand;
    __real__(r_oracle) = 0.0; __imag__(r_oracle) = 0.0;
    oracle->zdotu(&r_oracle, n, x, 1, y, 1);
    __real__(r_cand) = 0.0; __imag__(r_cand) = 0.0;
    cand->zdotu(&r_cand, n, x, 1, y, 1);
    double scale = fb_norm_frob_cf64((const double *)&r_oracle, 1);
    double relerr = fb_judge_relerr(&r_cand, &r_oracle, 1, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (ns_out) {
        fb_complex_double_t tmp;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zdotu(&tmp, n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); cand->zdotu(&tmp, n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zdotc(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zdotc || !cand->zdotc) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t r_oracle, r_cand;
    __real__(r_oracle) = 0.0; __imag__(r_oracle) = 0.0;
    oracle->zdotc(&r_oracle, n, x, 1, y, 1);
    __real__(r_cand) = 0.0; __imag__(r_cand) = 0.0;
    cand->zdotc(&r_cand, n, x, 1, y, 1);
    double scale = fb_norm_frob_cf64((const double *)&r_oracle, 1);
    double relerr = fb_judge_relerr(&r_cand, &r_oracle, 1, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (ns_out) {
        fb_complex_double_t tmp;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zdotc(&tmp, n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); cand->zdotc(&tmp, n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SDSDOT / DSDOT --------------------------------------------------- */
static fb_judge_status_t run_sdsdot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sdsdot || !cand->sdsdot) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    float sb = (float)tc->alpha;
    const float *x = (const float *)tc->A;
    const float *y = (const float *)tc->B;
    float r_oracle = oracle->sdsdot(n, sb, x, 1, y, 1);
    float r_cand   = cand->sdsdot(n, sb, x, 1, y, 1);
    double scale = fabs((double)r_oracle);
    result_from_relerr(res, scalar_relerr_f32(r_cand, r_oracle, scale));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->sdsdot(n, sb, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->sdsdot(n, sb, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsdot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsdot || !cand->dsdot) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const float *x = (const float *)tc->A;
    const float *y = (const float *)tc->B;
    double r_oracle = oracle->dsdot(n, x, 1, y, 1);
    double r_cand   = cand->dsdot(n, x, 1, y, 1);
    result_from_relerr(res, scalar_relerr_f64(r_cand, r_oracle, fabs(r_oracle)));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->dsdot(n, x, 1, y, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->dsdot(n, x, 1, y, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SCASUM / DZASUM -------------------------------------------------- */
static fb_judge_status_t run_scasum(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->scasum || !cand->scasum) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    float r_oracle = oracle->scasum(n, x, 1);
    float r_cand   = cand->scasum(n, x, 1);
    result_from_relerr(res, scalar_relerr_f32(r_cand, r_oracle, fabs((double)r_oracle)));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->scasum(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->scasum(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dzasum(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dzasum || !cand->dzasum) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    double r_oracle = oracle->dzasum(n, x, 1);
    double r_cand   = cand->dzasum(n, x, 1);
    result_from_relerr(res, scalar_relerr_f64(r_cand, r_oracle, fabs(r_oracle)));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->dzasum(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->dzasum(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SCNRM2 / DZNRM2 -------------------------------------------------- */
static fb_judge_status_t run_scnrm2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->scnrm2 || !cand->scnrm2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    float r_oracle = oracle->scnrm2(n, x, 1);
    float r_cand   = cand->scnrm2(n, x, 1);
    result_from_relerr(res, scalar_relerr_f32(r_cand, r_oracle, fabs((double)r_oracle)));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->scnrm2(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->scnrm2(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dznrm2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dznrm2 || !cand->dznrm2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    double r_oracle = oracle->dznrm2(n, x, 1);
    double r_cand   = cand->dznrm2(n, x, 1);
    result_from_relerr(res, scalar_relerr_f64(r_cand, r_oracle, fabs(r_oracle)));
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->dznrm2(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->dznrm2(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- ICAMAX / IZAMAX -------------------------------------------------- */
static fb_judge_status_t run_icamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->icamax || !cand->icamax) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    int oracle_idx = oracle->icamax(n, x, 1);
    int cand_idx   = cand->icamax(n, x, 1);
    if (cand_idx == oracle_idx) {
        res->digits = 16.0; res->relative_error = 0.0;
        res->is_fatal = false; res->is_oracle_fatal = false;
    } else {
        result_from_relerr(res, 1.0);
    }
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->icamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->icamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_izamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->izamax || !cand->izamax) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    int oracle_idx = oracle->izamax(n, x, 1);
    int cand_idx   = cand->izamax(n, x, 1);
    if (cand_idx == oracle_idx) {
        res->digits = 16.0; res->relative_error = 0.0;
        res->is_fatal = false; res->is_oracle_fatal = false;
    } else {
        result_from_relerr(res, 1.0);
    }
    if (ns_out) {
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) (void)cand->izamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns(); (void)cand->izamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — Rotation runners
 * ========================================================================= */

/* ---- SROT / DROT ------------------------------------------------------- */
static fb_judge_status_t run_srot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->srot || !cand->srot) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    float c = (float)tc->alpha, s = (float)tc->beta;
    const float *x_in = (const float *)tc->A;
    const float *y_in = (const float *)tc->B;

    float *xo = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
    float *yo = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->srot(n, xo, 1, yo, 1, c, s);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    float *xc = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
    float *yc = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->srot(n, xc, 1, yc, 1, c, s);
    double scale = fb_norm_frob_f32(yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F32) ||
        fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        float *xt = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
        float *yt = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->srot(n, xt, 1, yt, 1, c, s);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->srot(n, xt, 1, yt, 1, c, s);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_drot(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->drot || !cand->drot) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    double c = tc->alpha, s = tc->beta;
    const double *x_in = (const double *)tc->A;
    const double *y_in = (const double *)tc->B;

    double *xo = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
    double *yo = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->drot(n, xo, 1, yo, 1, c, s);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *xc = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
    double *yc = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->drot(n, xc, 1, yc, 1, c, s);
    double scale = fb_norm_frob_f64(yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F64) ||
        fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        double *xt = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
        double *yt = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->drot(n, xt, 1, yt, 1, c, s);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->drot(n, xt, 1, yt, 1, c, s);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

/* ---- CROT: complex float Givens rotation (real c, complex s) ----------- */
static fb_judge_status_t run_crot(const fb_backend_vtable_t *oracle,
                                  const fb_backend_vtable_t *cand,
                                  const fb_corpus_case_t *tc,
                                  fb_judge_case_result_t *res,
                                  uint64_t *ns_out) {
  if (!oracle->crot || !cand->crot)
    return FB_JUDGE_ERR_NOT_IMPL;
  int n = (int)tc->m;  /* L1: vector length lives in tc->m */
  float c = (float)tc->alpha;
  fb_complex_float_t s;
  __real__(s) = (float)tc->beta;
  __imag__(s) = 0.0f;
  const fb_complex_float_t *x_in = (const fb_complex_float_t *)tc->A;
  const fb_complex_float_t *y_in = (const fb_complex_float_t *)tc->B;

  fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_float_t));
  fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_float_t));
  if (!xo || !yo) {
    free(xo);
    free(yo);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  oracle->crot(n, xo, 1, yo, 1, c, s);
  if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) {
    free(xo);
    free(yo);
    result_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_float_t));
  fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_float_t));
  if (!xc || !yc) {
    free(xo);
    free(yo);
    free(xc);
    free(yc);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  cand->crot(n, xc, 1, yc, 1, c, s);
  double scale = fb_norm_frob_cf32((const float *)yo, (size_t)n);
  double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, scale);
  double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32,
                              fb_norm_frob_cf32((const float *)xo, (size_t)n));
  result_from_relerr(res, (ey > ex) ? ey : ex);
  if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32) ||
      fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32))
    res->is_fatal = true;
  free(xc);
  free(yc);
  if (ns_out) {
    fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(
        x_in, (size_t)n, sizeof(fb_complex_float_t));
    fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(
        y_in, (size_t)n, sizeof(fb_complex_float_t));
    if (xt && yt) {
      for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
        cand->crot(n, xt, 1, yt, 1, c, s);
      uint64_t best = UINT64_MAX;
      for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
        uint64_t t0 = fb_judge_time_ns();
        cand->crot(n, xt, 1, yt, 1, c, s);
        uint64_t dt = fb_judge_time_ns() - t0;
        if (dt < best)
          best = dt;
      }
      *ns_out = best;
    } else {
      *ns_out = 0;
    }
    free(xt);
    free(yt);
  }
  free(xo);
  free(yo);
  return FB_JUDGE_OK;
}

/* ---- ZROT: complex double Givens rotation (real c, complex double s) --- */
static fb_judge_status_t run_zrot(const fb_backend_vtable_t *oracle,
                                  const fb_backend_vtable_t *cand,
                                  const fb_corpus_case_t *tc,
                                  fb_judge_case_result_t *res,
                                  uint64_t *ns_out) {
  if (!oracle->zrot || !cand->zrot)
    return FB_JUDGE_ERR_NOT_IMPL;
  int n = (int)tc->m;  /* L1: vector length lives in tc->m */
  double c = tc->alpha;
  fb_complex_double_t s;
  __real__(s) = tc->beta;
  __imag__(s) = 0.0;
  const fb_complex_double_t *x_in = (const fb_complex_double_t *)tc->A;
  const fb_complex_double_t *y_in = (const fb_complex_double_t *)tc->B;

  fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_double_t));
  fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_double_t));
  if (!xo || !yo) {
    free(xo);
    free(yo);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  oracle->zrot(n, xo, 1, yo, 1, c, s);
  if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) {
    free(xo);
    free(yo);
    result_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_double_t));
  fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_double_t));
  if (!xc || !yc) {
    free(xo);
    free(yo);
    free(xc);
    free(yc);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  cand->zrot(n, xc, 1, yc, 1, c, s);
  double scale = fb_norm_frob_cf64((const double *)yo, (size_t)n);
  double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, scale);
  double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64,
                              fb_norm_frob_cf64((const double *)xo, (size_t)n));
  result_from_relerr(res, (ey > ex) ? ey : ex);
  if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64) ||
      fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64))
    res->is_fatal = true;
  free(xc);
  free(yc);
  if (ns_out) {
    fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(
        x_in, (size_t)n, sizeof(fb_complex_double_t));
    fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(
        y_in, (size_t)n, sizeof(fb_complex_double_t));
    if (xt && yt) {
      for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
        cand->zrot(n, xt, 1, yt, 1, c, s);
      uint64_t best = UINT64_MAX;
      for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
        uint64_t t0 = fb_judge_time_ns();
        cand->zrot(n, xt, 1, yt, 1, c, s);
        uint64_t dt = fb_judge_time_ns() - t0;
        if (dt < best)
          best = dt;
      }
      *ns_out = best;
    } else {
      *ns_out = 0;
    }
    free(xt);
    free(yt);
  }
  free(xo);
  free(yo);
  return FB_JUDGE_OK;
}

/* ---- ZDROT: complex double vectors, real double Givens rotation --------- */
static fb_judge_status_t run_zdrot(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_case_result_t *res,
                                   uint64_t *ns_out) {
  if (!oracle->zdrot || !cand->zdrot)
    return FB_JUDGE_ERR_NOT_IMPL;
  int n = (int)tc->m;  /* L1: vector length lives in tc->m */
  double c = tc->alpha, s = tc->beta;
  const fb_complex_double_t *x_in = (const fb_complex_double_t *)tc->A;
  const fb_complex_double_t *y_in = (const fb_complex_double_t *)tc->B;

  fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_double_t));
  fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_double_t));
  if (!xo || !yo) {
    free(xo);
    free(yo);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  oracle->zdrot(n, xo, 1, yo, 1, c, s);
  if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) {
    free(xo);
    free(yo);
    result_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(
      x_in, (size_t)n, sizeof(fb_complex_double_t));
  fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(
      y_in, (size_t)n, sizeof(fb_complex_double_t));
  if (!xc || !yc) {
    free(xo);
    free(yo);
    free(xc);
    free(yc);
    result_fatal(res);
    return FB_JUDGE_ERR_ALLOC;
  }
  cand->zdrot(n, xc, 1, yc, 1, c, s);
  double scale = fb_norm_frob_cf64((const double *)yo, (size_t)n);
  double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, scale);
  double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64,
                              fb_norm_frob_cf64((const double *)xo, (size_t)n));
  result_from_relerr(res, (ey > ex) ? ey : ex);
  if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64) ||
      fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64))
    res->is_fatal = true;
  free(xc);
  free(yc);
  if (ns_out) {
    fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(
        x_in, (size_t)n, sizeof(fb_complex_double_t));
    fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(
        y_in, (size_t)n, sizeof(fb_complex_double_t));
    if (xt && yt) {
      for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
        cand->zdrot(n, xt, 1, yt, 1, c, s);
      uint64_t best = UINT64_MAX;
      for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
        uint64_t t0 = fb_judge_time_ns();
        cand->zdrot(n, xt, 1, yt, 1, c, s);
        uint64_t dt = fb_judge_time_ns() - t0;
        if (dt < best)
          best = dt;
      }
      *ns_out = best;
    } else {
      *ns_out = 0;
    }
    free(xt);
    free(yt);
  }
  free(xo);
  free(yo);
  return FB_JUDGE_OK;
}

/* ---- SROTG / DROTG ----------------------------------------------------- */
static fb_judge_status_t run_srotg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->srotg || !cand->srotg) return FB_JUDGE_ERR_NOT_IMPL;
    /* Use first two elements of tc->A as a and b inputs */
    const float *ab = (const float *)tc->A;
    float ao = ab[0], bo = ab[1], co = 0.0f, so = 0.0f;
    oracle->srotg(&ao, &bo, &co, &so);
    float ac = ab[0], bc = ab[1], cc = 0.0f, sc = 0.0f;
    cand->srotg(&ac, &bc, &cc, &sc);
    /* Compare c and s (rotation angle) */
    double ec = scalar_relerr_f32(cc, co, fabs((double)co));
    double es = scalar_relerr_f32(sc, so, fabs((double)so));
    result_from_relerr(res, (ec > es) ? ec : es);
    if (ns_out) {
        float a2, b2, c2, s2;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            a2 = ab[0]; b2 = ab[1]; cand->srotg(&a2, &b2, &c2, &s2);
        }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            a2 = ab[0]; b2 = ab[1];
            uint64_t t0 = fb_judge_time_ns(); cand->srotg(&a2, &b2, &c2, &s2);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_drotg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->drotg || !cand->drotg) return FB_JUDGE_ERR_NOT_IMPL;
    const double *ab = (const double *)tc->A;
    double ao = ab[0], bo = ab[1], co = 0.0, so = 0.0;
    oracle->drotg(&ao, &bo, &co, &so);
    double ac = ab[0], bc = ab[1], cc = 0.0, sc = 0.0;
    cand->drotg(&ac, &bc, &cc, &sc);
    double ec = scalar_relerr_f64(cc, co, fabs(co));
    double es = scalar_relerr_f64(sc, so, fabs(so));
    result_from_relerr(res, (ec > es) ? ec : es);
    if (ns_out) {
        double a2, b2, c2, s2;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            a2 = ab[0]; b2 = ab[1]; cand->drotg(&a2, &b2, &c2, &s2);
        }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            a2 = ab[0]; b2 = ab[1];
            uint64_t t0 = fb_judge_time_ns(); cand->drotg(&a2, &b2, &c2, &s2);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* ---- SROTM / DROTM / SROTMG ------------------------------------------- */
static fb_judge_status_t run_srotm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->srotm || !cand->srotm) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const float *x_in = (const float *)tc->A;
    const float *y_in = (const float *)tc->B;
    /* Use identity modified rotation: param[0]=-2 (use identity, no-op) for safety */
    float param[5] = {-2.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    float *xo = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
    float *yo = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->srotm(n, xo, 1, yo, 1, param);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    float *xc = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
    float *yc = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->srotm(n, xc, 1, yc, 1, param);
    double scale = fb_norm_frob_f32(yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        float *xt = (float *)clone_buf(x_in, (size_t)n, sizeof(float));
        float *yt = (float *)clone_buf(y_in, (size_t)n, sizeof(float));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->srotm(n, xt, 1, yt, 1, param);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->srotm(n, xt, 1, yt, 1, param);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_drotm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->drotm || !cand->drotm) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->m;  /* L1: vector length lives in tc->m */
    const double *x_in = (const double *)tc->A;
    const double *y_in = (const double *)tc->B;
    double param[5] = {-2.0, 1.0, 0.0, 0.0, 1.0};

    double *xo = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
    double *yo = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
    if (!xo || !yo) { free(xo); free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->drotm(n, xo, 1, yo, 1, param);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) {
        free(xo); free(yo); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *xc = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
    double *yc = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
    if (!xc || !yc) { free(xo); free(yo); free(xc); free(yc); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->drotm(n, xc, 1, yc, 1, param);
    double scale = fb_norm_frob_f64(yo, (size_t)n);
    double ey = fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, scale);
    double ex = fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n));
    result_from_relerr(res, (ey > ex) ? ey : ex);
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc); free(yc);
    if (ns_out) {
        double *xt = (double *)clone_buf(x_in, (size_t)n, sizeof(double));
        double *yt = (double *)clone_buf(y_in, (size_t)n, sizeof(double));
        if (xt && yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->drotm(n, xt, 1, yt, 1, param);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->drotm(n, xt, 1, yt, 1, param);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            *ns_out = best;
        } else { *ns_out = 0; }
        free(xt); free(yt);
    }
    free(xo); free(yo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_srotmg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->srotmg || !cand->srotmg) return FB_JUDGE_ERR_NOT_IMPL;
    const float *in = (const float *)tc->A;
    /* Use first 3 elements as d1, d2, x1; y1 from alpha */
    float d1o = in[0], d2o = in[1], x1o = in[2];
    float y1 = (float)tc->alpha;
    float po[5] = {0.0f}; oracle->srotmg(&d1o, &d2o, &x1o, y1, po);
    float d1c = in[0], d2c = in[1], x1c = in[2];
    float pc[5] = {0.0f}; cand->srotmg(&d1c, &d2c, &x1c, y1, pc);
    double relerr = scalar_relerr_f32(pc[1], po[1], fabs((double)po[1]));
    for (int i = 2; i < 5; i++) {
        double e = scalar_relerr_f32(pc[i], po[i], fabs((double)po[i]));
        if (e > relerr) relerr = e;
    }
    result_from_relerr(res, relerr);
    if (ns_out) {
        float d1t, d2t, x1t; float pt[5];
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            d1t = in[0]; d2t = in[1]; x1t = in[2];
            cand->srotmg(&d1t, &d2t, &x1t, y1, pt);
        }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            d1t = in[0]; d2t = in[1]; x1t = in[2];
            uint64_t t0 = fb_judge_time_ns(); cand->srotmg(&d1t, &d2t, &x1t, y1, pt);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — CGEMV / ZGEMV
 * ========================================================================= */

static fb_judge_status_t run_cgemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgemv || !cand->cgemv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->B;

    fb_complex_float_t *y_oracle = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)m, FB_DTYPE_CF32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *y_cand = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)y_oracle, (size_t)m);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)m, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)m, FB_DTYPE_CF32)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_float_t *y_time = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgemv || !cand->zgemv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->B;

    fb_complex_double_t *y_oracle = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)m, FB_DTYPE_CF64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *y_cand = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)y_oracle, (size_t)m);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)m, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)m, FB_DTYPE_CF64)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_double_t *y_time = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A, lda, x, 1, beta, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(y_time); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(y_oracle);
    return FB_JUDGE_OK;
}

#define FB_DEFINE_GEMV_BATCH_RUNNER_REAL(fn_name, field, single_field, scalar_type, single_fn_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*batch_fn_t)(char, int, int, scalar_type, \
                               const scalar_type **, int, const scalar_type **, int, \
                               scalar_type, scalar_type **, int, int); \
    bool oracle_has_batch = (oracle->field != NULL); \
    bool cand_has_batch = (cand->field != NULL); \
    bool oracle_has_single = (oracle->single_field != NULL); \
    bool cand_has_single = (cand->single_field != NULL); \
    if ((!oracle_has_batch && !oracle_has_single) || (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL; \
    batch_fn_t oracle_batch_fn = (batch_fn_t)oracle->field; \
    batch_fn_t cand_batch_fn = (batch_fn_t)cand->field; \
    single_fn_type oracle_single_fn = (single_fn_type)oracle->single_field; \
    single_fn_type cand_single_fn = (single_fn_type)cand->single_field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    scalar_type beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, y_stride = tc->C_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **x_array = (const scalar_type **)make_const_batch_ptrs(x_batch, x_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **y_oracle_array = (scalar_type **)make_batch_ptrs(y_oracle, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !y_oracle || !A_array || !x_array || !y_oracle_array) { \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (oracle_has_batch) { \
        oracle_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_oracle_array, 1, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            oracle_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_oracle_array[b], 1); \
        } \
    } \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **y_cand_array = (scalar_type **)make_batch_ptrs(y_cand, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand || !y_cand_array) { \
        free(y_cand_array); free(y_cand); \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (cand_has_batch) { \
        cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_cand_array, 1, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_cand_array[b], 1); \
        } \
    } \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand_array); \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **y_time_array = (scalar_type **)make_batch_ptrs(y_time, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time && y_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_time_array, 1, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_time_array[b], 1); \
                    } \
                } \
            } \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_time_array, 1, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_time_array[b], 1); \
                    } \
                } \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(y_time_array); \
        free(y_time); \
    } \
    free((void *)A_array); \
    free((void *)x_array); \
    free(y_oracle_array); \
    free(A_batch); \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMV_BATCH_RUNNER_COMPLEX(fn_name, field, single_field, scalar_type, single_fn_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*batch_fn_t)(char, int, int, scalar_type, \
                               const scalar_type **, int, const scalar_type **, int, \
                               scalar_type, scalar_type **, int, int); \
    bool oracle_has_batch = (oracle->field != NULL); \
    bool cand_has_batch = (cand->field != NULL); \
    bool oracle_has_single = (oracle->single_field != NULL); \
    bool cand_has_single = (cand->single_field != NULL); \
    if ((!oracle_has_batch && !oracle_has_single) || (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL; \
    batch_fn_t oracle_batch_fn = (batch_fn_t)oracle->field; \
    batch_fn_t cand_batch_fn = (batch_fn_t)cand->field; \
    single_fn_type oracle_single_fn = (single_fn_type)oracle->single_field; \
    single_fn_type cand_single_fn = (single_fn_type)cand->single_field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, y_stride = tc->C_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **x_array = (const scalar_type **)make_const_batch_ptrs(x_batch, x_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **y_oracle_array = (scalar_type **)make_batch_ptrs(y_oracle, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !y_oracle || !A_array || !x_array || !y_oracle_array) { \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (oracle_has_batch) { \
        oracle_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_oracle_array, 1, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            oracle_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_oracle_array[b], 1); \
        } \
    } \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **y_cand_array = (scalar_type **)make_batch_ptrs(y_cand, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand || !y_cand_array) { \
        free(y_cand_array); free(y_cand); \
        free((void *)A_array); free((void *)x_array); free(y_oracle_array); \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (cand_has_batch) { \
        cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_cand_array, 1, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_cand_array[b], 1); \
        } \
    } \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand_array); \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **y_time_array = (scalar_type **)make_batch_ptrs(y_time, y_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time && y_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_time_array, 1, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_time_array[b], 1); \
                    } \
                } \
            } \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_array, lda, x_array, 1, beta, y_time_array, 1, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_array[b], lda, x_array[b], 1, beta, y_time_array[b], 1); \
                    } \
                } \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(y_time_array); \
        free(y_time); \
    } \
    free((void *)A_array); \
    free((void *)x_array); \
    free(y_oracle_array); \
    free(A_batch); \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_REAL(fn_name, field, single_field, scalar_type, single_fn_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*batch_fn_t)(char, int, int, scalar_type, const scalar_type *, int, long long, \
                               const scalar_type *, int, long long, scalar_type, scalar_type *, int, long long, int); \
    bool oracle_has_batch = (oracle->field != NULL); \
    bool cand_has_batch = (cand->field != NULL); \
    bool oracle_has_single = (oracle->single_field != NULL); \
    bool cand_has_single = (cand->single_field != NULL); \
    if ((!oracle_has_batch && !oracle_has_single) || (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL; \
    batch_fn_t oracle_batch_fn = (batch_fn_t)oracle->field; \
    batch_fn_t cand_batch_fn = (batch_fn_t)cand->field; \
    single_fn_type oracle_single_fn = (single_fn_type)oracle->single_field; \
    single_fn_type cand_single_fn = (single_fn_type)cand->single_field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    scalar_type beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, y_stride = tc->C_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !y_oracle) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (oracle_has_batch) { \
        oracle_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            oracle_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_oracle + ((size_t)b * y_stride), 1); \
        } \
    } \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (cand_has_batch) { \
        cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_cand + ((size_t)b * y_stride), 1); \
        } \
    } \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn(y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_time + ((size_t)b * y_stride), 1); \
                    } \
                } \
            } \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_time + ((size_t)b * y_stride), 1); \
                    } \
                } \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(y_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_COMPLEX(fn_name, field, single_field, scalar_type, single_fn_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*batch_fn_t)(char, int, int, scalar_type, const scalar_type *, int, long long, \
                               const scalar_type *, int, long long, scalar_type, scalar_type *, int, long long, int); \
    bool oracle_has_batch = (oracle->field != NULL); \
    bool cand_has_batch = (cand->field != NULL); \
    bool oracle_has_single = (oracle->single_field != NULL); \
    bool cand_has_single = (cand->single_field != NULL); \
    if ((!oracle_has_batch && !oracle_has_single) || (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL; \
    batch_fn_t oracle_batch_fn = (batch_fn_t)oracle->field; \
    batch_fn_t cand_batch_fn = (batch_fn_t)cand->field; \
    single_fn_type oracle_single_fn = (single_fn_type)oracle->single_field; \
    single_fn_type cand_single_fn = (single_fn_type)cand->single_field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, y_stride = tc->C_elems; \
    size_t total_y = y_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *y_oracle = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !y_oracle) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (oracle_has_batch) { \
        oracle_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_oracle, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            oracle_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_oracle + ((size_t)b * y_stride), 1); \
        } \
    } \
    if (fb_judge_has_nan_inf(y_oracle, total_y, dtype_enum)) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *y_cand = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!y_cand) { \
        free(A_batch); free(x_batch); free(y_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    if (cand_has_batch) { \
        cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_cand, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
    } else { \
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
            cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_cand + ((size_t)b * y_stride), 1); \
        } \
    } \
    result_from_relerr(res, fb_judge_relerr(y_cand, y_oracle, total_y, dtype_enum, norm_fn((const real_type *)y_oracle, total_y))); \
    if (fb_judge_has_nan_inf(y_cand, total_y, dtype_enum)) res->is_fatal = true; \
    free(y_cand); \
    if (ns_out) { \
        scalar_type *y_time = (scalar_type *)repeat_buf(tc->C_init, y_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (y_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_time + ((size_t)b * y_stride), 1); \
                    } \
                } \
            } \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                if (cand_has_batch) { \
                    cand_batch_fn('N', m, n, alpha, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, beta, y_time, 1, (long long)y_stride, FB_JUDGE_BATCH_COUNT); \
                } else { \
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) { \
                        cand_single_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, alpha, A_batch + ((size_t)b * a_stride), lda, x_batch + ((size_t)b * x_stride), 1, beta, y_time + ((size_t)b * y_stride), 1); \
                    } \
                } \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(y_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(y_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_GEMV_BATCH_RUNNER_REAL(run_sgemv_batch, sgemv_batch, sgemv, float, fb_sgemv_fn, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_GEMV_BATCH_RUNNER_REAL(run_dgemv_batch, dgemv_batch, dgemv, double, fb_dgemv_fn, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_GEMV_BATCH_RUNNER_COMPLEX(run_cgemv_batch, cgemv_batch, cgemv, fb_complex_float_t, fb_cgemv_fn, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMV_BATCH_RUNNER_COMPLEX(run_zgemv_batch, zgemv_batch, zgemv, fb_complex_double_t, fb_zgemv_fn, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_REAL(run_sgemv_batch_strided, sgemv_batch_strided, sgemv, float, fb_sgemv_fn, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_REAL(run_dgemv_batch_strided, dgemv_batch_strided, dgemv, double, fb_dgemv_fn, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_COMPLEX(run_cgemv_batch_strided, cgemv_batch_strided, cgemv, fb_complex_float_t, fb_cgemv_fn, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_COMPLEX(run_zgemv_batch_strided, zgemv_batch_strided, zgemv, fb_complex_double_t, fb_zgemv_fn, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

#undef FB_DEFINE_GEMV_BATCH_RUNNER_REAL
#undef FB_DEFINE_GEMV_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_REAL
#undef FB_DEFINE_GEMV_BATCH_STRIDED_RUNNER_COMPLEX

#define FB_DEFINE_DGMM_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, int, int, const scalar_type *, int, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    char side = 'R'; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldc; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, b_stride = tc->C_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *b_oracle = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !b_oracle) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(side, m, n, A_batch, lda, x_batch, 1, b_oracle, ldb, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(b_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *b_cand = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!b_cand) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_cand, ldb, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(b_cand, b_oracle, total_b, dtype_enum, norm_fn(b_oracle, total_b))); \
    if (fb_judge_has_nan_inf(b_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(b_cand); \
    if (ns_out) { \
        scalar_type *b_time = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (b_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_time, ldb, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_time, ldb, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(b_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(b_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_DGMM_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, int, int, const scalar_type *, int, const scalar_type *, int, scalar_type *, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    char side = 'R'; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldc; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, b_stride = tc->C_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *b_oracle = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !b_oracle) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(side, m, n, A_batch, lda, x_batch, 1, b_oracle, ldb, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(b_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *b_cand = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!b_cand) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_cand, ldb, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(b_cand, b_oracle, total_b, dtype_enum, norm_fn((const real_type *)b_oracle, total_b))); \
    if (fb_judge_has_nan_inf(b_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(b_cand); \
    if (ns_out) { \
        scalar_type *b_time = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (b_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_time, ldb, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(side, m, n, A_batch, lda, x_batch, 1, b_time, ldb, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(b_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(b_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, int, int, const scalar_type *, int, long long, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    char side = 'R'; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldc; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, b_stride = tc->C_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *b_oracle = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !b_oracle) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_oracle, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(b_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *b_cand = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!b_cand) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_cand, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(b_cand, b_oracle, total_b, dtype_enum, norm_fn(b_oracle, total_b))); \
    if (fb_judge_has_nan_inf(b_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(b_cand); \
    if (ns_out) { \
        scalar_type *b_time = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (b_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(b_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(b_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, int, int, const scalar_type *, int, long long, const scalar_type *, int, long long, scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    char side = 'R'; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldc; \
    size_t a_stride = tc->A_elems, x_stride = tc->B_elems, b_stride = tc->C_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *x_batch = (scalar_type *)repeat_buf(tc->B, x_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *b_oracle = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !x_batch || !b_oracle) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_oracle, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(b_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *b_cand = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!b_cand) { \
        free(A_batch); free(x_batch); free(b_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_cand, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(b_cand, b_oracle, total_b, dtype_enum, norm_fn((const real_type *)b_oracle, total_b))); \
    if (fb_judge_has_nan_inf(b_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(b_cand); \
    if (ns_out) { \
        scalar_type *b_time = (scalar_type *)repeat_buf(tc->C_init, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (b_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(side, m, n, A_batch, lda, (long long)a_stride, x_batch, 1, (long long)x_stride, b_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(b_time); \
    } \
    free(A_batch); \
    free(x_batch); \
    free(b_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_DGMM_BATCH_RUNNER_REAL(run_sdgmm_batch, sdgmm_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_DGMM_BATCH_RUNNER_REAL(run_ddgmm_batch, ddgmm_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_DGMM_BATCH_RUNNER_COMPLEX(run_cdgmm_batch, cdgmm_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_DGMM_BATCH_RUNNER_COMPLEX(run_zdgmm_batch, zdgmm_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_REAL(run_sdgmm_batch_strided, sdgmm_batch_strided, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_REAL(run_ddgmm_batch_strided, ddgmm_batch_strided, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_COMPLEX(run_cdgmm_batch_strided, cdgmm_batch_strided, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float)
FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_COMPLEX(run_zdgmm_batch_strided, zdgmm_batch_strided, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double)

#undef FB_DEFINE_DGMM_BATCH_RUNNER_REAL
#undef FB_DEFINE_DGMM_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_REAL
#undef FB_DEFINE_DGMM_BATCH_STRIDED_RUNNER_COMPLEX

#define FB_DEFINE_SYMM_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         const scalar_type **, int, scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **B_array = (const scalar_type **)make_const_batch_ptrs(B_batch, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle || !A_array || !B_array || !C_oracle_array) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_SYRK_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !C_oracle || !A_array || !C_oracle_array) { \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_SYR2K_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         const scalar_type **, int, scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **B_array = (const scalar_type **)make_const_batch_ptrs(B_batch, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle || !A_array || !B_array || !C_oracle_array) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_TRSM_BATCH_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, char, char, int, int, scalar_type, const scalar_type **, int, \
                         scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_oracle = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **B_oracle_array = (scalar_type **)make_batch_ptrs(B_oracle, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_oracle || !A_array || !B_oracle_array) { \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_oracle_array, ldb, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(B_oracle, total_b, dtype_enum)) { \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *B_cand = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **B_cand_array = (scalar_type **)make_batch_ptrs(B_cand, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!B_cand || !B_cand_array) { \
        free(B_cand_array); free(B_cand); \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_cand_array, ldb, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(B_cand, B_oracle, total_b, dtype_enum, norm_fn(B_oracle, total_b))); \
    if (fb_judge_has_nan_inf(B_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(B_cand_array); \
    free(B_cand); \
    if (ns_out) { \
        scalar_type *B_time = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **B_time_array = (scalar_type **)make_batch_ptrs(B_time, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (B_time && B_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_time_array, ldb, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_time_array, ldb, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(B_time_array); \
        free(B_time); \
    } \
    free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_REAL(fn_name, field, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, char, char, int, int, scalar_type, const scalar_type *, int, long long, \
                         scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_oracle = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_oracle) { \
        free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_oracle, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(B_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(B_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *B_cand = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!B_cand) { \
        free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_cand, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(B_cand, B_oracle, total_b, dtype_enum, norm_fn(B_oracle, total_b))); \
    if (fb_judge_has_nan_inf(B_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(B_cand); \
    if (ns_out) { \
        scalar_type *B_time = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (B_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(B_time); \
    } \
    free(A_batch); free(B_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_SYMM_BATCH_RUNNER_REAL(run_ssymm_batch, ssymm_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_SYMM_BATCH_RUNNER_REAL(run_dsymm_batch, dsymm_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_SYRK_BATCH_RUNNER_REAL(run_ssyrk_batch, ssyrk_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_SYRK_BATCH_RUNNER_REAL(run_dsyrk_batch, dsyrk_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_SYR2K_BATCH_RUNNER_REAL(run_ssyr2k_batch, ssyr2k_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_SYR2K_BATCH_RUNNER_REAL(run_dsyr2k_batch, dsyr2k_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_TRSM_BATCH_RUNNER_REAL(run_strsm_batch, strsm_batch, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_TRSM_BATCH_RUNNER_REAL(run_dtrsm_batch, dtrsm_batch, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_REAL(run_strsm_batch_strided, strsm_batch_strided, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_REAL(run_dtrsm_batch_strided, dtrsm_batch_strided, double, FB_DTYPE_F64, fb_norm_frob_f64)

#undef FB_DEFINE_SYMM_BATCH_RUNNER_REAL
#undef FB_DEFINE_SYRK_BATCH_RUNNER_REAL
#undef FB_DEFINE_SYR2K_BATCH_RUNNER_REAL
#undef FB_DEFINE_TRSM_BATCH_RUNNER_REAL
#undef FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_REAL

#define FB_DEFINE_SYMM_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         const scalar_type **, int, scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **B_array = (const scalar_type **)make_const_batch_ptrs(B_batch, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle || !A_array || !B_array || !C_oracle_array) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', m, n, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_SYRK_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda, ldc = (int)tc->ldc; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    size_t a_stride = tc->A_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !C_oracle || !A_array || !C_oracle_array) { \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free(C_oracle_array); free(A_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_SYR2K_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, int, int, scalar_type, const scalar_type **, int, \
                         const scalar_type **, int, scalar_type, scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; \
    scalar_type beta; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    __real__(beta) = (real_type)tc->beta; __imag__(beta) = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **B_array = (const scalar_type **)make_const_batch_ptrs(B_batch, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_oracle_array = (scalar_type **)make_batch_ptrs(C_oracle, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle || !A_array || !B_array || !C_oracle_array) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_oracle_array, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **C_cand_array = (scalar_type **)make_batch_ptrs(C_cand, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand || !C_cand_array) { \
        free(C_cand_array); free(C_cand); \
        free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_cand_array, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand_array); \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **C_time_array = (scalar_type **)make_batch_ptrs(C_time, c_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time && C_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', n, k, alpha, A_array, lda, B_array, ldb, beta, C_time_array, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(C_time_array); \
        free(C_time); \
    } \
    free((void *)A_array); free((void *)B_array); free(C_oracle_array); \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_TRSM_BATCH_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, char, char, int, int, scalar_type, const scalar_type **, int, \
                         scalar_type **, int, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb; \
    scalar_type alpha; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_oracle = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    const scalar_type **A_array = (const scalar_type **)make_const_batch_ptrs(A_batch, a_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **B_oracle_array = (scalar_type **)make_batch_ptrs(B_oracle, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_oracle || !A_array || !B_oracle_array) { \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_oracle_array, ldb, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(B_oracle, total_b, dtype_enum)) { \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *B_cand = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type **B_cand_array = (scalar_type **)make_batch_ptrs(B_cand, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!B_cand || !B_cand_array) { \
        free(B_cand_array); free(B_cand); \
        free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_cand_array, ldb, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(B_cand, B_oracle, total_b, dtype_enum, norm_fn((const real_type *)B_oracle, total_b))); \
    if (fb_judge_has_nan_inf(B_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(B_cand_array); \
    free(B_cand); \
    if (ns_out) { \
        scalar_type *B_time = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        scalar_type **B_time_array = (scalar_type **)make_batch_ptrs(B_time, b_stride * sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (B_time && B_time_array) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_time_array, ldb, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_array, lda, B_time_array, ldb, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(B_time_array); \
        free(B_time); \
    } \
    free((void *)A_array); free(B_oracle_array); free(A_batch); free(B_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_COMPLEX(fn_name, field, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    typedef void (*fn_t)(char, char, char, char, int, int, scalar_type, const scalar_type *, int, long long, \
                         scalar_type *, int, long long, int); \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_t oracle_fn = (fn_t)oracle->field; \
    fn_t cand_fn = (fn_t)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda, ldb = (int)tc->ldb; \
    scalar_type alpha; \
    __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems; \
    size_t total_b = b_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_oracle = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_oracle) { \
        free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_oracle, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(B_oracle, total_b, dtype_enum)) { \
        free(A_batch); free(B_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *B_cand = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!B_cand) { \
        free(A_batch); free(B_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_cand, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(B_cand, B_oracle, total_b, dtype_enum, norm_fn((const real_type *)B_oracle, total_b))); \
    if (fb_judge_has_nan_inf(B_cand, total_b, dtype_enum)) res->is_fatal = true; \
    free(B_cand); \
    if (ns_out) { \
        scalar_type *B_time = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (B_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('L', 'U', 'N', 'N', m, n, alpha, A_batch, lda, (long long)a_stride, B_time, ldb, (long long)b_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
        free(B_time); \
    } \
    free(A_batch); free(B_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_SYMM_BATCH_RUNNER_COMPLEX(run_csymm_batch, csymm_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_SYMM_BATCH_RUNNER_COMPLEX(run_zsymm_batch, zsymm_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_SYRK_BATCH_RUNNER_COMPLEX(run_csyrk_batch, csyrk_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_SYRK_BATCH_RUNNER_COMPLEX(run_zsyrk_batch, zsyrk_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_SYR2K_BATCH_RUNNER_COMPLEX(run_csyr2k_batch, csyr2k_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_SYR2K_BATCH_RUNNER_COMPLEX(run_zsyr2k_batch, zsyr2k_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_TRSM_BATCH_RUNNER_COMPLEX(run_ctrsm_batch, ctrsm_batch, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_TRSM_BATCH_RUNNER_COMPLEX(run_ztrsm_batch, ztrsm_batch, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_COMPLEX(run_ctrsm_batch_strided, ctrsm_batch_strided, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_COMPLEX(run_ztrsm_batch_strided, ztrsm_batch_strided, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

#undef FB_DEFINE_SYMM_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_SYRK_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_SYR2K_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_TRSM_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_TRSM_BATCH_STRIDED_RUNNER_COMPLEX

/* =========================================================================
 * Phase 2 — L2 SYMV/HEMV (symmetric/hermitian matrix-vector multiply)
 *
 * Corpus layout: A = n×n matrix (tc->A, lda), x = vector (tc->B, n),
 *                y_init = vector (tc->C_init, n)
 * ========================================================================= */

static fb_judge_status_t run_ssymv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssymv || !cand->ssymv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *A = (const float *)tc->A, *x = (const float *)tc->B;

    float *yo = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *yc = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        float *yt = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsymv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsymv || !cand->dsymv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    double alpha = tc->alpha, beta = tc->beta;
    const double *A = (const double *)tc->A, *x = (const double *)tc->B;

    double *yo = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *yc = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        double *yt = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_chemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chemv || !cand->chemv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->B;

    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->chemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->chemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zhemv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhemv || !cand->zhemv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->B;

    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zhemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zhemv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — L2 TRMV/TRSV (x in-place)
 *
 * Corpus: A = n×n matrix (tc->A, lda), x_init = vector (tc->B, n)
 * After call, x is overwritten with result.
 * ========================================================================= */

static fb_judge_status_t run_strmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->strmv || !cand->strmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const float *A = (const float *)tc->A;

    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->strmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->strmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->strmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->strmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dtrmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtrmv || !cand->dtrmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const double *A = (const double *)tc->A;

    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dtrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dtrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ctrmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctrmv || !cand->ctrmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;

    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ctrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ctrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ztrmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztrmv || !cand->ztrmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;

    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ztrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ztrmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* STRSV / DTRSV / CTRSV / ZTRSV — same pattern as TRMV but x = RHS → solution */

static fb_judge_status_t run_strsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->strsv || !cand->strsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const float *A = (const float *)tc->A;

    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->strsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->strsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->strsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->strsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dtrsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtrsv || !cand->dtrsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const double *A = (const double *)tc->A;

    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dtrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dtrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ctrsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctrsv || !cand->ctrsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;

    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ctrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ctrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ztrsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztrsv || !cand->ztrsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;

    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ztrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ztrsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, lda, xt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — L2 GER/GERU/GERC (rank-1 update, A in-place)
 *
 * Corpus: x = vector (tc->A, m), y = vector (tc->B, n),
 *         A_init = matrix (tc->C_init, m × lda)
 * ========================================================================= */

static fb_judge_status_t run_sger(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sger || !cand->sger) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    float alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A;
    const float *y = (const float *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    float *Ao = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Ac = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Ao, (int)m, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        float *At = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->sger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->sger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dger(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dger || !cand->dger) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    double alpha = tc->alpha;
    const double *x = (const double *)tc->A;
    const double *y = (const double *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    double *Ao = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Ac = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Ao, (int)m, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        double *At = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dger(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeru(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgeru || !cand->cgeru) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    fb_complex_float_t *Ao = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Ac = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_float_t *At = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgerc(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgerc || !cand->cgerc) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    fb_complex_float_t *Ao = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Ac = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_float_t *At = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeru(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgeru || !cand->zgeru) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    fb_complex_double_t *Ao = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Ac = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_double_t *At = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zgeru(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgerc(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgerc || !cand->zgerc) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    size_t A_sz = (size_t)(m * lda);

    fb_complex_double_t *Ao = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Ac = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)m, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_double_t *At = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zgerc(FB_LAYOUT_ROW_MAJOR, m, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* =========================================================================
 * Phase 2 — L3 CGEMM / ZGEMM
 * ========================================================================= */

static fb_judge_status_t run_cgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgemm || !cand->cgemm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *B = (const fb_complex_float_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);

    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        fb_complex_float_t *Ct = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgemm || !cand->zgemm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *B = (const fb_complex_double_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);

    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        fb_complex_double_t *Ct = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

typedef void (*fb_sgemm_batch_fn_t)(char, char, int, int, int, float,
                                    const float *, int, const float *, int,
                                    float, float *, int, int);
typedef void (*fb_dgemm_batch_fn_t)(char, char, int, int, int, double,
                                    const double *, int, const double *, int,
                                    double, double *, int, int);
typedef void (*fb_cgemm_batch_fn_t)(char, char, int, int, int,
                                    fb_complex_float_t,
                                    const fb_complex_float_t *, int,
                                    const fb_complex_float_t *, int,
                                    fb_complex_float_t,
                                    fb_complex_float_t *, int, int);
typedef void (*fb_zgemm_batch_fn_t)(char, char, int, int, int,
                                    fb_complex_double_t,
                                    const fb_complex_double_t *, int,
                                    const fb_complex_double_t *, int,
                                    fb_complex_double_t,
                                    fb_complex_double_t *, int, int);

typedef void (*fb_plain_sgemm_strided_fn_t)(char, char, int, int, int, float,
                                            const float *, int, long long,
                                            const float *, int, long long,
                                            float, float *, int, long long, int);
typedef void (*fb_plain_dgemm_strided_fn_t)(char, char, int, int, int, double,
                                            const double *, int, long long,
                                            const double *, int, long long,
                                            double, double *, int, long long, int);
typedef void (*fb_plain_cgemm_strided_fn_t)(char, char, int, int, int,
                                            fb_complex_float_t,
                                            const fb_complex_float_t *, int, long long,
                                            const fb_complex_float_t *, int, long long,
                                            fb_complex_float_t,
                                            fb_complex_float_t *, int, long long, int);
typedef void (*fb_plain_zgemm_strided_fn_t)(char, char, int, int, int,
                                            fb_complex_double_t,
                                            const fb_complex_double_t *, int, long long,
                                            const fb_complex_double_t *, int, long long,
                                            fb_complex_double_t,
                                            fb_complex_double_t *, int, long long, int);

typedef void (*fb_cblas_sgemm_batch_fn_t)(int, char, char, int, int, int,
                                          float,
                                          const float *, int,
                                          const float *, int,
                                          float,
                                          float *, int, int);
typedef void (*fb_cblas_dgemm_batch_fn_t)(int, char, char, int, int, int,
                                          double,
                                          const double *, int,
                                          const double *, int,
                                          double,
                                          double *, int, int);
typedef void (*fb_cblas_cgemm_batch_fn_t)(int, char, char, int, int, int,
                                          fb_complex_float_t,
                                          const fb_complex_float_t *, int,
                                          const fb_complex_float_t *, int,
                                          fb_complex_float_t,
                                          fb_complex_float_t *, int, int);
typedef void (*fb_cblas_zgemm_batch_fn_t)(int, char, char, int, int, int,
                                          fb_complex_double_t,
                                          const fb_complex_double_t *, int,
                                          const fb_complex_double_t *, int,
                                          fb_complex_double_t,
                                          fb_complex_double_t *, int, int);

typedef void (*fb_sgemm_strided_fn_t)(int, int, int, int, int, int, float,
                                      const float *, int, long long,
                                      const float *, int, long long,
                                      float, float *, int, long long, int);
typedef void (*fb_dgemm_strided_fn_t)(int, int, int, int, int, int, double,
                                      const double *, int, long long,
                                      const double *, int, long long,
                                      double, double *, int, long long, int);
typedef void (*fb_cgemm_strided_fn_t)(int, int, int, int, int, int,
                                      fb_complex_float_t,
                                      const fb_complex_float_t *, int, long long,
                                      const fb_complex_float_t *, int, long long,
                                      fb_complex_float_t,
                                      fb_complex_float_t *, int, long long, int);
typedef void (*fb_zgemm_strided_fn_t)(int, int, int, int, int, int,
                                      fb_complex_double_t,
                                      const fb_complex_double_t *, int, long long,
                                      const fb_complex_double_t *, int, long long,
                                      fb_complex_double_t,
                                      fb_complex_double_t *, int, long long, int);

typedef void (*fb_cblas_sgemm_strided_fn_t)(int, char, char, int, int, int,
                                            float,
                                            const float *, int, long long,
                                            const float *, int, long long,
                                            float,
                                            float *, int, long long, int);
typedef void (*fb_cblas_dgemm_strided_fn_t)(int, char, char, int, int, int,
                                            double,
                                            const double *, int, long long,
                                            const double *, int, long long,
                                            double,
                                            double *, int, long long, int);
typedef void (*fb_cblas_cgemm_strided_fn_t)(int, char, char, int, int, int,
                                            fb_complex_float_t,
                                            const fb_complex_float_t *, int, long long,
                                            const fb_complex_float_t *, int, long long,
                                            fb_complex_float_t,
                                            fb_complex_float_t *, int, long long, int);
typedef void (*fb_cblas_zgemm_strided_fn_t)(int, char, char, int, int, int,
                                            fb_complex_double_t,
                                            const fb_complex_double_t *, int, long long,
                                            const fb_complex_double_t *, int, long long,
                                            fb_complex_double_t,
                                            fb_complex_double_t *, int, long long, int);

static fb_judge_status_t run_sgemm_batch(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    bool oracle_has_batch = (oracle->sgemm_batch != NULL);
    bool cand_has_batch = (cand->sgemm_batch != NULL);
    bool oracle_has_single = (oracle->sgemm != NULL);
    bool cand_has_single = (cand->sgemm != NULL);
    if ((!oracle_has_batch && !oracle_has_single) ||
        (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL;

    fb_sgemm_batch_fn_t oracle_fn = (fb_sgemm_batch_fn_t)oracle->sgemm_batch;
    fb_sgemm_batch_fn_t cand_fn = (fb_sgemm_batch_fn_t)cand->sgemm_batch;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    float *A_batch = (float *)repeat_buf(tc->A, a_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    float *B_batch = (float *)repeat_buf(tc->B, b_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    float *C_oracle = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    if (oracle_has_batch) {
        oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                  C_oracle, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            oracle->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                          m, n, k, alpha,
                          A_batch + ((size_t)b * a_stride), lda,
                          B_batch + ((size_t)b * b_stride), ldb,
                          beta,
                          C_oracle + ((size_t)b * c_stride), ldc);
        }
    }
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_F32)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *C_cand = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    if (cand_has_batch) {
        cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                C_cand, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                        m, n, k, alpha,
                        A_batch + ((size_t)b * a_stride), lda,
                        B_batch + ((size_t)b * b_stride), ldb,
                        beta,
                        C_cand + ((size_t)b * c_stride), ldc);
        }
    }
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_F32, fb_norm_frob_f32(C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_F32)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        float *C_time = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgemm_batch(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    bool oracle_has_batch = (oracle->dgemm_batch != NULL);
    bool cand_has_batch = (cand->dgemm_batch != NULL);
    bool oracle_has_single = (oracle->dgemm != NULL);
    bool cand_has_single = (cand->dgemm != NULL);
    if ((!oracle_has_batch && !oracle_has_single) ||
        (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL;

    fb_dgemm_batch_fn_t oracle_fn = (fb_dgemm_batch_fn_t)oracle->dgemm_batch;
    fb_dgemm_batch_fn_t cand_fn = (fb_dgemm_batch_fn_t)cand->dgemm_batch;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    double alpha = tc->alpha, beta = tc->beta;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    double *A_batch = (double *)repeat_buf(tc->A, a_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    double *B_batch = (double *)repeat_buf(tc->B, b_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    double *C_oracle = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    if (oracle_has_batch) {
        oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                  C_oracle, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            oracle->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                          m, n, k, alpha,
                          A_batch + ((size_t)b * a_stride), lda,
                          B_batch + ((size_t)b * b_stride), ldb,
                          beta,
                          C_oracle + ((size_t)b * c_stride), ldc);
        }
    }
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_F64)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *C_cand = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    if (cand_has_batch) {
        cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                C_cand, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                        m, n, k, alpha,
                        A_batch + ((size_t)b * a_stride), lda,
                        B_batch + ((size_t)b * b_stride), ldb,
                        beta,
                        C_cand + ((size_t)b * c_stride), ldc);
        }
    }
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_F64, fb_norm_frob_f64(C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_F64)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        double *C_time = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgemm_batch(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    bool oracle_has_batch = (oracle->cgemm_batch != NULL);
    bool cand_has_batch = (cand->cgemm_batch != NULL);
    bool oracle_has_single = (oracle->cgemm != NULL);
    bool cand_has_single = (cand->cgemm != NULL);
    if ((!oracle_has_batch && !oracle_has_single) ||
        (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL;

    fb_cgemm_batch_fn_t oracle_fn = (fb_cgemm_batch_fn_t)oracle->cgemm_batch;
    fb_cgemm_batch_fn_t cand_fn = (fb_cgemm_batch_fn_t)cand->cgemm_batch;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    fb_complex_float_t *A_batch = (fb_complex_float_t *)repeat_buf(tc->A, a_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_float_t *B_batch = (fb_complex_float_t *)repeat_buf(tc->B, b_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_float_t *C_oracle = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    if (oracle_has_batch) {
        oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                  C_oracle, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            oracle->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                          m, n, k, alpha,
                          A_batch + ((size_t)b * a_stride), lda,
                          B_batch + ((size_t)b * b_stride), ldb,
                          beta,
                          C_oracle + ((size_t)b * c_stride), ldc);
        }
    }
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_CF32)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_float_t *C_cand = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    if (cand_has_batch) {
        cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                C_cand, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                        m, n, k, alpha,
                        A_batch + ((size_t)b * a_stride), lda,
                        B_batch + ((size_t)b * b_stride), ldb,
                        beta,
                        C_cand + ((size_t)b * c_stride), ldc);
        }
    }
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_CF32)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        fb_complex_float_t *C_time = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->cgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgemm_batch(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    bool oracle_has_batch = (oracle->zgemm_batch != NULL);
    bool cand_has_batch = (cand->zgemm_batch != NULL);
    bool oracle_has_single = (oracle->zgemm != NULL);
    bool cand_has_single = (cand->zgemm != NULL);
    if ((!oracle_has_batch && !oracle_has_single) ||
        (!cand_has_batch && !cand_has_single)) return FB_JUDGE_ERR_NOT_IMPL;

    fb_zgemm_batch_fn_t oracle_fn = (fb_zgemm_batch_fn_t)oracle->zgemm_batch;
    fb_zgemm_batch_fn_t cand_fn = (fb_zgemm_batch_fn_t)cand->zgemm_batch;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    fb_complex_double_t *A_batch = (fb_complex_double_t *)repeat_buf(tc->A, a_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_double_t *B_batch = (fb_complex_double_t *)repeat_buf(tc->B, b_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_double_t *C_oracle = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    if (oracle_has_batch) {
        oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                  C_oracle, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            oracle->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                          m, n, k, alpha,
                          A_batch + ((size_t)b * a_stride), lda,
                          B_batch + ((size_t)b * b_stride), ldb,
                          beta,
                          C_oracle + ((size_t)b * c_stride), ldc);
        }
    }
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_CF64)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_double_t *C_cand = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    if (cand_has_batch) {
        cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta,
                C_cand, ldc, FB_JUDGE_BATCH_COUNT);
    } else {
        for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
            cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                        m, n, k, alpha,
                        A_batch + ((size_t)b * a_stride), lda,
                        B_batch + ((size_t)b * b_stride), ldb,
                        beta,
                        C_cand + ((size_t)b * c_stride), ldc);
        }
    }
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_CF64)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        fb_complex_double_t *C_time = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                if (cand_has_batch) {
                    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb,
                            beta, C_time, ldc, FB_JUDGE_BATCH_COUNT);
                } else {
                    for (int b = 0; b < FB_JUDGE_BATCH_COUNT; b++) {
                        cand->zgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                                    m, n, k, alpha,
                                    A_batch + ((size_t)b * a_stride), lda,
                                    B_batch + ((size_t)b * b_stride), ldb,
                                    beta,
                                    C_time + ((size_t)b * c_stride), ldc);
                    }
                }
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgemm_strided(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgemm_strided || !cand->sgemm_strided) return FB_JUDGE_ERR_NOT_IMPL;

    fb_sgemm_strided_fn_t oracle_fn = (fb_sgemm_strided_fn_t)oracle->sgemm_strided;
    fb_sgemm_strided_fn_t cand_fn = (fb_sgemm_strided_fn_t)cand->sgemm_strided;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    float *A_batch = (float *)repeat_buf(tc->A, a_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    float *B_batch = (float *)repeat_buf(tc->B, b_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    float *C_oracle = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
              A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
              beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_F32)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    float *C_cand = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
            A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
            beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_F32, fb_norm_frob_f32(C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_F32)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        float *C_time = (float *)repeat_buf(tc->C_init, c_stride, sizeof(float), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgemm_strided(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgemm_strided || !cand->dgemm_strided) return FB_JUDGE_ERR_NOT_IMPL;

    fb_dgemm_strided_fn_t oracle_fn = (fb_dgemm_strided_fn_t)oracle->dgemm_strided;
    fb_dgemm_strided_fn_t cand_fn = (fb_dgemm_strided_fn_t)cand->dgemm_strided;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    double alpha = tc->alpha, beta = tc->beta;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    double *A_batch = (double *)repeat_buf(tc->A, a_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    double *B_batch = (double *)repeat_buf(tc->B, b_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    double *C_oracle = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
              A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
              beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_F64)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *C_cand = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
            A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
            beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_F64, fb_norm_frob_f64(C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_F64)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        double *C_time = (double *)repeat_buf(tc->C_init, c_stride, sizeof(double), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgemm_strided(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgemm_strided || !cand->cgemm_strided) return FB_JUDGE_ERR_NOT_IMPL;

    fb_cgemm_strided_fn_t oracle_fn = (fb_cgemm_strided_fn_t)oracle->cgemm_strided;
    fb_cgemm_strided_fn_t cand_fn = (fb_cgemm_strided_fn_t)cand->cgemm_strided;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    fb_complex_float_t *A_batch = (fb_complex_float_t *)repeat_buf(tc->A, a_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_float_t *B_batch = (fb_complex_float_t *)repeat_buf(tc->B, b_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_float_t *C_oracle = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
              A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
              beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_CF32)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_float_t *C_cand = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
            A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
            beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_CF32)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        fb_complex_float_t *C_time = (fb_complex_float_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_float_t), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgemm_strided(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgemm_strided || !cand->zgemm_strided) return FB_JUDGE_ERR_NOT_IMPL;

    fb_zgemm_strided_fn_t oracle_fn = (fb_zgemm_strided_fn_t)oracle->zgemm_strided;
    fb_zgemm_strided_fn_t cand_fn = (fb_zgemm_strided_fn_t)cand->zgemm_strided;
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems;
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT;

    fb_complex_double_t *A_batch = (fb_complex_double_t *)repeat_buf(tc->A, a_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_double_t *B_batch = (fb_complex_double_t *)repeat_buf(tc->B, b_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    fb_complex_double_t *C_oracle = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    if (!A_batch || !B_batch || !C_oracle) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }

    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
              A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
              beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    if (fb_judge_has_nan_inf(C_oracle, total_c, FB_DTYPE_CF64)) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_double_t *C_cand = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
    if (!C_cand) {
        free(A_batch); free(B_batch); free(C_oracle);
        result_fatal(res); return FB_JUDGE_ERR_ALLOC;
    }
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha,
            A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride,
            beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT);
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)C_oracle, total_c)));
    if (fb_judge_has_nan_inf(C_cand, total_c, FB_DTYPE_CF64)) res->is_fatal = true;
    free(C_cand);

    if (ns_out) {
        fb_complex_double_t *C_time = (fb_complex_double_t *)repeat_buf(tc->C_init, c_stride, sizeof(fb_complex_double_t), FB_JUDGE_BATCH_COUNT);
        if (C_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
            }
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns();
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k,
                        alpha, A_batch, lda, (long long)a_stride,
                        B_batch, ldb, (long long)b_stride,
                        beta, C_time, ldc, (long long)c_stride,
                        FB_JUDGE_BATCH_COUNT);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(C_time);
            *ns_out = best;
        } else {
            *ns_out = 0;
        }
    }

    free(A_batch); free(B_batch); free(C_oracle);
    return FB_JUDGE_OK;
}

#define FB_DEFINE_GEMM3M_BATCH_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_oracle, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_cand, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM3M_BATCH_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    scalar_type beta;  __real__(beta)  = (real_type)tc->beta;  __imag__(beta)  = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_oracle, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_cand, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_oracle, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_cand, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    scalar_type beta;  __real__(beta)  = (real_type)tc->beta;  __imag__(beta)  = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_oracle, ldc, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_cand, ldc, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(FB_LAYOUT_ROW_MAJOR, 'N', 'N', m, n, k, alpha, A_batch, lda, B_batch, ldb, beta, C_time, ldc, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM3M_STRIDED_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMM3M_STRIDED_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    scalar_type beta;  __real__(beta)  = (real_type)tc->beta;  __imag__(beta)  = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha, beta = (scalar_type)tc->beta; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn(C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int m = (int)tc->m, n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    scalar_type beta;  __real__(beta)  = (real_type)tc->beta;  __imag__(beta)  = imag_zero; \
    size_t a_stride = tc->A_elems, b_stride = tc->B_elems, c_stride = tc->C_elems; \
    size_t total_c = c_stride * (size_t)FB_JUDGE_BATCH_COUNT; \
    scalar_type *A_batch = (scalar_type *)repeat_buf(tc->A, a_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *B_batch = (scalar_type *)repeat_buf(tc->B, b_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    scalar_type *C_oracle = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!A_batch || !B_batch || !C_oracle) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    oracle_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_oracle, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    if (fb_judge_has_nan_inf(C_oracle, total_c, dtype_enum)) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_oracle_fatal(res); return FB_JUDGE_OK; \
    } \
    scalar_type *C_cand = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
    if (!C_cand) { \
        free(A_batch); free(B_batch); free(C_oracle); \
        result_fatal(res); return FB_JUDGE_ERR_ALLOC; \
    } \
    cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_cand, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
    result_from_relerr(res, fb_judge_relerr(C_cand, C_oracle, total_c, dtype_enum, norm_fn((const real_type *)C_oracle, total_c))); \
    if (fb_judge_has_nan_inf(C_cand, total_c, dtype_enum)) res->is_fatal = true; \
    free(C_cand); \
    if (ns_out) { \
        scalar_type *C_time = (scalar_type *)repeat_buf(tc->C_init, c_stride, sizeof(scalar_type), FB_JUDGE_BATCH_COUNT); \
        if (C_time) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
            uint64_t best = UINT64_MAX; \
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('N', 'N', m, n, k, alpha, A_batch, lda, (long long)a_stride, B_batch, ldb, (long long)b_stride, beta, C_time, ldc, (long long)c_stride, FB_JUDGE_BATCH_COUNT); \
                uint64_t dt = fb_judge_time_ns() - t0; \
                if (dt < best) best = dt; \
            } \
            free(C_time); \
            *ns_out = best; \
        } else { \
            *ns_out = 0; \
        } \
    } \
    free(A_batch); free(B_batch); free(C_oracle); \
    return FB_JUDGE_OK; \
}

FB_DEFINE_GEMM3M_BATCH_RUNNER_REAL(run_sgemm3m_batch, sgemm3m_batch, fb_sgemm_batch_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_GEMM3M_BATCH_RUNNER_REAL(run_dgemm3m_batch, dgemm3m_batch, fb_dgemm_batch_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_GEMM3M_BATCH_RUNNER_COMPLEX(run_cgemm3m_batch, cgemm3m_batch, fb_cgemm_batch_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMM3M_BATCH_RUNNER_COMPLEX(run_zgemm3m_batch, zgemm3m_batch, fb_zgemm_batch_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_REAL(run_cblas_sgemm3m_batch, cblas_sgemm3m_batch, fb_cblas_sgemm_batch_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_REAL(run_cblas_dgemm3m_batch, cblas_dgemm3m_batch, fb_cblas_dgemm_batch_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_COMPLEX(run_cblas_cgemm3m_batch, cblas_cgemm3m_batch, fb_cblas_cgemm_batch_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_COMPLEX(run_cblas_zgemm3m_batch, cblas_zgemm3m_batch, fb_cblas_zgemm_batch_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_REAL(run_sgemm3m_batch_strided, sgemm3m_batch_strided, fb_plain_sgemm_strided_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_REAL(run_dgemm3m_batch_strided, dgemm3m_batch_strided, fb_plain_dgemm_strided_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_COMPLEX(run_cgemm3m_batch_strided, cgemm3m_batch_strided, fb_plain_cgemm_strided_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_COMPLEX(run_zgemm3m_batch_strided, zgemm3m_batch_strided, fb_plain_zgemm_strided_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

FB_DEFINE_GEMM3M_STRIDED_RUNNER_REAL(run_cblas_sgemm3m_batch_strided, cblas_sgemm3m_batch_strided, fb_cblas_sgemm_strided_fn_t, float, FB_DTYPE_F32, fb_norm_frob_f32)
FB_DEFINE_GEMM3M_STRIDED_RUNNER_REAL(run_cblas_dgemm3m_batch_strided, cblas_dgemm3m_batch_strided, fb_cblas_dgemm_strided_fn_t, double, FB_DTYPE_F64, fb_norm_frob_f64)
FB_DEFINE_GEMM3M_STRIDED_RUNNER_COMPLEX(run_cblas_cgemm3m_batch_strided, cblas_cgemm3m_batch_strided, fb_cblas_cgemm_strided_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMM3M_STRIDED_RUNNER_COMPLEX(run_cblas_zgemm3m_batch_strided, cblas_zgemm3m_batch_strided, fb_cblas_zgemm_strided_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

#undef FB_DEFINE_GEMM3M_BATCH_RUNNER_REAL
#undef FB_DEFINE_GEMM3M_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_REAL
#undef FB_DEFINE_CBLAS_GEMM3M_BATCH_RUNNER_COMPLEX
#undef FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_REAL
#undef FB_DEFINE_PLAIN_GEMM3M_STRIDED_RUNNER_COMPLEX
#undef FB_DEFINE_GEMM3M_STRIDED_RUNNER_REAL
#undef FB_DEFINE_GEMM3M_STRIDED_RUNNER_COMPLEX

typedef void (*fb_sgemmt_fn_t)(char, char, char, int, int, float,
                               const float *, int, const float *, int,
                               float, float *, int);
typedef void (*fb_dgemmt_fn_t)(char, char, char, int, int, double,
                               const double *, int, const double *, int,
                               double, double *, int);
typedef void (*fb_cgemmt_fn_t)(char, char, char, int, int,
                               fb_complex_float_t,
                               const fb_complex_float_t *, int,
                               const fb_complex_float_t *, int,
                               fb_complex_float_t,
                               fb_complex_float_t *, int);
typedef void (*fb_zgemmt_fn_t)(char, char, char, int, int,
                               fb_complex_double_t,
                               const fb_complex_double_t *, int,
                               const fb_complex_double_t *, int,
                               fb_complex_double_t,
                               fb_complex_double_t *, int);

#define FB_DEFINE_GEMMT_RUNNER_REAL(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha = (scalar_type)tc->alpha; \
    scalar_type beta = (scalar_type)tc->beta; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    size_t C_sz = (size_t)n * (size_t)ldc; \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Co, ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Cc, ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, n, n, ldc, ldc, dtype_enum, norm_fn(Co, n, n, ldc))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Ct, ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Ct, ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEMMT_RUNNER_COMPLEX(fn_name, field, fn_type, scalar_type, dtype_enum, norm_fn, real_type, imag_zero) \
static fb_judge_status_t fn_name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out) \
{ \
    if (!oracle->field || !cand->field) return FB_JUDGE_ERR_NOT_IMPL; \
    fn_type oracle_fn = (fn_type)oracle->field; \
    fn_type cand_fn = (fn_type)cand->field; \
    int n = (int)tc->n, k = (int)tc->k; \
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc; \
    scalar_type alpha; __real__(alpha) = (real_type)tc->alpha; __imag__(alpha) = imag_zero; \
    scalar_type beta;  __real__(beta)  = (real_type)tc->beta;  __imag__(beta)  = imag_zero; \
    const scalar_type *A = (const scalar_type *)tc->A; \
    const scalar_type *B = (const scalar_type *)tc->B; \
    size_t C_sz = (size_t)n * (size_t)ldc; \
    scalar_type *Co = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    oracle_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Co, ldc); \
    if (fb_judge_has_nan_inf(Co, C_sz, dtype_enum)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; } \
    scalar_type *Cc = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; } \
    cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Cc, ldc); \
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, n, n, ldc, ldc, dtype_enum, norm_fn((const real_type *)Co, C_sz))); \
    if (fb_judge_has_nan_inf(Cc, C_sz, dtype_enum)) res->is_fatal = true; \
    free(Cc); \
    if (ns_out) { \
        scalar_type *Ct = (scalar_type *)clone_buf(tc->C_init, C_sz, sizeof(scalar_type)); \
        if (Ct) { \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) \
                cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Ct, ldc); \
            uint64_t best = UINT64_MAX; \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { \
                uint64_t t0 = fb_judge_time_ns(); \
                cand_fn('U', 'N', 'N', n, k, alpha, A, lda, B, ldb, beta, Ct, ldc); \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt; \
            } \
            free(Ct); *ns_out = best; \
        } else { *ns_out = 0; } \
    } \
    free(Co); return FB_JUDGE_OK; \
}

FB_DEFINE_GEMMT_RUNNER_REAL(run_sgemmt, sgemmt, fb_sgemmt_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_GEMMT_RUNNER_REAL(run_dgemmt, dgemmt, fb_dgemmt_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_GEMMT_RUNNER_COMPLEX(run_cgemmt, cgemmt, fb_cgemmt_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMMT_RUNNER_COMPLEX(run_zgemmt, zgemmt, fb_zgemmt_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)
FB_DEFINE_GEMMT_RUNNER_REAL(run_cblas_sgemmt, cblas_sgemmt, fb_sgemmt_fn_t, float, FB_DTYPE_F32, fb_matrix_norm_frob_f32)
FB_DEFINE_GEMMT_RUNNER_REAL(run_cblas_dgemmt, cblas_dgemmt, fb_dgemmt_fn_t, double, FB_DTYPE_F64, fb_matrix_norm_frob_f64)
FB_DEFINE_GEMMT_RUNNER_COMPLEX(run_cblas_cgemmt, cblas_cgemmt, fb_cgemmt_fn_t, fb_complex_float_t, FB_DTYPE_CF32, fb_norm_frob_cf32, float, 0.0f)
FB_DEFINE_GEMMT_RUNNER_COMPLEX(run_cblas_zgemmt, cblas_zgemmt, fb_zgemmt_fn_t, fb_complex_double_t, FB_DTYPE_CF64, fb_norm_frob_cf64, double, 0.0)

#undef FB_DEFINE_GEMMT_RUNNER_REAL
#undef FB_DEFINE_GEMMT_RUNNER_COMPLEX

/* =========================================================================
 * Phase 2 — L3 SYMM / SYRK / TRMM / TRSM
 * ========================================================================= */

static fb_judge_status_t run_ssymm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssymm || !cand->ssymm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *A = (const float *)tc->A, *B = (const float *)tc->B;
    size_t C_sz = (size_t)(m * ldc);

    float *Co = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Cc = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Co, (int)m, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        float *Ct = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsymm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsymm || !cand->dsymm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    double alpha = tc->alpha, beta = tc->beta;
    const double *A = (const double *)tc->A, *B = (const double *)tc->B;
    size_t C_sz = (size_t)(m * ldc);

    double *Co = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Cc = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Co, (int)m, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        double *Ct = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ssyrk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssyrk || !cand->ssyrk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *A = (const float *)tc->A;
    size_t C_sz = (size_t)(n * ldc);

    float *Co = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Cc = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Co, (int)n, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        float *Ct = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyrk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsyrk || !cand->dsyrk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    double alpha = tc->alpha, beta = tc->beta;
    const double *A = (const double *)tc->A;
    size_t C_sz = (size_t)(n * ldc);

    double *Co = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Cc = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    double relerr = fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Co, (int)n, (int)n, (int)ldc));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        double *Ct = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

/* =========================================================================
 * L3 Complex Mirror Runners (slots 136–159)
 * CSYMM/ZSYMM/CHEMM/ZHEMM, CSYRK/ZSYRK/CHERK/ZHERK,
 * SSYR2K/DSYR2K/CSYR2K/ZSYR2K/CHER2K/ZHER2K, CTRMM/ZTRMM, CTRSM/ZTRSM
 * ========================================================================= */

/* --- CSYMM / ZSYMM / CHEMM / ZHEMM --- */

#define TIMED_ALLOC_AND_RUN_CF32(func_call, sz_sym, dtype_sym)                        \
    if (ns_out) {                                                                       \
        fb_complex_float_t *Ct = (fb_complex_float_t *)                                \
            clone_buf(tc->C_init, (sz_sym), sizeof(fb_complex_float_t));               \
        if (Ct) {                                                                       \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) func_call;                 \
            uint64_t best = UINT64_MAX;                                                 \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {                       \
                uint64_t t0 = fb_judge_time_ns(); func_call;                           \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;      \
            }                                                                           \
            free(Ct); *ns_out = best;                                                  \
        } else { *ns_out = 0; }                                                        \
    }

#define TIMED_ALLOC_AND_RUN_CF64(func_call, sz_sym, dtype_sym)                        \
    if (ns_out) {                                                                       \
        fb_complex_double_t *Ct = (fb_complex_double_t *)                              \
            clone_buf(tc->C_init, (sz_sym), sizeof(fb_complex_double_t));              \
        if (Ct) {                                                                       \
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) func_call;                 \
            uint64_t best = UINT64_MAX;                                                 \
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {                       \
                uint64_t t0 = fb_judge_time_ns(); func_call;                           \
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;      \
            }                                                                           \
            free(Ct); *ns_out = best;                                                  \
        } else { *ns_out = 0; }                                                        \
    }

static fb_judge_status_t run_csymm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csymm || !cand->csymm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *B = (const fb_complex_float_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->csymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsymm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zsymm || !cand->zsymm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *B = (const fb_complex_double_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zsymm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_chemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chemm || !cand->chemm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *B = (const fb_complex_float_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->chemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zhemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhemm || !cand->zhemm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *B = (const fb_complex_double_t *)tc->B;
    size_t C_sz = (size_t)(m * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)m, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zhemm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

/* --- CSYRK / ZSYRK: complex alpha+beta --- */

static fb_judge_status_t run_csyrk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csyrk || !cand->csyrk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->csyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsyrk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zsyrk || !cand->zsyrk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zsyrk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

/* --- CHERK / ZHERK: REAL alpha and beta (Hermitian rank-k) --- */

static fb_judge_status_t run_cherk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cherk || !cand->cherk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha;  /* REAL float — not complex! */
    float beta  = (float)tc->beta;   /* REAL float — not complex! */
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->cherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zherk(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zherk || !cand->zherk) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldc = (int)tc->ldc;
    double alpha = tc->alpha;  /* REAL double — not complex! */
    double beta  = tc->beta;   /* REAL double — not complex! */
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zherk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

/* --- SYR2K variants (real and complex) --- */

static fb_judge_status_t run_ssyr2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssyr2k || !cand->ssyr2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *A = (const float *)tc->A, *B = (const float *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    float *Co = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Cc = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_F32, fb_matrix_norm_frob_f32(Co, (int)n, (int)n, (int)ldc)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        float *Ct = (float *)clone_buf(tc->C_init, C_sz, sizeof(float));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyr2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsyr2k || !cand->dsyr2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    double alpha = tc->alpha, beta = tc->beta;
    const double *A = (const double *)tc->A, *B = (const double *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    double *Co = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_F64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Cc = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_F64, fb_matrix_norm_frob_f64(Co, (int)n, (int)n, (int)ldc)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Cc);
    if (ns_out) {
        double *Ct = (double *)clone_buf(tc->C_init, C_sz, sizeof(double));
        if (Ct) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Ct); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_csyr2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csyr2k || !cand->csyr2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *B = (const fb_complex_float_t *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->csyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsyr2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zsyr2k || !cand->zsyr2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *B = (const fb_complex_double_t *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zsyr2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

/* --- CHER2K: complex alpha, REAL float beta --- */

static fb_judge_status_t run_cher2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cher2k || !cand->cher2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    float beta = (float)tc->beta;  /* REAL float — not complex! */
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *B = (const fb_complex_float_t *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_float_t *Co = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF32)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Cc = (fb_complex_float_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_float_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF32(
        cand->cher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF32)
    free(Co); return FB_JUDGE_OK;
}

/* --- ZHER2K: complex alpha, REAL double beta --- */

static fb_judge_status_t run_zher2k(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zher2k || !cand->zher2k) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb, ldc = (int)tc->ldc;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    double beta = tc->beta;  /* REAL double — not complex! */
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *B = (const fb_complex_double_t *)tc->B;
    size_t C_sz = (size_t)(n * ldc);
    fb_complex_double_t *Co = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Co) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Co, ldc);
    if (fb_judge_has_nan_inf(Co, C_sz, FB_DTYPE_CF64)) { free(Co); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Cc = (fb_complex_double_t *)clone_buf(tc->C_init, C_sz, sizeof(fb_complex_double_t));
    if (!Cc) { free(Co); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Cc, ldc);
    result_from_relerr(res, fb_judge_relerr_matrix(Cc, Co, (int)n, (int)n, (int)ldc, (int)ldc,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Co, C_sz)));
    if (fb_judge_has_nan_inf(Cc, C_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Cc);
    TIMED_ALLOC_AND_RUN_CF64(
        cand->zher2k(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, lda, B, ldb, beta, Ct, ldc),
        C_sz, FB_DTYPE_CF64)
    free(Co); return FB_JUDGE_OK;
}

/* --- CTRMM / ZTRMM: B = alpha * A * B (B in-place, complex) --- */

static fb_judge_status_t run_ctrmm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctrmm || !cand->ctrmm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    size_t B_sz = (size_t)(m * ldb);
    fb_complex_float_t *Bo = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_CF32)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Bc = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    result_from_relerr(res, fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Bo, B_sz)));
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        fb_complex_float_t *Bt = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ctrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ctrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ztrmm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztrmm || !cand->ztrmm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    size_t B_sz = (size_t)(m * ldb);
    fb_complex_double_t *Bo = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_CF64)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Bc = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    result_from_relerr(res, fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Bo, B_sz)));
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        fb_complex_double_t *Bt = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ztrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ztrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

/* --- CTRSM / ZTRSM: B overwritten with solution (complex) --- */

static fb_judge_status_t run_ctrsm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctrsm || !cand->ctrsm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    size_t B_sz = (size_t)(m * ldb);
    fb_complex_float_t *Bo = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_CF32)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Bc = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    result_from_relerr(res, fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
        FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Bo, B_sz)));
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        fb_complex_float_t *Bt = (fb_complex_float_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_float_t));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ctrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ctrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ztrsm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztrsm || !cand->ztrsm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    size_t B_sz = (size_t)(m * ldb);
    fb_complex_double_t *Bo = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_CF64)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Bc = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    result_from_relerr(res, fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
        FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Bo, B_sz)));
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        fb_complex_double_t *Bt = (fb_complex_double_t *)clone_buf(tc->B, B_sz, sizeof(fb_complex_double_t));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ztrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ztrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

/* STRMM / DTRMM: B = alpha * A * B  (B in-place) */
static fb_judge_status_t run_strmm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->strmm || !cand->strmm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    float alpha = (float)tc->alpha;
    const float *A = (const float *)tc->A;
    size_t B_sz = (size_t)(m * ldb);

    float *Bo = (float *)clone_buf(tc->B, B_sz, sizeof(float));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->strmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_F32)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Bc = (float *)clone_buf(tc->B, B_sz, sizeof(float));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->strmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    double relerr = fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Bo, (int)m, (int)n, (int)ldb));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        float *Bt = (float *)clone_buf(tc->B, B_sz, sizeof(float));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->strmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->strmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dtrmm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtrmm || !cand->dtrmm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    double alpha = tc->alpha;
    const double *A = (const double *)tc->A;
    size_t B_sz = (size_t)(m * ldb);

    double *Bo = (double *)clone_buf(tc->B, B_sz, sizeof(double));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_F64)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Bc = (double *)clone_buf(tc->B, B_sz, sizeof(double));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    double relerr = fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Bo, (int)m, (int)n, (int)ldb));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        double *Bt = (double *)clone_buf(tc->B, B_sz, sizeof(double));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dtrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dtrmm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

/* STRSM / DTRSM: solve A*X = alpha*B for X, B overwritten with X */
static fb_judge_status_t run_strsm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->strsm || !cand->strsm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    float alpha = (float)tc->alpha;
    const float *A = (const float *)tc->A;
    size_t B_sz = (size_t)(m * ldb);

    float *Bo = (float *)clone_buf(tc->B, B_sz, sizeof(float));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->strsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_F32)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Bc = (float *)clone_buf(tc->B, B_sz, sizeof(float));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->strsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    double relerr = fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Bo, (int)m, (int)n, (int)ldb));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        float *Bt = (float *)clone_buf(tc->B, B_sz, sizeof(float));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->strsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->strsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dtrsm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtrsm || !cand->dtrsm) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    double alpha = tc->alpha;
    const double *A = (const double *)tc->A;
    size_t B_sz = (size_t)(m * ldb);

    double *Bo = (double *)clone_buf(tc->B, B_sz, sizeof(double));
    if (!Bo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bo, ldb);
    if (fb_judge_has_nan_inf(Bo, B_sz, FB_DTYPE_F64)) { free(Bo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Bc = (double *)clone_buf(tc->B, B_sz, sizeof(double));
    if (!Bc) { free(Bo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bc, ldb);
    double relerr = fb_judge_relerr_matrix(Bc, Bo, (int)m, (int)n, (int)ldb, (int)ldb,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Bo, (int)m, (int)n, (int)ldb));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Bc, B_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Bc);
    if (ns_out) {
        double *Bt = (double *)clone_buf(tc->B, B_sz, sizeof(double));
        if (Bt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dtrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dtrsm(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, lda, Bt, ldb);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(Bt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Bo); return FB_JUDGE_OK;
}

/* =========================================================================
 * L2 Packed / Banded Runners  (slots 78–109)
 *
 * Corpus layout:
 *   SPMV/SBMV/HPMV/HBMV: tc->A = matrix, tc->B = x, tc->C_init = initial y
 *   TBMV/TBSV/TPMV/TPSV:  tc->A = matrix, tc->B = x (modified in-place)
 *   SPR/HPR/SPR2/HPR2:    tc->A = x, tc->B = y (rank-2 only), tc->C_init = AP
 *
 * NOTE: Corpus sets k=0 for L2 ops, producing degenerate-but-valid banded
 * tests (bandwidth=0 = diagonal only). Full bandwidth testing requires
 * corpus generator updates.
 * ========================================================================= */

/* --- SSPMV / DSPMV / CHPMV / ZHPMV (packed symmetric/Hermitian MV) --- */

static fb_judge_status_t run_sspmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sspmv || !cand->sspmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *AP = (const float *)tc->A, *x = (const float *)tc->B;
    float *yo = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *yc = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        float *yt = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->sspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->sspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dspmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dspmv || !cand->dspmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    double alpha = tc->alpha, beta = tc->beta;
    const double *AP = (const double *)tc->A, *x = (const double *)tc->B;
    double *yo = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *yc = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        double *yt = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dspmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_chpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chpmv || !cand->chpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *AP = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x  = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->chpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->chpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_zhpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhpmv || !cand->zhpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *AP = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x  = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zhpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->zhpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, AP, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

/* --- SSBMV / DSBMV / CHBMV / ZHBMV (banded symmetric/Hermitian MV) --- */

typedef void (*fb_sbmv_cblas_fn_t)(int layout, int uplo, int n, int k,
                                   float alpha, const float *a, int lda,
                                   const float *x, int incx, float beta,
                                   float *y, int incy);
typedef void (*fb_dbmv_cblas_fn_t)(int layout, int uplo, int n, int k,
                                   double alpha, const double *a, int lda,
                                   const double *x, int incx, double beta,
                                   double *y, int incy);

static fb_judge_status_t run_sbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SBMV][FB_CONV_CBLAS];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SBMV][FB_CONV_CBLAS];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    const int col_major = 102;
    const int upper = 121;
    const int lower = 122;
    const float case1_a[6] = {0.0f, 1.0f, 0.5f, 3.0f, 4.0f, 5.0f};
    const float case1_x[3] = {1.0f, -2.0f, 0.75f};
    float oracle_case1_y[3] = {1.0f, 0.5f, -1.5f};
    float cand_case1_y[3] = {1.0f, 0.5f, -1.5f};
    const float case2_a[6] = {2.0f, 1.0f, 3.0f, -1.5f, 6.0f, 0.0f};
    const float case2_x[3] = {3.0f, -2.0f, 1.0f};
    float oracle_case2_y[3] = {1.0f, -3.0f, 2.0f};
    float cand_case2_y[3] = {1.0f, -3.0f, 2.0f};
    float oracle_out[6];
    float cand_out[6];

    ((fb_sbmv_cblas_fn_t)oracle_fn)(col_major, upper, 3, 1, 1.5f, case1_a, 2,
                                    case1_x, 1, -0.25f, oracle_case1_y, 1);
    ((fb_sbmv_cblas_fn_t)oracle_fn)(col_major, lower, 3, 1, -0.5f, case2_a, 2,
                                    case2_x, -1, 0.75f, oracle_case2_y, -1);
    memcpy(oracle_out, oracle_case1_y, sizeof(oracle_case1_y));
    memcpy(oracle_out + 3, oracle_case2_y, sizeof(oracle_case2_y));
    if (fb_judge_has_nan_inf(oracle_out, 6u, FB_DTYPE_F32)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5f, case1_a, 2,
                                  case1_x, 1, -0.25f, cand_case1_y, 1);
    ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5f, case2_a, 2,
                                  case2_x, -1, 0.75f, cand_case2_y, -1);
    memcpy(cand_out, cand_case1_y, sizeof(cand_case1_y));
    memcpy(cand_out + 3, cand_case2_y, sizeof(cand_case2_y));
    result_from_relerr(res,
                       fb_judge_relerr(cand_out, oracle_out, 6u, FB_DTYPE_F32,
                                       fb_norm_frob_f32(oracle_out, 6u)));
    if (fb_judge_has_nan_inf(cand_out, 6u, FB_DTYPE_F32)) {
        res->is_fatal = true;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            float y1[3] = {1.0f, 0.5f, -1.5f};
            float y2[3] = {1.0f, -3.0f, 2.0f};
            ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5f,
                                          case1_a, 2, case1_x, 1, -0.25f,
                                          y1, 1);
            ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5f,
                                          case2_a, 2, case2_x, -1, 0.75f,
                                          y2, -1);
        }
        for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
            float y1[3] = {1.0f, 0.5f, -1.5f};
            float y2[3] = {1.0f, -3.0f, 2.0f};
            uint64_t t0 = fb_judge_time_ns();
            ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5f,
                                          case1_a, 2, case1_x, 1, -0.25f,
                                          y1, 1);
            ((fb_sbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5f,
                                          case2_a, 2, case2_x, -1, 0.75f,
                                          y2, -1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DBMV][FB_CONV_CBLAS];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DBMV][FB_CONV_CBLAS];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    const int col_major = 102;
    const int upper = 121;
    const int lower = 122;
    const double case1_a[6] = {0.0, 1.0, 0.5, 3.0, 4.0, 5.0};
    const double case1_x[3] = {1.0, -2.0, 0.75};
    double oracle_case1_y[3] = {1.0, 0.5, -1.5};
    double cand_case1_y[3] = {1.0, 0.5, -1.5};
    const double case2_a[6] = {2.0, 1.0, 3.0, -1.5, 6.0, 0.0};
    const double case2_x[3] = {3.0, -2.0, 1.0};
    double oracle_case2_y[3] = {1.0, -3.0, 2.0};
    double cand_case2_y[3] = {1.0, -3.0, 2.0};
    double oracle_out[6];
    double cand_out[6];

    ((fb_dbmv_cblas_fn_t)oracle_fn)(col_major, upper, 3, 1, 1.5, case1_a, 2,
                                    case1_x, 1, -0.25, oracle_case1_y, 1);
    ((fb_dbmv_cblas_fn_t)oracle_fn)(col_major, lower, 3, 1, -0.5, case2_a, 2,
                                    case2_x, -1, 0.75, oracle_case2_y, -1);
    memcpy(oracle_out, oracle_case1_y, sizeof(oracle_case1_y));
    memcpy(oracle_out + 3, oracle_case2_y, sizeof(oracle_case2_y));
    if (fb_judge_has_nan_inf(oracle_out, 6u, FB_DTYPE_F64)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5, case1_a, 2,
                                  case1_x, 1, -0.25, cand_case1_y, 1);
    ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5, case2_a, 2,
                                  case2_x, -1, 0.75, cand_case2_y, -1);
    memcpy(cand_out, cand_case1_y, sizeof(cand_case1_y));
    memcpy(cand_out + 3, cand_case2_y, sizeof(cand_case2_y));
    result_from_relerr(res,
                       fb_judge_relerr(cand_out, oracle_out, 6u, FB_DTYPE_F64,
                                       fb_norm_frob_f64(oracle_out, 6u)));
    if (fb_judge_has_nan_inf(cand_out, 6u, FB_DTYPE_F64)) {
        res->is_fatal = true;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            double y1[3] = {1.0, 0.5, -1.5};
            double y2[3] = {1.0, -3.0, 2.0};
            ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5,
                                          case1_a, 2, case1_x, 1, -0.25,
                                          y1, 1);
            ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5,
                                          case2_a, 2, case2_x, -1, 0.75,
                                          y2, -1);
        }
        for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
            double y1[3] = {1.0, 0.5, -1.5};
            double y2[3] = {1.0, -3.0, 2.0};
            uint64_t t0 = fb_judge_time_ns();
            ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, upper, 3, 1, 1.5,
                                          case1_a, 2, case1_x, 1, -0.25,
                                          y1, 1);
            ((fb_dbmv_cblas_fn_t)cand_fn)(col_major, lower, 3, 1, -0.5,
                                          case2_a, 2, case2_x, -1, 0.75,
                                          y2, -1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_ssbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssbmv || !cand->ssbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    const float *A = (const float *)tc->A, *x = (const float *)tc->B;
    float *yo = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *yc = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        float *yt = (float *)clone_buf(tc->C_init, (size_t)n, sizeof(float));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ssbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ssbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dsbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsbmv || !cand->dsbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    double alpha = tc->alpha, beta = tc->beta;
    const double *A = (const double *)tc->A, *x = (const double *)tc->B;
    double *yo = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_F64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *yc = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        double *yt = (double *)clone_buf(tc->C_init, (size_t)n, sizeof(double));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dsbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dsbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_chbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chbmv || !cand->chbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    fb_complex_float_t beta;  __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->chbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->chbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_zhbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhbmv || !cand->zhbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    fb_complex_double_t beta;  __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zhbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->zhbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, k, alpha, A, lda, x, 1, beta, yt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

/* --- STBMV / DTBMV / CTBMV / ZTBMV (banded triangular MV, x in-place) --- */

static fb_judge_status_t run_stbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->stbmv || !cand->stbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const float *A = (const float *)tc->A;
    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->stbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->stbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->stbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->stbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dtbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtbmv || !cand->dtbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const double *A = (const double *)tc->A;
    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dtbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dtbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ctbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctbmv || !cand->ctbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ctbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ctbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ztbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztbmv || !cand->ztbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ztbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ztbmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* --- STBSV / DTBSV / CTBSV / ZTBSV (banded triangular solve, x in-place) --- */

static fb_judge_status_t run_stbsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->stbsv || !cand->stbsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const float *A = (const float *)tc->A;
    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->stbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->stbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->stbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->stbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dtbsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtbsv || !cand->dtbsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const double *A = (const double *)tc->A;
    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dtbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dtbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ctbsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctbsv || !cand->ctbsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ctbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ctbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ztbsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztbsv || !cand->ztbsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, k = (int)tc->k, lda = (int)tc->lda;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ztbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ztbsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, k, A, lda, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* --- STPMV / DTPMV / CTPMV / ZTPMV (packed triangular MV, x in-place) --- */

static fb_judge_status_t run_stpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->stpmv || !cand->stpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const float *AP = (const float *)tc->A;
    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->stpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->stpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->stpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->stpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dtpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtpmv || !cand->dtpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const double *AP = (const double *)tc->A;
    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dtpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dtpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ctpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctpmv || !cand->ctpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *AP = (const fb_complex_float_t *)tc->A;
    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ctpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ctpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ztpmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztpmv || !cand->ztpmv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *AP = (const fb_complex_double_t *)tc->A;
    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ztpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ztpmv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* --- STPSV / DTPSV / CTPSV / ZTPSV (packed triangular solve, x in-place) --- */

static fb_judge_status_t run_stpsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->stpsv || !cand->stpsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const float *AP = (const float *)tc->A;
    float *xo = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->stpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *xc = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->stpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F32, fb_norm_frob_f32(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        float *xt = (float *)clone_buf(tc->B, (size_t)n, sizeof(float));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->stpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->stpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dtpsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dtpsv || !cand->dtpsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const double *AP = (const double *)tc->A;
    double *xo = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dtpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_F64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *xc = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dtpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_F64, fb_norm_frob_f64(xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_F64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        double *xt = (double *)clone_buf(tc->B, (size_t)n, sizeof(double));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dtpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dtpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ctpsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ctpsv || !cand->ctpsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_float_t *AP = (const fb_complex_float_t *)tc->A;
    fb_complex_float_t *xo = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ctpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF32)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *xc = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ctpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_float_t *xt = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ctpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ctpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_ztpsv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ztpsv || !cand->ztpsv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    const fb_complex_double_t *AP = (const fb_complex_double_t *)tc->A;
    fb_complex_double_t *xo = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ztpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xo, 1);
    if (fb_judge_has_nan_inf(xo, (size_t)n, FB_DTYPE_CF64)) { free(xo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *xc = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!xc) { free(xo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ztpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xc, 1);
    result_from_relerr(res, fb_judge_relerr(xc, xo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)xo, (size_t)n)));
    if (fb_judge_has_nan_inf(xc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(xc);
    if (ns_out) {
        fb_complex_double_t *xt = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (xt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->ztpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->ztpsv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, AP, xt, 1); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(xt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(xo); return FB_JUDGE_OK;
}

/* --- SSPR / DSPR / CHPR / ZHPR (packed rank-1 update, AP in-place) --- */
/* CHPR/ZHPR: alpha is REAL float/double */

static fb_judge_status_t run_sspr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sspr || !cand->sspr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    float alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    float *APo = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_F32)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *APc = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_F32, fb_norm_frob_f32(APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        float *APt = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->sspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->sspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dspr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dspr || !cand->dspr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    double alpha = tc->alpha;
    const double *x = (const double *)tc->A;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    double *APo = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_F64)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *APc = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_F64, fb_norm_frob_f64(APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        double *APt = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dspr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_chpr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chpr || !cand->chpr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    float alpha = (float)tc->alpha;  /* REAL float — intentional */
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    fb_complex_float_t *APo = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_CF32)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *APc = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        fb_complex_float_t *APt = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->chpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->chpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_zhpr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhpr || !cand->zhpr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    double alpha = tc->alpha;  /* REAL double — intentional */
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    fb_complex_double_t *APo = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_CF64)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *APc = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        fb_complex_double_t *APt = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zhpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->zhpr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}

/* --- SSPR2 / DSPR2 / CHPR2 / ZHPR2 (packed rank-2 update, AP in-place) --- */
/* CHPR2/ZHPR2: alpha is COMPLEX */

static fb_judge_status_t run_sspr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sspr2 || !cand->sspr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    float alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A, *y = (const float *)tc->B;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    float *APo = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_F32)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *APc = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_F32, fb_norm_frob_f32(APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        float *APt = (float *)clone_buf(tc->C_init, AP_sz, sizeof(float));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->sspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->sspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_dspr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dspr2 || !cand->dspr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    double alpha = tc->alpha;
    const double *x = (const double *)tc->A, *y = (const double *)tc->B;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    double *APo = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_F64)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *APc = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_F64, fb_norm_frob_f64(APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        double *APt = (double *)clone_buf(tc->C_init, AP_sz, sizeof(double));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->dspr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_chpr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->chpr2 || !cand->chpr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    fb_complex_float_t *APo = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->chpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_CF32)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *APc = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->chpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        fb_complex_float_t *APt = (fb_complex_float_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_float_t));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->chpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->chpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}
static fb_judge_status_t run_zhpr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zhpr2 || !cand->zhpr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    size_t AP_sz = (size_t)(n * (n + 1) / 2);
    fb_complex_double_t *APo = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
    if (!APo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zhpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APo);
    if (fb_judge_has_nan_inf(APo, AP_sz, FB_DTYPE_CF64)) { free(APo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *APc = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
    if (!APc) { free(APo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zhpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APc);
    result_from_relerr(res, fb_judge_relerr(APc, APo, AP_sz, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)APo, AP_sz)));
    if (fb_judge_has_nan_inf(APc, AP_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(APc);
    if (ns_out) {
        fb_complex_double_t *APt = (fb_complex_double_t *)clone_buf(tc->C_init, AP_sz, sizeof(fb_complex_double_t));
        if (APt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zhpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) { uint64_t t0 = fb_judge_time_ns(); cand->zhpr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, APt); uint64_t dt = fb_judge_time_ns()-t0; if (dt<best) best=dt; }
            free(APt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(APo); return FB_JUDGE_OK;
}

/* =========================================================================
 * L2 Rank-Update Runners: SYR, DSYR, CHER, ZHER, SSYR2, DSYR2, CHER2, ZHER2
 * Corpus layout: x in tc->A, y (for rank-2) in tc->B, A start in tc->C_init
 * ========================================================================= */

static fb_judge_status_t run_ssyr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssyr || !cand->ssyr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    float alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    float *Ao = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Ac = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Ao, (int)n, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        float *At = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsyr || !cand->dsyr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    double alpha = tc->alpha;
    const double *x = (const double *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    double *Ao = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Ac = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Ao, (int)n, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        double *At = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* CHER: alpha is REAL float (not complex) — Hermitian rank-1 update */
static fb_judge_status_t run_cher(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cher || !cand->cher) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    float alpha = (float)tc->alpha;   /* REAL float — intentional */
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_float_t *Ao = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Ac = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_float_t *At = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* ZHER: alpha is REAL double */
static fb_judge_status_t run_zher(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zher || !cand->zher) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    double alpha = tc->alpha;   /* REAL double — intentional */
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_double_t *Ao = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Ac = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_double_t *At = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zher(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_ssyr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssyr2 || !cand->ssyr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    float alpha = (float)tc->alpha;
    const float *x = (const float *)tc->A, *y = (const float *)tc->B;
    size_t A_sz = (size_t)(n * lda);
    float *Ao = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->ssyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *Ac = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->ssyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F32, fb_matrix_norm_frob_f32(Ao, (int)n, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        float *At = (float *)clone_buf(tc->C_init, A_sz, sizeof(float));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->ssyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->ssyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyr2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dsyr2 || !cand->dsyr2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    double alpha = tc->alpha;
    const double *x = (const double *)tc->A, *y = (const double *)tc->B;
    size_t A_sz = (size_t)(n * lda);
    double *Ao = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dsyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_F64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *Ac = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dsyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_F64, fb_matrix_norm_frob_f64(Ao, (int)n, (int)n, (int)lda));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_F64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        double *At = (double *)clone_buf(tc->C_init, A_sz, sizeof(double));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->dsyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->dsyr2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* CHER2: alpha is COMPLEX float */
static fb_judge_status_t run_cher2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cher2 || !cand->cher2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha; __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_float_t *Ao = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Ac = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_float_t *At = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->cher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->cher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* =========================================================================
 * ZHER2: alpha is COMPLEX double */
static fb_judge_status_t run_zher2(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zher2 || !cand->zher2) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha; __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_double_t *Ao = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Ac = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, (int)n, (int)n, (int)lda, (int)lda,
                                           FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)Ao, A_sz));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_double_t *At = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++)
                cand->zher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns();
                cand->zher2(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}
/* =========================================================================
 * SGBMV/DGBMV/CGBMV/ZGBMV — band matrix-vector multiply
 * ========================================================================= */

typedef void (*fb_sgbmvx_fortran_fn_t)(char *trans, int *m, int *n, int *kl,
                                       int *ku, float *alpha, float *ab,
                                       int *ldab, float *x, int *incx,
                                       float *beta, float *y, int *incy);
typedef void (*fb_dgbmvx_fortran_fn_t)(char *trans, int *m, int *n, int *kl,
                                       int *ku, double *alpha, double *ab,
                                       int *ldab, double *x, int *incx,
                                       double *beta, double *y, int *incy);
typedef void (*fb_cgbmvx_fortran_fn_t)(char *trans, int *m, int *n, int *kl,
                                       int *ku, fb_complex_float_t *alpha,
                                       fb_complex_float_t *ab, int *ldab,
                                       fb_complex_float_t *x, int *incx,
                                       fb_complex_float_t *beta,
                                       fb_complex_float_t *y, int *incy);
typedef void (*fb_zgbmvx_fortran_fn_t)(char *trans, int *m, int *n, int *kl,
                                       int *ku, fb_complex_double_t *alpha,
                                       fb_complex_double_t *ab, int *ldab,
                                       fb_complex_double_t *x, int *incx,
                                       fb_complex_double_t *beta,
                                       fb_complex_double_t *y, int *incy);
typedef void (*fb_sgbcon_fortran_fn_t)(const char *norm, const int *n,
                                       const int *kl, const int *ku,
                                       const float *ab, const int *ldab,
                                       const int *ipiv, const float *anorm,
                                       float *rcond, float *work,
                                       int *iwork, int *info);
typedef void (*fb_dgbcon_fortran_fn_t)(const char *norm, const int *n,
                                       const int *kl, const int *ku,
                                       const double *ab, const int *ldab,
                                       const int *ipiv, const double *anorm,
                                       double *rcond, double *work,
                                       int *iwork, int *info);
typedef void (*fb_cgbcon_fortran_fn_t)(const char *norm, const int *n,
                                       const int *kl, const int *ku,
                                       const fb_complex_float_t *ab,
                                       const int *ldab, const int *ipiv,
                                       const float *anorm, float *rcond,
                                       fb_complex_float_t *work,
                                       int *iwork, int *info);
typedef void (*fb_zgbcon_fortran_fn_t)(const char *norm, const int *n,
                                       const int *kl, const int *ku,
                                       const fb_complex_double_t *ab,
                                       const int *ldab, const int *ipiv,
                                       const double *anorm, double *rcond,
                                       fb_complex_double_t *work,
                                       int *iwork, int *info);
typedef void (*fb_sgbequ_fortran_fn_t)(int *m, int *n, int *kl, int *ku,
                                       float *ab, int *ldab, float *r,
                                       float *c, float *rowcnd,
                                       float *colcnd, float *amax,
                                       int *info);
typedef void (*fb_dgbequ_fortran_fn_t)(int *m, int *n, int *kl, int *ku,
                                       double *ab, int *ldab, double *r,
                                       double *c, double *rowcnd,
                                       double *colcnd, double *amax,
                                       int *info);
typedef void (*fb_cgbequ_fortran_fn_t)(int *m, int *n, int *kl, int *ku,
                                       fb_complex_float_t *ab, int *ldab,
                                       float *r, float *c, float *rowcnd,
                                       float *colcnd, float *amax,
                                       int *info);
typedef void (*fb_zgbequ_fortran_fn_t)(int *m, int *n, int *kl, int *ku,
                                       fb_complex_double_t *ab, int *ldab,
                                       double *r, double *c,
                                       double *rowcnd, double *colcnd,
                                       double *amax, int *info);
typedef void (*fb_sgbrfs_fortran_fn_t)(const char *trans, int *n, int *kl,
                                       int *ku, int *nrhs, float *ab,
                                       int *ldab, float *afb, int *ldafb,
                                       int *ipiv, float *b, int *ldb,
                                       float *x, int *ldx, float *ferr,
                                       float *berr, float *work,
                                       int *iwork, int *info);
typedef void (*fb_dgbrfs_fortran_fn_t)(const char *trans, int *n, int *kl,
                                       int *ku, int *nrhs, double *ab,
                                       int *ldab, double *afb, int *ldafb,
                                       int *ipiv, double *b, int *ldb,
                                       double *x, int *ldx, double *ferr,
                                       double *berr, double *work,
                                       int *iwork, int *info);
typedef void (*fb_cgbrfs_fortran_fn_t)(char *trans, int *n, int *kl,
                                       int *ku, int *nrhs,
                                       fb_complex_float_t *ab, int *ldab,
                                       fb_complex_float_t *afb, int *ldafb,
                                       int *ipiv, fb_complex_float_t *b,
                                       int *ldb, fb_complex_float_t *x,
                                       int *ldx, float *ferr, float *berr,
                                       fb_complex_float_t *work,
                                       float *rwork, int *info);
typedef void (*fb_zgbrfs_fortran_fn_t)(char *trans, int *n, int *kl,
                                       int *ku, int *nrhs,
                                       fb_complex_double_t *ab, int *ldab,
                                       fb_complex_double_t *afb, int *ldafb,
                                       int *ipiv, fb_complex_double_t *b,
                                       int *ldb, fb_complex_double_t *x,
                                       int *ldx, double *ferr,
                                       double *berr,
                                       fb_complex_double_t *work,
                                       double *rwork, int *info);
typedef void (*fb_sgebak_fortran_fn_t)(char *job, char *side, int *n,
                                       int *ilo, int *ihi, float *scale,
                                       int *m, float *v, int *ldv,
                                       int *info);
typedef void (*fb_dgebak_fortran_fn_t)(char *job, char *side, int *n,
                                       int *ilo, int *ihi, double *scale,
                                       int *m, double *v, int *ldv,
                                       int *info);
typedef void (*fb_cgebak_fortran_fn_t)(char *job, char *side, int *n,
                                       int *ilo, int *ihi, float *scale,
                                       int *m, fb_complex_float_t *v,
                                       int *ldv, int *info);
typedef void (*fb_zgebak_fortran_fn_t)(char *job, char *side, int *n,
                                       int *ilo, int *ihi, double *scale,
                                       int *m, fb_complex_double_t *v,
                                       int *ldv, int *info);
typedef void (*fb_sgebal_fortran_fn_t)(char *job, int *n, float *a,
                                       int *lda, int *ilo, int *ihi,
                                       float *scale, int *info);
typedef void (*fb_dgebal_fortran_fn_t)(char *job, int *n, double *a,
                                       int *lda, int *ilo, int *ihi,
                                       double *scale, int *info);
typedef void (*fb_cgebal_fortran_fn_t)(char *job, int *n,
                                       fb_complex_float_t *a, int *lda,
                                       int *ilo, int *ihi, float *scale,
                                       int *info);
typedef void (*fb_zgebal_fortran_fn_t)(char *job, int *n,
                                       fb_complex_double_t *a, int *lda,
                                       int *ilo, int *ihi, double *scale,
                                       int *info);
typedef void (*fb_sgebrd_fortran_fn_t)(int *m, int *n, float *a, int *lda,
                                       float *d, float *e, float *tauq,
                                       float *taup, float *work, int *lwork,
                                       int *info);
typedef void (*fb_dgebrd_fortran_fn_t)(int *m, int *n, double *a, int *lda,
                                       double *d, double *e, double *tauq,
                                       double *taup, double *work,
                                       int *lwork, int *info);
typedef void (*fb_cgebrd_fortran_fn_t)(int *m, int *n,
                                       fb_complex_float_t *a, int *lda,
                                       float *d, float *e,
                                       fb_complex_float_t *tauq,
                                       fb_complex_float_t *taup,
                                       fb_complex_float_t *work,
                                       int *lwork, int *info);
typedef void (*fb_zgebrd_fortran_fn_t)(int *m, int *n,
                                       fb_complex_double_t *a, int *lda,
                                       double *d, double *e,
                                       fb_complex_double_t *tauq,
                                       fb_complex_double_t *taup,
                                       fb_complex_double_t *work,
                                       int *lwork, int *info);
typedef void (*fb_sgecon_fortran_fn_t)(const char *norm, int *n,
                                       const float *a, int *lda,
                                       float *anorm, float *rcond,
                                       float *work, int *iwork,
                                       int *info);
typedef void (*fb_dgecon_fortran_fn_t)(const char *norm, int *n,
                                       const double *a, int *lda,
                                       double *anorm, double *rcond,
                                       double *work, int *iwork,
                                       int *info);
typedef void (*fb_cgecon_fortran_fn_t)(const char *norm, int *n,
                                       const fb_complex_float_t *a,
                                       int *lda, float *anorm,
                                       float *rcond,
                                       fb_complex_float_t *work,
                                       float *rwork, int *info);
typedef void (*fb_zgecon_fortran_fn_t)(const char *norm, int *n,
                                       const fb_complex_double_t *a,
                                       int *lda, double *anorm,
                                       double *rcond,
                                       fb_complex_double_t *work,
                                       double *rwork, int *info);
typedef void (*fb_sgeequ_fortran_fn_t)(int *m, int *n, const float *a,
                                       int *lda, float *r, float *c,
                                       float *rowcnd, float *colcnd,
                                       float *amax, int *info);
typedef void (*fb_dgeequ_fortran_fn_t)(int *m, int *n, const double *a,
                                       int *lda, double *r, double *c,
                                       double *rowcnd, double *colcnd,
                                       double *amax, int *info);

static fb_complex_float_t fb_direct_make_cf32(float real_part, float imag_part)
{
    fb_complex_float_t value = 0.0f;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

static fb_complex_double_t fb_direct_make_cf64(double real_part,
                                               double imag_part)
{
    fb_complex_double_t value = 0.0;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

#define FB_DEFINE_GBCON_RUNNER(name, op_id, fn_type, scalar_type, real_type, \
                               relerr_fn, output_dtype, ab_init, zero_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char norm = '1'; \
    int n = 1, kl = 0, ku = 0, ldab = 1; \
    int ipiv[1] = {1}; \
    int info_oracle = -1, info_cand = -1; \
    real_type anorm = (real_type)4.0; \
    real_type rcond_oracle = (real_type)-1.0; \
    real_type rcond_cand = (real_type)-1.0; \
    scalar_type ab[1] = {ab_init}; \
    scalar_type work_oracle[1] = {zero_init}; \
    scalar_type work_cand[1] = {zero_init}; \
    int iwork_oracle[1] = {0}; \
    int iwork_cand[1] = {0}; \
    ((fn_type)oracle_fn)(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, \
                         &rcond_oracle, work_oracle, iwork_oracle, \
                         &info_oracle); \
    if (info_oracle != 0 || !isfinite((double)rcond_oracle)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, \
                       &rcond_cand, work_cand, iwork_cand, &info_cand); \
    if (info_cand != 0 || !isfinite((double)rcond_cand)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, relerr_fn(rcond_cand, rcond_oracle, \
                                      (double)rcond_oracle)); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[1] = {zero_init}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&norm, &n, &kl, &ku, ab, &ldab, ipiv, \
                               &anorm, &rcond_time, work_time, iwork_time, \
                               &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[1] = {zero_init}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&norm, &n, &kl, &ku, ab, &ldab, ipiv, \
                               &anorm, &rcond_time, work_time, iwork_time, \
                               &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    (void)output_dtype; \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GECON_RUNNER_REAL(name, op_id, fn_type, scalar_type, \
                                    real_type, relerr_fn, a_init, zero_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char norm = '1'; \
    int n = 1, lda = 1; \
    int info_oracle = -1, info_cand = -1; \
    real_type anorm = (real_type)4.0; \
    real_type rcond_oracle = (real_type)-1.0; \
    real_type rcond_cand = (real_type)-1.0; \
    scalar_type a[1] = {a_init}; \
    scalar_type work_oracle[4] = {zero_init, zero_init, zero_init, zero_init}; \
    scalar_type work_cand[4] = {zero_init, zero_init, zero_init, zero_init}; \
    int iwork_oracle[1] = {0}; \
    int iwork_cand[1] = {0}; \
    ((fn_type)oracle_fn)(&norm, &n, a, &lda, &anorm, &rcond_oracle, \
                         work_oracle, iwork_oracle, &info_oracle); \
    if (info_oracle != 0 || !isfinite((double)rcond_oracle)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_cand, \
                       work_cand, iwork_cand, &info_cand); \
    if (info_cand != 0 || !isfinite((double)rcond_cand)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, relerr_fn(rcond_cand, rcond_oracle, \
                                      (double)rcond_oracle)); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[4] = {zero_init, zero_init, zero_init, zero_init}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_time, \
                               work_time, iwork_time, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[4] = {zero_init, zero_init, zero_init, zero_init}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_time, \
                               work_time, iwork_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GECON_RUNNER_COMPLEX(name, op_id, fn_type, scalar_type, \
                                       real_type, relerr_fn, a_init, zero_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char norm = '1'; \
    int n = 1, lda = 1; \
    int info_oracle = -1, info_cand = -1; \
    real_type anorm = (real_type)4.0; \
    real_type rcond_oracle = (real_type)-1.0; \
    real_type rcond_cand = (real_type)-1.0; \
    scalar_type a[1] = {a_init}; \
    scalar_type work_oracle[2] = {zero_init, zero_init}; \
    scalar_type work_cand[2] = {zero_init, zero_init}; \
    real_type rwork_oracle[2] = {(real_type)0, (real_type)0}; \
    real_type rwork_cand[2] = {(real_type)0, (real_type)0}; \
    ((fn_type)oracle_fn)(&norm, &n, a, &lda, &anorm, &rcond_oracle, \
                         work_oracle, rwork_oracle, &info_oracle); \
    if (info_oracle != 0 || !isfinite((double)rcond_oracle)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_cand, \
                       work_cand, rwork_cand, &info_cand); \
    if (info_cand != 0 || !isfinite((double)rcond_cand)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, relerr_fn(rcond_cand, rcond_oracle, \
                                      (double)rcond_oracle)); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[2] = {zero_init, zero_init}; \
            real_type rwork_time[2] = {(real_type)0, (real_type)0}; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_time, \
                               work_time, rwork_time, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            real_type rcond_time = (real_type)-1.0; \
            scalar_type work_time[2] = {zero_init, zero_init}; \
            real_type rwork_time[2] = {(real_type)0, (real_type)0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&norm, &n, a, &lda, &anorm, &rcond_time, \
                               work_time, rwork_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GBEQU_RUNNER(name, op_id, fn_type, scalar_type, real_type, \
                               output_dtype, norm_fn, ab_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    int m = 1, n = 1, kl = 0, ku = 0, ldab = 1; \
    int info_oracle = -1, info_cand = -1; \
    scalar_type ab[1] = {ab_init}; \
    real_type r_oracle[1] = {(real_type)0}; \
    real_type c_oracle[1] = {(real_type)0}; \
    real_type rowcnd_oracle = (real_type)0; \
    real_type colcnd_oracle = (real_type)0; \
    real_type amax_oracle = (real_type)0; \
    real_type r_cand[1] = {(real_type)0}; \
    real_type c_cand[1] = {(real_type)0}; \
    real_type rowcnd_cand = (real_type)0; \
    real_type colcnd_cand = (real_type)0; \
    real_type amax_cand = (real_type)0; \
    real_type oracle_out[5]; \
    real_type cand_out[5]; \
    ((fn_type)oracle_fn)(&m, &n, &kl, &ku, ab, &ldab, r_oracle, c_oracle, \
                         &rowcnd_oracle, &colcnd_oracle, &amax_oracle, \
                         &info_oracle); \
    oracle_out[0] = r_oracle[0]; \
    oracle_out[1] = c_oracle[0]; \
    oracle_out[2] = rowcnd_oracle; \
    oracle_out[3] = colcnd_oracle; \
    oracle_out[4] = amax_oracle; \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(oracle_out, 5u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&m, &n, &kl, &ku, ab, &ldab, r_cand, c_cand, \
                       &rowcnd_cand, &colcnd_cand, &amax_cand, &info_cand); \
    cand_out[0] = r_cand[0]; \
    cand_out[1] = c_cand[0]; \
    cand_out[2] = rowcnd_cand; \
    cand_out[3] = colcnd_cand; \
    cand_out[4] = amax_cand; \
    if (info_cand != 0 || fb_judge_has_nan_inf(cand_out, 5u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 5u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 5u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            real_type r_time[1] = {(real_type)0}; \
            real_type c_time[1] = {(real_type)0}; \
            real_type rowcnd_time = (real_type)0; \
            real_type colcnd_time = (real_type)0; \
            real_type amax_time = (real_type)0; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&m, &n, &kl, &ku, ab, &ldab, r_time, c_time, \
                               &rowcnd_time, &colcnd_time, &amax_time, \
                               &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            real_type r_time[1] = {(real_type)0}; \
            real_type c_time[1] = {(real_type)0}; \
            real_type rowcnd_time = (real_type)0; \
            real_type colcnd_time = (real_type)0; \
            real_type amax_time = (real_type)0; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&m, &n, &kl, &ku, ab, &ldab, r_time, c_time, \
                               &rowcnd_time, &colcnd_time, &amax_time, \
                               &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEEQU_RUNNER(name, op_id, fn_type, scalar_type, real_type, \
                               output_dtype, norm_fn, a_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    int m = 1, n = 1, lda = 1; \
    int info_oracle = -1, info_cand = -1; \
    scalar_type a[1] = {a_init}; \
    real_type r_oracle[1] = {(real_type)0}; \
    real_type c_oracle[1] = {(real_type)0}; \
    real_type rowcnd_oracle = (real_type)0; \
    real_type colcnd_oracle = (real_type)0; \
    real_type amax_oracle = (real_type)0; \
    real_type r_cand[1] = {(real_type)0}; \
    real_type c_cand[1] = {(real_type)0}; \
    real_type rowcnd_cand = (real_type)0; \
    real_type colcnd_cand = (real_type)0; \
    real_type amax_cand = (real_type)0; \
    real_type oracle_out[5]; \
    real_type cand_out[5]; \
    ((fn_type)oracle_fn)(&m, &n, a, &lda, r_oracle, c_oracle, \
                         &rowcnd_oracle, &colcnd_oracle, &amax_oracle, \
                         &info_oracle); \
    oracle_out[0] = r_oracle[0]; \
    oracle_out[1] = c_oracle[0]; \
    oracle_out[2] = rowcnd_oracle; \
    oracle_out[3] = colcnd_oracle; \
    oracle_out[4] = amax_oracle; \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(oracle_out, 5u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&m, &n, a, &lda, r_cand, c_cand, \
                       &rowcnd_cand, &colcnd_cand, &amax_cand, &info_cand); \
    cand_out[0] = r_cand[0]; \
    cand_out[1] = c_cand[0]; \
    cand_out[2] = rowcnd_cand; \
    cand_out[3] = colcnd_cand; \
    cand_out[4] = amax_cand; \
    if (info_cand != 0 || fb_judge_has_nan_inf(cand_out, 5u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 5u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 5u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            real_type r_time[1] = {(real_type)0}; \
            real_type c_time[1] = {(real_type)0}; \
            real_type rowcnd_time = (real_type)0; \
            real_type colcnd_time = (real_type)0; \
            real_type amax_time = (real_type)0; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&m, &n, a, &lda, r_time, c_time, \
                               &rowcnd_time, &colcnd_time, &amax_time, \
                               &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            real_type r_time[1] = {(real_type)0}; \
            real_type c_time[1] = {(real_type)0}; \
            real_type rowcnd_time = (real_type)0; \
            real_type colcnd_time = (real_type)0; \
            real_type amax_time = (real_type)0; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&m, &n, a, &lda, r_time, c_time, \
                               &rowcnd_time, &colcnd_time, &amax_time, \
                               &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GBRFS_RUNNER_REAL(name, op_id, fn_type, scalar_type, \
                                    real_type, output_dtype, norm_fn, \
                                    value_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char trans = 'N'; \
    int n = 1, kl = 0, ku = 0, nrhs = 1; \
    int ldab = 1, ldafb = 1, ldb = 1, ldx = 1; \
    int info_oracle = -1, info_cand = -1; \
    int ipiv[1] = {1}; \
    scalar_type ab[1] = {value_init((real_type)4.0, (real_type)0.0)}; \
    scalar_type afb[1] = {value_init((real_type)4.0, (real_type)0.0)}; \
    scalar_type b[1] = {value_init((real_type)8.0, (real_type)0.0)}; \
    scalar_type x_oracle[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
    scalar_type x_cand[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
    real_type ferr_oracle[1] = {(real_type)-1.0}; \
    real_type berr_oracle[1] = {(real_type)-1.0}; \
    real_type ferr_cand[1] = {(real_type)-1.0}; \
    real_type berr_cand[1] = {(real_type)-1.0}; \
    scalar_type work_oracle[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type work_cand[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    int iwork_oracle[1] = {0}; \
    int iwork_cand[1] = {0}; \
    real_type oracle_out[3]; \
    real_type cand_out[3]; \
    ((fn_type)oracle_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                         &ldafb, ipiv, b, &ldb, x_oracle, &ldx, \
                         ferr_oracle, berr_oracle, work_oracle, \
                         iwork_oracle, &info_oracle); \
    oracle_out[0] = (real_type)x_oracle[0]; \
    oracle_out[1] = ferr_oracle[0]; \
    oracle_out[2] = berr_oracle[0]; \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(oracle_out, 3u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, \
                       ipiv, b, &ldb, x_cand, &ldx, ferr_cand, berr_cand, \
                       work_cand, iwork_cand, &info_cand); \
    cand_out[0] = (real_type)x_cand[0]; \
    cand_out[1] = ferr_cand[0]; \
    cand_out[2] = berr_cand[0]; \
    if (info_cand != 0 || fb_judge_has_nan_inf(cand_out, 3u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 3u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 3u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type x_time[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
            real_type ferr_time[1] = {(real_type)-1.0}; \
            real_type berr_time[1] = {(real_type)-1.0}; \
            scalar_type work_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                               &ldafb, ipiv, b, &ldb, x_time, &ldx, \
                               ferr_time, berr_time, work_time, iwork_time, \
                               &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type x_time[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
            real_type ferr_time[1] = {(real_type)-1.0}; \
            real_type berr_time[1] = {(real_type)-1.0}; \
            scalar_type work_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            int iwork_time[1] = {0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                               &ldafb, ipiv, b, &ldb, x_time, &ldx, \
                               ferr_time, berr_time, work_time, iwork_time, \
                               &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GBRFS_RUNNER_COMPLEX(name, op_id, fn_type, scalar_type, \
                                       real_type, output_dtype, norm_fn, \
                                       value_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char trans = 'N'; \
    int n = 1, kl = 0, ku = 0, nrhs = 1; \
    int ldab = 1, ldafb = 1, ldb = 1, ldx = 1; \
    int info_oracle = -1, info_cand = -1; \
    int ipiv[1] = {1}; \
    scalar_type ab[1] = {value_init((real_type)4.0, (real_type)0.0)}; \
    scalar_type afb[1] = {value_init((real_type)4.0, (real_type)0.0)}; \
    scalar_type b[1] = {value_init((real_type)8.0, (real_type)0.0)}; \
    scalar_type x_oracle[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
    scalar_type x_cand[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
    real_type ferr_oracle[1] = {(real_type)-1.0}; \
    real_type berr_oracle[1] = {(real_type)-1.0}; \
    real_type ferr_cand[1] = {(real_type)-1.0}; \
    real_type berr_cand[1] = {(real_type)-1.0}; \
    scalar_type work_oracle[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type work_cand[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    real_type rwork_oracle[1] = {(real_type)0.0}; \
    real_type rwork_cand[1] = {(real_type)0.0}; \
    real_type oracle_out[4]; \
    real_type cand_out[4]; \
    ((fn_type)oracle_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                         &ldafb, ipiv, b, &ldb, x_oracle, &ldx, \
                         ferr_oracle, berr_oracle, work_oracle, \
                         rwork_oracle, &info_oracle); \
    oracle_out[0] = (real_type)__real__(x_oracle[0]); \
    oracle_out[1] = (real_type)__imag__(x_oracle[0]); \
    oracle_out[2] = ferr_oracle[0]; \
    oracle_out[3] = berr_oracle[0]; \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(oracle_out, 4u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, \
                       ipiv, b, &ldb, x_cand, &ldx, ferr_cand, berr_cand, \
                       work_cand, rwork_cand, &info_cand); \
    cand_out[0] = (real_type)__real__(x_cand[0]); \
    cand_out[1] = (real_type)__imag__(x_cand[0]); \
    cand_out[2] = ferr_cand[0]; \
    cand_out[3] = berr_cand[0]; \
    if (info_cand != 0 || fb_judge_has_nan_inf(cand_out, 4u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 4u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 4u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type x_time[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
            real_type ferr_time[1] = {(real_type)-1.0}; \
            real_type berr_time[1] = {(real_type)-1.0}; \
            scalar_type work_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            real_type rwork_time[1] = {(real_type)0.0}; \
            int info_time = -1; \
            ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                               &ldafb, ipiv, b, &ldb, x_time, &ldx, \
                               ferr_time, berr_time, work_time, rwork_time, \
                               &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type x_time[1] = {value_init((real_type)2.0, (real_type)0.0)}; \
            real_type ferr_time[1] = {(real_type)-1.0}; \
            real_type berr_time[1] = {(real_type)-1.0}; \
            scalar_type work_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            real_type rwork_time[1] = {(real_type)0.0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, \
                               &ldafb, ipiv, b, &ldb, x_time, &ldx, \
                               ferr_time, berr_time, work_time, rwork_time, \
                               &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBAK_RUNNER_REAL(name, op_id, fn_type, scalar_type, \
                                    output_dtype, norm_fn) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char job = 'S'; \
    int n = 2, ilo = 1, ihi = 2, m = 2, ldv = 2; \
    scalar_type scale[2] = {(scalar_type)2.0, (scalar_type)0.5}; \
    scalar_type right_template[4] = {(scalar_type)1.0, (scalar_type)2.0, \
                                     (scalar_type)-3.0, (scalar_type)4.0}; \
    scalar_type left_template[4] = {(scalar_type)4.0, (scalar_type)1.0, \
                                    (scalar_type)-2.0, (scalar_type)0.5}; \
    scalar_type right_oracle[4]; \
    scalar_type right_cand[4]; \
    scalar_type left_oracle[4]; \
    scalar_type left_cand[4]; \
    scalar_type oracle_out[8]; \
    scalar_type cand_out[8]; \
    int info_oracle = -1; \
    int info_cand = -1; \
    char side = 'R'; \
    memcpy(right_oracle, right_template, sizeof(right_oracle)); \
    ((fn_type)oracle_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, \
                         right_oracle, &ldv, &info_oracle); \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(right_oracle, 4u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(right_cand, right_template, sizeof(right_cand)); \
    ((fn_type)cand_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, right_cand, \
                       &ldv, &info_cand); \
    if (info_cand != 0 || fb_judge_has_nan_inf(right_cand, 4u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    side = 'L'; \
    info_oracle = -1; \
    info_cand = -1; \
    memcpy(left_oracle, left_template, sizeof(left_oracle)); \
    ((fn_type)oracle_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, left_oracle, \
                         &ldv, &info_oracle); \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(left_oracle, 4u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(left_cand, left_template, sizeof(left_cand)); \
    ((fn_type)cand_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, left_cand, \
                       &ldv, &info_cand); \
    if (info_cand != 0 || fb_judge_has_nan_inf(left_cand, 4u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(oracle_out, right_oracle, sizeof(right_oracle)); \
    memcpy(oracle_out + 4, left_oracle, sizeof(left_oracle)); \
    memcpy(cand_out, right_cand, sizeof(right_cand)); \
    memcpy(cand_out + 4, left_cand, sizeof(left_cand)); \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 8u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 8u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type v_time[4]; \
            char side_time = 'R'; \
            int info_time = -1; \
            memcpy(v_time, right_template, sizeof(v_time)); \
            ((fn_type)cand_fn)(&job, &side_time, &n, &ilo, &ihi, scale, &m, \
                               v_time, &ldv, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type v_time[4]; \
            char side_time = 'R'; \
            int info_time = -1; \
            memcpy(v_time, right_template, sizeof(v_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&job, &side_time, &n, &ilo, &ihi, scale, &m, \
                               v_time, &ldv, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBAK_RUNNER_COMPLEX(name, op_id, fn_type, scalar_type, \
                                       real_type, output_dtype, norm_fn, \
                                       norm_base_type, value_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char job = 'S'; \
    int n = 2, ilo = 1, ihi = 2, m = 2, ldv = 2; \
    real_type scale[2] = {(real_type)2.0, (real_type)0.5}; \
    scalar_type right_template[4] = { \
        value_init((real_type)1.0, (real_type)1.0), \
        value_init((real_type)2.0, (real_type)-1.0), \
        value_init((real_type)-3.0, (real_type)0.5), \
        value_init((real_type)4.0, (real_type)2.0) \
    }; \
    scalar_type left_template[4] = { \
        value_init((real_type)4.0, (real_type)2.0), \
        value_init((real_type)1.0, (real_type)0.5), \
        value_init((real_type)-2.0, (real_type)-1.0), \
        value_init((real_type)0.5, (real_type)1.5) \
    }; \
    scalar_type right_oracle[4]; \
    scalar_type right_cand[4]; \
    scalar_type left_oracle[4]; \
    scalar_type left_cand[4]; \
    scalar_type oracle_out[8]; \
    scalar_type cand_out[8]; \
    int info_oracle = -1; \
    int info_cand = -1; \
    char side = 'R'; \
    memcpy(right_oracle, right_template, sizeof(right_oracle)); \
    ((fn_type)oracle_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, \
                         right_oracle, &ldv, &info_oracle); \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(right_oracle, 4u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(right_cand, right_template, sizeof(right_cand)); \
    ((fn_type)cand_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, right_cand, \
                       &ldv, &info_cand); \
    if (info_cand != 0 || fb_judge_has_nan_inf(right_cand, 4u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    side = 'L'; \
    info_oracle = -1; \
    info_cand = -1; \
    memcpy(left_oracle, left_template, sizeof(left_oracle)); \
    ((fn_type)oracle_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, left_oracle, \
                         &ldv, &info_oracle); \
    if (info_oracle != 0 || \
        fb_judge_has_nan_inf(left_oracle, 4u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(left_cand, left_template, sizeof(left_cand)); \
    ((fn_type)cand_fn)(&job, &side, &n, &ilo, &ihi, scale, &m, left_cand, \
                       &ldv, &info_cand); \
    if (info_cand != 0 || fb_judge_has_nan_inf(left_cand, 4u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(oracle_out, right_oracle, sizeof(right_oracle)); \
    memcpy(oracle_out + 4, left_oracle, sizeof(left_oracle)); \
    memcpy(cand_out, right_cand, sizeof(right_cand)); \
    memcpy(cand_out + 4, left_cand, sizeof(left_cand)); \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 8u, \
                                            output_dtype, \
                                            norm_fn((const norm_base_type *)oracle_out, \
                                                    8u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type v_time[4]; \
            char side_time = 'R'; \
            int info_time = -1; \
            memcpy(v_time, right_template, sizeof(v_time)); \
            ((fn_type)cand_fn)(&job, &side_time, &n, &ilo, &ihi, scale, &m, \
                               v_time, &ldv, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type v_time[4]; \
            char side_time = 'R'; \
            int info_time = -1; \
            memcpy(v_time, right_template, sizeof(v_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&job, &side_time, &n, &ilo, &ihi, scale, &m, \
                               v_time, &ldv, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBAL_RUNNER_REAL(name, op_id, fn_type, scalar_type, \
                                    output_dtype, norm_fn) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char job_n = 'N'; \
    char job_s = 'S'; \
    int n = 2, lda = 2; \
    scalar_type a_n_template[4] = { \
        (scalar_type)1.0, (scalar_type)3.0, \
        (scalar_type)2.0, (scalar_type)4.0 \
    }; \
    scalar_type a_s_template[4] = { \
        (scalar_type)1.0, (scalar_type)0.001, \
        (scalar_type)1000.0, (scalar_type)1.0 \
    }; \
    scalar_type a_n_oracle[4]; \
    scalar_type a_n_cand[4]; \
    scalar_type a_s_oracle[4]; \
    scalar_type a_s_cand[4]; \
    scalar_type scale_n_oracle[2] = {(scalar_type)-1.0, (scalar_type)-1.0}; \
    scalar_type scale_n_cand[2] = {(scalar_type)-1.0, (scalar_type)-1.0}; \
    scalar_type scale_s_oracle[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type scale_s_cand[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type oracle_out[16]; \
    scalar_type cand_out[16]; \
    int ilo_n_oracle = -1, ihi_n_oracle = -1, info_n_oracle = -1; \
    int ilo_n_cand = -1, ihi_n_cand = -1, info_n_cand = -1; \
    int ilo_s_oracle = -1, ihi_s_oracle = -1, info_s_oracle = -1; \
    int ilo_s_cand = -1, ihi_s_cand = -1, info_s_cand = -1; \
    size_t idx = 0; \
    memcpy(a_n_oracle, a_n_template, sizeof(a_n_oracle)); \
    ((fn_type)oracle_fn)(&job_n, &n, a_n_oracle, &lda, &ilo_n_oracle, \
                         &ihi_n_oracle, scale_n_oracle, &info_n_oracle); \
    if (info_n_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_n_cand, a_n_template, sizeof(a_n_cand)); \
    ((fn_type)cand_fn)(&job_n, &n, a_n_cand, &lda, &ilo_n_cand, &ihi_n_cand, \
                       scale_n_cand, &info_n_cand); \
    if (info_n_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_s_oracle, a_s_template, sizeof(a_s_oracle)); \
    ((fn_type)oracle_fn)(&job_s, &n, a_s_oracle, &lda, &ilo_s_oracle, \
                         &ihi_s_oracle, scale_s_oracle, &info_s_oracle); \
    if (info_s_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_s_cand, a_s_template, sizeof(a_s_cand)); \
    ((fn_type)cand_fn)(&job_s, &n, a_s_cand, &lda, &ilo_s_cand, &ihi_s_cand, \
                       scale_s_cand, &info_s_cand); \
    if (info_s_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    for (size_t i = 0; i < 4u; i++) oracle_out[idx++] = a_n_oracle[i]; \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = scale_n_oracle[i]; \
    oracle_out[idx++] = (scalar_type)ilo_n_oracle; \
    oracle_out[idx++] = (scalar_type)ihi_n_oracle; \
    for (size_t i = 0; i < 4u; i++) oracle_out[idx++] = a_s_oracle[i]; \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = scale_s_oracle[i]; \
    oracle_out[idx++] = (scalar_type)ilo_s_oracle; \
    oracle_out[idx++] = (scalar_type)ihi_s_oracle; \
    idx = 0; \
    for (size_t i = 0; i < 4u; i++) cand_out[idx++] = a_n_cand[i]; \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = scale_n_cand[i]; \
    cand_out[idx++] = (scalar_type)ilo_n_cand; \
    cand_out[idx++] = (scalar_type)ihi_n_cand; \
    for (size_t i = 0; i < 4u; i++) cand_out[idx++] = a_s_cand[i]; \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = scale_s_cand[i]; \
    cand_out[idx++] = (scalar_type)ilo_s_cand; \
    cand_out[idx++] = (scalar_type)ihi_s_cand; \
    if (fb_judge_has_nan_inf(oracle_out, 16u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (fb_judge_has_nan_inf(cand_out, 16u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 16u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 16u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type a_time[4]; \
            scalar_type scale_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            int ilo_time = -1, ihi_time = -1, info_time = -1; \
            memcpy(a_time, a_s_template, sizeof(a_time)); \
            ((fn_type)cand_fn)(&job_s, &n, a_time, &lda, &ilo_time, &ihi_time, \
                               scale_time, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type a_time[4]; \
            scalar_type scale_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            int ilo_time = -1, ihi_time = -1, info_time = -1; \
            memcpy(a_time, a_s_template, sizeof(a_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&job_s, &n, a_time, &lda, &ilo_time, &ihi_time, \
                               scale_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBAL_RUNNER_COMPLEX(name, op_id, fn_type, scalar_type, \
                                       real_type, output_dtype, norm_fn, \
                                       value_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    char job_n = 'N'; \
    char job_s = 'S'; \
    int n = 2, lda = 2; \
    scalar_type a_n_template[4] = { \
        value_init((real_type)1.0, (real_type)0.5), \
        value_init((real_type)3.0, (real_type)-1.0), \
        value_init((real_type)2.0, (real_type)1.5), \
        value_init((real_type)4.0, (real_type)-0.25) \
    }; \
    scalar_type a_s_template[4] = { \
        value_init((real_type)1.0, (real_type)0.0), \
        value_init((real_type)0.0, (real_type)0.001), \
        value_init((real_type)0.0, (real_type)1000.0), \
        value_init((real_type)1.0, (real_type)0.0) \
    }; \
    scalar_type a_n_oracle[4]; \
    scalar_type a_n_cand[4]; \
    scalar_type a_s_oracle[4]; \
    scalar_type a_s_cand[4]; \
    real_type scale_n_oracle[2] = {(real_type)-1.0, (real_type)-1.0}; \
    real_type scale_n_cand[2] = {(real_type)-1.0, (real_type)-1.0}; \
    real_type scale_s_oracle[2] = {(real_type)0.0, (real_type)0.0}; \
    real_type scale_s_cand[2] = {(real_type)0.0, (real_type)0.0}; \
    real_type oracle_out[24]; \
    real_type cand_out[24]; \
    int ilo_n_oracle = -1, ihi_n_oracle = -1, info_n_oracle = -1; \
    int ilo_n_cand = -1, ihi_n_cand = -1, info_n_cand = -1; \
    int ilo_s_oracle = -1, ihi_s_oracle = -1, info_s_oracle = -1; \
    int ilo_s_cand = -1, ihi_s_cand = -1, info_s_cand = -1; \
    size_t idx = 0; \
    memcpy(a_n_oracle, a_n_template, sizeof(a_n_oracle)); \
    ((fn_type)oracle_fn)(&job_n, &n, a_n_oracle, &lda, &ilo_n_oracle, \
                         &ihi_n_oracle, scale_n_oracle, &info_n_oracle); \
    if (info_n_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_n_cand, a_n_template, sizeof(a_n_cand)); \
    ((fn_type)cand_fn)(&job_n, &n, a_n_cand, &lda, &ilo_n_cand, &ihi_n_cand, \
                       scale_n_cand, &info_n_cand); \
    if (info_n_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_s_oracle, a_s_template, sizeof(a_s_oracle)); \
    ((fn_type)oracle_fn)(&job_s, &n, a_s_oracle, &lda, &ilo_s_oracle, \
                         &ihi_s_oracle, scale_s_oracle, &info_s_oracle); \
    if (info_s_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_s_cand, a_s_template, sizeof(a_s_cand)); \
    ((fn_type)cand_fn)(&job_s, &n, a_s_cand, &lda, &ilo_s_cand, &ihi_s_cand, \
                       scale_s_cand, &info_s_cand); \
    if (info_s_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    for (size_t i = 0; i < 4u; i++) { \
        oracle_out[idx++] = (real_type)__real__(a_n_oracle[i]); \
        oracle_out[idx++] = (real_type)__imag__(a_n_oracle[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = scale_n_oracle[i]; \
    oracle_out[idx++] = (real_type)ilo_n_oracle; \
    oracle_out[idx++] = (real_type)ihi_n_oracle; \
    for (size_t i = 0; i < 4u; i++) { \
        oracle_out[idx++] = (real_type)__real__(a_s_oracle[i]); \
        oracle_out[idx++] = (real_type)__imag__(a_s_oracle[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = scale_s_oracle[i]; \
    oracle_out[idx++] = (real_type)ilo_s_oracle; \
    oracle_out[idx++] = (real_type)ihi_s_oracle; \
    idx = 0; \
    for (size_t i = 0; i < 4u; i++) { \
        cand_out[idx++] = (real_type)__real__(a_n_cand[i]); \
        cand_out[idx++] = (real_type)__imag__(a_n_cand[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = scale_n_cand[i]; \
    cand_out[idx++] = (real_type)ilo_n_cand; \
    cand_out[idx++] = (real_type)ihi_n_cand; \
    for (size_t i = 0; i < 4u; i++) { \
        cand_out[idx++] = (real_type)__real__(a_s_cand[i]); \
        cand_out[idx++] = (real_type)__imag__(a_s_cand[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = scale_s_cand[i]; \
    cand_out[idx++] = (real_type)ilo_s_cand; \
    cand_out[idx++] = (real_type)ihi_s_cand; \
    if (fb_judge_has_nan_inf(oracle_out, 24u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (fb_judge_has_nan_inf(cand_out, 24u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 24u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 24u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type a_time[4]; \
            real_type scale_time[2] = {(real_type)0.0, (real_type)0.0}; \
            int ilo_time = -1, ihi_time = -1, info_time = -1; \
            memcpy(a_time, a_s_template, sizeof(a_time)); \
            ((fn_type)cand_fn)(&job_s, &n, a_time, &lda, &ilo_time, &ihi_time, \
                               scale_time, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type a_time[4]; \
            real_type scale_time[2] = {(real_type)0.0, (real_type)0.0}; \
            int ilo_time = -1, ihi_time = -1, info_time = -1; \
            memcpy(a_time, a_s_template, sizeof(a_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&job_s, &n, a_time, &lda, &ilo_time, &ihi_time, \
                               scale_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBRD_RUNNER_REAL(name, op_id, fn_type, scalar_type, \
                                    output_dtype, norm_fn) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    int m = 2, n = 2, lda = 2, lwork = 4; \
    scalar_type a_template[4] = { \
        (scalar_type)1.0, (scalar_type)3.0, \
        (scalar_type)2.0, (scalar_type)4.0 \
    }; \
    scalar_type a_oracle[4]; \
    scalar_type a_cand[4]; \
    scalar_type d_oracle[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type d_cand[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type e_oracle[1] = {(scalar_type)0.0}; \
    scalar_type e_cand[1] = {(scalar_type)0.0}; \
    scalar_type tauq_oracle[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type tauq_cand[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type taup_oracle[1] = {(scalar_type)0.0}; \
    scalar_type taup_cand[1] = {(scalar_type)0.0}; \
    scalar_type work_oracle[4] = {(scalar_type)0.0, (scalar_type)0.0, \
                                  (scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type work_cand[4] = {(scalar_type)0.0, (scalar_type)0.0, \
                                (scalar_type)0.0, (scalar_type)0.0}; \
    scalar_type oracle_out[10]; \
    scalar_type cand_out[10]; \
    int info_oracle = -1; \
    int info_cand = -1; \
    size_t idx = 0; \
    memcpy(a_oracle, a_template, sizeof(a_oracle)); \
    ((fn_type)oracle_fn)(&m, &n, a_oracle, &lda, d_oracle, e_oracle, \
                         tauq_oracle, taup_oracle, work_oracle, &lwork, \
                         &info_oracle); \
    if (info_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_cand, a_template, sizeof(a_cand)); \
    ((fn_type)cand_fn)(&m, &n, a_cand, &lda, d_cand, e_cand, tauq_cand, \
                       taup_cand, work_cand, &lwork, &info_cand); \
    if (info_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    for (size_t i = 0; i < 4u; i++) oracle_out[idx++] = a_oracle[i]; \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = d_oracle[i]; \
    oracle_out[idx++] = e_oracle[0]; \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = tauq_oracle[i]; \
    oracle_out[idx++] = taup_oracle[0]; \
    idx = 0; \
    for (size_t i = 0; i < 4u; i++) cand_out[idx++] = a_cand[i]; \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = d_cand[i]; \
    cand_out[idx++] = e_cand[0]; \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = tauq_cand[i]; \
    cand_out[idx++] = taup_cand[0]; \
    if (fb_judge_has_nan_inf(oracle_out, 10u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (fb_judge_has_nan_inf(cand_out, 10u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 10u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 10u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type a_time[4]; \
            scalar_type d_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            scalar_type e_time[1] = {(scalar_type)0.0}; \
            scalar_type tauq_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            scalar_type taup_time[1] = {(scalar_type)0.0}; \
            scalar_type work_time[4] = {(scalar_type)0.0, (scalar_type)0.0, \
                                        (scalar_type)0.0, (scalar_type)0.0}; \
            int info_time = -1; \
            memcpy(a_time, a_template, sizeof(a_time)); \
            ((fn_type)cand_fn)(&m, &n, a_time, &lda, d_time, e_time, tauq_time, \
                               taup_time, work_time, &lwork, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type a_time[4]; \
            scalar_type d_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            scalar_type e_time[1] = {(scalar_type)0.0}; \
            scalar_type tauq_time[2] = {(scalar_type)0.0, (scalar_type)0.0}; \
            scalar_type taup_time[1] = {(scalar_type)0.0}; \
            scalar_type work_time[4] = {(scalar_type)0.0, (scalar_type)0.0, \
                                        (scalar_type)0.0, (scalar_type)0.0}; \
            int info_time = -1; \
            memcpy(a_time, a_template, sizeof(a_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&m, &n, a_time, &lda, d_time, e_time, tauq_time, \
                               taup_time, work_time, &lwork, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_GEBRD_RUNNER_COMPLEX(name, op_id, fn_type, scalar_type, \
                                       real_type, output_dtype, norm_fn, \
                                       value_init) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fb_generic_fn oracle_fn = oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fb_generic_fn cand_fn = cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL; \
    int m = 2, n = 2, lda = 2, lwork = 4; \
    scalar_type a_template[4] = { \
        value_init((real_type)1.0, (real_type)0.5), \
        value_init((real_type)3.0, (real_type)-1.0), \
        value_init((real_type)2.0, (real_type)1.5), \
        value_init((real_type)4.0, (real_type)-0.25) \
    }; \
    scalar_type a_oracle[4]; \
    scalar_type a_cand[4]; \
    real_type d_oracle[2] = {(real_type)0.0, (real_type)0.0}; \
    real_type d_cand[2] = {(real_type)0.0, (real_type)0.0}; \
    real_type e_oracle[1] = {(real_type)0.0}; \
    real_type e_cand[1] = {(real_type)0.0}; \
    scalar_type tauq_oracle[2] = {value_init((real_type)0.0, (real_type)0.0), \
                                  value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type tauq_cand[2] = {value_init((real_type)0.0, (real_type)0.0), \
                                value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type taup_oracle[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type taup_cand[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type work_oracle[4] = {value_init((real_type)0.0, (real_type)0.0), \
                                  value_init((real_type)0.0, (real_type)0.0), \
                                  value_init((real_type)0.0, (real_type)0.0), \
                                  value_init((real_type)0.0, (real_type)0.0)}; \
    scalar_type work_cand[4] = {value_init((real_type)0.0, (real_type)0.0), \
                                value_init((real_type)0.0, (real_type)0.0), \
                                value_init((real_type)0.0, (real_type)0.0), \
                                value_init((real_type)0.0, (real_type)0.0)}; \
    real_type oracle_out[17]; \
    real_type cand_out[17]; \
    int info_oracle = -1; \
    int info_cand = -1; \
    size_t idx = 0; \
    memcpy(a_oracle, a_template, sizeof(a_oracle)); \
    ((fn_type)oracle_fn)(&m, &n, a_oracle, &lda, d_oracle, e_oracle, \
                         tauq_oracle, taup_oracle, work_oracle, &lwork, \
                         &info_oracle); \
    if (info_oracle != 0) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    memcpy(a_cand, a_template, sizeof(a_cand)); \
    ((fn_type)cand_fn)(&m, &n, a_cand, &lda, d_cand, e_cand, tauq_cand, \
                       taup_cand, work_cand, &lwork, &info_cand); \
    if (info_cand != 0) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    for (size_t i = 0; i < 4u; i++) { \
        oracle_out[idx++] = (real_type)__real__(a_oracle[i]); \
        oracle_out[idx++] = (real_type)__imag__(a_oracle[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) oracle_out[idx++] = d_oracle[i]; \
    oracle_out[idx++] = e_oracle[0]; \
    for (size_t i = 0; i < 2u; i++) { \
        oracle_out[idx++] = (real_type)__real__(tauq_oracle[i]); \
        oracle_out[idx++] = (real_type)__imag__(tauq_oracle[i]); \
    } \
    oracle_out[idx++] = (real_type)__real__(taup_oracle[0]); \
    oracle_out[idx++] = (real_type)__imag__(taup_oracle[0]); \
    idx = 0; \
    for (size_t i = 0; i < 4u; i++) { \
        cand_out[idx++] = (real_type)__real__(a_cand[i]); \
        cand_out[idx++] = (real_type)__imag__(a_cand[i]); \
    } \
    for (size_t i = 0; i < 2u; i++) cand_out[idx++] = d_cand[i]; \
    cand_out[idx++] = e_cand[0]; \
    for (size_t i = 0; i < 2u; i++) { \
        cand_out[idx++] = (real_type)__real__(tauq_cand[i]); \
        cand_out[idx++] = (real_type)__imag__(tauq_cand[i]); \
    } \
    cand_out[idx++] = (real_type)__real__(taup_cand[0]); \
    cand_out[idx++] = (real_type)__imag__(taup_cand[0]); \
    if (fb_judge_has_nan_inf(oracle_out, 17u, output_dtype)) { \
        result_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    if (fb_judge_has_nan_inf(cand_out, 17u, output_dtype)) { \
        result_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    result_from_relerr(res, fb_judge_relerr(cand_out, oracle_out, 17u, \
                                            output_dtype, \
                                            norm_fn(oracle_out, 17u))); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { \
            scalar_type a_time[4]; \
            real_type d_time[2] = {(real_type)0.0, (real_type)0.0}; \
            real_type e_time[1] = {(real_type)0.0}; \
            scalar_type tauq_time[2] = {value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0)}; \
            scalar_type taup_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            scalar_type work_time[4] = {value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0)}; \
            int info_time = -1; \
            memcpy(a_time, a_template, sizeof(a_time)); \
            ((fn_type)cand_fn)(&m, &n, a_time, &lda, d_time, e_time, tauq_time, \
                               taup_time, work_time, &lwork, &info_time); \
        } \
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) { \
            scalar_type a_time[4]; \
            real_type d_time[2] = {(real_type)0.0, (real_type)0.0}; \
            real_type e_time[1] = {(real_type)0.0}; \
            scalar_type tauq_time[2] = {value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0)}; \
            scalar_type taup_time[1] = {value_init((real_type)0.0, (real_type)0.0)}; \
            scalar_type work_time[4] = {value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0), \
                                        value_init((real_type)0.0, (real_type)0.0)}; \
            int info_time = -1; \
            memcpy(a_time, a_template, sizeof(a_time)); \
            uint64_t t0 = fb_judge_time_ns(); \
            ((fn_type)cand_fn)(&m, &n, a_time, &lda, d_time, e_time, tauq_time, \
                               taup_time, work_time, &lwork, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

static fb_judge_status_t run_sgbmvx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGBMVX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGBMVX][FB_CONV_FORTRAN];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    char trans = 'N';
    int m = 3, n = 3, kl = 1, ku = 1, ldab = 3, incx = 1, incy = 1;
    float alpha = 2.0f, beta = 0.5f;
    float ab[9] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 0.0f};
    float x[3] = {1.0f, 2.0f, 3.0f};
    float y_template[3] = {10.0f, 20.0f, 30.0f};
    float y_oracle[3];
    float y_cand[3];

    memcpy(y_oracle, y_template, sizeof(y_oracle));
    ((fb_sgbmvx_fortran_fn_t)oracle_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                        &ldab, x, &incx, &beta, y_oracle,
                                        &incy);
    if (fb_judge_has_nan_inf(y_oracle, 3u, FB_DTYPE_F32)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    memcpy(y_cand, y_template, sizeof(y_cand));
    ((fb_sgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                      &ldab, x, &incx, &beta, y_cand,
                                      &incy);
    result_from_relerr(
        res,
        fb_judge_relerr(y_cand, y_oracle, 3u, FB_DTYPE_F32,
                        fb_norm_frob_f32(y_oracle, 3u)));
    if (fb_judge_has_nan_inf(y_cand, 3u, FB_DTYPE_F32)) res->is_fatal = true;

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            float y_time[3];
            memcpy(y_time, y_template, sizeof(y_time));
            ((fb_sgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
        }
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            float y_time[3];
            memcpy(y_time, y_template, sizeof(y_time));
            uint64_t t0 = fb_judge_time_ns();
            ((fb_sgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgbmvx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGBMVX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGBMVX][FB_CONV_FORTRAN];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    char trans = 'C';
    int m = 3, n = 2, kl = 1, ku = 0, ldab = 2, incx = -1, incy = -1;
    double alpha = 1.0, beta = 0.5;
    double ab[4] = {1.0, 2.0, 3.0, 4.0};
    double x[3] = {5.0, -1.0, 2.0};
    double y_template[2] = {10.0, 20.0};
    double y_oracle[2];
    double y_cand[2];

    memcpy(y_oracle, y_template, sizeof(y_oracle));
    ((fb_dgbmvx_fortran_fn_t)oracle_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                        &ldab, x, &incx, &beta, y_oracle,
                                        &incy);
    if (fb_judge_has_nan_inf(y_oracle, 2u, FB_DTYPE_F64)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    memcpy(y_cand, y_template, sizeof(y_cand));
    ((fb_dgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                      &ldab, x, &incx, &beta, y_cand,
                                      &incy);
    result_from_relerr(
        res,
        fb_judge_relerr(y_cand, y_oracle, 2u, FB_DTYPE_F64,
                        fb_norm_frob_f64(y_oracle, 2u)));
    if (fb_judge_has_nan_inf(y_cand, 2u, FB_DTYPE_F64)) res->is_fatal = true;

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            double y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            ((fb_dgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
        }
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            double y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            uint64_t t0 = fb_judge_time_ns();
            ((fb_dgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgbmvx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_CGBMVX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_CGBMVX][FB_CONV_FORTRAN];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    char trans = 'T';
    int m = 2, n = 2, kl = 1, ku = 1, ldab = 3, incx = 1, incy = 1;
    fb_complex_float_t alpha = fb_direct_make_cf32(1.0f, 0.0f);
    fb_complex_float_t beta = fb_direct_make_cf32(0.0f, 0.0f);
    fb_complex_float_t ab[6] = {
        fb_direct_make_cf32(0.0f, 0.0f),
        fb_direct_make_cf32(1.0f, 1.0f),
        fb_direct_make_cf32(3.0f, 0.0f),
        fb_direct_make_cf32(2.0f, -1.0f),
        fb_direct_make_cf32(4.0f, 2.0f),
        fb_direct_make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t x[2] = {
        fb_direct_make_cf32(1.0f, 1.0f),
        fb_direct_make_cf32(2.0f, -1.0f)
    };
    fb_complex_float_t y_template[2] = {
        fb_direct_make_cf32(0.0f, 0.0f),
        fb_direct_make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t y_oracle[2];
    fb_complex_float_t y_cand[2];

    memcpy(y_oracle, y_template, sizeof(y_oracle));
    ((fb_cgbmvx_fortran_fn_t)oracle_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                        &ldab, x, &incx, &beta, y_oracle,
                                        &incy);
    if (fb_judge_has_nan_inf(y_oracle, 2u, FB_DTYPE_CF32)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    memcpy(y_cand, y_template, sizeof(y_cand));
    ((fb_cgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                      &ldab, x, &incx, &beta, y_cand,
                                      &incy);
    result_from_relerr(
        res,
        fb_judge_relerr(y_cand, y_oracle, 2u, FB_DTYPE_CF32,
                        fb_norm_frob_cf32((const float *)y_oracle, 2u)));
    if (fb_judge_has_nan_inf(y_cand, 2u, FB_DTYPE_CF32)) res->is_fatal = true;

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            fb_complex_float_t y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            ((fb_cgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
        }
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            fb_complex_float_t y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            uint64_t t0 = fb_judge_time_ns();
            ((fb_cgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgbmvx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    (void)tc;
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_ZGBMVX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_ZGBMVX][FB_CONV_FORTRAN];
    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;

    char trans = 'C';
    int m = 2, n = 2, kl = 1, ku = 1, ldab = 3, incx = 1, incy = 1;
    fb_complex_double_t alpha = fb_direct_make_cf64(1.0, 0.0);
    fb_complex_double_t beta = fb_direct_make_cf64(0.0, 0.0);
    fb_complex_double_t ab[6] = {
        fb_direct_make_cf64(0.0, 0.0),
        fb_direct_make_cf64(1.0, 1.0),
        fb_direct_make_cf64(3.0, 0.0),
        fb_direct_make_cf64(2.0, -1.0),
        fb_direct_make_cf64(4.0, 2.0),
        fb_direct_make_cf64(0.0, 0.0)
    };
    fb_complex_double_t x[2] = {
        fb_direct_make_cf64(1.0, 1.0),
        fb_direct_make_cf64(2.0, -1.0)
    };
    fb_complex_double_t y_template[2] = {
        fb_direct_make_cf64(0.0, 0.0),
        fb_direct_make_cf64(0.0, 0.0)
    };
    fb_complex_double_t y_oracle[2];
    fb_complex_double_t y_cand[2];

    memcpy(y_oracle, y_template, sizeof(y_oracle));
    ((fb_zgbmvx_fortran_fn_t)oracle_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                        &ldab, x, &incx, &beta, y_oracle,
                                        &incy);
    if (fb_judge_has_nan_inf(y_oracle, 2u, FB_DTYPE_CF64)) {
        result_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    memcpy(y_cand, y_template, sizeof(y_cand));
    ((fb_zgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku, &alpha, ab,
                                      &ldab, x, &incx, &beta, y_cand,
                                      &incy);
    result_from_relerr(
        res,
        fb_judge_relerr(y_cand, y_oracle, 2u, FB_DTYPE_CF64,
                        fb_norm_frob_cf64((const double *)y_oracle, 2u)));
    if (fb_judge_has_nan_inf(y_cand, 2u, FB_DTYPE_CF64)) res->is_fatal = true;

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) {
            fb_complex_double_t y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            ((fb_zgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
        }
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            fb_complex_double_t y_time[2];
            memcpy(y_time, y_template, sizeof(y_time));
            uint64_t t0 = fb_judge_time_ns();
            ((fb_zgbmvx_fortran_fn_t)cand_fn)(&trans, &m, &n, &kl, &ku,
                                              &alpha, ab, &ldab, x, &incx,
                                              &beta, y_time, &incy);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    return FB_JUDGE_OK;
}

FB_DEFINE_GBCON_RUNNER(run_sgbcon, FB_OP_SGBCON, fb_sgbcon_fortran_fn_t,
                       float, float, scalar_relerr_f32, FB_DTYPE_F32,
                       4.0f, 0.0f)
FB_DEFINE_GBCON_RUNNER(run_dgbcon, FB_OP_DGBCON, fb_dgbcon_fortran_fn_t,
                       double, double, scalar_relerr_f64, FB_DTYPE_F64,
                       4.0, 0.0)
FB_DEFINE_GBCON_RUNNER(run_cgbcon, FB_OP_CGBCON, fb_cgbcon_fortran_fn_t,
                       fb_complex_float_t, float, scalar_relerr_f32,
                       FB_DTYPE_F32, fb_direct_make_cf32(4.0f, 0.0f),
                       fb_direct_make_cf32(0.0f, 0.0f))
FB_DEFINE_GBCON_RUNNER(run_zgbcon, FB_OP_ZGBCON, fb_zgbcon_fortran_fn_t,
                       fb_complex_double_t, double, scalar_relerr_f64,
                       FB_DTYPE_F64, fb_direct_make_cf64(4.0, 0.0),
                       fb_direct_make_cf64(0.0, 0.0))

FB_DEFINE_GBEQU_RUNNER(run_sgbequ, FB_OP_SGBEQU, fb_sgbequ_fortran_fn_t,
                       float, float, FB_DTYPE_F32, fb_norm_frob_f32, 4.0f)
FB_DEFINE_GBEQU_RUNNER(run_dgbequ, FB_OP_DGBEQU, fb_dgbequ_fortran_fn_t,
                       double, double, FB_DTYPE_F64, fb_norm_frob_f64, 4.0)
FB_DEFINE_GBEQU_RUNNER(run_cgbequ, FB_OP_CGBEQU, fb_cgbequ_fortran_fn_t,
                       fb_complex_float_t, float, FB_DTYPE_F32,
                       fb_norm_frob_f32, fb_direct_make_cf32(4.0f, 0.0f))
FB_DEFINE_GBEQU_RUNNER(run_zgbequ, FB_OP_ZGBEQU, fb_zgbequ_fortran_fn_t,
                       fb_complex_double_t, double, FB_DTYPE_F64,
                       fb_norm_frob_f64, fb_direct_make_cf64(4.0, 0.0))

FB_DEFINE_GBRFS_RUNNER_REAL(run_sgbrfs, FB_OP_SGBRFS,
                            fb_sgbrfs_fortran_fn_t, float, float,
                            FB_DTYPE_F32, fb_norm_frob_f32,
                            fb_direct_make_cf32)
FB_DEFINE_GBRFS_RUNNER_REAL(run_dgbrfs, FB_OP_DGBRFS,
                            fb_dgbrfs_fortran_fn_t, double, double,
                            FB_DTYPE_F64, fb_norm_frob_f64,
                            fb_direct_make_cf64)
FB_DEFINE_GBRFS_RUNNER_COMPLEX(run_cgbrfs, FB_OP_CGBRFS,
                               fb_cgbrfs_fortran_fn_t, fb_complex_float_t,
                               float, FB_DTYPE_F32, fb_norm_frob_f32,
                               fb_direct_make_cf32)
FB_DEFINE_GBRFS_RUNNER_COMPLEX(run_zgbrfs, FB_OP_ZGBRFS,
                               fb_zgbrfs_fortran_fn_t, fb_complex_double_t,
                               double, FB_DTYPE_F64, fb_norm_frob_f64,
                               fb_direct_make_cf64)

FB_DEFINE_GEBAK_RUNNER_REAL(run_sgebak, FB_OP_SGEBAK,
                            fb_sgebak_fortran_fn_t, float, FB_DTYPE_F32,
                            fb_norm_frob_f32)
FB_DEFINE_GEBAK_RUNNER_REAL(run_dgebak, FB_OP_DGEBAK,
                            fb_dgebak_fortran_fn_t, double, FB_DTYPE_F64,
                            fb_norm_frob_f64)
FB_DEFINE_GEBAK_RUNNER_COMPLEX(run_cgebak, FB_OP_CGEBAK,
                               fb_cgebak_fortran_fn_t, fb_complex_float_t,
                               float, FB_DTYPE_CF32, fb_norm_frob_cf32,
                               float, fb_direct_make_cf32)
FB_DEFINE_GEBAK_RUNNER_COMPLEX(run_zgebak, FB_OP_ZGEBAK,
                               fb_zgebak_fortran_fn_t, fb_complex_double_t,
                               double, FB_DTYPE_CF64, fb_norm_frob_cf64,
                               double, fb_direct_make_cf64)
FB_DEFINE_GEBAL_RUNNER_REAL(run_sgebal, FB_OP_SGEBAL,
                            fb_sgebal_fortran_fn_t, float, FB_DTYPE_F32,
                            fb_norm_frob_f32)
FB_DEFINE_GEBAL_RUNNER_REAL(run_dgebal, FB_OP_DGEBAL,
                            fb_dgebal_fortran_fn_t, double, FB_DTYPE_F64,
                            fb_norm_frob_f64)
FB_DEFINE_GEBAL_RUNNER_COMPLEX(run_cgebal, FB_OP_CGEBAL,
                               fb_cgebal_fortran_fn_t, fb_complex_float_t,
                               float, FB_DTYPE_F32, fb_norm_frob_f32,
                               fb_direct_make_cf32)
FB_DEFINE_GEBAL_RUNNER_COMPLEX(run_zgebal, FB_OP_ZGEBAL,
                               fb_zgebal_fortran_fn_t, fb_complex_double_t,
                               double, FB_DTYPE_F64, fb_norm_frob_f64,
                               fb_direct_make_cf64)
FB_DEFINE_GEBRD_RUNNER_REAL(run_sgebrd, FB_OP_SGEBRD,
                            fb_sgebrd_fortran_fn_t, float, FB_DTYPE_F32,
                            fb_norm_frob_f32)
FB_DEFINE_GEBRD_RUNNER_REAL(run_dgebrd, FB_OP_DGEBRD,
                            fb_dgebrd_fortran_fn_t, double, FB_DTYPE_F64,
                            fb_norm_frob_f64)
FB_DEFINE_GEBRD_RUNNER_COMPLEX(run_cgebrd, FB_OP_CGEBRD,
                               fb_cgebrd_fortran_fn_t, fb_complex_float_t,
                               float, FB_DTYPE_F32, fb_norm_frob_f32,
                               fb_direct_make_cf32)
FB_DEFINE_GEBRD_RUNNER_COMPLEX(run_zgebrd, FB_OP_ZGEBRD,
                               fb_zgebrd_fortran_fn_t, fb_complex_double_t,
                               double, FB_DTYPE_F64, fb_norm_frob_f64,
                               fb_direct_make_cf64)
FB_DEFINE_GECON_RUNNER_REAL(run_sgecon, FB_OP_SGECON,
                            fb_sgecon_fortran_fn_t, float, float,
                            scalar_relerr_f32, 4.0f, 0.0f)
FB_DEFINE_GECON_RUNNER_REAL(run_dgecon, FB_OP_DGECON,
                            fb_dgecon_fortran_fn_t, double, double,
                            scalar_relerr_f64, 4.0, 0.0)
FB_DEFINE_GECON_RUNNER_COMPLEX(run_cgecon, FB_OP_CGECON,
                               fb_cgecon_fortran_fn_t,
                               fb_complex_float_t, float,
                               scalar_relerr_f32,
                               fb_direct_make_cf32(4.0f, 0.0f),
                               fb_direct_make_cf32(0.0f, 0.0f))
FB_DEFINE_GECON_RUNNER_COMPLEX(run_zgecon, FB_OP_ZGECON,
                               fb_zgecon_fortran_fn_t,
                               fb_complex_double_t, double,
                               scalar_relerr_f64,
                               fb_direct_make_cf64(4.0, 0.0),
                               fb_direct_make_cf64(0.0, 0.0))
FB_DEFINE_GEEQU_RUNNER(run_sgeequ, FB_OP_SGEEQU, fb_sgeequ_fortran_fn_t,
                       float, float, FB_DTYPE_F32, fb_norm_frob_f32, 4.0f)
FB_DEFINE_GEEQU_RUNNER(run_dgeequ, FB_OP_DGEEQU, fb_dgeequ_fortran_fn_t,
                       double, double, FB_DTYPE_F64, fb_norm_frob_f64, 4.0)

static fb_judge_status_t run_sgbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgbmv || !cand->sgbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int kl = (tc->k > 0 && (int)tc->k < n) ? (int)tc->k : 2;
    int ku = kl, lda_band = kl + ku + 1;
    float alpha = (float)tc->alpha, beta = (float)tc->beta;
    if ((size_t)(lda_band * n) > tc->A_elems || m <= 0 || n <= 0) { result_fatal(res); return FB_JUDGE_OK; }
    const float *AB = (const float *)tc->A, *x = (const float *)tc->B;
    float *yo = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->sgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)m, FB_DTYPE_F32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    float *yc = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->sgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)m, FB_DTYPE_F32, fb_norm_frob_f32(yo, (size_t)m)));
    if (fb_judge_has_nan_inf(yc, (size_t)m, FB_DTYPE_F32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        float *yt = (float *)clone_buf(tc->C_init, (size_t)m, sizeof(float));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->sgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->sgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgbmv || !cand->dgbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int kl = (tc->k > 0 && (int)tc->k < n) ? (int)tc->k : 2;
    int ku = kl, lda_band = kl + ku + 1;
    double alpha = tc->alpha, beta = tc->beta;
    if ((size_t)(lda_band * n) > tc->A_elems || m <= 0 || n <= 0) { result_fatal(res); return FB_JUDGE_OK; }
    const double *AB = (const double *)tc->A, *x = (const double *)tc->B;
    double *yo = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->dgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)m, FB_DTYPE_F64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    double *yc = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->dgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)m, FB_DTYPE_F64, fb_norm_frob_f64(yo, (size_t)m)));
    if (fb_judge_has_nan_inf(yc, (size_t)m, FB_DTYPE_F64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        double *yt = (double *)clone_buf(tc->C_init, (size_t)m, sizeof(double));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->dgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->dgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgbmv || !cand->cgbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int kl = (tc->k > 0 && (int)tc->k < n) ? (int)tc->k : 2;
    int ku = kl, lda_band = kl + ku + 1;
    fb_complex_float_t alpha, beta;
    __real__(alpha) = (float)tc->alpha; __imag__(alpha) = 0.0f;
    __real__(beta)  = (float)tc->beta;  __imag__(beta)  = 0.0f;
    if ((size_t)(lda_band * n) > tc->A_elems || m <= 0 || n <= 0) { result_fatal(res); return FB_JUDGE_OK; }
    const fb_complex_float_t *AB = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x  = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)m, FB_DTYPE_CF32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)m, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)yo, (size_t)m)));
    if (fb_judge_has_nan_inf(yc, (size_t)m, FB_DTYPE_CF32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_float_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->cgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->cgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgbmv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgbmv || !cand->zgbmv) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n;
    int kl = (tc->k > 0 && (int)tc->k < n) ? (int)tc->k : 2;
    int ku = kl, lda_band = kl + ku + 1;
    fb_complex_double_t alpha, beta;
    __real__(alpha) = tc->alpha; __imag__(alpha) = 0.0;
    __real__(beta)  = tc->beta;  __imag__(beta)  = 0.0;
    if ((size_t)(lda_band * n) > tc->A_elems || m <= 0 || n <= 0) { result_fatal(res); return FB_JUDGE_OK; }
    const fb_complex_double_t *AB = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x  = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)m, FB_DTYPE_CF64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)m, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)yo, (size_t)m)));
    if (fb_judge_has_nan_inf(yc, (size_t)m, FB_DTYPE_CF64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)m, sizeof(fb_complex_double_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zgbmv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, AB, lda_band, x, 1, beta, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

/* =========================================================================
 * CSYMV/ZSYMV — complex symmetric matrix-vector multiply (alpha/beta void*)
 * ========================================================================= */
static fb_judge_status_t run_csymv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csymv || !cand->csymv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha_v, beta_v;
    __real__(alpha_v) = (float)tc->alpha; __imag__(alpha_v) = 0.0f;
    __real__(beta_v)  = (float)tc->beta;  __imag__(beta_v)  = 0.0f;
    const void *alpha = &alpha_v, *beta_p = &beta_v;
    const fb_complex_float_t *A = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t *yo = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF32)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *yc = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF32, fb_norm_frob_cf32((const float *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_float_t *yt = (fb_complex_float_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_float_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->csymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->csymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsymv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zsymv || !cand->zsymv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha_v, beta_v;
    __real__(alpha_v) = tc->alpha; __imag__(alpha_v) = 0.0;
    __real__(beta_v)  = tc->beta;  __imag__(beta_v)  = 0.0;
    const void *alpha = &alpha_v, *beta_p = &beta_v;
    const fb_complex_double_t *A = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t *yo = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yo) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yo, 1);
    if (fb_judge_has_nan_inf(yo, (size_t)n, FB_DTYPE_CF64)) { free(yo); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *yc = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
    if (!yc) { free(yo); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yc, 1);
    result_from_relerr(res, fb_judge_relerr(yc, yo, (size_t)n, FB_DTYPE_CF64, fb_norm_frob_cf64((const double *)yo, (size_t)n)));
    if (fb_judge_has_nan_inf(yc, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(yc);
    if (ns_out) {
        fb_complex_double_t *yt = (fb_complex_double_t *)clone_buf(tc->C_init, (size_t)n, sizeof(fb_complex_double_t));
        if (yt) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yt, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zsymv(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, A, lda, x, 1, beta_p, yt, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(yt); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(yo); return FB_JUDGE_OK;
}

/* =========================================================================
 * CSYR/ZSYR — complex symmetric rank-1 update (alpha as void*)
 * ========================================================================= */
static fb_judge_status_t run_csyr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->csyr || !cand->csyr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_float_t alpha_v;
    __real__(alpha_v) = (float)tc->alpha; __imag__(alpha_v) = 0.0f;
    const void *alpha = &alpha_v;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_float_t *Ao = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->csyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF32)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *Ac = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->csyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, n, n, lda, lda, FB_DTYPE_CF32,
        fb_norm_frob_cf32((const float *)Ao, (size_t)(n * lda)));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF32)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_float_t *At = (fb_complex_float_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_float_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->csyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns(); cand->csyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsyr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zsyr || !cand->zsyr) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    fb_complex_double_t alpha_v;
    __real__(alpha_v) = tc->alpha; __imag__(alpha_v) = 0.0;
    const void *alpha = &alpha_v;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    size_t A_sz = (size_t)(n * lda);
    fb_complex_double_t *Ao = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ao) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ao, lda);
    if (fb_judge_has_nan_inf(Ao, A_sz, FB_DTYPE_CF64)) { free(Ao); result_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *Ac = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
    if (!Ac) { free(Ao); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, Ac, lda);
    double relerr = fb_judge_relerr_matrix(Ac, Ao, n, n, lda, lda, FB_DTYPE_CF64,
        fb_norm_frob_cf64((const double *)Ao, (size_t)(n * lda)));
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(Ac, A_sz, FB_DTYPE_CF64)) res->is_fatal = true;
    free(Ac);
    if (ns_out) {
        fb_complex_double_t *At = (fb_complex_double_t *)clone_buf(tc->C_init, A_sz, sizeof(fb_complex_double_t));
        if (At) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
            uint64_t best = UINT64_MAX;
            for (int t2 = 0; t2 < FB_JUDGE_TIMING_RUNS; t2++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zsyr(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, alpha, x, 1, At, lda);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
            }
            free(At); *ns_out = best;
        } else { *ns_out = 0; }
    }
    free(Ao); return FB_JUDGE_OK;
}

/* =========================================================================
 * CROTG/ZROTG — complex Givens rotation construction
 * ========================================================================= */
static fb_judge_status_t run_crotg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->crotg || !cand->crotg) return FB_JUDGE_ERR_NOT_IMPL;
    const fb_complex_float_t *ab = (const fb_complex_float_t *)tc->A;
    fb_complex_float_t ao = ab[0], bo = ab[1], so;
    __real__(so) = 0.0f; __imag__(so) = 0.0f;
    float co = 0.0f;
    oracle->crotg(&ao, &bo, &co, &so);
    fb_complex_float_t ac = ab[0], bc = ab[1], sc;
    __real__(sc) = 0.0f; __imag__(sc) = 0.0f;
    float cc = 0.0f;
    cand->crotg(&ac, &bc, &cc, &sc);
    float so_abs = hypotf(__real__(so), __imag__(so));
    double ec = scalar_relerr_f32(cc, co, (double)fabsf(co));
    double es_re = scalar_relerr_f32(__real__(sc), __real__(so), (double)so_abs);
    double es_im = scalar_relerr_f32(__imag__(sc), __imag__(so), (double)so_abs);
    double emax = ec > es_re ? ec : es_re; if (es_im > emax) emax = es_im;
    result_from_relerr(res, emax);
    if (ns_out) {
        fb_complex_float_t a2, b2, s2; float c2;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { a2 = ab[0]; b2 = ab[1]; cand->crotg(&a2, &b2, &c2, &s2); }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            a2 = ab[0]; b2 = ab[1];
            uint64_t t0 = fb_judge_time_ns(); cand->crotg(&a2, &b2, &c2, &s2);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zrotg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zrotg || !cand->zrotg) return FB_JUDGE_ERR_NOT_IMPL;
    const fb_complex_double_t *ab = (const fb_complex_double_t *)tc->A;
    fb_complex_double_t ao = ab[0], bo = ab[1], so;
    __real__(so) = 0.0; __imag__(so) = 0.0;
    double co = 0.0;
    oracle->zrotg(&ao, &bo, &co, &so);
    fb_complex_double_t ac = ab[0], bc = ab[1], sc;
    __real__(sc) = 0.0; __imag__(sc) = 0.0;
    double cc = 0.0;
    cand->zrotg(&ac, &bc, &cc, &sc);
    double so_abs = hypot(__real__(so), __imag__(so));
    double ec = scalar_relerr_f64(cc, co, fabs(co));
    double es_re = scalar_relerr_f64(__real__(sc), __real__(so), so_abs);
    double es_im = scalar_relerr_f64(__imag__(sc), __imag__(so), so_abs);
    double emax = ec > es_re ? ec : es_re; if (es_im > emax) emax = es_im;
    result_from_relerr(res, emax);
    if (ns_out) {
        fb_complex_double_t a2, b2, s2; double c2;
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { a2 = ab[0]; b2 = ab[1]; cand->zrotg(&a2, &b2, &c2, &s2); }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            a2 = ab[0]; b2 = ab[1];
            uint64_t t0 = fb_judge_time_ns(); cand->zrotg(&a2, &b2, &c2, &s2);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DROTMG — double modified Givens rotation construction
 * ========================================================================= */
static fb_judge_status_t run_drotmg(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->drotmg || !cand->drotmg) return FB_JUDGE_ERR_NOT_IMPL;
    const double *ab = (const double *)tc->A;
    if (tc->A_elems < 4) { result_fatal(res); return FB_JUDGE_OK; }
    double d1o = ab[0], d2o = ab[1], x1o = ab[2]; const double y1 = ab[3];
    double param_o[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    oracle->drotmg(&d1o, &d2o, &x1o, y1, param_o);
    double d1c = ab[0], d2c = ab[1], x1c = ab[2];
    double param_c[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    cand->drotmg(&d1c, &d2c, &x1c, y1, param_c);
    double max_err = 0.0;
    for (int i = 0; i < 5; i++) {
        double ar = fabs(param_o[i]);
        double e = (ar > 1e-16) ? fabs(param_o[i] - param_c[i]) / ar : fabs(param_o[i] - param_c[i]);
        if (e > max_err) max_err = e;
    }
    result_from_relerr(res, max_err);
    if (ns_out) {
        double a2, b2, c2; double p2[5];
        for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) { a2=ab[0]; b2=ab[1]; c2=ab[2]; cand->drotmg(&a2,&b2,&c2,y1,p2); }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
            a2=ab[0]; b2=ab[1]; c2=ab[2];
            uint64_t t0 = fb_judge_time_ns(); cand->drotmg(&a2,&b2,&c2,y1,p2);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}


/* =========================================================================
 * Dispatch table
 *
 * Indexed by op_id (see judge_op_ids.h). NULL entries return
 * FB_JUDGE_ERR_NOT_IMPL. The table covers all op IDs up to batched GEMM.
 * ========================================================================= */

/* Dispatch table spans all op IDs — size is FB_JUDGE_MAX_OPERATIONS */

static const fb_direct_runner_fn fb_direct_dispatch[FB_JUDGE_MAX_OPERATIONS] = {
    /* BLAS Level 1 */
    [FB_OP_SAXPY] = run_saxpy,
    [FB_OP_DAXPY] = run_daxpy,
    [FB_OP_CAXPY] = run_caxpy,
    [FB_OP_ZAXPY] = run_zaxpy,
    [FB_OP_SAXPBY] = run_saxpby,
    [FB_OP_DAXPBY] = run_daxpby,
    [FB_OP_CAXPBY] = run_caxpby,
    [FB_OP_ZAXPBY] = run_zaxpby,
    [FB_OP_CBLAS_SAXPBY] = run_saxpby,
    [FB_OP_CBLAS_DAXPBY] = run_daxpby,
    [FB_OP_CBLAS_CAXPBY] = run_caxpby,
    [FB_OP_CBLAS_ZAXPBY] = run_zaxpby,
    [FB_OP_SAXPY_BATCH] = run_saxpy_batch,
    [FB_OP_DAXPY_BATCH] = run_daxpy_batch,
    [FB_OP_CAXPY_BATCH] = run_caxpy_batch,
    [FB_OP_ZAXPY_BATCH] = run_zaxpy_batch,
    [FB_OP_CBLAS_SAXPY_BATCH] = run_saxpy_batch,
    [FB_OP_CBLAS_DAXPY_BATCH] = run_daxpy_batch,
    [FB_OP_CBLAS_CAXPY_BATCH] = run_caxpy_batch,
    [FB_OP_CBLAS_ZAXPY_BATCH] = run_zaxpy_batch,
    [FB_OP_SAXPY_BATCH_STRIDED] = run_saxpy_batch_strided,
    [FB_OP_DAXPY_BATCH_STRIDED] = run_daxpy_batch_strided,
    [FB_OP_CAXPY_BATCH_STRIDED] = run_caxpy_batch_strided,
    [FB_OP_ZAXPY_BATCH_STRIDED] = run_zaxpy_batch_strided,
    [FB_OP_CBLAS_SAXPY_BATCH_STRIDED] = run_saxpy_batch_strided,
    [FB_OP_CBLAS_DAXPY_BATCH_STRIDED] = run_daxpy_batch_strided,
    [FB_OP_CBLAS_CAXPY_BATCH_STRIDED] = run_caxpy_batch_strided,
    [FB_OP_CBLAS_ZAXPY_BATCH_STRIDED] = run_zaxpy_batch_strided,
    [FB_OP_SCOPY_BATCH] = run_scopy_batch,
    [FB_OP_DCOPY_BATCH] = run_dcopy_batch,
    [FB_OP_CCOPY_BATCH] = run_ccopy_batch,
    [FB_OP_ZCOPY_BATCH] = run_zcopy_batch,
    [FB_OP_CBLAS_SCOPY_BATCH] = run_scopy_batch,
    [FB_OP_CBLAS_DCOPY_BATCH] = run_dcopy_batch,
    [FB_OP_CBLAS_CCOPY_BATCH] = run_ccopy_batch,
    [FB_OP_CBLAS_ZCOPY_BATCH] = run_zcopy_batch,
    [FB_OP_SCOPY_BATCH_STRIDED] = run_scopy_batch_strided,
    [FB_OP_DCOPY_BATCH_STRIDED] = run_dcopy_batch_strided,
    [FB_OP_CCOPY_BATCH_STRIDED] = run_ccopy_batch_strided,
    [FB_OP_ZCOPY_BATCH_STRIDED] = run_zcopy_batch_strided,
    [FB_OP_CBLAS_SCOPY_BATCH_STRIDED] = run_scopy_batch_strided,
    [FB_OP_CBLAS_DCOPY_BATCH_STRIDED] = run_dcopy_batch_strided,
    [FB_OP_CBLAS_CCOPY_BATCH_STRIDED] = run_ccopy_batch_strided,
    [FB_OP_CBLAS_ZCOPY_BATCH_STRIDED] = run_zcopy_batch_strided,
    [FB_OP_SSCAL] = run_sscal,
    [FB_OP_DSCAL] = run_dscal,
    [FB_OP_CSCAL] = run_cscal,
    [FB_OP_ZSCAL] = run_zscal,
    [FB_OP_CSSCAL] = run_csscal,
    [FB_OP_ZDSCAL] = run_zdscal,
    [FB_OP_SCOPY] = run_scopy,
    [FB_OP_DCOPY] = run_dcopy,
    [FB_OP_CCOPY] = run_ccopy,
    [FB_OP_ZCOPY] = run_zcopy,
    [FB_OP_SSWAP] = run_sswap,
    [FB_OP_DSWAP] = run_dswap,
    [FB_OP_CSWAP] = run_cswap,
    [FB_OP_ZSWAP] = run_zswap,
    [FB_OP_SDOT] = run_sdot,
    [FB_OP_DDOT] = run_ddot,
    [FB_OP_CDOTC] = run_cdotc,
    [FB_OP_CDOTU] = run_cdotu,
    [FB_OP_ZDOTC] = run_zdotc,
    [FB_OP_ZDOTU] = run_zdotu,
    [FB_OP_SDSDOT] = run_sdsdot,
    [FB_OP_DSDOT] = run_dsdot,
    [FB_OP_SASUM] = run_sasum,
    [FB_OP_DASUM] = run_dasum,
    [FB_OP_SCASUM] = run_scasum,
    [FB_OP_DZASUM] = run_dzasum,
    [FB_OP_SNRM2] = run_snrm2,
    [FB_OP_DNRM2] = run_dnrm2,
    [FB_OP_SCNRM2] = run_scnrm2,
    [FB_OP_DZNRM2] = run_dznrm2,
    [FB_OP_ISAMAX] = run_isamax,
    [FB_OP_IDAMAX] = run_idamax,
    [FB_OP_ICAMAX] = run_icamax,
    [FB_OP_IZAMAX] = run_izamax,
    [FB_OP_SROT] = run_srot,
    [FB_OP_DROT] = run_drot,
    [FB_OP_CROT] = run_crot,
    [FB_OP_ZROT] = run_zrot,
    [FB_OP_ZDROT] = run_zdrot,
    [FB_OP_SROTG] = run_srotg,
    [FB_OP_DROTG] = run_drotg,
    [FB_OP_SROTM] = run_srotm,
    [FB_OP_DROTM] = run_drotm,
    [FB_OP_SROTMG] = run_srotmg,
    /* BLAS Level 2 */
    [FB_OP_SGEMV] = run_sgemv,
    [FB_OP_DGEMV] = run_dgemv,
    [FB_OP_CGEMV] = run_cgemv,
    [FB_OP_ZGEMV] = run_zgemv,
    [FB_OP_SGEMV_BATCH] = run_sgemv_batch,
    [FB_OP_DGEMV_BATCH] = run_dgemv_batch,
    [FB_OP_CGEMV_BATCH] = run_cgemv_batch,
    [FB_OP_ZGEMV_BATCH] = run_zgemv_batch,
    [FB_OP_CBLAS_SGEMV_BATCH] = run_sgemv_batch,
    [FB_OP_CBLAS_DGEMV_BATCH] = run_dgemv_batch,
    [FB_OP_CBLAS_CGEMV_BATCH] = run_cgemv_batch,
    [FB_OP_CBLAS_ZGEMV_BATCH] = run_zgemv_batch,
    [FB_OP_SGEMV_BATCH_STRIDED] = run_sgemv_batch_strided,
    [FB_OP_DGEMV_BATCH_STRIDED] = run_dgemv_batch_strided,
    [FB_OP_CGEMV_BATCH_STRIDED] = run_cgemv_batch_strided,
    [FB_OP_ZGEMV_BATCH_STRIDED] = run_zgemv_batch_strided,
    [FB_OP_CBLAS_SGEMV_BATCH_STRIDED] = run_sgemv_batch_strided,
    [FB_OP_CBLAS_DGEMV_BATCH_STRIDED] = run_dgemv_batch_strided,
    [FB_OP_CBLAS_CGEMV_BATCH_STRIDED] = run_cgemv_batch_strided,
    [FB_OP_CBLAS_ZGEMV_BATCH_STRIDED] = run_zgemv_batch_strided,
    [FB_OP_SDGMM_BATCH] = run_sdgmm_batch,
    [FB_OP_DDGMM_BATCH] = run_ddgmm_batch,
    [FB_OP_CDGMM_BATCH] = run_cdgmm_batch,
    [FB_OP_ZDGMM_BATCH] = run_zdgmm_batch,
    [FB_OP_CBLAS_SDGMM_BATCH] = run_sdgmm_batch,
    [FB_OP_CBLAS_DDGMM_BATCH] = run_ddgmm_batch,
    [FB_OP_CBLAS_CDGMM_BATCH] = run_cdgmm_batch,
    [FB_OP_CBLAS_ZDGMM_BATCH] = run_zdgmm_batch,
    [FB_OP_SDGMM_BATCH_STRIDED] = run_sdgmm_batch_strided,
    [FB_OP_DDGMM_BATCH_STRIDED] = run_ddgmm_batch_strided,
    [FB_OP_CDGMM_BATCH_STRIDED] = run_cdgmm_batch_strided,
    [FB_OP_ZDGMM_BATCH_STRIDED] = run_zdgmm_batch_strided,
    [FB_OP_CBLAS_SDGMM_BATCH_STRIDED] = run_sdgmm_batch_strided,
    [FB_OP_CBLAS_DDGMM_BATCH_STRIDED] = run_ddgmm_batch_strided,
    [FB_OP_CBLAS_CDGMM_BATCH_STRIDED] = run_cdgmm_batch_strided,
    [FB_OP_CBLAS_ZDGMM_BATCH_STRIDED] = run_zdgmm_batch_strided,
    [FB_OP_SSYMM_BATCH] = run_ssymm_batch,
    [FB_OP_DSYMM_BATCH] = run_dsymm_batch,
    [FB_OP_CSYMM_BATCH] = run_csymm_batch,
    [FB_OP_ZSYMM_BATCH] = run_zsymm_batch,
    [FB_OP_CBLAS_SSYMM_BATCH] = run_ssymm_batch,
    [FB_OP_CBLAS_DSYMM_BATCH] = run_dsymm_batch,
    [FB_OP_CBLAS_CSYMM_BATCH] = run_csymm_batch,
    [FB_OP_CBLAS_ZSYMM_BATCH] = run_zsymm_batch,
    [FB_OP_SSYR2K_BATCH] = run_ssyr2k_batch,
    [FB_OP_DSYR2K_BATCH] = run_dsyr2k_batch,
    [FB_OP_CSYR2K_BATCH] = run_csyr2k_batch,
    [FB_OP_ZSYR2K_BATCH] = run_zsyr2k_batch,
    [FB_OP_CBLAS_SSYR2K_BATCH] = run_ssyr2k_batch,
    [FB_OP_CBLAS_DSYR2K_BATCH] = run_dsyr2k_batch,
    [FB_OP_CBLAS_CSYR2K_BATCH] = run_csyr2k_batch,
    [FB_OP_CBLAS_ZSYR2K_BATCH] = run_zsyr2k_batch,
    [FB_OP_SSYRK_BATCH] = run_ssyrk_batch,
    [FB_OP_DSYRK_BATCH] = run_dsyrk_batch,
    [FB_OP_CSYRK_BATCH] = run_csyrk_batch,
    [FB_OP_ZSYRK_BATCH] = run_zsyrk_batch,
    [FB_OP_CBLAS_SSYRK_BATCH] = run_ssyrk_batch,
    [FB_OP_CBLAS_DSYRK_BATCH] = run_dsyrk_batch,
    [FB_OP_CBLAS_CSYRK_BATCH] = run_csyrk_batch,
    [FB_OP_CBLAS_ZSYRK_BATCH] = run_zsyrk_batch,
    [FB_OP_STRSM_BATCH] = run_strsm_batch,
    [FB_OP_DTRSM_BATCH] = run_dtrsm_batch,
    [FB_OP_CTRSM_BATCH] = run_ctrsm_batch,
    [FB_OP_ZTRSM_BATCH] = run_ztrsm_batch,
    [FB_OP_CBLAS_STRSM_BATCH] = run_strsm_batch,
    [FB_OP_CBLAS_DTRSM_BATCH] = run_dtrsm_batch,
    [FB_OP_CBLAS_CTRSM_BATCH] = run_ctrsm_batch,
    [FB_OP_CBLAS_ZTRSM_BATCH] = run_ztrsm_batch,
    [FB_OP_STRSM_BATCH_STRIDED] = run_strsm_batch_strided,
    [FB_OP_DTRSM_BATCH_STRIDED] = run_dtrsm_batch_strided,
    [FB_OP_CTRSM_BATCH_STRIDED] = run_ctrsm_batch_strided,
    [FB_OP_ZTRSM_BATCH_STRIDED] = run_ztrsm_batch_strided,
    [FB_OP_CBLAS_STRSM_BATCH_STRIDED] = run_strsm_batch_strided,
    [FB_OP_CBLAS_DTRSM_BATCH_STRIDED] = run_dtrsm_batch_strided,
    [FB_OP_CBLAS_CTRSM_BATCH_STRIDED] = run_ctrsm_batch_strided,
    [FB_OP_CBLAS_ZTRSM_BATCH_STRIDED] = run_ztrsm_batch_strided,
    [FB_OP_SSYMV] = run_ssymv,
    [FB_OP_DSYMV] = run_dsymv,
    [FB_OP_CHEMV] = run_chemv,
    [FB_OP_ZHEMV] = run_zhemv,
    [FB_OP_STRMV] = run_strmv,
    [FB_OP_DTRMV] = run_dtrmv,
    [FB_OP_CTRMV] = run_ctrmv,
    [FB_OP_ZTRMV] = run_ztrmv,
    [FB_OP_STRSV] = run_strsv,
    [FB_OP_DTRSV] = run_dtrsv,
    [FB_OP_CTRSV] = run_ctrsv,
    [FB_OP_ZTRSV] = run_ztrsv,
    [FB_OP_SGER] = run_sger,
    [FB_OP_DGER] = run_dger,
    [FB_OP_CGERU] = run_cgeru,
    [FB_OP_CGERC] = run_cgerc,
    [FB_OP_ZGERU] = run_zgeru,
    [FB_OP_ZGERC] = run_zgerc,
    [FB_OP_SSYR] = run_ssyr,
    [FB_OP_DSYR] = run_dsyr,
    [FB_OP_CHER] = run_cher,
    [FB_OP_ZHER] = run_zher,
    [FB_OP_SSYR2] = run_ssyr2,
    [FB_OP_DSYR2] = run_dsyr2,
    [FB_OP_CHER2] = run_cher2,
    [FB_OP_ZHER2] = run_zher2,
    [FB_OP_SSPMV] = run_sspmv,
    [FB_OP_DSPMV] = run_dspmv,
    [FB_OP_CHPMV] = run_chpmv,
    [FB_OP_ZHPMV] = run_zhpmv,
    [FB_OP_SBMV] = run_sbmv,
    [FB_OP_DBMV] = run_dbmv,
    [FB_OP_SSBMV] = run_ssbmv,
    [FB_OP_DSBMV] = run_dsbmv,
    [FB_OP_CHBMV] = run_chbmv,
    [FB_OP_ZHBMV] = run_zhbmv,
    [FB_OP_STBMV] = run_stbmv,
    [FB_OP_DTBMV] = run_dtbmv,
    [FB_OP_CTBMV] = run_ctbmv,
    [FB_OP_ZTBMV] = run_ztbmv,
    [FB_OP_STBSV] = run_stbsv,
    [FB_OP_DTBSV] = run_dtbsv,
    [FB_OP_CTBSV] = run_ctbsv,
    [FB_OP_ZTBSV] = run_ztbsv,
    [FB_OP_STPMV] = run_stpmv,
    [FB_OP_DTPMV] = run_dtpmv,
    [FB_OP_CTPMV] = run_ctpmv,
    [FB_OP_ZTPMV] = run_ztpmv,
    [FB_OP_STPSV] = run_stpsv,
    [FB_OP_DTPSV] = run_dtpsv,
    [FB_OP_CTPSV] = run_ctpsv,
    [FB_OP_ZTPSV] = run_ztpsv,
    [FB_OP_SSPR] = run_sspr,
    [FB_OP_DSPR] = run_dspr,
    [FB_OP_CHPR] = run_chpr,
    [FB_OP_ZHPR] = run_zhpr,
    [FB_OP_SSPR2] = run_sspr2,
    [FB_OP_DSPR2] = run_dspr2,
    [FB_OP_CHPR2] = run_chpr2,
    [FB_OP_ZHPR2] = run_zhpr2,
    /* BLAS Level 3 */
    [FB_OP_SGEMM_COMPUTE] = run_sgemm_compute,
    [FB_OP_DGEMM_COMPUTE] = run_dgemm_compute,
    [FB_OP_CGEMM_COMPUTE] = run_cgemm_compute,
    [FB_OP_ZGEMM_COMPUTE] = run_zgemm_compute,
    [FB_OP_CBLAS_SGEMM_COMPUTE] = run_cblas_sgemm_compute,
    [FB_OP_CBLAS_DGEMM_COMPUTE] = run_cblas_dgemm_compute,
    [FB_OP_CBLAS_CGEMM_COMPUTE] = run_cblas_cgemm_compute,
    [FB_OP_CBLAS_ZGEMM_COMPUTE] = run_cblas_zgemm_compute,
    [FB_OP_SGEMM_PACK] = run_sgemm_pack,
    [FB_OP_DGEMM_PACK] = run_dgemm_pack,
    [FB_OP_CGEMM_PACK] = run_cgemm_pack,
    [FB_OP_ZGEMM_PACK] = run_zgemm_pack,
    [FB_OP_CBLAS_SGEMM_PACK] = run_cblas_sgemm_pack,
    [FB_OP_CBLAS_DGEMM_PACK] = run_cblas_dgemm_pack,
    [FB_OP_CBLAS_CGEMM_PACK] = run_cblas_cgemm_pack,
    [FB_OP_CBLAS_ZGEMM_PACK] = run_cblas_zgemm_pack,
    [FB_OP_SGEMM_PACK_GET_SIZE] = run_sgemm_pack_get_size,
    [FB_OP_DGEMM_PACK_GET_SIZE] = run_dgemm_pack_get_size,
    [FB_OP_CGEMM_PACK_GET_SIZE] = run_cgemm_pack_get_size,
    [FB_OP_ZGEMM_PACK_GET_SIZE] = run_zgemm_pack_get_size,
    [FB_OP_CBLAS_SGEMM_PACK_GET_SIZE] = run_cblas_sgemm_pack_get_size,
    [FB_OP_CBLAS_DGEMM_PACK_GET_SIZE] = run_cblas_dgemm_pack_get_size,
    [FB_OP_CBLAS_CGEMM_PACK_GET_SIZE] = run_cblas_cgemm_pack_get_size,
    [FB_OP_CBLAS_ZGEMM_PACK_GET_SIZE] = run_cblas_zgemm_pack_get_size,
    [FB_OP_SGEMM_PTR] = run_sgemm_ptr,
    [FB_OP_DGEMM_PTR] = run_dgemm_ptr,
    [FB_OP_CGEMM_PTR] = run_cgemm_ptr,
    [FB_OP_ZGEMM_PTR] = run_zgemm_ptr,
    [FB_OP_MKL_JIT_CREATE_CGEMM] = run_mkl_jit_create_cgemm,
    [FB_OP_MKL_JIT_CREATE_DGEMM] = run_mkl_jit_create_dgemm,
    [FB_OP_MKL_JIT_CREATE_SGEMM] = run_mkl_jit_create_sgemm,
    [FB_OP_MKL_JIT_CREATE_ZGEMM] = run_mkl_jit_create_zgemm,
    [FB_OP_MKL_JIT_DESTROY] = run_mkl_jit_destroy,
    [FB_OP_MKL_JIT_GET_CGEMM_PTR] = run_mkl_jit_get_cgemm_ptr,
    [FB_OP_MKL_JIT_GET_DGEMM_PTR] = run_mkl_jit_get_dgemm_ptr,
    [FB_OP_MKL_JIT_GET_SGEMM_PTR] = run_mkl_jit_get_sgemm_ptr,
    [FB_OP_MKL_JIT_GET_ZGEMM_PTR] = run_mkl_jit_get_zgemm_ptr,
    [FB_OP_SGEMM] = run_sgemm,
    [FB_OP_DGEMM] = run_dgemm,
    [FB_OP_CGEMM] = run_cgemm,
    [FB_OP_ZGEMM] = run_zgemm,
    [FB_OP_SGEMM_BATCH] = run_sgemm_batch,
    [FB_OP_DGEMM_BATCH] = run_dgemm_batch,
    [FB_OP_CGEMM_BATCH] = run_cgemm_batch,
    [FB_OP_ZGEMM_BATCH] = run_zgemm_batch,
    [FB_OP_CBLAS_SGEMM_BATCH] = run_sgemm_batch,
    [FB_OP_CBLAS_DGEMM_BATCH] = run_dgemm_batch,
    [FB_OP_CBLAS_CGEMM_BATCH] = run_cgemm_batch,
    [FB_OP_CBLAS_ZGEMM_BATCH] = run_zgemm_batch,
    [FB_OP_SGEMM_STRIDED] = run_sgemm_strided,
    [FB_OP_DGEMM_STRIDED] = run_dgemm_strided,
    [FB_OP_CGEMM_STRIDED] = run_cgemm_strided,
    [FB_OP_ZGEMM_STRIDED] = run_zgemm_strided,
    [FB_OP_CBLAS_SGEMM_BATCH_STRIDED] = run_sgemm_strided,
    [FB_OP_CBLAS_DGEMM_BATCH_STRIDED] = run_dgemm_strided,
    [FB_OP_CBLAS_CGEMM_BATCH_STRIDED] = run_cgemm_strided,
    [FB_OP_CBLAS_ZGEMM_BATCH_STRIDED] = run_zgemm_strided,
    [FB_OP_SGEMM3M_BATCH] = run_sgemm3m_batch,
    [FB_OP_DGEMM3M_BATCH] = run_dgemm3m_batch,
    [FB_OP_CGEMM3M_BATCH] = run_cgemm3m_batch,
    [FB_OP_ZGEMM3M_BATCH] = run_zgemm3m_batch,
    [FB_OP_CBLAS_SGEMM3M_BATCH] = run_cblas_sgemm3m_batch,
    [FB_OP_CBLAS_DGEMM3M_BATCH] = run_cblas_dgemm3m_batch,
    [FB_OP_CBLAS_CGEMM3M_BATCH] = run_cblas_cgemm3m_batch,
    [FB_OP_CBLAS_ZGEMM3M_BATCH] = run_cblas_zgemm3m_batch,
    [FB_OP_SGEMM3M_BATCH_STRIDED] = run_sgemm3m_batch_strided,
    [FB_OP_DGEMM3M_BATCH_STRIDED] = run_dgemm3m_batch_strided,
    [FB_OP_CGEMM3M_BATCH_STRIDED] = run_cgemm3m_batch_strided,
    [FB_OP_ZGEMM3M_BATCH_STRIDED] = run_zgemm3m_batch_strided,
    [FB_OP_CBLAS_SGEMM3M_BATCH_STRIDED] = run_cblas_sgemm3m_batch_strided,
    [FB_OP_CBLAS_DGEMM3M_BATCH_STRIDED] = run_cblas_dgemm3m_batch_strided,
    [FB_OP_CBLAS_CGEMM3M_BATCH_STRIDED] = run_cblas_cgemm3m_batch_strided,
    [FB_OP_CBLAS_ZGEMM3M_BATCH_STRIDED] = run_cblas_zgemm3m_batch_strided,
    [FB_OP_SGEMMT] = run_sgemmt,
    [FB_OP_DGEMMT] = run_dgemmt,
    [FB_OP_CGEMMT] = run_cgemmt,
    [FB_OP_ZGEMMT] = run_zgemmt,
    [FB_OP_CBLAS_SGEMMT] = run_cblas_sgemmt,
    [FB_OP_CBLAS_DGEMMT] = run_cblas_dgemmt,
    [FB_OP_CBLAS_CGEMMT] = run_cblas_cgemmt,
    [FB_OP_CBLAS_ZGEMMT] = run_cblas_zgemmt,
    [FB_OP_SSYMM] = run_ssymm,
    [FB_OP_DSYMM] = run_dsymm,
    [FB_OP_CSYMM] = run_csymm,
    [FB_OP_ZSYMM] = run_zsymm,
    [FB_OP_CHEMM] = run_chemm,
    [FB_OP_ZHEMM] = run_zhemm,
    [FB_OP_SSYRK] = run_ssyrk,
    [FB_OP_DSYRK] = run_dsyrk,
    [FB_OP_CSYRK] = run_csyrk,
    [FB_OP_ZSYRK] = run_zsyrk,
    [FB_OP_CHERK] = run_cherk,
    [FB_OP_ZHERK] = run_zherk,
    [FB_OP_SSYR2K] = run_ssyr2k,
    [FB_OP_DSYR2K] = run_dsyr2k,
    [FB_OP_CSYR2K] = run_csyr2k,
    [FB_OP_ZSYR2K] = run_zsyr2k,
    [FB_OP_CHER2K] = run_cher2k,
    [FB_OP_ZHER2K] = run_zher2k,
    [FB_OP_STRMM] = run_strmm,
    [FB_OP_DTRMM] = run_dtrmm,
    [FB_OP_CTRMM] = run_ctrmm,
    [FB_OP_ZTRMM] = run_ztrmm,
    [FB_OP_STRSM] = run_strsm,
    [FB_OP_DTRSM] = run_dtrsm,
    [FB_OP_CTRSM] = run_ctrsm,
    [FB_OP_ZTRSM] = run_ztrsm,
    [FB_OP_SGBMV] = run_sgbmv,
    [FB_OP_DGBMV] = run_dgbmv,
    [FB_OP_CGBMV] = run_cgbmv,
    [FB_OP_ZGBMV] = run_zgbmv,
    [FB_OP_SGBMVX] = run_sgbmvx,
    [FB_OP_DGBMVX] = run_dgbmvx,
    [FB_OP_CGBMVX] = run_cgbmvx,
    [FB_OP_ZGBMVX] = run_zgbmvx,
    [FB_OP_SGBCON] = run_sgbcon,
    [FB_OP_DGBCON] = run_dgbcon,
    [FB_OP_CGBCON] = run_cgbcon,
    [FB_OP_ZGBCON] = run_zgbcon,
    [FB_OP_SGEBAL] = run_sgebal,
    [FB_OP_DGEBAL] = run_dgebal,
    [FB_OP_CGEBAL] = run_cgebal,
    [FB_OP_ZGEBAL] = run_zgebal,
    [FB_OP_SGEBRD] = run_sgebrd,
    [FB_OP_DGEBRD] = run_dgebrd,
    [FB_OP_CGEBRD] = run_cgebrd,
    [FB_OP_ZGEBRD] = run_zgebrd,
    [FB_OP_SGECON] = run_sgecon,
    [FB_OP_DGECON] = run_dgecon,
    [FB_OP_CGECON] = run_cgecon,
    [FB_OP_ZGECON] = run_zgecon,
    [FB_OP_SGEEQU] = run_sgeequ,
    [FB_OP_DGEEQU] = run_dgeequ,
    [FB_OP_SGBEQU] = run_sgbequ,
    [FB_OP_DGBEQU] = run_dgbequ,
    [FB_OP_CGBEQU] = run_cgbequ,
    [FB_OP_ZGBEQU] = run_zgbequ,
    [FB_OP_SGBRFS] = run_sgbrfs,
    [FB_OP_DGBRFS] = run_dgbrfs,
    [FB_OP_CGBRFS] = run_cgbrfs,
    [FB_OP_ZGBRFS] = run_zgbrfs,
    [FB_OP_SGEBAK] = run_sgebak,
    [FB_OP_DGEBAK] = run_dgebak,
    [FB_OP_CGEBAK] = run_cgebak,
    [FB_OP_ZGEBAK] = run_zgebak,
    [FB_OP_CSYMV] = run_csymv,
    [FB_OP_ZSYMV] = run_zsymv,
    [FB_OP_CSYR]  = run_csyr,
    [FB_OP_ZSYR]  = run_zsyr,
    [FB_OP_CROTG] = run_crotg,
    [FB_OP_ZROTG] = run_zrotg,
    [FB_OP_DROTMG] = run_drotmg,
};

/* =========================================================================
 * Public entry point
 * ========================================================================= */

fb_judge_status_t fb_judge_run_direct_case(
    const fb_backend_vtable_t *oracle_vtable,
    const fb_backend_vtable_t *candidate_vtable,
    const fb_corpus_case_t    *tc,
    fb_judge_case_result_t    *result_out,
    uint64_t                  *elapsed_ns_out)
{
    if (!oracle_vtable || !candidate_vtable || !tc || !result_out)
        return FB_JUDGE_ERR_INVALID_OP;

    uint32_t op_id = tc->meta.op_id;

    /* Zero the result before dispatch so partial-fill is safe. */
    *result_out = (fb_judge_case_result_t){ 0 };

    if (op_id >= FB_JUDGE_MAX_OPERATIONS || fb_direct_dispatch[op_id] == NULL)
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_direct_dispatch[op_id](
        oracle_vtable, candidate_vtable, tc, result_out, elapsed_ns_out);
}