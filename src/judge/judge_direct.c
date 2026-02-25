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

    int64_t n    = (int64_t)tc->n;
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

    int64_t  n     = (int64_t)tc->n;
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

    int64_t n     = (int64_t)tc->n;
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

    int64_t n     = (int64_t)tc->n;
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

    int64_t      n = (int64_t)tc->n;
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

    int64_t       n = (int64_t)tc->n;
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

    int64_t n = (int64_t)tc->n;

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

    int64_t n = (int64_t)tc->n;

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

    int64_t      n = (int64_t)tc->n;
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

    int64_t       n = (int64_t)tc->n;
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

    int64_t      n = (int64_t)tc->n;
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

    int64_t       n = (int64_t)tc->n;
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

    int64_t      n = (int64_t)tc->n;
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

    int64_t       n = (int64_t)tc->n;
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

    int64_t      n = (int64_t)tc->n;
    const float *x = (const float *)tc->A;

    int64_t oracle_idx = oracle->isamax(n, x, 1);
    int64_t cand_idx   = cand->isamax(n, x, 1);

    if (oracle_idx < 0 || oracle_idx >= n) {
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    if (cand_idx == oracle_idx) {
        /* Perfect match */
        res->digits         = 16.0;
        res->relative_error = 0.0;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    } else {
        /* Check whether the magnitudes are within tolerance (could be a tie). */
        float abs_ref  = fabsf(x[oracle_idx]);
        float abs_cand = (cand_idx >= 0 && cand_idx < n) ? fabsf(x[cand_idx]) : 0.0f;
        /* If |x[cand]| < |x[oracle]| * (1 - eps), the candidate picked the wrong max. */
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

    int64_t       n = (int64_t)tc->n;
    const double *x = (const double *)tc->A;

    int64_t oracle_idx = oracle->idamax(n, x, 1);
    int64_t cand_idx   = cand->idamax(n, x, 1);

    if (oracle_idx < 0 || oracle_idx >= n) {
        result_oracle_fatal(res); return FB_JUDGE_OK;
    }

    if (cand_idx == oracle_idx) {
        res->digits = 16.0; res->relative_error = 0.0;
        res->is_fatal = false; res->is_oracle_fatal = false;
    } else {
        double abs_ref  = fabs(x[oracle_idx]);
        double abs_cand = (cand_idx >= 0 && cand_idx < n) ? fabs(x[cand_idx]) : 0.0;
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

    int64_t      m    = (int64_t)tc->m;
    int64_t      n    = (int64_t)tc->n;
    int64_t      lda  = (int64_t)tc->lda;
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

    int64_t       m    = (int64_t)tc->m;
    int64_t       n    = (int64_t)tc->n;
    int64_t       lda  = (int64_t)tc->lda;
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

/* ---- SGEMM -------------------------------------------------------------- */
static fb_judge_status_t run_sgemm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgemm || !cand->sgemm) return FB_JUDGE_ERR_NOT_IMPL;

    int64_t      m    = (int64_t)tc->m;
    int64_t      n    = (int64_t)tc->n;
    int64_t      k    = (int64_t)tc->k;
    int64_t      lda  = (int64_t)tc->lda;
    int64_t      ldb  = (int64_t)tc->ldb;
    int64_t      ldc  = (int64_t)tc->ldc;
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

    int64_t       m    = (int64_t)tc->m;
    int64_t       n    = (int64_t)tc->n;
    int64_t       k    = (int64_t)tc->k;
    int64_t       lda  = (int64_t)tc->lda;
    int64_t       ldb  = (int64_t)tc->ldb;
    int64_t       ldc  = (int64_t)tc->ldc;
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
    int64_t n = (int64_t)tc->n;
    fb_complex_float_t alpha;
    alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;

    fb_complex_float_t *y_oracle = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->caxpy(n, alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *y_cand = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->caxpy(n, alpha, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_float_t *y_time = (fb_complex_float_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_float_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->caxpy(n, alpha, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->caxpy(n, alpha, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
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
    int64_t n = (int64_t)tc->n;
    fb_complex_double_t alpha;
    alpha.real = tc->alpha; alpha.imag = 0.0;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;

    fb_complex_double_t *y_oracle = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zaxpy(n, alpha, x, 1, y_oracle, 1);
    if (fb_judge_has_nan_inf(y_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(y_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *y_cand = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
    if (!y_cand) { free(y_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zaxpy(n, alpha, x, 1, y_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)y_oracle, (size_t)n);
    double relerr = fb_judge_relerr(y_cand, y_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(y_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(y_cand);
    if (ns_out) {
        fb_complex_double_t *y_time = (fb_complex_double_t *)clone_buf(tc->B, (size_t)n, sizeof(fb_complex_double_t));
        if (y_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zaxpy(n, alpha, x, 1, y_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zaxpy(n, alpha, x, 1, y_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
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
    int64_t n = (int64_t)tc->n;
    fb_complex_float_t alpha;
    alpha.real = (float)tc->alpha; alpha.imag = 0.0f;

    fb_complex_float_t *x_oracle = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->cscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF32)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *x_cand = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->cscal(n, alpha, x_cand, 1);
    double scale = fb_norm_frob_cf32((const float *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF32, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF32)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_float_t *x_time = (fb_complex_float_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_float_t));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->cscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->cscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
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
    int64_t n = (int64_t)tc->n;
    fb_complex_double_t alpha;
    alpha.real = tc->alpha; alpha.imag = 0.0;

    fb_complex_double_t *x_oracle = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_oracle) { result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    oracle->zscal(n, alpha, x_oracle, 1);
    if (fb_judge_has_nan_inf(x_oracle, (size_t)n, FB_DTYPE_CF64)) {
        free(x_oracle); result_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *x_cand = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
    if (!x_cand) { free(x_oracle); result_fatal(res); return FB_JUDGE_ERR_ALLOC; }
    cand->zscal(n, alpha, x_cand, 1);
    double scale = fb_norm_frob_cf64((const double *)x_oracle, (size_t)n);
    double relerr = fb_judge_relerr(x_cand, x_oracle, (size_t)n, FB_DTYPE_CF64, scale);
    result_from_relerr(res, relerr);
    if (fb_judge_has_nan_inf(x_cand, (size_t)n, FB_DTYPE_CF64)) res->is_fatal = true;
    free(x_cand);
    if (ns_out) {
        fb_complex_double_t *x_time = (fb_complex_double_t *)clone_buf(tc->A, (size_t)n, sizeof(fb_complex_double_t));
        if (x_time) {
            for (int w = 0; w < FB_JUDGE_WARMUP_RUNS; w++) cand->zscal(n, alpha, x_time, 1);
            uint64_t best = UINT64_MAX;
            for (int t = 0; t < FB_JUDGE_TIMING_RUNS; t++) {
                uint64_t t0 = fb_judge_time_ns(); cand->zscal(n, alpha, x_time, 1);
                uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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

/* ---- CSWAP: swap x,y (complex single) --------------------------------- */
static fb_judge_status_t run_cswap(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cswap || !cand->cswap) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n;

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
    int64_t n = (int64_t)tc->n;

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
    int64_t n = (int64_t)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t r_oracle, r_cand;
    r_oracle.real = 0.0f; r_oracle.imag = 0.0f;
    oracle->cdotu(&r_oracle, n, x, 1, y, 1);
    r_cand.real = 0.0f; r_cand.imag = 0.0f;
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
    int64_t n = (int64_t)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    const fb_complex_float_t *y = (const fb_complex_float_t *)tc->B;
    fb_complex_float_t r_oracle, r_cand;
    r_oracle.real = 0.0f; r_oracle.imag = 0.0f;
    oracle->cdotc(&r_oracle, n, x, 1, y, 1);
    r_cand.real = 0.0f; r_cand.imag = 0.0f;
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
    int64_t n = (int64_t)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t r_oracle, r_cand;
    r_oracle.real = 0.0; r_oracle.imag = 0.0;
    oracle->zdotu(&r_oracle, n, x, 1, y, 1);
    r_cand.real = 0.0; r_cand.imag = 0.0;
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
    int64_t n = (int64_t)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    const fb_complex_double_t *y = (const fb_complex_double_t *)tc->B;
    fb_complex_double_t r_oracle, r_cand;
    r_oracle.real = 0.0; r_oracle.imag = 0.0;
    oracle->zdotc(&r_oracle, n, x, 1, y, 1);
    r_cand.real = 0.0; r_cand.imag = 0.0;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
    const fb_complex_float_t *x = (const fb_complex_float_t *)tc->A;
    int64_t oracle_idx = oracle->icamax(n, x, 1);
    int64_t cand_idx   = cand->icamax(n, x, 1);
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
    int64_t n = (int64_t)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;
    int64_t oracle_idx = oracle->izamax(n, x, 1);
    int64_t cand_idx   = cand->izamax(n, x, 1);
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t n = (int64_t)tc->n;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_float_t alpha; alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
    fb_complex_float_t beta;  beta.real  = (float)tc->beta;  beta.imag  = 0.0f;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_double_t alpha; alpha.real = tc->alpha; alpha.imag = 0.0;
    fb_complex_double_t beta;  beta.real  = tc->beta;  beta.imag  = 0.0;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_float_t alpha; alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
    fb_complex_float_t beta;  beta.real  = (float)tc->beta;  beta.imag  = 0.0f;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_double_t alpha; alpha.real = tc->alpha; alpha.imag = 0.0;
    fb_complex_double_t beta;  beta.real  = tc->beta;  beta.imag  = 0.0;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_float_t alpha; alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_float_t alpha; alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_double_t alpha; alpha.real = tc->alpha; alpha.imag = 0.0;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, lda = (int64_t)tc->lda;
    fb_complex_double_t alpha; alpha.real = tc->alpha; alpha.imag = 0.0;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, k = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb, ldc = (int64_t)tc->ldc;
    fb_complex_float_t alpha; alpha.real = (float)tc->alpha; alpha.imag = 0.0f;
    fb_complex_float_t beta;  beta.real  = (float)tc->beta;  beta.imag  = 0.0f;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, k = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb, ldc = (int64_t)tc->ldc;
    fb_complex_double_t alpha; alpha.real = tc->alpha; alpha.imag = 0.0;
    fb_complex_double_t beta;  beta.real  = tc->beta;  beta.imag  = 0.0;
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

/* =========================================================================
 * Phase 2 — L3 SYMM / SYRK / TRMM / TRSM
 * ========================================================================= */

static fb_judge_status_t run_ssymm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->ssymm || !cand->ssymm) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb, ldc = (int64_t)tc->ldc;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb, ldc = (int64_t)tc->ldc;
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
    int64_t n = (int64_t)tc->n, k = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldc = (int64_t)tc->ldc;
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
    int64_t n = (int64_t)tc->n, k = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldc = (int64_t)tc->ldc;
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

/* STRMM / DTRMM: B = alpha * A * B  (B in-place) */
static fb_judge_status_t run_strmm(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_case_result_t *res, uint64_t *ns_out)
{
    if (!oracle->strmm || !cand->strmm) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
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
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
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
 * Dispatch table
 *
 * Indexed by op_id (see judge_op_ids.h). NULL entries return
 * FB_JUDGE_ERR_NOT_IMPL. The table is sized to cover all currently
 * implemented ops (FB_OP_DGEMM = 131; round to nearest 8).
 * ========================================================================= */

#define FB_DIRECT_DISPATCH_SIZE  160

static const fb_direct_runner_fn fb_direct_dispatch[FB_DIRECT_DISPATCH_SIZE] = {
    /* 0  */ run_saxpy,   /* FB_OP_SAXPY  */
    /* 1  */ run_daxpy,   /* FB_OP_DAXPY  */
    /* 2  */ run_caxpy,   /* FB_OP_CAXPY  */
    /* 3  */ run_zaxpy,   /* FB_OP_ZAXPY  */
    /* 4  */ run_sscal,   /* FB_OP_SSCAL  */
    /* 5  */ run_dscal,   /* FB_OP_DSCAL  */
    /* 6  */ run_cscal,   /* FB_OP_CSCAL  */
    /* 7  */ run_zscal,   /* FB_OP_ZSCAL  */
    /* 8  */ run_csscal,  /* FB_OP_CSSCAL */
    /* 9  */ run_zdscal,  /* FB_OP_ZDSCAL */
    /* 10 */ run_scopy,   /* FB_OP_SCOPY  */
    /* 11 */ run_dcopy,   /* FB_OP_DCOPY  */
    /* 12 */ run_ccopy,   /* FB_OP_CCOPY  */
    /* 13 */ run_zcopy,   /* FB_OP_ZCOPY  */
    /* 14 */ run_sswap,   /* FB_OP_SSWAP  */
    /* 15 */ run_dswap,   /* FB_OP_DSWAP  */
    /* 16 */ run_cswap,   /* FB_OP_CSWAP  */
    /* 17 */ run_zswap,   /* FB_OP_ZSWAP  */
    /* 18 */ run_sdot,    /* FB_OP_SDOT   */
    /* 19 */ run_ddot,    /* FB_OP_DDOT   */
    /* 20 */ run_cdotc,   /* FB_OP_CDOTC  */
    /* 21 */ run_cdotu,   /* FB_OP_CDOTU  */
    /* 22 */ run_zdotc,   /* FB_OP_ZDOTC  */
    /* 23 */ run_zdotu,   /* FB_OP_ZDOTU  */
    /* 24 */ run_sdsdot,  /* FB_OP_SDSDOT */
    /* 25 */ run_dsdot,   /* FB_OP_DSDOT  */
    /* 26 */ run_sasum,   /* FB_OP_SASUM  */
    /* 27 */ run_dasum,   /* FB_OP_DASUM  */
    /* 28 */ run_scasum,  /* FB_OP_SCASUM */
    /* 29 */ run_dzasum,  /* FB_OP_DZASUM */
    /* 30 */ run_snrm2,   /* FB_OP_SNRM2  */
    /* 31 */ run_dnrm2,   /* FB_OP_DNRM2  */
    /* 32 */ run_scnrm2,  /* FB_OP_SCNRM2 */
    /* 33 */ run_dznrm2,  /* FB_OP_DZNRM2 */
    /* 34 */ run_isamax,  /* FB_OP_ISAMAX */
    /* 35 */ run_idamax,  /* FB_OP_IDAMAX */
    /* 36 */ run_icamax,  /* FB_OP_ICAMAX */
    /* 37 */ run_izamax,  /* FB_OP_IZAMAX */
    /* 38 */ run_srot,    /* FB_OP_SROT   */
    /* 39 */ run_drot,    /* FB_OP_DROT   */
    /* 40 */ NULL,        /* FB_OP_CROT   — not yet implemented */
    /* 41 */ NULL,        /* FB_OP_ZROT   — not yet implemented */
    /* 42 */ NULL,        /* FB_OP_ZDROT  — not yet implemented */
    /* 43 */ run_srotg,   /* FB_OP_SROTG  */
    /* 44 */ run_drotg,   /* FB_OP_DROTG  */
    /* 45 */ run_srotm,   /* FB_OP_SROTM  */
    /* 46 */ run_drotm,   /* FB_OP_DROTM  */
    /* 47 */ run_srotmg,  /* FB_OP_SROTMG */
    /* 48 */ run_sgemv,   /* FB_OP_SGEMV  */
    /* 49 */ run_dgemv,   /* FB_OP_DGEMV  */
    /* 50 */ run_cgemv,   /* FB_OP_CGEMV  */
    /* 51 */ run_zgemv,   /* FB_OP_ZGEMV  */
    [52]  = run_ssymv,   /* FB_OP_SSYMV  */
    [53]  = run_dsymv,   /* FB_OP_DSYMV  */
    [54]  = run_chemv,   /* FB_OP_CHEMV  */
    [55]  = run_zhemv,   /* FB_OP_ZHEMV  */
    [56]  = run_strmv,   /* FB_OP_STRMV  */
    [57]  = run_dtrmv,   /* FB_OP_DTRMV  */
    [58]  = run_ctrmv,   /* FB_OP_CTRMV  */
    [59]  = run_ztrmv,   /* FB_OP_ZTRMV  */
    [60]  = run_strsv,   /* FB_OP_STRSV  */
    [61]  = run_dtrsv,   /* FB_OP_DTRSV  */
    [62]  = run_ctrsv,   /* FB_OP_CTRSV  */
    [63]  = run_ztrsv,   /* FB_OP_ZTRSV  */
    [64]  = run_sger,    /* FB_OP_SGER   */
    [65]  = run_dger,    /* FB_OP_DGER   */
    [66]  = run_cgeru,   /* FB_OP_CGERU  */
    [67]  = run_cgerc,   /* FB_OP_CGERC  */
    [68]  = run_zgeru,   /* FB_OP_ZGERU  */
    [69]  = run_zgerc,   /* FB_OP_ZGERC  */
    [70]  = NULL, [71]  = NULL, [72]  = NULL, [73]  = NULL,
    [74]  = NULL, [75]  = NULL, [76]  = NULL, [77]  = NULL,
    [78]  = NULL, [79]  = NULL, [80]  = NULL, [81]  = NULL,
    [82]  = NULL, [83]  = NULL, [84]  = NULL, [85]  = NULL,
    [86]  = NULL, [87]  = NULL, [88]  = NULL, [89]  = NULL,
    [90]  = NULL, [91]  = NULL, [92]  = NULL, [93]  = NULL,
    [94]  = NULL, [95]  = NULL, [96]  = NULL, [97]  = NULL,
    [98]  = NULL, [99]  = NULL,
    [100] = NULL, [101] = NULL, [102] = NULL, [103] = NULL,
    [104] = NULL, [105] = NULL, [106] = NULL, [107] = NULL,
    [108] = NULL, [109] = NULL, [110] = NULL, [111] = NULL,
    [112] = NULL, [113] = NULL, [114] = NULL, [115] = NULL,
    [116] = NULL, [117] = NULL, [118] = NULL, [119] = NULL,
    [120] = NULL, [121] = NULL, [122] = NULL, [123] = NULL,
    [124] = NULL, [125] = NULL, [126] = NULL, [127] = NULL,
    [128] = NULL, [129] = NULL,
    /* L3 */
    [130] = run_sgemm,   /* FB_OP_SGEMM  */
    [131] = run_dgemm,   /* FB_OP_DGEMM  */
    [132] = run_cgemm,   /* FB_OP_CGEMM  */
    [133] = run_zgemm,   /* FB_OP_ZGEMM  */
    [134] = run_ssymm,   /* FB_OP_SSYMM  */
    [135] = run_dsymm,   /* FB_OP_DSYMM  */
    [136] = NULL, [137] = NULL,
    [138] = NULL, [139] = NULL,
    [140] = run_ssyrk,   /* FB_OP_SSYRK  */
    [141] = run_dsyrk,   /* FB_OP_DSYRK  */
    [142] = NULL, [143] = NULL,
    [144] = NULL, [145] = NULL, [146] = NULL, [147] = NULL,
    [148] = NULL, [149] = NULL, [150] = NULL, [151] = NULL,
    [152] = run_strmm,   /* FB_OP_STRMM  */
    [153] = run_dtrmm,   /* FB_OP_DTRMM  */
    [154] = NULL, [155] = NULL,
    [156] = run_strsm,   /* FB_OP_STRSM  */
    [157] = run_dtrsm,   /* FB_OP_DTRSM  */
    [158] = NULL, [159] = NULL,
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

    if (op_id >= FB_DIRECT_DISPATCH_SIZE || fb_direct_dispatch[op_id] == NULL)
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_direct_dispatch[op_id](
        oracle_vtable, candidate_vtable, tc, result_out, elapsed_ns_out);
}
