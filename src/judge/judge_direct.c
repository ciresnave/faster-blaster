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
    /* 2  */ NULL,        /* FB_OP_CAXPY  — Phase 2 */
    /* 3  */ NULL,        /* FB_OP_ZAXPY  — Phase 2 */
    /* 4  */ run_sscal,   /* FB_OP_SSCAL  */
    /* 5  */ run_dscal,   /* FB_OP_DSCAL  */
    /* 6  */ NULL,        /* FB_OP_CSCAL  — Phase 2 */
    /* 7  */ NULL,        /* FB_OP_ZSCAL  — Phase 2 */
    /* 8  */ NULL,        /* FB_OP_CSSCAL — Phase 2 */
    /* 9  */ NULL,        /* FB_OP_ZDSCAL — Phase 2 */
    /* 10 */ run_scopy,   /* FB_OP_SCOPY  */
    /* 11 */ run_dcopy,   /* FB_OP_DCOPY  */
    /* 12 */ NULL,        /* FB_OP_CCOPY  — Phase 2 */
    /* 13 */ NULL,        /* FB_OP_ZCOPY  — Phase 2 */
    /* 14 */ run_sswap,   /* FB_OP_SSWAP  */
    /* 15 */ run_dswap,   /* FB_OP_DSWAP  */
    /* 16 */ NULL,        /* FB_OP_CSWAP  — Phase 2 */
    /* 17 */ NULL,        /* FB_OP_ZSWAP  — Phase 2 */
    /* 18 */ run_sdot,    /* FB_OP_SDOT   */
    /* 19 */ run_ddot,    /* FB_OP_DDOT   */
    /* 20 */ NULL,        /* FB_OP_CDOTC  — Phase 2 */
    /* 21 */ NULL,        /* FB_OP_CDOTU  — Phase 2 */
    /* 22 */ NULL,        /* FB_OP_ZDOTC  — Phase 2 */
    /* 23 */ NULL,        /* FB_OP_ZDOTU  — Phase 2 */
    /* 24 */ NULL,        /* FB_OP_SDSDOT — Phase 2 */
    /* 25 */ NULL,        /* FB_OP_DSDOT  — Phase 2 */
    /* 26 */ run_sasum,   /* FB_OP_SASUM  */
    /* 27 */ run_dasum,   /* FB_OP_DASUM  */
    /* 28 */ NULL,        /* FB_OP_SCASUM — Phase 2 */
    /* 29 */ NULL,        /* FB_OP_DZASUM — Phase 2 */
    /* 30 */ run_snrm2,   /* FB_OP_SNRM2  */
    /* 31 */ run_dnrm2,   /* FB_OP_DNRM2  */
    /* 32 */ NULL,        /* FB_OP_SCNRM2 — Phase 2 */
    /* 33 */ NULL,        /* FB_OP_DZNRM2 — Phase 2 */
    /* 34 */ run_isamax,  /* FB_OP_ISAMAX */
    /* 35 */ run_idamax,  /* FB_OP_IDAMAX */
    /* 36 */ NULL,        /* FB_OP_ICAMAX — Phase 2 */
    /* 37 */ NULL,        /* FB_OP_IZAMAX — Phase 2 */
    /* 38 */ NULL,        /* FB_OP_SROT   — Phase 2 */
    /* 39 */ NULL,        /* FB_OP_DROT   — Phase 2 */
    /* 40 */ NULL,        /* FB_OP_CROT   — Phase 2 */
    /* 41 */ NULL,        /* FB_OP_ZROT   — Phase 2 */
    /* 42 */ NULL,        /* FB_OP_ZDROT  — Phase 2 */
    /* 43 */ NULL,        /* FB_OP_SROTG  — Phase 2 */
    /* 44 */ NULL,        /* FB_OP_DROTG  — Phase 2 */
    /* 45 */ NULL,        /* FB_OP_SROTM  — Phase 2 */
    /* 46 */ NULL,        /* FB_OP_DROTM  — Phase 2 */
    /* 47 */ NULL,        /* FB_OP_SROTMG — Phase 2 */
    /* 48 */ run_sgemv,   /* FB_OP_SGEMV  */
    /* 49 */ run_dgemv,   /* FB_OP_DGEMV  */
    /* 50 */ NULL,        /* FB_OP_CGEMV  — Phase 2 */
    /* 51 */ NULL,        /* FB_OP_ZGEMV  — Phase 2 */
    /* 52..129 : L2 variants not yet implemented */
    [52]  = NULL, [53]  = NULL, [54]  = NULL, [55]  = NULL,
    [56]  = NULL, [57]  = NULL, [58]  = NULL, [59]  = NULL,
    [60]  = NULL, [61]  = NULL, [62]  = NULL, [63]  = NULL,
    [64]  = NULL, [65]  = NULL, [66]  = NULL, [67]  = NULL,
    [68]  = NULL, [69]  = NULL, [70]  = NULL, [71]  = NULL,
    [72]  = NULL, [73]  = NULL, [74]  = NULL, [75]  = NULL,
    [76]  = NULL, [77]  = NULL, [78]  = NULL, [79]  = NULL,
    [80]  = NULL, [81]  = NULL, [82]  = NULL, [83]  = NULL,
    [84]  = NULL, [85]  = NULL, [86]  = NULL, [87]  = NULL,
    [88]  = NULL, [89]  = NULL, [90]  = NULL, [91]  = NULL,
    [92]  = NULL, [93]  = NULL, [94]  = NULL, [95]  = NULL,
    [96]  = NULL, [97]  = NULL, [98]  = NULL, [99]  = NULL,
    [100] = NULL, [101] = NULL, [102] = NULL, [103] = NULL,
    [104] = NULL, [105] = NULL, [106] = NULL, [107] = NULL,
    [108] = NULL, [109] = NULL, [110] = NULL, [111] = NULL,
    [112] = NULL, [113] = NULL, [114] = NULL, [115] = NULL,
    [116] = NULL, [117] = NULL, [118] = NULL, [119] = NULL,
    [120] = NULL, [121] = NULL, [122] = NULL, [123] = NULL,
    [124] = NULL, [125] = NULL, [126] = NULL, [127] = NULL,
    [128] = NULL, [129] = NULL,
    /* L3 */
    [130] = run_sgemm,   /* FB_OP_SGEMM */
    [131] = run_dgemm,   /* FB_OP_DGEMM */
    [132] = NULL,        /* FB_OP_CGEMM — Phase 2 */
    [133] = NULL,        /* FB_OP_ZGEMM — Phase 2 */
    /* 134..159: other L3 ops — Phase 2 */
    [134] = NULL, [135] = NULL, [136] = NULL, [137] = NULL,
    [138] = NULL, [139] = NULL, [140] = NULL, [141] = NULL,
    [142] = NULL, [143] = NULL, [144] = NULL, [145] = NULL,
    [146] = NULL, [147] = NULL, [148] = NULL, [149] = NULL,
    [150] = NULL, [151] = NULL, [152] = NULL, [153] = NULL,
    [154] = NULL, [155] = NULL, [156] = NULL, [157] = NULL,
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
