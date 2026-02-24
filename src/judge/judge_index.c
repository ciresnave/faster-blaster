/**
 * @file judge_index.c
 * @brief FB_JUDGE_INDEX archetype evaluator — discrete argmax/argmin ops.
 *
 * Covers: ISAMAX, IDAMAX, ICAMAX, IZAMAX.
 *
 * Two sub-results per test case:
 *
 *   result->index   Discrete position quality (pass/fail with tie credit).
 *                   digits == 16        → exact index match.
 *                   0 < digits < 16     → tie-correct (value credit applied).
 *                   digits == 0         → wrong (candidate missed the max).
 *
 *   result->value   Magnitude agreement at the oracle's index.
 *                   Always computed; measures how closely |x[cand_idx]|
 *                   matches |x[oracle_idx]|. A wrong-index candidate with a
 *                   close magnitude may still score well here, indicating a
 *                   degenerate (tie) input rather than an implementation error.
 *
 * Tie detection threshold:
 *   float :  4 × FLT_EPSILON
 *   double:  4 × DBL_EPSILON
 *   complex: same thresholds applied to the BLAS sum-of-absolutes magnitude.
 *
 * Complex magnitude convention (ICAMAX / IZAMAX):
 *   |z|₁ = |Re(z)| + |Im(z)|
 * This matches the BLAS reference norm used to select the maximum index.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_index.h"
#include "judge_op_ids.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* =========================================================================
 * Internal constants
 * ========================================================================= */

/** Warmup calls before the timed window (same policy as judge_direct.c). */
#define FB_INDEX_WARMUP_RUNS   2
/** Timed calls; best (minimum) is recorded. */
#define FB_INDEX_TIMING_RUNS   5

/* =========================================================================
 * Private helpers
 * ========================================================================= */

/**
 * BLAS sum-of-absolutes magnitude for complex float.
 * Matches reference ICAMAX selection criterion: |Re(z)| + |Im(z)|.
 */
static inline float cmag_f32(fb_complex_float_t z)
{
    return fabsf(crealf(z)) + fabsf(cimagf(z));
}

/**
 * BLAS sum-of-absolutes magnitude for complex double.
 * Matches reference IZAMAX selection criterion: |Re(z)| + |Im(z)|.
 */
static inline double cmag_f64(fb_complex_double_t z)
{
    return fabs(creal(z)) + fabs(cimag(z));
}

/**
 * Build the value sub-result from absolute oracle and candidate magnitudes.
 *
 * @param abs_oracle  Magnitude of x[oracle_idx] (the true maximum).
 * @param abs_cand    Magnitude of x[cand_idx]   (what candidate found).
 * @param tau         Zero-guard: typically eps_type × abs_oracle.
 * @param res         Output case result.
 */
static void make_value_result(double abs_oracle, double abs_cand,
                              double tau, fb_judge_case_result_t *res)
{
    double denom = (abs_oracle > tau) ? abs_oracle : tau;
    double rel   = fabs(abs_oracle - abs_cand) / denom;

    if (rel == 0.0) {
        res->digits         = 16.0;
        res->relative_error = 0.0;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    } else {
        double d = -log10(rel);
        res->digits         = (d < 0.0) ? 0.0 : (d > 16.0 ? 16.0 : d);
        res->relative_error = rel;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    }
}

/**
 * Build the index sub-result.
 *
 *   exact match   → digits = 16
 *   tie-correct   → digits = value->digits (partial credit)
 *   clearly wrong → digits = 0
 *
 * @param match       True if oracle_idx == cand_idx.
 * @param is_tie      True if magnitudes are within tie_eps (even if indices differ).
 * @param value       Pre-computed value sub-result (used for tie credit).
 * @param res         Output index case result.
 */
static void make_index_result(bool match, bool is_tie,
                              const fb_judge_case_result_t *value,
                              fb_judge_case_result_t *res)
{
    if (match) {
        res->digits         = 16.0;
        res->relative_error = 0.0;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    } else if (is_tie) {
        /* Candidate found a different-but-equivalent element. Credit proportional
         * to how closely its magnitude matches the oracle's maximum. */
        res->digits         = value->digits;
        res->relative_error = value->relative_error;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    } else {
        /* Candidate clearly returned the wrong maximum. */
        res->digits         = 0.0;
        res->relative_error = 1.0;
        res->is_fatal       = false;
        res->is_oracle_fatal = false;
    }
}

/** Mark oracle-fatal result (both sub-results). */
static void make_oracle_fatal(fb_judge_index_result_t *r)
{
    r->index.digits  = 0.0; r->index.relative_error  = 1.0;
    r->value.digits  = 0.0; r->value.relative_error  = 1.0;
    r->index.is_oracle_fatal = true;
    r->value.is_oracle_fatal = true;
    r->index.is_fatal = false;
    r->value.is_fatal = false;
}

/** Mark candidate-fatal result (both sub-results). */
static void make_cand_fatal(fb_judge_index_result_t *r)
{
    r->index.digits  = 0.0; r->index.relative_error  = 1.0;
    r->value.digits  = 0.0; r->value.relative_error  = 1.0;
    r->index.is_oracle_fatal = false;
    r->value.is_oracle_fatal = false;
    r->index.is_fatal = true;
    r->value.is_fatal = true;
}

/* =========================================================================
 * Runner type
 * ========================================================================= */

typedef fb_judge_status_t (*fb_index_runner_fn)(
    const fb_backend_vtable_t *oracle,
    const fb_backend_vtable_t *cand,
    const fb_corpus_case_t    *tc,
    fb_judge_index_result_t   *result,
    uint64_t                  *ns_out);

/* =========================================================================
 * ISAMAX — single-precision float
 * ========================================================================= */

static fb_judge_status_t run_isamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_index_result_t *res, uint64_t *ns_out)
{
    if (!oracle->isamax || !cand->isamax) return FB_JUDGE_ERR_NOT_IMPL;

    int64_t      n = (int64_t)tc->n;
    const float *x = (const float *)tc->A;

    if (n <= 0 || !x) { make_oracle_fatal(res); return FB_JUDGE_OK; }

    int64_t oracle_idx = oracle->isamax(n, x, 1);
    int64_t cand_idx   = cand->isamax(n, x, 1);

    /* Oracle validity check. */
    if (oracle_idx < 0 || oracle_idx >= n) {
        make_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double abs_oracle = (double)fabsf(x[oracle_idx]);
    double tau        = (double)FLT_EPSILON * abs_oracle;

    /* Candidate out-of-bounds → treat as fatal for the candidate. */
    if (cand_idx < 0 || cand_idx >= n) {
        make_cand_fatal(res); return FB_JUDGE_OK;
    }

    double abs_cand = (double)fabsf(x[cand_idx]);

    /* Value sub-result (always computed). */
    make_value_result(abs_oracle, abs_cand, tau, &res->value);

    /* Tie check: |oracle_max - cand_val| / oracle_max < 4 × eps. */
    bool is_tie = (fabs(abs_oracle - abs_cand) / fmax(abs_oracle, (double)FLT_MIN))
                  <= (double)(FLT_EPSILON * 4.0f);
    make_index_result(cand_idx == oracle_idx, is_tie, &res->value, &res->index);

    /* Timing: best-of-N on candidate only. */
    if (ns_out) {
        for (int w = 0; w < FB_INDEX_WARMUP_RUNS; w++) (void)cand->isamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_INDEX_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->isamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * IDAMAX — double-precision float
 * ========================================================================= */

static fb_judge_status_t run_idamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_index_result_t *res, uint64_t *ns_out)
{
    if (!oracle->idamax || !cand->idamax) return FB_JUDGE_ERR_NOT_IMPL;

    int64_t       n = (int64_t)tc->n;
    const double *x = (const double *)tc->A;

    if (n <= 0 || !x) { make_oracle_fatal(res); return FB_JUDGE_OK; }

    int64_t oracle_idx = oracle->idamax(n, x, 1);
    int64_t cand_idx   = cand->idamax(n, x, 1);

    if (oracle_idx < 0 || oracle_idx >= n) { make_oracle_fatal(res); return FB_JUDGE_OK; }

    if (cand_idx < 0 || cand_idx >= n) { make_cand_fatal(res); return FB_JUDGE_OK; }

    double abs_oracle = fabs(x[oracle_idx]);
    double abs_cand   = fabs(x[cand_idx]);
    double tau        = DBL_EPSILON * abs_oracle;

    make_value_result(abs_oracle, abs_cand, tau, &res->value);

    bool is_tie = (fabs(abs_oracle - abs_cand) / fmax(abs_oracle, DBL_MIN))
                  <= (DBL_EPSILON * 4.0);
    make_index_result(cand_idx == oracle_idx, is_tie, &res->value, &res->index);

    if (ns_out) {
        for (int w = 0; w < FB_INDEX_WARMUP_RUNS; w++) (void)cand->idamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_INDEX_TIMING_RUNS; t++) {
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
 * ICAMAX — complex single-precision (magnitude = |Re| + |Im|)
 * ========================================================================= */

static fb_judge_status_t run_icamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_index_result_t *res, uint64_t *ns_out)
{
    if (!oracle->icamax || !cand->icamax) return FB_JUDGE_ERR_NOT_IMPL;

    int64_t                    n = (int64_t)tc->n;
    const fb_complex_float_t  *x = (const fb_complex_float_t *)tc->A;

    if (n <= 0 || !x) { make_oracle_fatal(res); return FB_JUDGE_OK; }

    int64_t oracle_idx = oracle->icamax(n, x, 1);
    int64_t cand_idx   = cand->icamax(n, x, 1);

    if (oracle_idx < 0 || oracle_idx >= n) { make_oracle_fatal(res); return FB_JUDGE_OK; }
    if (cand_idx   < 0 || cand_idx   >= n) { make_cand_fatal(res);   return FB_JUDGE_OK; }

    double abs_oracle = (double)cmag_f32(x[oracle_idx]);
    double abs_cand   = (double)cmag_f32(x[cand_idx]);
    double tau        = (double)FLT_EPSILON * abs_oracle;

    make_value_result(abs_oracle, abs_cand, tau, &res->value);

    bool is_tie = (fabs(abs_oracle - abs_cand) / fmax(abs_oracle, (double)FLT_MIN))
                  <= (double)(FLT_EPSILON * 4.0f);
    make_index_result(cand_idx == oracle_idx, is_tie, &res->value, &res->index);

    if (ns_out) {
        for (int w = 0; w < FB_INDEX_WARMUP_RUNS; w++) (void)cand->icamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_INDEX_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->icamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * IZAMAX — complex double-precision (magnitude = |Re| + |Im|)
 * ========================================================================= */

static fb_judge_status_t run_izamax(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_index_result_t *res, uint64_t *ns_out)
{
    if (!oracle->izamax || !cand->izamax) return FB_JUDGE_ERR_NOT_IMPL;

    int64_t                    n = (int64_t)tc->n;
    const fb_complex_double_t *x = (const fb_complex_double_t *)tc->A;

    if (n <= 0 || !x) { make_oracle_fatal(res); return FB_JUDGE_OK; }

    int64_t oracle_idx = oracle->izamax(n, x, 1);
    int64_t cand_idx   = cand->izamax(n, x, 1);

    if (oracle_idx < 0 || oracle_idx >= n) { make_oracle_fatal(res); return FB_JUDGE_OK; }
    if (cand_idx   < 0 || cand_idx   >= n) { make_cand_fatal(res);   return FB_JUDGE_OK; }

    double abs_oracle = cmag_f64(x[oracle_idx]);
    double abs_cand   = cmag_f64(x[cand_idx]);
    double tau        = DBL_EPSILON * abs_oracle;

    make_value_result(abs_oracle, abs_cand, tau, &res->value);

    bool is_tie = (fabs(abs_oracle - abs_cand) / fmax(abs_oracle, DBL_MIN))
                  <= (DBL_EPSILON * 4.0);
    make_index_result(cand_idx == oracle_idx, is_tie, &res->value, &res->index);

    if (ns_out) {
        for (int w = 0; w < FB_INDEX_WARMUP_RUNS; w++) (void)cand->izamax(n, x, 1);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_INDEX_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->izamax(n, x, 1);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Dispatch
 * ========================================================================= */

/**
 * Dispatch table for INDEX archetype operations.
 * Indexed by op_id; only the 4 AMAX ops are supported.
 */
static const fb_index_runner_fn fb_index_dispatch[] = {
    [FB_OP_ISAMAX] = run_isamax,
    [FB_OP_IDAMAX] = run_idamax,
    [FB_OP_ICAMAX] = run_icamax,
    [FB_OP_IZAMAX] = run_izamax,
};

#define FB_INDEX_DISPATCH_SIZE \
    (sizeof(fb_index_dispatch) / sizeof(fb_index_dispatch[0]))

/* =========================================================================
 * Public entry point
 * ========================================================================= */

fb_judge_status_t fb_judge_run_index_case(
    const fb_backend_vtable_t  *oracle,
    const fb_backend_vtable_t  *cand,
    const fb_corpus_case_t     *tc,
    fb_judge_index_result_t    *result_out,
    uint64_t                   *elapsed_ns_out)
{
    if (!oracle || !cand || !tc || !result_out)
        return FB_JUDGE_ERR_INVALID_OP;

    memset(result_out, 0, sizeof(*result_out));

    uint32_t op = tc->meta.op_id;
    if (op >= FB_INDEX_DISPATCH_SIZE || !fb_index_dispatch[op])
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_index_dispatch[op](oracle, cand, tc, result_out, elapsed_ns_out);
}
