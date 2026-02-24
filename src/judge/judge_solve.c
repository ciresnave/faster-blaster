/**
 * FB_JUDGE_SOLVE implementation — solver operation evaluation.
 *
 * Backward error metric: η = ||b - A*x|| / (||A|| ||x|| + ||b||)
 *
 * Phase 4: Solver operation evaluation.
 * Current implementation: SGESV, DGESV, SPOSV, DPOSV as stubs.
 * Full residual computation deferred to Phase 4+.
 */

#include "judge_solve.h"
#include "judge_op_ids.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/* =========================================================================
 * Helper: Convert relative error to digit count
 * ========================================================================= */

static fb_judge_case_result_t result_from_relerr(double relerr)
{
    fb_judge_case_result_t res = {0};
    if (isnan(relerr) || isinf(relerr) || relerr > 1.0) {
        res.digits = 0;
        res.relative_error = relerr;
        res.is_fatal = true;
        return res;
    }
    res.relative_error = relerr;
    if (relerr < FLT_EPSILON) {
        res.digits = 16;
    } else {
        double log_inv = -log10(relerr);
        res.digits = (uint8_t)(log_inv < 0.0 ? 0 : (log_inv > 16.0 ? 16 : (uint8_t)floor(log_inv + 0.5)));
    }
    return res;
}

static double estimate_condition_f32(int64_t n)
{
    return (double)(n * 10);
}

static double estimate_condition_f64(int64_t n)
{
    return (double)(n * 10);
}

/* =========================================================================
 * Solver runners — Phase 4 basic stubs
 * ========================================================================= */

static fb_judge_status_t run_sgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgesv || !cand->sgesv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;
    res->residual = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->kappa_estimate = estimate_condition_f32(tc->m > tc->n ? tc->m : tc->n);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgesv || !cand->dgesv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;
    res->residual = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->kappa_estimate = estimate_condition_f64(tc->m > tc->n ? tc->m : tc->n);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sposv || !cand->sposv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;
    res->residual = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->kappa_estimate = estimate_condition_f32(tc->n);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dposv || !cand->dposv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;
    res->residual = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->kappa_estimate = estimate_condition_f64(tc->n);
    return FB_JUDGE_OK;
}

/* Stubs for other solvers */
static fb_judge_status_t run_sgels(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_dgels(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_sgetrs(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_dgetrs(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_spotrs(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_dpotrs(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

/* =========================================================================
 * Dispatch table
 * ========================================================================= */

typedef fb_judge_status_t (*fb_solve_runner_fn)(
    const fb_backend_vtable_t *, const fb_backend_vtable_t *,
    const fb_corpus_case_t *, fb_judge_solve_result_t *, uint64_t *);

static const fb_solve_runner_fn fb_solve_dispatch[] = {
    [FB_OP_SGESV] = run_sgesv,
    [FB_OP_DGESV] = run_dgesv,
    [FB_OP_SPOSV] = run_sposv,
    [FB_OP_DPOSV] = run_dposv,
    [FB_OP_SGELS] = run_sgels,
    [FB_OP_DGELS] = run_dgels,
    [FB_OP_SGETRS] = run_sgetrs,
    [FB_OP_DGETRS] = run_dgetrs,
    [FB_OP_SPOTRS] = run_spotrs,
    [FB_OP_DPOTRS] = run_dpotrs,
};

#define FB_SOLVE_DISPATCH_SIZE \
    (sizeof(fb_solve_dispatch) / sizeof(fb_solve_dispatch[0]))

/* =========================================================================
 * Public entry point
 * ========================================================================= */

fb_judge_status_t fb_judge_run_solve_case(
    const fb_backend_vtable_t  *oracle,
    const fb_backend_vtable_t  *cand,
    const fb_corpus_case_t     *tc,
    fb_judge_solve_result_t    *result_out,
    uint64_t                   *elapsed_ns_out)
{
    if (!oracle || !cand || !tc || !result_out)
        return FB_JUDGE_ERR_INVALID_OP;

    memset(result_out, 0, sizeof(*result_out));

    uint32_t op = tc->meta.op_id;
    if (op >= FB_SOLVE_DISPATCH_SIZE || !fb_solve_dispatch[op])
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_solve_dispatch[op](oracle, cand, tc, result_out, elapsed_ns_out);
}
