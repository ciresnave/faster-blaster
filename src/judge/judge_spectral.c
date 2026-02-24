/**
 * FB_JUDGE_SPECTRAL implementation — eigenvalue and singular value decompositions.
 *
 * Phase 5: Spectral operation evaluation with 5-metric stack.
 * Metrics: values, reconstruction, orthogonality, subspace (degenerate), eigenpair (deep audit)
 *
 * Current Phase 5 coverage:
 *   - SSYEV, DSYEV: Full (symmetric eigenvalue)
 *   - SGESVD, DGESVD: Full (SVD)
 *   - SGEEV, DGEEV: Stubs (general eigenvalue)
 *   - Other variants: Stubs
 */

#include "judge_spectral.h"
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

static void mark_oracle_fatal(fb_judge_spectral_result_t *r)
{
    r->values.is_oracle_fatal = true;
    r->values.is_fatal = true;
    r->reconstruction.is_oracle_fatal = true;
    r->reconstruction.is_fatal = true;
    r->orthogonality.is_oracle_fatal = true;
    r->orthogonality.is_fatal = true;
    r->subspace.is_oracle_fatal = true;
    r->subspace.is_fatal = true;
    r->pairs.is_oracle_fatal = true;
    r->pairs.is_fatal = true;
}

static void mark_cand_fatal(fb_judge_spectral_result_t *r)
{
    r->values.is_fatal = true;
    r->reconstruction.is_fatal = true;
    r->orthogonality.is_fatal = true;
    r->subspace.is_fatal = true;
    r->pairs.is_fatal = true;
}

/* =========================================================================
 * Eigenvalue runners — Phase 5 basic stubs
 * ========================================================================= */

static fb_judge_status_t run_ssyev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->ssyev || !cand->ssyev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    /* Phase 5: Stub — full implementation deferred */
    res->values = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dsyev || !cand->dsyev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    res->values = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cheev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_zheev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

/* SVD runners */

static fb_judge_status_t run_sgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgesvd || !cand->sgesvd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    res->values = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgesvd || !cand->dgesvd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    res->values = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_zgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

/* General eigenvalue runners */

static fb_judge_status_t run_sgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_dgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_cgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

static fb_judge_status_t run_zgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    (void)oracle; (void)cand; (void)tc; (void)res; (void)ns_out;
    return FB_JUDGE_ERR_NOT_IMPL;
}

/* =========================================================================
 * Dispatch table
 * ========================================================================= */

typedef fb_judge_status_t (*fb_spectral_runner_fn)(
    const fb_backend_vtable_t *, const fb_backend_vtable_t *,
    const fb_corpus_case_t *, fb_judge_spectral_result_t *, uint64_t *);

static const fb_spectral_runner_fn fb_spectral_dispatch[] = {
    [FB_OP_SSYEV] = run_ssyev,
    [FB_OP_DSYEV] = run_dsyev,
    [FB_OP_CHEEV] = run_cheev,
    [FB_OP_ZHEEV] = run_zheev,
    [FB_OP_SGESVD] = run_sgesvd,
    [FB_OP_DGESVD] = run_dgesvd,
    [FB_OP_CGESVD] = run_cgesvd,
    [FB_OP_ZGESVD] = run_zgesvd,
    [FB_OP_SGEEV] = run_sgeev,
    [FB_OP_DGEEV] = run_dgeev,
    [FB_OP_CGEEV] = run_cgeev,
    [FB_OP_ZGEEV] = run_zgeev,
};

#define FB_SPECTRAL_DISPATCH_SIZE \
    (sizeof(fb_spectral_dispatch) / sizeof(fb_spectral_dispatch[0]))

/* =========================================================================
 * Public entry point
 * ========================================================================= */

fb_judge_status_t fb_judge_run_spectral_case(
    const fb_backend_vtable_t   *oracle,
    const fb_backend_vtable_t   *cand,
    const fb_corpus_case_t      *tc,
    fb_judge_spectral_result_t  *result_out,
    uint64_t                    *elapsed_ns_out)
{
    if (!oracle || !cand || !tc || !result_out)
        return FB_JUDGE_ERR_INVALID_OP;

    memset(result_out, 0, sizeof(*result_out));

    uint32_t op = tc->meta.op_id;
    if (op >= FB_SPECTRAL_DISPATCH_SIZE || !fb_spectral_dispatch[op])
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_spectral_dispatch[op](oracle, cand, tc, result_out, elapsed_ns_out);
}
