/**
 * @file judge_solve.c
 * @brief FB_JUDGE_SOLVE archetype — real backward-error residual for all 10 runners.
 *
 * Metric: η = ‖b − A·x‖_F / (‖A‖_F · ‖x‖_F + ‖b‖_F)
 *
 * Coverage (Phase 4+):
 *   GESV:  SGESV,  DGESV  — general square driver
 *   POSV:  SPOSV,  DPOSV  — SPD driver (internally Cholesky)
 *   GELS:  SGELS,  DGELS  — overdetermined/underdetermined QR/LQ driver
 *   GETRS: SGETRS, DGETRS — triangular back-sub on oracle-factored LU
 *   POTRS: SPOTRS, DPOTRS — triangular back-sub on oracle-factored Cholesky
 */

#include "judge_solve.h"
#include "judge_op_ids.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#define FB_SOLVE_WARMUP_RUNS  2
#define FB_SOLVE_TIMING_RUNS  5

/* =========================================================================
 * Private helpers
 * ========================================================================= */

/** Convert relative error to digit count. */
static fb_judge_case_result_t make_result(double relerr)
{
    fb_judge_case_result_t r = {0};
    if (isnan(relerr) || isinf(relerr) || relerr > 1.0) {
        r.digits = 0; r.relative_error = relerr; r.is_fatal = true; return r;
    }
    r.relative_error = relerr;
    if (relerr < FLT_EPSILON) {
        r.digits = 16;
    } else {
        double d = -log10(relerr);
        r.digits = (uint8_t)(d < 0.0 ? 0 : (d > 16.0 ? 16 : (uint8_t)(d + 0.5)));
    }
    return r;
}

/**
 * Backward error η = ‖b − A·x‖_F / (‖A‖_F · ‖x‖_F + ‖b‖_F).
 * A: m×n row-major lda; b: m×nrhs ldb; x: n×nrhs ldx.
 */
static double bwerr_f32(const float  *A, int64_t m, int64_t n, int64_t lda,
                         const float  *b, int64_t ldb,
                         const float  *x, int64_t ldx, int64_t nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int64_t j = 0; j < nrhs; j++) {
        for (int64_t i = 0; i < m; i++) {
            double rij = (double)b[i*ldb + j];
            for (int64_t k = 0; k < n; k++)
                rij -= (double)A[i*lda + k] * (double)x[k*ldx + j];
            nr2 += rij * rij;
            nb2 += (double)b[i*ldb+j] * (double)b[i*ldb+j];
        }
        for (int64_t i = 0; i < n; i++) { double v = (double)x[i*ldx+j]; nx2 += v*v; }
    }
    for (int64_t i = 0; i < m; i++)
        for (int64_t k = 0; k < n; k++) { double v = (double)A[i*lda+k]; nA2 += v*v; }
    double denom = sqrt(nA2)*sqrt(nx2) + sqrt(nb2);
    if (denom < (double)FLT_EPSILON)
        return (nr2 < (double)FLT_EPSILON*(double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(nr2) / denom;
}

static double bwerr_f64(const double *A, int64_t m, int64_t n, int64_t lda,
                         const double *b, int64_t ldb,
                         const double *x, int64_t ldx, int64_t nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int64_t j = 0; j < nrhs; j++) {
        for (int64_t i = 0; i < m; i++) {
            double rij = b[i*ldb+j];
            for (int64_t k = 0; k < n; k++)
                rij -= A[i*lda+k] * x[k*ldx+j];
            nr2 += rij*rij; nb2 += b[i*ldb+j]*b[i*ldb+j];
        }
        for (int64_t i = 0; i < n; i++) { double v = x[i*ldx+j]; nx2 += v*v; }
    }
    for (int64_t i = 0; i < m; i++)
        for (int64_t k = 0; k < n; k++) { double v = A[i*lda+k]; nA2 += v*v; }
    double denom = sqrt(nA2)*sqrt(nx2) + sqrt(nb2);
    if (denom < DBL_EPSILON) return (nr2 < DBL_EPSILON*DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(nr2) / denom;
}

static void mark_oc_fatal(fb_judge_solve_result_t *r)
    { memset(r, 0, sizeof(*r)); r->residual.is_oracle_fatal = true; }
static void mark_ca_fatal(fb_judge_solve_result_t *r)
    { memset(r, 0, sizeof(*r)); r->residual.is_fatal = true; }

/* =========================================================================
 * SGESV — A·X = B  (general square F32 driver)
 * ========================================================================= */
static fb_judge_status_t run_sgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgesv || !cand->sgesv) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Ao = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    int64_t *piv = (int64_t*)malloc((size_t)n*sizeof(int64_t));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    float *Ac = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    int64_t *pivc = (int64_t*)malloc((size_t)n*sizeof(int64_t));
    if (!Ac || !Bc || !pivc) { free(Ac); free(Bc); free(pivc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->sgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->sgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(pivc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(pivc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGESV — A·X = B  (general square F64 driver)
 * ========================================================================= */
static fb_judge_status_t run_dgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgesv || !cand->dgesv) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Ao = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    int64_t *piv = (int64_t*)malloc((size_t)n*sizeof(int64_t));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    double *Ac = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    int64_t *pivc = (int64_t*)malloc((size_t)n*sizeof(int64_t));
    if (!Ac || !Bc || !pivc) { free(Ac); free(Bc); free(pivc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(pivc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(pivc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SPOSV / DPOSV — SPD driver
 * ========================================================================= */
static fb_judge_status_t run_sposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sposv || !cand->sposv) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Ao = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    float *Ac = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->sposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->sposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dposv || !cand->dposv) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Ao = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    double *Ac = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGELS / DGELS — QR/LQ least-squares driver (overdetermined or underdetermined)
 * ========================================================================= */
static fb_judge_status_t run_sgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgels || !cand->sgels) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Ao = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    float *Ac = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->sgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->sgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    /* For overdetermined (m>=n): solution in Bc[0:n,:]; compare ‖b−Ax‖ over m rows. */
    res->residual = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)((m > n ? m : n) * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgels || !cand->dgels) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t m = (int64_t)tc->m, n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Ao = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    double *Ac = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)((m > n ? m : n) * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGETRS / DGETRS — triangular back-substitution on oracle-factored LU.
 *
 * Strategy: use oracle->sgetrf to produce (LU, ipiv); give same factored form
 * to both oracle->sgetrs and cand->sgetrs so GETRS is tested in isolation.
 * ========================================================================= */
static fb_judge_status_t run_sgetrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgetrs || !cand->sgetrs || !oracle->sgetrf) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    /* Produce reference LU factorization with oracle->sgetrf. */
    float *LU = (float*)malloc(Asz);
    int64_t *ipiv = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!LU || !ipiv) { free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LU, A0, Asz);
    if (oracle->sgetrf(FB_LAYOUT_ROW_MAJOR, n, n, LU, lda, ipiv) != 0) {
        free(LU); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }

    /* Oracle GETRS (verify oracle path). */
    float *LUo = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    if (!LUo || !Bo) { free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LUo, LU, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUo, lda, ipiv, Bo, ldb) != 0) {
        free(LU); free(ipiv); free(LUo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(LUo);

    /* Candidate GETRS. */
    float *LUc = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    if (!LUc || !Bc) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->sgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
    if (cand->sgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb) != 0) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(LU); free(ipiv); free(LUc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgetrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgetrs || !cand->dgetrs || !oracle->dgetrf) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *LU = (double*)malloc(Asz);
    int64_t *ipiv = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!LU || !ipiv) { free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LU, A0, Asz);
    if (oracle->dgetrf(FB_LAYOUT_ROW_MAJOR, n, n, LU, lda, ipiv) != 0) {
        free(LU); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    double *LUo = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    if (!LUo || !Bo) { free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LUo, LU, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUo, lda, ipiv, Bo, ldb) != 0) {
        free(LU); free(ipiv); free(LUo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(LUo);
    double *LUc = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    if (!LUc || !Bc) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb) != 0) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(LU); free(ipiv); free(LUc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SPOTRS / DPOTRS — back-substitution on oracle-factored Cholesky.
 *
 * Strategy: oracle->spotrf produces the Cholesky factor L (or U); same
 * factored matrix given to both oracle and candidate POTRS.
 * ========================================================================= */
static fb_judge_status_t run_spotrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->spotrs || !cand->spotrs || !oracle->spotrf) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Lf = (float*)malloc(Asz);
    if (!Lf) return FB_JUDGE_ERR_ALLOC;
    memcpy(Lf, A0, Asz);
    if (oracle->spotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, Lf, lda) != 0) {
        free(Lf); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    float *Lfo = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    if (!Lfo || !Bo) { free(Lf); free(Lfo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Lfo, Lf, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->spotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfo, lda, Bo, ldb) != 0) {
        free(Lf); free(Lfo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Lfo);
    float *Lfc = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    if (!Lfc || !Bc) { free(Lf); free(Lfc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->spotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->spotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
    if (cand->spotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb) != 0) {
        free(Lf); free(Lfc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Lf); free(Lfc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dpotrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dpotrs || !cand->dpotrs || !oracle->dpotrf) return FB_JUDGE_ERR_NOT_IMPL;
    int64_t n = (int64_t)tc->n, nrhs = (int64_t)tc->k;
    int64_t lda = (int64_t)tc->lda, ldb = (int64_t)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Lf = (double*)malloc(Asz);
    if (!Lf) return FB_JUDGE_ERR_ALLOC;
    memcpy(Lf, A0, Asz);
    if (oracle->dpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, Lf, lda) != 0) {
        free(Lf); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    double *Lfo = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    if (!Lfo || !Bo) { free(Lf); free(Lfo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Lfo, Lf, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfo, lda, Bo, ldb) != 0) {
        free(Lf); free(Lfo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Lfo);
    double *Lfc = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    if (!Lfc || !Bc) { free(Lf); free(Lfc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb) != 0) {
        free(Lf); free(Lfc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Lf); free(Lfc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Dispatch table
 * ========================================================================= */

typedef fb_judge_status_t (*fb_solve_runner_fn)(
    const fb_backend_vtable_t *, const fb_backend_vtable_t *,
    const fb_corpus_case_t *, fb_judge_solve_result_t *, uint64_t *);

static const fb_solve_runner_fn fb_solve_dispatch[] = {
    [FB_OP_SGESV]  = run_sgesv,
    [FB_OP_DGESV]  = run_dgesv,
    [FB_OP_SPOSV]  = run_sposv,
    [FB_OP_DPOSV]  = run_dposv,
    [FB_OP_SGELS]  = run_sgels,
    [FB_OP_DGELS]  = run_dgels,
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
