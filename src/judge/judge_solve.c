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
static double bwerr_f32(const float  *A, int m, int n, int lda,
                         const float  *b, int ldb,
                         const float  *x, int ldx, int nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij = (double)b[i*ldb + j];
            for (int k = 0; k < n; k++)
                rij -= (double)A[i*lda + k] * (double)x[k*ldx + j];
            nr2 += rij * rij;
            nb2 += (double)b[i*ldb+j] * (double)b[i*ldb+j];
        }
        for (int i = 0; i < n; i++) { double v = (double)x[i*ldx+j]; nx2 += v*v; }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) { double v = (double)A[i*lda+k]; nA2 += v*v; }
    double denom = sqrt(nA2)*sqrt(nx2) + sqrt(nb2);
    if (denom < (double)FLT_EPSILON)
        return (nr2 < (double)FLT_EPSILON*(double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(nr2) / denom;
}

static double bwerr_f64(const double *A, int m, int n, int lda,
                         const double *b, int ldb,
                         const double *x, int ldx, int nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij = b[i*ldb+j];
            for (int k = 0; k < n; k++)
                rij -= A[i*lda+k] * x[k*ldx+j];
            nr2 += rij*rij; nb2 += b[i*ldb+j]*b[i*ldb+j];
        }
        for (int i = 0; i < n; i++) { double v = x[i*ldx+j]; nx2 += v*v; }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) { double v = A[i*lda+k]; nA2 += v*v; }
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Ao = (float*)malloc(Asz), *Bo = (float*)malloc(Bsz);
    int *piv = (int*)malloc((size_t)n*sizeof(int));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    float *Ac = (float*)malloc(Asz), *Bc = (float*)malloc(Bsz);
    int *pivc = (int*)malloc((size_t)n*sizeof(int));
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Ao = (double*)malloc(Asz), *Bo = (double*)malloc(Bsz);
    int *piv = (int*)malloc((size_t)n*sizeof(int));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    double *Ac = (double*)malloc(Asz), *Bc = (double*)malloc(Bsz);
    int *pivc = (int*)malloc((size_t)n*sizeof(int));
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    /* Produce reference LU factorization with oracle->sgetrf. */
    float *LU = (float*)malloc(Asz);
    int *ipiv = (int*)malloc((size_t)n * sizeof(int));
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *LU = (double*)malloc(Asz);
    int *ipiv = (int*)malloc((size_t)n * sizeof(int));
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
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
 * Complex backward-error helpers
 * η = ‖b − A·x‖_F / (‖A‖_F · ‖x‖_F + ‖b‖_F)  (complex arithmetic)
 * ========================================================================= */

static double bwerr_cf32(const fb_complex_float_t *A, int m, int n, int lda,
                          const fb_complex_float_t *b, int ldb,
                          const fb_complex_float_t *x, int ldx, int nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij_re = (double)__real__(b[i*ldb + j]);
            double rij_im = (double)__imag__(b[i*ldb + j]);
            for (int k = 0; k < n; k++) {
                double ar = (double)__real__(A[i*lda + k]), ai = (double)__imag__(A[i*lda + k]);
                double xr = (double)__real__(x[k*ldx + j]), xi = (double)__imag__(x[k*ldx + j]);
                rij_re -= ar*xr - ai*xi;
                rij_im -= ar*xi + ai*xr;
            }
            nr2 += rij_re*rij_re + rij_im*rij_im;
            nb2 += (double)__real__(b[i*ldb+j])*(double)__real__(b[i*ldb+j])
                 + (double)__imag__(b[i*ldb+j])*(double)__imag__(b[i*ldb+j]);
        }
        for (int i = 0; i < n; i++) {
            double xr = (double)__real__(x[i*ldx+j]), xi = (double)__imag__(x[i*ldx+j]);
            nx2 += xr*xr + xi*xi;
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            double ar = (double)__real__(A[i*lda+k]), ai = (double)__imag__(A[i*lda+k]);
            nA2 += ar*ar + ai*ai;
        }
    double denom = sqrt(nA2)*sqrt(nx2) + sqrt(nb2);
    if (denom < (double)FLT_EPSILON)
        return (nr2 < (double)FLT_EPSILON*(double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(nr2) / denom;
}

static double bwerr_cf64(const fb_complex_double_t *A, int m, int n, int lda,
                          const fb_complex_double_t *b, int ldb,
                          const fb_complex_double_t *x, int ldx, int nrhs)
{
    double nr2 = 0.0, nA2 = 0.0, nx2 = 0.0, nb2 = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij_re = __real__(b[i*ldb + j]), rij_im = __imag__(b[i*ldb + j]);
            for (int k = 0; k < n; k++) {
                double ar = __real__(A[i*lda+k]), ai = __imag__(A[i*lda+k]);
                double xr = __real__(x[k*ldx+j]), xi = __imag__(x[k*ldx+j]);
                rij_re -= ar*xr - ai*xi;
                rij_im -= ar*xi + ai*xr;
            }
            nr2 += rij_re*rij_re + rij_im*rij_im;
            nb2 += __real__(b[i*ldb+j])*__real__(b[i*ldb+j]) + __imag__(b[i*ldb+j])*__imag__(b[i*ldb+j]);
        }
        for (int i = 0; i < n; i++) {
            nx2 += __real__(x[i*ldx+j])*__real__(x[i*ldx+j]) + __imag__(x[i*ldx+j])*__imag__(x[i*ldx+j]);
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            nA2 += __real__(A[i*lda+k])*__real__(A[i*lda+k]) + __imag__(A[i*lda+k])*__imag__(A[i*lda+k]);
    double denom = sqrt(nA2)*sqrt(nx2) + sqrt(nb2);
    if (denom < DBL_EPSILON) return (nr2 < DBL_EPSILON*DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(nr2) / denom;
}

/* =========================================================================
 * CGESV / ZGESV — complex general square driver
 * ========================================================================= */

static fb_judge_status_t run_cgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgesv || !cand->cgesv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    int *piv = (int *)malloc((size_t)n*sizeof(int));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    int *pivc = (int *)malloc((size_t)n*sizeof(int));
    if (!Ac || !Bc || !pivc) { free(Ac); free(Bc); free(pivc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(pivc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(pivc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgesv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgesv || !cand->zgesv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    int *piv = (int *)malloc((size_t)n*sizeof(int));
    if (!Ao || !Bo || !piv) { free(Ao); free(Bo); free(piv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ao, lda, piv, Bo, ldb) != 0) {
        free(Ao); free(Bo); free(piv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(piv);
    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    int *pivc = (int *)malloc((size_t)n*sizeof(int));
    if (!Ac || !Bc || !pivc) { free(Ac); free(Bc); free(pivc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zgesv(FB_LAYOUT_ROW_MAJOR, n, nrhs, Ac, lda, pivc, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(pivc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(pivc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CPOSV / ZPOSV — complex Hermitian positive-definite driver
 * ========================================================================= */

static fb_judge_status_t run_cposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cposv || !cand->cposv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zposv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zposv || !cand->zposv) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zposv(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGELS / ZGELS — complex least-squares driver
 * ========================================================================= */

static fb_judge_status_t run_cgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgels || !cand->cgels) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgels || !cand->zgels) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo) { free(Ao); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ao, lda, Bo, ldb) != 0) {
        free(Ao); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao);
    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zgels(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, nrhs, Ac, lda, Bc, ldb) != 0) {
        free(Ac); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGETRS / ZGETRS — back-substitution on oracle-factored complex LU
 * ========================================================================= */

static fb_judge_status_t run_cgetrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgetrs || !cand->cgetrs || !oracle->cgetrf) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *LU = (fb_complex_float_t *)malloc(Asz);
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!LU || !ipiv) { free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LU, A0, Asz);
    if (oracle->cgetrf(FB_LAYOUT_ROW_MAJOR, n, n, LU, lda, ipiv) != 0) {
        free(LU); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *LUo = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo  = (fb_complex_float_t *)malloc(Bsz);
    if (!LUo || !Bo) { free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LUo, LU, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUo, lda, ipiv, Bo, ldb) != 0) {
        free(LU); free(ipiv); free(LUo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(LUo);
    fb_complex_float_t *LUc = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc  = (fb_complex_float_t *)malloc(Bsz);
    if (!LUc || !Bc) { free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb) != 0) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(LU); free(ipiv); free(LUc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgetrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgetrs || !cand->zgetrs || !oracle->zgetrf) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *LU = (fb_complex_double_t *)malloc(Asz);
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!LU || !ipiv) { free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LU, A0, Asz);
    if (oracle->zgetrf(FB_LAYOUT_ROW_MAJOR, n, n, LU, lda, ipiv) != 0) {
        free(LU); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *LUo = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo  = (fb_complex_double_t *)malloc(Bsz);
    if (!LUo || !Bo) { free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(LUo, LU, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUo, lda, ipiv, Bo, ldb) != 0) {
        free(LU); free(ipiv); free(LUo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(LUo);
    fb_complex_double_t *LUc = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc  = (fb_complex_double_t *)malloc(Bsz);
    if (!LUc || !Bc) { free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(LUc, LU, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zgetrs(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, n, nrhs, LUc, lda, ipiv, Bc, ldb) != 0) {
        free(LU); free(ipiv); free(LUc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(LU); free(ipiv); free(LUc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CPOTRS / ZPOTRS — back-substitution on oracle-factored complex Cholesky
 * ========================================================================= */

static fb_judge_status_t run_cpotrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cpotrs || !cand->cpotrs || !oracle->cpotrf) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *Lf = (fb_complex_float_t *)malloc(Asz);
    if (!Lf) return FB_JUDGE_ERR_ALLOC;
    memcpy(Lf, A0, Asz);
    if (oracle->cpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, Lf, lda) != 0) {
        free(Lf); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *Lfo = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo  = (fb_complex_float_t *)malloc(Bsz);
    if (!Lfo || !Bo) { free(Lf); free(Lfo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Lfo, Lf, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfo, lda, Bo, ldb) != 0) {
        free(Lf); free(Lfo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Lfo);
    fb_complex_float_t *Lfc = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc  = (fb_complex_float_t *)malloc(Bsz);
    if (!Lfc || !Bc) { free(Lf); free(Lfc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb) != 0) {
        free(Lf); free(Lfc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Lf); free(Lfc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zpotrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zpotrs || !cand->zpotrs || !oracle->zpotrf) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    fb_uplo_t uplo = FB_UPPER;
    size_t Asz = (size_t)n*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *Lf = (fb_complex_double_t *)malloc(Asz);
    if (!Lf) return FB_JUDGE_ERR_ALLOC;
    memcpy(Lf, A0, Asz);
    if (oracle->zpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, Lf, lda) != 0) {
        free(Lf); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *Lfo = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo  = (fb_complex_double_t *)malloc(Bsz);
    if (!Lfo || !Bo) { free(Lf); free(Lfo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(Lfo, Lf, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfo, lda, Bo, ldb) != 0) {
        free(Lf); free(Lfo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Lfo);
    fb_complex_double_t *Lfc = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc  = (fb_complex_double_t *)malloc(Bsz);
    if (!Lfc || !Bc) { free(Lf); free(Lfc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Lfc, Lf, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zpotrs(FB_LAYOUT_ROW_MAJOR, uplo, n, nrhs, Lfc, lda, Bc, ldb) != 0) {
        free(Lf); free(Lfc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Lf); free(Lfc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}
/* =========================================================================
 * SGELSD — minimum-norm least squares via SVD (divide and conquer)
 * ========================================================================= */
static fb_judge_status_t run_sgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgelsd || !cand->sgelsd) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;

    /* Oracle path: verify oracle succeeds */
    float *Ao = (float *)malloc(Asz), *Bo = (float *)malloc(Bsz);
    if (!Ao || !Bo || !s_work) {
        free(Ao); free(Bo); free(s_work); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->sgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       s_work, -1.0f, &rank_out) != 0) {
        free(Ao); free(Bo); free(s_work); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);

    /* Candidate path */
    float *Ac = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(s_work); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->sgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0f, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0f, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->sgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     s_work, -1.0f, &rank_out) != 0) {
        free(Ac); free(Bc); free(s_work); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Ac); free(Bc); free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGELSY — minimum-norm least squares via complete orthogonal factorization
 * ========================================================================= */
static fb_judge_status_t run_sgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->sgelsy || !cand->sgelsy) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    /* jpvt must be zeroed before each call (zero = no preferred column ordering) */
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;

    /* Oracle path: verify oracle succeeds */
    float *Ao = (float *)malloc(Asz), *Bo = (float *)malloc(Bsz);
    if (!Ao || !Bo || !jpvt) {
        free(Ao); free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    memset(jpvt, 0, (size_t)n * sizeof(int));
    if (oracle->sgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       jpvt, -1.0f, &rank_out) != 0) {
        free(Ao); free(Bo); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);

    /* Candidate path */
    float *Ac = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            memset(jpvt, 0, (size_t)n * sizeof(int));
            (void)cand->sgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0f, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0f, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    memset(jpvt, 0, (size_t)n * sizeof(int));
    if (cand->sgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     jpvt, -1.0f, &rank_out) != 0) {
        free(Ac); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}


/* =========================================================================
 * DGELSD — double precision min-norm least squares via divide-and-conquer SVD
 * ========================================================================= */
static fb_judge_status_t run_dgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgelsd || !cand->dgelsd) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    double *Ao = (double *)malloc(Asz), *Bo = (double *)malloc(Bsz);
    if (!Ao || !Bo || !s_work) {
        free(Ao); free(Bo); free(s_work); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->dgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       s_work, -1.0, &rank_out) != 0) {
        free(Ao); free(Bo); free(s_work); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    double *Ac = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(s_work); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->dgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->dgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     s_work, -1.0, &rank_out) != 0) {
        free(Ac); free(Bc); free(s_work); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Ac); free(Bc); free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGELSD — complex single precision min-norm least squares via SVD
 * ========================================================================= */
static fb_judge_status_t run_cgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgelsd || !cand->cgelsd) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;
    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo || !s_work) {
        free(Ao); free(Bo); free(s_work); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->cgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       s_work, -1.0f, &rank_out) != 0) {
        free(Ao); free(Bo); free(s_work); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(s_work); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->cgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0f, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0f, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->cgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     s_work, -1.0f, &rank_out) != 0) {
        free(Ac); free(Bc); free(s_work); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Ac); free(Bc); free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGELSD — complex double precision min-norm least squares via SVD
 * ========================================================================= */
static fb_judge_status_t run_zgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgelsd || !cand->zgelsd) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo || !s_work) {
        free(Ao); free(Bo); free(s_work); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz);
    if (oracle->zgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       s_work, -1.0, &rank_out) != 0) {
        free(Ao); free(Bo); free(s_work); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(s_work); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            (void)cand->zgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               s_work, -1.0, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz);
    if (cand->zgelsd(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     s_work, -1.0, &rank_out) != 0) {
        free(Ac); free(Bc); free(s_work); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Ac); free(Bc); free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGELSY — double precision min-norm via complete orthogonal factorization
 * ========================================================================= */
static fb_judge_status_t run_dgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->dgelsy || !cand->dgelsy) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    double *Ao = (double *)malloc(Asz), *Bo = (double *)malloc(Bsz);
    if (!Ao || !Bo || !jpvt) {
        free(Ao); free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (oracle->dgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       jpvt, -1.0, &rank_out) != 0) {
        free(Ao); free(Bo); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    double *Ac = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            (void)cand->dgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (cand->dgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     jpvt, -1.0, &rank_out) != 0) {
        free(Ac); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGELSY — complex single precision min-norm via complete orthogonal factorization
 * ========================================================================= */
static fb_judge_status_t run_cgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->cgelsy || !cand->cgelsy) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo || !jpvt) {
        free(Ao); free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (oracle->cgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       jpvt, -1.0f, &rank_out) != 0) {
        free(Ao); free(Bo); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            (void)cand->cgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0f, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0f, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (cand->cgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     jpvt, -1.0f, &rank_out) != 0) {
        free(Ac); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGELSY — complex double precision min-norm via complete orthogonal factorization
 * ========================================================================= */
static fb_judge_status_t run_zgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    if (!oracle->zgelsy || !cand->zgelsy) return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo || !jpvt) {
        free(Ao); free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, A0, Asz); memcpy(Bo, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (oracle->zgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ao, lda, Bo, ldb,
                       jpvt, -1.0, &rank_out) != 0) {
        free(Ao); free(Bo); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(Ao); free(Bo);
    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) { free(Ac); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            (void)cand->zgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0, &rank_out);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                               jpvt, -1.0, &rank_out);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, A0, Asz); memcpy(Bc, b0, Bsz); memset(jpvt, 0, (size_t)n * sizeof(int));
    if (cand->zgelsy(FB_LAYOUT_ROW_MAJOR, m, n, nrhs, Ac, lda, Bc, ldb,
                     jpvt, -1.0, &rank_out) != 0) {
        free(Ac); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Ac); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * STRTRS / DTRTRS / CTRTRS / ZTRTRS
 * Triangular system solve: T·x = B  (T already triangular — no pre-factor)
 * Synthetic upper triangular T: forced positive diagonal + scaled off-diag
 * ========================================================================= */

static fb_judge_status_t run_strtrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_strtrs_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                  fb_diag_t, int, int, const float *, int, float *, int);
    if (!oracle->strtrs || !cand->strtrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    /* Build synthetic upper triangular T: forced diagonal, bounded off-diag.
     * Off-diagonal uses deterministic 0.5/(j-i+1) — independent of corpus
     * scale — to avoid float overflow in extreme-scale corpus cases. */
    float *T = (float *)malloc(Asz);
    if (!T) return FB_JUDGE_ERR_ALLOC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < lda; j++) {
            if (j > i && j < n) T[i*lda+j] = 0.5f / (float)(j - i + 1);
            else if (j == i)    T[i*lda+j] = (float)(i + 2);
            else                T[i*lda+j] = 0.0f;
        }
    }
    float *To = (float *)malloc(Asz), *Bo = (float *)malloc(Bsz);
    if (!To || !Bo) { free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(To, T, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_strtrs_fn_t)oracle->strtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, To, lda, Bo, ldb) != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    float *Tc = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_strtrs_fn_t)cand->strtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_strtrs_fn_t)cand->strtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_strtrs_fn_t)cand->strtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, Tc, lda, Bc, ldb) != 0) {
        free(T); free(Tc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f32(T, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(T); free(Tc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dtrtrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_dtrtrs_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                  fb_diag_t, int, int, const double *, int, double *, int);
    if (!oracle->dtrtrs || !cand->dtrtrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *T = (double *)malloc(Asz);
    if (!T) return FB_JUDGE_ERR_ALLOC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < lda; j++) {
            if (j > i && j < n) T[i*lda+j] = 0.5 / (double)(j - i + 1);
            else if (j == i)    T[i*lda+j] = (double)(i + 2);
            else                T[i*lda+j] = 0.0;
        }
    }
    double *To = (double *)malloc(Asz), *Bo = (double *)malloc(Bsz);
    if (!To || !Bo) { free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(To, T, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_dtrtrs_fn_t)oracle->dtrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, To, lda, Bo, ldb) != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    double *Tc = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_dtrtrs_fn_t)cand->dtrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_dtrtrs_fn_t)cand->dtrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_dtrtrs_fn_t)cand->dtrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, Tc, lda, Bc, ldb) != 0) {
        free(T); free(Tc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f64(T, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(T); free(Tc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_ctrtrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_ctrtrs_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                  fb_diag_t, int, int,
                                  const fb_complex_float_t *, int,
                                  fb_complex_float_t *, int);
    if (!oracle->ctrtrs || !cand->ctrtrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 =
        (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *T = (fb_complex_float_t *)malloc(Asz);
    if (!T) return FB_JUDGE_ERR_ALLOC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < lda; j++) {
            if (j > i && j < n) T[i*lda+j] = (fb_complex_float_t)(0.5f / (float)(j - i + 1));
            else if (j == i)    T[i*lda+j] = (fb_complex_float_t)(float)(i + 2);
            else                T[i*lda+j] = (fb_complex_float_t)0.0f;
        }
    }
    fb_complex_float_t *To = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!To || !Bo) { free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(To, T, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_ctrtrs_fn_t)oracle->ctrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, To, lda, Bo, ldb) != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    fb_complex_float_t *Tc = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_ctrtrs_fn_t)cand->ctrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_ctrtrs_fn_t)cand->ctrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_ctrtrs_fn_t)cand->ctrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, Tc, lda, Bc, ldb) != 0) {
        free(T); free(Tc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf32(T, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(T); free(Tc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_ztrtrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_ztrtrs_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                  fb_diag_t, int, int,
                                  const fb_complex_double_t *, int,
                                  fb_complex_double_t *, int);
    if (!oracle->ztrtrs || !cand->ztrtrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 =
        (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *T = (fb_complex_double_t *)malloc(Asz);
    if (!T) return FB_JUDGE_ERR_ALLOC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < lda; j++) {
            if (j > i && j < n) T[i*lda+j] = (fb_complex_double_t)(0.5 / (double)(j - i + 1));
            else if (j == i)    T[i*lda+j] = (fb_complex_double_t)(double)(i + 2);
            else                T[i*lda+j] = (fb_complex_double_t)0.0;
        }
    }
    fb_complex_double_t *To = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!To || !Bo) { free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(To, T, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_ztrtrs_fn_t)oracle->ztrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, To, lda, Bo, ldb) != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    fb_complex_double_t *Tc = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_ztrtrs_fn_t)cand->ztrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_ztrtrs_fn_t)cand->ztrtrs)(
                FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
                n, nrhs, Tc, lda, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_ztrtrs_fn_t)cand->ztrtrs)(
            FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT,
            n, nrhs, Tc, lda, Bc, ldb) != 0) {
        free(T); free(Tc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf64(T, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(T); free(Tc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SSYTRS / DSYTRS / CSYTRS / ZSYTRS runners
 * Strategy: build a well-conditioned SPD matrix S; factor via oracle->spotrf
 * (FB_LOWER) to get Cholesky L; convert L→LDL^T in-place; then call ssytrs
 * with uplo=FB_LOWER.  Backward error is checked against the original S.
 * ========================================================================= */

static fb_judge_status_t run_ssytrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_ssytrs_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                  const float *, int, const int *, float *, int);
    if (!oracle->ssytrs || !cand->ssytrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *b0 = (const float *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    /* Build LDL^T factored form directly (no Cholesky needed):
     *   D[k] = (float)(k+2)  (positive diagonal)
     *   L_unit[i,k] = 0.1f/(i-k)  for i>k  (small unit-lower entries)
     * Factored storage (lower, uplo=L): fact[i*lda+i]=D[i], fact[i*lda+k]=L[i,k]
     * Then compute Sorig = L_unit * D * L_unit^T explicitly so bwerr is exact. */
    float *fact  = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
    float *Sorig = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
    int   *ipiv  = (int   *)malloc((size_t)n * sizeof(int));
    if (!fact || !Sorig || !ipiv) { free(fact); free(Sorig); free(ipiv); return FB_JUDGE_ERR_ALLOC; }

    for (int i = 0; i < n; i++) {
        fact[i*lda+i] = (float)(i + 2);
        ipiv[i] = i + 1;
        for (int k = 0; k < i; k++)
            fact[i*lda+k] = 0.1f / (float)(i - k);
    }
    /* Sorig[i,j] = sum_{k=0}^{min(i,j)} L_unit[i,k]*D[k]*L_unit[j,k] */
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                float lik = (k == i) ? 1.0f : fact[i*lda+k];
                float ljk = (k == j) ? 1.0f : fact[j*lda+k];
                sum += lik * fact[k*lda+k] * ljk;
            }
            Sorig[i*lda+j] = sum;
            Sorig[j*lda+i] = sum;
        }
    }

    /* Oracle solve */
    float *fo = (float *)malloc(Asz), *Bo = (float *)malloc(Bsz);
    if (!fo || !Bo) { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(fo, fact, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_ssytrs_fn_t)oracle->ssytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fo, lda, ipiv, Bo, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);

    /* Candidate solve */
    float *fc = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_ssytrs_fn_t)cand->ssytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_ssytrs_fn_t)cand->ssytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_ssytrs_fn_t)cand->ssytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fc, lda, ipiv, Bc, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK; }

    res->residual = make_result(bwerr_f32(Sorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));

    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsytrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_dsytrs_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                  const double *, int, const int *, double *, int);
    if (!oracle->dsytrs || !cand->dsytrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *b0 = (const double *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);

    /* Direct LDL^T construction: D[k]=(k+2), L_unit[i,k]=0.1/(i-k) for i>k */
    double *fact  = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
    double *Sorig = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
    int    *ipiv  = (int    *)malloc((size_t)n * sizeof(int));
    if (!fact || !Sorig || !ipiv) { free(fact); free(Sorig); free(ipiv); return FB_JUDGE_ERR_ALLOC; }

    for (int i = 0; i < n; i++) {
        fact[i*lda+i] = (double)(i + 2);
        ipiv[i] = i + 1;
        for (int k = 0; k < i; k++)
            fact[i*lda+k] = 0.1 / (double)(i - k);
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                double lik = (k == i) ? 1.0 : fact[i*lda+k];
                double ljk = (k == j) ? 1.0 : fact[j*lda+k];
                sum += lik * fact[k*lda+k] * ljk;
            }
            Sorig[i*lda+j] = sum;
            Sorig[j*lda+i] = sum;
        }
    }

    double *fo = (double *)malloc(Asz), *Bo = (double *)malloc(Bsz);
    if (!fo || !Bo) { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(fo, fact, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_dsytrs_fn_t)oracle->dsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fo, lda, ipiv, Bo, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);

    double *fc = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_dsytrs_fn_t)cand->dsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_dsytrs_fn_t)cand->dsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_dsytrs_fn_t)cand->dsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fc, lda, ipiv, Bc, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK; }
    res->residual       = make_result(bwerr_f64(Sorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_csytrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_csytrs_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                  const fb_complex_float_t *, int,
                                  const int *, fb_complex_float_t *, int);
    if (!oracle->csytrs || !cand->csytrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);

    /* Direct LDL^T construction with real D and real L (valid for both SYTRS and HETRS):
     * D[k]=(k+2) stored as real complex, L_unit[i,k]=0.1/(i-k) for i>k (real).      */
    fb_complex_float_t *fact  = (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(fb_complex_float_t));
    fb_complex_float_t *Sorig = (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(fb_complex_float_t));
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!fact || !Sorig || !ipiv) { free(fact); free(Sorig); free(ipiv); return FB_JUDGE_ERR_ALLOC; }

    for (int i = 0; i < n; i++) {
        __real__(fact[i*lda+i]) = (float)(i + 2);
        ipiv[i] = i + 1;
        for (int k = 0; k < i; k++)
            __real__(fact[i*lda+k]) = 0.1f / (float)(i - k);
    }
    /* Sorig = L_unit * D * L_unit^H (real L,D so ^H = ^T) */
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                float lik = (k == i) ? 1.0f : (float)__real__(fact[i*lda+k]);
                float ljk = (k == j) ? 1.0f : (float)__real__(fact[j*lda+k]);
                sum += lik * (float)__real__(fact[k*lda+k]) * ljk;
            }
            __real__(Sorig[i*lda+j]) = sum;
            __real__(Sorig[j*lda+i]) = sum;
        }
    }

    fb_complex_float_t *fo = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!fo || !Bo) { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(fo, fact, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_csytrs_fn_t)oracle->csytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fo, lda, ipiv, Bo, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);
    fb_complex_float_t *fc = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_csytrs_fn_t)cand->csytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_csytrs_fn_t)cand->csytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_csytrs_fn_t)cand->csytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fc, lda, ipiv, Bc, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK; }
    res->residual       = make_result(bwerr_cf32(Sorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsytrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    typedef int (*fb_zsytrs_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                  const fb_complex_double_t *, int,
                                  const int *, fb_complex_double_t *, int);
    if (!oracle->zsytrs || !cand->zsytrs) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);

    /* Direct LDL^T construction with real D and real L (valid for both ZSYTRS and ZHETRS) */
    fb_complex_double_t *fact  = (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(fb_complex_double_t));
    fb_complex_double_t *Sorig = (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(fb_complex_double_t));
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!fact || !Sorig || !ipiv) { free(fact); free(Sorig); free(ipiv); return FB_JUDGE_ERR_ALLOC; }

    for (int i = 0; i < n; i++) {
        __real__(fact[i*lda+i]) = (double)(i + 2);
        ipiv[i] = i + 1;
        for (int k = 0; k < i; k++)
            __real__(fact[i*lda+k]) = 0.1 / (double)(i - k);
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                double lik = (k == i) ? 1.0 : (double)__real__(fact[i*lda+k]);
                double ljk = (k == j) ? 1.0 : (double)__real__(fact[j*lda+k]);
                sum += lik * (double)__real__(fact[k*lda+k]) * ljk;
            }
            __real__(Sorig[i*lda+j]) = sum;
            __real__(Sorig[j*lda+i]) = sum;
        }
    }

    fb_complex_double_t *fo = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!fo || !Bo) { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    memcpy(fo, fact, Asz); memcpy(Bo, b0, Bsz);
    if (((fb_zsytrs_fn_t)oracle->zsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fo, lda, ipiv, Bo, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);
    fb_complex_double_t *fc = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            (void)((fb_zsytrs_fn_t)cand->zsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_zsytrs_fn_t)cand->zsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER, n, nrhs, fc, lda, ipiv, Bc, ldb);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    if (((fb_zsytrs_fn_t)cand->zsytrs)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,
            n, nrhs, fc, lda, ipiv, Bc, ldb) != 0)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK; }
    res->residual       = make_result(bwerr_cf64(Sorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo);
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
    [FB_OP_CGESV]  = run_cgesv,
    [FB_OP_ZGESV]  = run_zgesv,
    [FB_OP_SPOSV]  = run_sposv,
    [FB_OP_DPOSV]  = run_dposv,
    [FB_OP_CPOSV]  = run_cposv,
    [FB_OP_ZPOSV]  = run_zposv,
    [FB_OP_SGELS]  = run_sgels,
    [FB_OP_DGELS]  = run_dgels,
    [FB_OP_CGELS]  = run_cgels,
    [FB_OP_ZGELS]  = run_zgels,
    [FB_OP_SGETRS] = run_sgetrs,
    [FB_OP_DGETRS] = run_dgetrs,
    [FB_OP_CGETRS] = run_cgetrs,
    [FB_OP_ZGETRS] = run_zgetrs,
    [FB_OP_SPOTRS] = run_spotrs,
    [FB_OP_DPOTRS] = run_dpotrs,
    [FB_OP_CPOTRS] = run_cpotrs,
    [FB_OP_ZPOTRS] = run_zpotrs,
    [FB_OP_STRTRS] = run_strtrs,
    [FB_OP_DTRTRS] = run_dtrtrs,
    [FB_OP_CTRTRS] = run_ctrtrs,
    [FB_OP_ZTRTRS] = run_ztrtrs,
    [FB_OP_SSYTRS] = run_ssytrs,
    [FB_OP_DSYTRS] = run_dsytrs,
    [FB_OP_CSYTRS] = run_csytrs,
    [FB_OP_ZSYTRS] = run_zsytrs,
    [FB_OP_SGELSD] = run_sgelsd,
    [FB_OP_DGELSD] = run_dgelsd,
    [FB_OP_CGELSD] = run_cgelsd,
    [FB_OP_ZGELSD] = run_zgelsd,
    [FB_OP_SGELSY] = run_sgelsy,
    [FB_OP_DGELSY] = run_dgelsy,
    [FB_OP_CGELSY] = run_cgelsy,
    [FB_OP_ZGELSY] = run_zgelsy,
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