/**
 * @file judge_factorization.c
 * @brief FB_JUDGE_FACTORIZATION archetype evaluator — LU, Cholesky, QR.
 *
 * Implements oracle-vs-candidate comparison for matrix factorizations with
 * two primary metrics: reconstruction residual and (for QR) orthogonality.
 *
 * Phase 3 coverage:
 *   LU (GETRF):      SGETRF, DGETRF
 *   Cholesky (POTRF): SPOTRF, DPOTRF
 *   QR (GEQRF):      SGEQRF, DGEQRF
 *
 * Complex variants (CGETRF, ZGETRF, etc.) return FB_JUDGE_ERR_NOT_IMPL; they
 * will be added in a later phase.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_factorization.h"
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

#define FB_FACTORIZATION_WARMUP_RUNS   2
#define FB_FACTORIZATION_TIMING_RUNS   5

/* =========================================================================
 * Private utilities
 * ========================================================================= */

/** Clone an m × n matrix (row-major, leading dimension lda). */
static float *clone_matrix_f32(const float *A, int m, int n, int lda)
{
    size_t sz = (size_t)m * (size_t)lda * sizeof(float);
    float *cpy = (float *)malloc(sz);
    if (!cpy) return NULL;
    memcpy(cpy, A, sz);
    return cpy;
}

/** Clone an m × n matrix (double precision). */
static double *clone_matrix_f64(const double *A, int m, int n, int lda)
{
    size_t sz = (size_t)m * (size_t)lda * sizeof(double);
    double *cpy = (double *)malloc(sz);
    if (!cpy) return NULL;
    memcpy(cpy, A, sz);
    return cpy;
}

/** Result builder: mark a case as fatal for oracle. */
static void mark_oracle_fatal(fb_judge_factorization_result_t *r)
{
    r->reconstruction.digits  = 0.0;
    r->reconstruction.relative_error = 1.0;
    r->reconstruction.is_oracle_fatal = true;
    r->reconstruction.is_fatal = false;
    r->orthogonality.digits = 0.0;
    r->orthogonality.relative_error = 1.0;
    r->orthogonality.is_oracle_fatal = true;
    r->orthogonality.is_fatal = false;
}

/** Result builder: mark a case as fatal for candidate. */
static void mark_cand_fatal(fb_judge_factorization_result_t *r)
{
    r->reconstruction.digits  = 0.0;
    r->reconstruction.relative_error = 1.0;
    r->reconstruction.is_oracle_fatal = false;
    r->reconstruction.is_fatal = true;
    r->orthogonality.digits = 0.0;
    r->orthogonality.relative_error = 1.0;
    r->orthogonality.is_oracle_fatal = false;
    r->orthogonality.is_fatal = true;
}

/** Build a result from relative error. */
static void result_from_relerr(fb_judge_case_result_t *res, double relerr)
{
    if (relerr == 0.0) {
        res->digits = 16.0; res->relative_error = 0.0;
    } else {
        double d = -log10(relerr);
        res->digits = (d < 0.0) ? 0.0 : (d > 16.0 ? 16.0 : d);
        res->relative_error = relerr;
    }
    res->is_fatal = false; res->is_oracle_fatal = false;
}

/* =========================================================================
 * Runner type
 * ========================================================================= */

typedef fb_judge_status_t (*fb_factorization_runner_fn)(
    const fb_backend_vtable_t *oracle,
    const fb_backend_vtable_t *cand,
    const fb_corpus_case_t    *tc,
    fb_judge_factorization_result_t *result,
    uint64_t                  *ns_out);

/* =========================================================================
 * LU Factorization (SGETRF, DGETRF)
 * ========================================================================= */

/**
 * Reconstruction for LU: ||P·A - L·U||_F / ||A||_F
 *
 * A_factored: GETRF in-place output (L in lower triangle with unit diagonal; U in upper).
 * ipiv: pivot indices (1-based LAPACKE convention).
 * Applies row permutation to A_orig then computes ||L·U - P·A||_F.
 */
static double fb_lu_reconstruction_f32(const float *A_orig, const float *A_factored,
                                       const int *ipiv,
                                       int m, int n, int lda)
{
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    float *PA = (float*)malloc(Asz);
    if (!PA) return 1.0;
    memcpy(PA, A_orig, Asz);
    /* Apply row permutation from GETRF. */
    for (int i = 0; i < minmn; i++) {
        int piv = ipiv[i] - 1;  /* LAPACKE 1-based → 0-based */
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                float tmp = PA[i*lda + j];
                PA[i*lda + j] = PA[piv*lda + j];
                PA[piv*lda + j] = tmp;
            }
        }
    }
    /* Compute ||L·U - PA||_F / ||PA||_F. */
    double norm_diff = 0.0, norm_PA = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            /* LU[i][j] = sum_{k=0}^{min(i,j)} L[i][k] · U[k][j] */
            int kmax = i < j ? i : j;
            double lij = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double Lik = (k < i) ? (double)A_factored[i*lda + k] : 1.0;
                double Ukj = (double)A_factored[k*lda + j];
                lij += Lik * Ukj;
            }
            double diff = lij - (double)PA[i*lda + j];
            norm_diff += diff * diff;
            double pa = (double)PA[i*lda + j];
            norm_PA += pa * pa;
        }
    }
    free(PA);
    if (norm_PA < (double)FLT_EPSILON)
        return (norm_diff < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_PA);
}

static double fb_lu_reconstruction_f64(const double *A_orig, const double *A_factored,
                                       const int *ipiv,
                                       int m, int n, int lda)
{
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    double *PA = (double*)malloc(Asz);
    if (!PA) return 1.0;
    memcpy(PA, A_orig, Asz);
    for (int i = 0; i < minmn; i++) {
        int piv = ipiv[i] - 1;
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                double tmp = PA[i*lda + j];
                PA[i*lda + j] = PA[piv*lda + j];
                PA[piv*lda + j] = tmp;
            }
        }
    }
    double norm_diff = 0.0, norm_PA = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = i < j ? i : j;
            double lij = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double Lik = (k < i) ? A_factored[i*lda + k] : 1.0;
                double Ukj = A_factored[k*lda + j];
                lij += Lik * Ukj;
            }
            double diff = lij - PA[i*lda + j];
            norm_diff += diff * diff;
            norm_PA += PA[i*lda + j] * PA[i*lda + j];
        }
    }
    free(PA);
    if (norm_PA < DBL_EPSILON) return (norm_diff < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_PA);
}

static fb_judge_status_t run_sgetrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgetrf || !cand->sgetrf) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;

    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    /* Clone for oracle. */
    float *A_oracle = clone_matrix_f32(A_orig, m, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;

    int *ipiv_oracle = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_oracle) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }

    /* Run oracle. */
    int info_oracle = oracle->sgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oracle, lda, ipiv_oracle);
    if (info_oracle != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(ipiv_oracle);
        return FB_JUDGE_OK;
    }

    /* Clone for candidate. */
    float *A_cand = clone_matrix_f32(A_orig, m, n, lda);
    if (!A_cand) { free(A_oracle); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }

    int *ipiv_cand = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_cand) { free(A_oracle); free(A_cand); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }

    /* Run candidate. */
    int info_cand = cand->sgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);

    if (info_cand != 0) {
        mark_cand_fatal(res);
        free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
        return FB_JUDGE_OK;
    }

    /* Reconstruction residual: ||P·A - L·U||_F / ||A||_F using candidate pivot. */
    double recon_err = fb_lu_reconstruction_f32(A_orig, A_cand, ipiv_cand, m, n, lda);
    result_from_relerr(&res->reconstruction, recon_err);

    /* QR only: orthogonality not applicable for LU. */
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    res->orthogonality.is_fatal = false;
    res->orthogonality.is_oracle_fatal = false;

    /* Timing. */
    if (ns_out) {
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++)
            (void)cand->sgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Cholesky Factorization (SPOTRF, DPOTRF)
 *
 * Reconstruction: ||A - L*L^T|| / ||A|| (lower) or ||A - U^T*U|| / ||A|| (upper)
 * ========================================================================= */

/**
 * Cholesky reconstruction: ||A - L·L^T||_F / ||A||_F (lower triangle).
 * A_factored contains L in lower triangle; upper triangle is irrelevant.
 * For upper triangle storage, L = U^T so result is the same formula.
 */
static double fb_cholesky_reconstruction_f32(const float *A_orig,
                                             const float *A_factored,
                                             int n, int lda,
                                             fb_uplo_t uplo)
{
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* (L·L^T)[i][j] = sum_{k=0}^{min(i,j)} L[i][k] · L[j][k] */
            int kmax = i < j ? i : j;
            double llt = 0.0;
            if (uplo == FB_LOWER) {
                for (int k = 0; k <= kmax; k++)
                    llt += (double)A_factored[i*lda+k] * (double)A_factored[j*lda+k];
            } else {
                /* Upper: L = U^T, L[i][k] = U[k][i] = A_factored[k*lda+i]. */
                for (int k = 0; k <= kmax; k++)
                    llt += (double)A_factored[k*lda+i] * (double)A_factored[k*lda+j];
            }
            double diff = llt - (double)A_orig[i*lda + j];
            norm_diff += diff * diff;
            double a = (double)A_orig[i*lda + j];
            norm_A += a * a;
        }
    }
    if (norm_A < (double)FLT_EPSILON) return 0.0;
    return sqrt(norm_diff / norm_A);
}

static double fb_cholesky_reconstruction_f64(const double *A_orig,
                                             const double *A_factored,
                                             int n, int lda,
                                             fb_uplo_t uplo)
{
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = i < j ? i : j;
            double llt = 0.0;
            if (uplo == FB_LOWER) {
                for (int k = 0; k <= kmax; k++)
                    llt += A_factored[i*lda+k] * A_factored[j*lda+k];
            } else {
                for (int k = 0; k <= kmax; k++)
                    llt += A_factored[k*lda+i] * A_factored[k*lda+j];
            }
            double diff = llt - A_orig[i*lda + j];
            norm_diff += diff * diff;
            norm_A += A_orig[i*lda+j] * A_orig[i*lda+j];
        }
    }
    if (norm_A < DBL_EPSILON) return 0.0;
    return sqrt(norm_diff / norm_A);
}

static fb_judge_status_t run_spotrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->spotrf || !cand->spotrf) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n;
    int lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;
    fb_uplo_t uplo = FB_UPPER;  /* Upper triangle; row-major Cholesky convention */

    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    /* Clone for oracle. */
    float *A_oracle = clone_matrix_f32(A_orig, n, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;

    int info_oracle = oracle->spotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_oracle, lda);
    if (info_oracle != 0) { mark_oracle_fatal(res); free(A_oracle); return FB_JUDGE_OK; }

    /* Clone for candidate. */
    float *A_cand = clone_matrix_f32(A_orig, n, n, lda);
    if (!A_cand) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }

    int info_cand = cand->spotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_cand, lda);
    if (info_cand != 0) {
        mark_cand_fatal(res);
        free(A_oracle); free(A_cand);
        return FB_JUDGE_OK;
    }

    double recon_err = fb_cholesky_reconstruction_f32(A_orig, A_cand, n, lda, uplo);
    result_from_relerr(&res->reconstruction, recon_err);

    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    res->orthogonality.is_fatal = false;
    res->orthogonality.is_oracle_fatal = false;

    if (ns_out) {
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            float *A_tmp = clone_matrix_f32(A_orig, n, n, lda);
            if (A_tmp) { (void)cand->spotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda); free(A_tmp); }
        }
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            float *A_tmp = clone_matrix_f32(A_orig, n, n, lda);
            if (!A_tmp) break;
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->spotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(A_tmp);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    free(A_oracle); free(A_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * QR Factorization (SGEQRF, DGEQRF)
 *
 * Reconstruction: ||A - Q*R|| / ||A||
 * Orthogonality:  ||Q^T*Q - I|| / ||I||  (requires extracting Q via ORGQR)
 * ========================================================================= */

/* =========================================================================
 * DGETRF — double-precision LU factorization
 * ========================================================================= */
static fb_judge_status_t run_dgetrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgetrf || !cand->dgetrf) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double*)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    double *A_oracle = clone_matrix_f64(A_orig, m, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    int *ipiv_oracle = (int*)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_oracle) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oracle, lda, ipiv_oracle) != 0) {
        mark_oracle_fatal(res); free(A_oracle); free(ipiv_oracle); return FB_JUDGE_OK;
    }
    double *A_cand = clone_matrix_f64(A_orig, m, n, lda);
    if (!A_cand) { free(A_oracle); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    int *ipiv_cand = (int*)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_cand) { free(A_oracle); free(A_cand); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->dgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand) != 0) {
        mark_cand_fatal(res);
        free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
        return FB_JUDGE_OK;
    }
    double recon_err = fb_lu_reconstruction_f64(A_orig, A_cand, ipiv_cand, m, n, lda);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++)
            (void)cand->dgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DPOTRF — double-precision Cholesky factorization
 * ========================================================================= */
static fb_judge_status_t run_dpotrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dpotrf || !cand->dpotrf) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double*)tc->A;
    fb_uplo_t uplo = FB_UPPER;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    double *A_oracle = clone_matrix_f64(A_orig, n, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    if (oracle->dpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_oracle, lda) != 0) {
        mark_oracle_fatal(res); free(A_oracle); return FB_JUDGE_OK;
    }
    double *A_cand = clone_matrix_f64(A_orig, n, n, lda);
    if (!A_cand) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->dpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_cand, lda) != 0) {
        mark_cand_fatal(res); free(A_oracle); free(A_cand); return FB_JUDGE_OK;
    }
    double recon_err = fb_cholesky_reconstruction_f64(A_orig, A_cand, n, lda, uplo);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            double *A_tmp = clone_matrix_f64(A_orig, n, n, lda);
            if (A_tmp) { (void)cand->dpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda); free(A_tmp); }
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            double *A_tmp = clone_matrix_f64(A_orig, n, n, lda);
            if (!A_tmp) break;
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(A_tmp);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGEQRF — single-precision QR factorization.
 *
 * Uses GEQRF to factor A = Q·R, then ORGQR to extract Q explicitly.
 * Metrics: reconstruction (‖A − Q·R‖_F / ‖A‖_F) and orthogonality (‖Q^T·Q − I‖_F / √k).
 * ========================================================================= */
static fb_judge_status_t run_sgeqrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgeqrf || !cand->sgeqrf || !cand->sorgqr) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const float *A_orig = (const float*)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;  /* min(m,n) */

    /* Verify oracle runs without error. */
    float *A_oc = clone_matrix_f32(A_orig, m, n, lda);
    float *tau_oc = (float*)malloc((size_t)k * sizeof(float));
    if (!A_oc || !tau_oc) { free(A_oc); free(tau_oc); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oc, lda, tau_oc) != 0) {
        free(A_oc); free(tau_oc); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    free(A_oc); free(tau_oc);

    /* Candidate path. */
    float *A_qr = clone_matrix_f32(A_orig, m, n, lda);
    float *tau  = (float*)malloc((size_t)k * sizeof(float));
    if (!A_qr || !tau) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(float));
            (void)cand->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(float));
    if (cand->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Extract R from the upper triangle of A_qr (before ORGQR overwrites it). */
    float *R = (float*)calloc((size_t)k * (size_t)n, sizeof(float));
    if (!R) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }
    for (int i = 0; i < k; i++)
        for (int j = i; j < n; j++)
            R[i*n + j] = A_qr[i*lda + j];

    /* Call ORGQR to extract Q (m×k) in-place into A_qr. */
    if (cand->sorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); free(R); mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    /* A_qr is now Q (m×k). */

    /* Reconstruction: ‖A_orig - Q·R‖_F / ‖A_orig‖_F */
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double qr = 0.0;
            for (int l = 0; l < k; l++)
                qr += (double)A_qr[i*lda + l] * (double)R[l*n + j];
            double diff = (double)A_orig[i*lda + j] - qr;
            norm_diff += diff * diff;
            double a = (double)A_orig[i*lda + j];
            norm_A += a * a;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_A < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_A));

    /* Orthogonality: ‖Q^T·Q - I‖_F / √k */
    double ortho_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double qtq = 0.0;
            for (int l = 0; l < m; l++)
                qtq += (double)A_qr[l*lda + i] * (double)A_qr[l*lda + j];
            double delta = qtq - (i == j ? 1.0 : 0.0);
            ortho_err += delta * delta;
        }
    }
    result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

    free(A_qr); free(tau); free(R);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGEQRF — double-precision QR factorization
 * ========================================================================= */
static fb_judge_status_t run_dgeqrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgeqrf || !cand->dgeqrf || !cand->dorgqr) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double*)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    double *A_oc = clone_matrix_f64(A_orig, m, n, lda);
    double *tau_oc = (double*)malloc((size_t)k * sizeof(double));
    if (!A_oc || !tau_oc) { free(A_oc); free(tau_oc); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oc, lda, tau_oc) != 0) {
        free(A_oc); free(tau_oc); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    free(A_oc); free(tau_oc);

    double *A_qr = clone_matrix_f64(A_orig, m, n, lda);
    double *tau  = (double*)malloc((size_t)k * sizeof(double));
    if (!A_qr || !tau) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(double));
            (void)cand->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(double));
    if (cand->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double *R = (double*)calloc((size_t)k * (size_t)n, sizeof(double));
    if (!R) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }
    for (int i = 0; i < k; i++)
        for (int j = i; j < n; j++)
            R[i*n + j] = A_qr[i*lda + j];

    if (cand->dorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); free(R); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double qr = 0.0;
            for (int l = 0; l < k; l++)
                qr += A_qr[i*lda + l] * R[l*n + j];
            double diff = A_orig[i*lda + j] - qr;
            norm_diff += diff * diff;
            norm_A += A_orig[i*lda + j] * A_orig[i*lda + j];
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_A < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_A));

    double ortho_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double qtq = 0.0;
            for (int l = 0; l < m; l++)
                qtq += A_qr[l*lda + i] * A_qr[l*lda + j];
            double delta = qtq - (i == j ? 1.0 : 0.0);
            ortho_err += delta * delta;
        }
    }
    result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

    free(A_qr); free(tau); free(R);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Complex helpers: clone, LU/Cholesky reconstruction, QR helpers
 * ========================================================================= */

static fb_complex_float_t *clone_matrix_cf32(const fb_complex_float_t *A,
                                              int m, int n, int lda)
{
    (void)n;
    size_t sz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    fb_complex_float_t *cpy = (fb_complex_float_t *)malloc(sz);
    if (!cpy) return NULL;
    memcpy(cpy, A, sz);
    return cpy;
}

static fb_complex_double_t *clone_matrix_cf64(const fb_complex_double_t *A,
                                               int m, int n, int lda)
{
    (void)n;
    size_t sz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    fb_complex_double_t *cpy = (fb_complex_double_t *)malloc(sz);
    if (!cpy) return NULL;
    memcpy(cpy, A, sz);
    return cpy;
}

/**
 * Complex LU reconstruction: ||PA - L·U||_F / ||PA||_F
 * LU is stored in-place: L in lower (unit diagonal), U in upper.
 * Arithmetic: complex multiply + modulus squared for norm.
 */
static double fb_lu_reconstruction_cf32(const fb_complex_float_t *A_orig,
                                        const fb_complex_float_t *A_factored,
                                        const int *ipiv,
                                        int m, int n, int lda)
{
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    fb_complex_float_t *PA = (fb_complex_float_t *)malloc(Asz);
    if (!PA) return 1.0;
    memcpy(PA, A_orig, Asz);
    /* Row permutation. */
    for (int i = 0; i < minmn; i++) {
        int piv = ipiv[i] - 1;
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                fb_complex_float_t tmp = PA[i*lda + j];
                PA[i*lda + j] = PA[piv*lda + j];
                PA[piv*lda + j] = tmp;
            }
        }
    }
    double norm_diff = 0.0, norm_PA = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = i < j ? i : j;
            double re = 0.0, im = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double Lr = (k < i) ? (double)__real__(A_factored[i*lda + k]) : 1.0;
                double Li = (k < i) ? (double)__imag__(A_factored[i*lda + k]) : 0.0;
                double Ur = (double)__real__(A_factored[k*lda + j]);
                double Ui = (double)__imag__(A_factored[k*lda + j]);
                re += Lr*Ur - Li*Ui;
                im += Lr*Ui + Li*Ur;
            }
            double dr = re - (double)__real__(PA[i*lda + j]);
            double di = im - (double)__imag__(PA[i*lda + j]);
            norm_diff += dr*dr + di*di;
            double pr = (double)__real__(PA[i*lda + j]), pi2 = (double)__imag__(PA[i*lda + j]);
            norm_PA += pr*pr + pi2*pi2;
        }
    }
    free(PA);
    if (norm_PA < (double)FLT_EPSILON)
        return (norm_diff < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_PA);
}

static double fb_lu_reconstruction_cf64(const fb_complex_double_t *A_orig,
                                        const fb_complex_double_t *A_factored,
                                        const int *ipiv,
                                        int m, int n, int lda)
{
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    fb_complex_double_t *PA = (fb_complex_double_t *)malloc(Asz);
    if (!PA) return 1.0;
    memcpy(PA, A_orig, Asz);
    for (int i = 0; i < minmn; i++) {
        int piv = ipiv[i] - 1;
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                fb_complex_double_t tmp = PA[i*lda + j];
                PA[i*lda + j] = PA[piv*lda + j];
                PA[piv*lda + j] = tmp;
            }
        }
    }
    double norm_diff = 0.0, norm_PA = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = i < j ? i : j;
            double re = 0.0, im = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double Lr = (k < i) ? __real__(A_factored[i*lda + k]) : 1.0;
                double Li = (k < i) ? __imag__(A_factored[i*lda + k]) : 0.0;
                double Ur = __real__(A_factored[k*lda + j]);
                double Ui = __imag__(A_factored[k*lda + j]);
                re += Lr*Ur - Li*Ui;
                im += Lr*Ui + Li*Ur;
            }
            double dr = re - __real__(PA[i*lda + j]);
            double di = im - __imag__(PA[i*lda + j]);
            norm_diff += dr*dr + di*di;
            norm_PA += __real__(PA[i*lda + j]) * __real__(PA[i*lda + j])
                     + __imag__(PA[i*lda + j]) * __imag__(PA[i*lda + j]);
        }
    }
    free(PA);
    if (norm_PA < DBL_EPSILON) return (norm_diff < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_PA);
}

/**
 * Complex Cholesky reconstruction: ||A - L·L^H||_F / ||A||_F
 * CPOTRF stores L (lower tri) when uplo=LOWER, or L^H (upper tri) when uplo=UPPER.
 * We always use UPPER → A_factored stores U (= L^H), reconstruction: ||A - U^H·U||_F / ||A||_F.
 */
static double fb_cholesky_reconstruction_cf32(const fb_complex_float_t *A_orig,
                                              const fb_complex_float_t *A_factored,
                                              int n, int lda, fb_uplo_t uplo)
{
    /* uplo==UPPER: U stored → reconstruct A_ij = sum_k conj(U_ki) * U_kj (since A = U^H * U). */
    /* uplo==LOWER: L stored → reconstruct A_ij = sum_k L_ik * conj(L_jk) (since A = L * L^H). */
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < n; k++) {
                double Ar, Ai, Br, Bi;
                if (uplo == FB_UPPER) {
                    /* A = U^H * U: rows k of U^H = cols k of U, but upper tri */
                    if (k > i || k > j) continue;  /* U_ki or U_kj is zero if k > i or k > j */
                    /* conj(U[k][i]) * U[k][j] */
                    Ar = (double)__real__(A_factored[k*lda + i]);
                    Ai = -(double)__imag__(A_factored[k*lda + i]);  /* conjugate */
                    Br = (double)__real__(A_factored[k*lda + j]);
                    Bi = (double)__imag__(A_factored[k*lda + j]);
                } else {
                    /* A = L * L^H: L_ik * conj(L_jk) */
                    if (k > i || k > j) continue;
                    Ar = (double)__real__(A_factored[i*lda + k]);
                    Ai = (double)__imag__(A_factored[i*lda + k]);
                    Br = (double)__real__(A_factored[j*lda + k]);
                    Bi = -(double)__imag__(A_factored[j*lda + k]);  /* conjugate */
                }
                re += Ar*Br - Ai*Bi;
                im += Ar*Bi + Ai*Br;
            }
            double dr = re - (double)__real__(A_orig[i*lda + j]);
            double di = im - (double)__imag__(A_orig[i*lda + j]);
            norm_diff += dr*dr + di*di;
            double oar = (double)__real__(A_orig[i*lda + j]), oai = (double)__imag__(A_orig[i*lda + j]);
            norm_A += oar*oar + oai*oai;
        }
    }
    if (norm_A < (double)FLT_EPSILON)
        return (norm_diff < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_A);
}

static double fb_cholesky_reconstruction_cf64(const fb_complex_double_t *A_orig,
                                              const fb_complex_double_t *A_factored,
                                              int n, int lda, fb_uplo_t uplo)
{
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < n; k++) {
                double Ar, Ai, Br, Bi;
                if (uplo == FB_UPPER) {
                    if (k > i || k > j) continue;
                    Ar =  __real__(A_factored[k*lda + i]);
                    Ai = -__imag__(A_factored[k*lda + i]);
                    Br =  __real__(A_factored[k*lda + j]);
                    Bi =  __imag__(A_factored[k*lda + j]);
                } else {
                    if (k > i || k > j) continue;
                    Ar =  __real__(A_factored[i*lda + k]);
                    Ai =  __imag__(A_factored[i*lda + k]);
                    Br =  __real__(A_factored[j*lda + k]);
                    Bi = -__imag__(A_factored[j*lda + k]);
                }
                re += Ar*Br - Ai*Bi;
                im += Ar*Bi + Ai*Br;
            }
            double dr = re - __real__(A_orig[i*lda + j]);
            double di = im - __imag__(A_orig[i*lda + j]);
            norm_diff += dr*dr + di*di;
            norm_A += __real__(A_orig[i*lda + j]) * __real__(A_orig[i*lda + j])
                    + __imag__(A_orig[i*lda + j]) * __imag__(A_orig[i*lda + j]);
        }
    }
    if (norm_A < DBL_EPSILON) return (norm_diff < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_A);
}

/* =========================================================================
 * CGETRF / ZGETRF — complex LU factorization
 * ========================================================================= */

static fb_judge_status_t run_cgetrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgetrf || !cand->cgetrf) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    fb_complex_float_t *A_oracle = clone_matrix_cf32(A_orig, m, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    int *ipiv_oracle = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_oracle) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oracle, lda, ipiv_oracle) != 0) {
        mark_oracle_fatal(res); free(A_oracle); free(ipiv_oracle); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_cand = clone_matrix_cf32(A_orig, m, n, lda);
    if (!A_cand) { free(A_oracle); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    int *ipiv_cand = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_cand) { free(A_oracle); free(A_cand); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->cgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand) != 0) {
        mark_cand_fatal(res);
        free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
        return FB_JUDGE_OK;
    }
    double recon_err = fb_lu_reconstruction_cf32(A_orig, A_cand, ipiv_cand, m, n, lda);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++)
            (void)cand->cgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgetrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgetrf || !cand->zgetrf) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    fb_complex_double_t *A_oracle = clone_matrix_cf64(A_orig, m, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    int *ipiv_oracle = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_oracle) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oracle, lda, ipiv_oracle) != 0) {
        mark_oracle_fatal(res); free(A_oracle); free(ipiv_oracle); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_cand = clone_matrix_cf64(A_orig, m, n, lda);
    if (!A_cand) { free(A_oracle); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    int *ipiv_cand = (int *)malloc((size_t)(m < n ? m : n) * sizeof(int));
    if (!ipiv_cand) { free(A_oracle); free(A_cand); free(ipiv_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->zgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand) != 0) {
        mark_cand_fatal(res);
        free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
        return FB_JUDGE_OK;
    }
    double recon_err = fb_lu_reconstruction_cf64(A_orig, A_cand, ipiv_cand, m, n, lda);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++)
            (void)cand->zgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
        uint64_t best = UINT64_MAX;
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgetrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cand, lda, ipiv_cand);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand); free(ipiv_oracle); free(ipiv_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CPOTRF / ZPOTRF — complex Cholesky factorization
 * ========================================================================= */

static fb_judge_status_t run_cpotrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cpotrf || !cand->cpotrf) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    fb_uplo_t uplo = FB_UPPER;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    fb_complex_float_t *A_oracle = clone_matrix_cf32(A_orig, n, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    if (oracle->cpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_oracle, lda) != 0) {
        mark_oracle_fatal(res); free(A_oracle); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_cand = clone_matrix_cf32(A_orig, n, n, lda);
    if (!A_cand) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->cpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_cand, lda) != 0) {
        mark_cand_fatal(res); free(A_oracle); free(A_cand); return FB_JUDGE_OK;
    }
    double recon_err = fb_cholesky_reconstruction_cf32(A_orig, A_cand, n, lda, uplo);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            fb_complex_float_t *A_tmp = clone_matrix_cf32(A_orig, n, n, lda);
            if (A_tmp) { (void)cand->cpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda); free(A_tmp); }
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            fb_complex_float_t *A_tmp = clone_matrix_cf32(A_orig, n, n, lda);
            if (!A_tmp) break;
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(A_tmp);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zpotrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zpotrf || !cand->zpotrf) return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    fb_uplo_t uplo = FB_UPPER;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    fb_complex_double_t *A_oracle = clone_matrix_cf64(A_orig, n, n, lda);
    if (!A_oracle) return FB_JUDGE_ERR_ALLOC;
    if (oracle->zpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_oracle, lda) != 0) {
        mark_oracle_fatal(res); free(A_oracle); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_cand = clone_matrix_cf64(A_orig, n, n, lda);
    if (!A_cand) { free(A_oracle); return FB_JUDGE_ERR_ALLOC; }
    if (cand->zpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_cand, lda) != 0) {
        mark_cand_fatal(res); free(A_oracle); free(A_cand); return FB_JUDGE_OK;
    }
    double recon_err = fb_cholesky_reconstruction_cf64(A_orig, A_cand, n, lda, uplo);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality.digits = 16.0;
    res->orthogonality.relative_error = 0.0;
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            fb_complex_double_t *A_tmp = clone_matrix_cf64(A_orig, n, n, lda);
            if (A_tmp) { (void)cand->zpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda); free(A_tmp); }
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            fb_complex_double_t *A_tmp = clone_matrix_cf64(A_orig, n, n, lda);
            if (!A_tmp) break;
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zpotrf(FB_LAYOUT_ROW_MAJOR, uplo, n, A_tmp, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(A_tmp);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    free(A_oracle); free(A_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGEQRF / ZGEQRF — complex QR factorization.
 *
 * Uses GEQRF → CUNGQR/ZUNGQR to extract Q; same metrics as real variants.
 * ========================================================================= */

static fb_judge_status_t run_cgeqrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgeqrf || !cand->cgeqrf || !cand->cungqr) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    /* Oracle sanity check. */
    fb_complex_float_t *A_oc  = clone_matrix_cf32(A_orig, m, n, lda);
    fb_complex_float_t *tau_oc = (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!A_oc || !tau_oc) { free(A_oc); free(tau_oc); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oc, lda, tau_oc) != 0) {
        free(A_oc); free(tau_oc); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    free(A_oc); free(tau_oc);

    /* Candidate path. */
    fb_complex_float_t *A_qr = clone_matrix_cf32(A_orig, m, n, lda);
    fb_complex_float_t *tau  = (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!A_qr || !tau) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            (void)cand->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
    if (cand->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Extract R from upper triangle. */
    fb_complex_float_t *R = (fb_complex_float_t *)calloc((size_t)k * (size_t)n, sizeof(fb_complex_float_t));
    if (!R) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }
    for (int i = 0; i < k; i++)
        for (int j = i; j < n; j++)
            R[i*n + j] = A_qr[i*lda + j];

    /* Extract Q via CUNGQR. */
    if (cand->cungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); free(R); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Reconstruction metric: ||A - Q·R||_F / ||A||_F. */
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            /* QR[i][j] = sum_l Q[i][l] * R[l][j] (complex multiply). */
            double qr_re = 0.0, qr_im = 0.0;
            for (int l = 0; l < k; l++) {
                double qr2 = (double)__real__(A_qr[i*lda + l]), qi2 = (double)__imag__(A_qr[i*lda + l]);
                double rr  = (double)__real__(R[l*n + j]),      ri  = (double)__imag__(R[l*n + j]);
                qr_re += qr2*rr - qi2*ri;
                qr_im += qr2*ri + qi2*rr;
            }
            double dr = (double)__real__(A_orig[i*lda + j]) - qr_re;
            double di = (double)__imag__(A_orig[i*lda + j]) - qr_im;
            norm_diff += dr*dr + di*di;
            double ar = (double)__real__(A_orig[i*lda + j]), ai = (double)__imag__(A_orig[i*lda + j]);
            norm_A += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_A < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_A));

    /* Orthogonality: ||Q^H·Q - I||_F / sqrt(k). */
    double ortho_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double re = 0.0, im = 0.0;
            for (int l = 0; l < m; l++) {
                /* conj(Q[l][i]) * Q[l][j]. */
                double qir =  (double)__real__(A_qr[l*lda + i]);
                double qii = -(double)__imag__(A_qr[l*lda + i]);
                double qjr =  (double)__real__(A_qr[l*lda + j]);
                double qji =  (double)__imag__(A_qr[l*lda + j]);
                re += qir*qjr - qii*qji;
                im += qir*qji + qii*qjr;
            }
            double dr = re - (i == j ? 1.0 : 0.0);
            double di = im;
            ortho_err += dr*dr + di*di;
        }
    }
    result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

    free(A_qr); free(tau); free(R);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeqrf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgeqrf || !cand->zgeqrf || !cand->zungqr) return FB_JUDGE_ERR_NOT_IMPL;

    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    fb_complex_double_t *A_oc  = clone_matrix_cf64(A_orig, m, n, lda);
    fb_complex_double_t *tau_oc = (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
    if (!A_oc || !tau_oc) { free(A_oc); free(tau_oc); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oc, lda, tau_oc) != 0) {
        free(A_oc); free(tau_oc); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    free(A_oc); free(tau_oc);

    fb_complex_double_t *A_qr = clone_matrix_cf64(A_orig, m, n, lda);
    fb_complex_double_t *tau  = (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
    if (!A_qr || !tau) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
            (void)cand->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_qr, A_orig, (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
    if (cand->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_double_t *R = (fb_complex_double_t *)calloc((size_t)k * (size_t)n, sizeof(fb_complex_double_t));
    if (!R) { free(A_qr); free(tau); return FB_JUDGE_ERR_ALLOC; }
    for (int i = 0; i < k; i++)
        for (int j = i; j < n; j++)
            R[i*n + j] = A_qr[i*lda + j];

    if (cand->zungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_qr, lda, tau) != 0) {
        free(A_qr); free(tau); free(R); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double qr_re = 0.0, qr_im = 0.0;
            for (int l = 0; l < k; l++) {
                double qrr = __real__(A_qr[i*lda + l]), qri = __imag__(A_qr[i*lda + l]);
                double rr  = __real__(R[l*n + j]),       ri  = __imag__(R[l*n + j]);
                qr_re += qrr*rr - qri*ri;
                qr_im += qrr*ri + qri*rr;
            }
            double dr = __real__(A_orig[i*lda + j]) - qr_re;
            double di = __imag__(A_orig[i*lda + j]) - qr_im;
            norm_diff += dr*dr + di*di;
            norm_A += __real__(A_orig[i*lda + j]) * __real__(A_orig[i*lda + j])
                    + __imag__(A_orig[i*lda + j]) * __imag__(A_orig[i*lda + j]);
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_A < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_A));

    double ortho_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double re = 0.0, im = 0.0;
            for (int l = 0; l < m; l++) {
                double qir =  __real__(A_qr[l*lda + i]), qii = -__imag__(A_qr[l*lda + i]);
                double qjr =  __real__(A_qr[l*lda + j]), qji =  __imag__(A_qr[l*lda + j]);
                re += qir*qjr - qii*qji;
                im += qir*qji + qii*qjr;
            }
            double dr = re - (i == j ? 1.0 : 0.0), di = im;
            ortho_err += dr*dr + di*di;
        }
    }
    result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

    free(A_qr); free(tau); free(R);
    return FB_JUDGE_OK;
}
/* =========================================================================
 * SORGQR — generate Q from single-precision QR factorization
 * ========================================================================= */
static fb_judge_status_t run_sorgqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgeqrf || !oracle->sorgqr || !cand->sgeqrf || !cand->sorgqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    /* Oracle path: SGEQRF then SORGQR */
    float *A_oq = clone_matrix_f32(A_orig, m, n, lda);
    float *tau_o = (float *)malloc((size_t)k * sizeof(float));
    if (!A_oq || !tau_o) { free(A_oq); free(tau_o); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oq, lda, tau_o) != 0) {
        free(A_oq); free(tau_o); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    if (oracle->sorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_oq, lda, tau_o) != 0) {
        free(A_oq); free(tau_o); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* A_oq now holds Q_oracle (m x k columns within m x lda buffer) */

    /* Candidate path: SGEQRF, then timed SORGQR */
    float *A_cq = clone_matrix_f32(A_orig, m, n, lda);
    float *tau_c = (float *)malloc((size_t)k * sizeof(float));
    if (!A_cq || !tau_c) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (cand->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cq, lda, tau_c) != 0) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    /* Save post-GEQRF state for timing loops */
    float *A_saved = clone_matrix_f32(A_cq, m, n, lda);
    float *tau_saved = (float *)malloc((size_t)k * sizeof(float));
    if (!A_saved || !tau_saved) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        free(A_saved); free(tau_saved);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_saved, tau_c, (size_t)k * sizeof(float));

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(float));
            memcpy(tau_c, tau_saved, (size_t)k * sizeof(float));
            (void)cand->sorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(float));
            memcpy(tau_c, tau_saved, (size_t)k * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(float));
    memcpy(tau_c, tau_saved, (size_t)k * sizeof(float));
    free(A_saved); free(tau_saved);
    if (cand->sorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c) != 0) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    /* A_cq now holds Q_cand (m x k) */

    /* Reconstruction: ||Q_oracle - Q_cand||_F / ||Q_oracle||_F */
    double norm_diff = 0.0, norm_Q = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double diff = (double)A_oq[i * lda + j] - (double)A_cq[i * lda + j];
            norm_diff += diff * diff;
            double qo = (double)A_oq[i * lda + j];
            norm_Q += qo * qo;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_Q < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_Q));

    /* Orthogonality of Q_cand: ||Q^T Q - I||_F / sqrt(k) */
    double ortho_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double qtq = 0.0;
            for (int l = 0; l < m; l++)
                qtq += (double)A_cq[l * lda + i] * (double)A_cq[l * lda + j];
            double delta = qtq - (i == j ? 1.0 : 0.0);
            ortho_err += delta * delta;
        }
    }
    result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

    free(A_oq); free(tau_o); free(A_cq); free(tau_c);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CUNGQR — generate Q from complex single-precision QR factorization
 * ========================================================================= */
static fb_judge_status_t run_cungqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgeqrf || !oracle->cungqr || !cand->cgeqrf || !cand->cungqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig ||
            tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    int k = m < n ? m : n;

    /* Oracle path: CGEQRF then CUNGQR */
    fb_complex_float_t *A_oq = clone_matrix_cf32(A_orig, m, n, lda);
    fb_complex_float_t *tau_o =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!A_oq || !tau_o) { free(A_oq); free(tau_o); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oq, lda, tau_o) != 0) {
        free(A_oq); free(tau_o); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    if (oracle->cungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_oq, lda, tau_o) != 0) {
        free(A_oq); free(tau_o); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Candidate path */
    fb_complex_float_t *A_cq = clone_matrix_cf32(A_orig, m, n, lda);
    fb_complex_float_t *tau_c =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!A_cq || !tau_c) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (cand->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cq, lda, tau_c) != 0) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_saved = clone_matrix_cf32(A_cq, m, n, lda);
    fb_complex_float_t *tau_saved =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!A_saved || !tau_saved) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        free(A_saved); free(tau_saved);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_saved, tau_c, (size_t)k * sizeof(fb_complex_float_t));

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_cq, A_saved, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            memcpy(tau_c, tau_saved, (size_t)k*sizeof(fb_complex_float_t));
            (void)cand->cungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_cq, A_saved, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            memcpy(tau_c, tau_saved, (size_t)k*sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_cq, A_saved, (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
    memcpy(tau_c, tau_saved, (size_t)k*sizeof(fb_complex_float_t));
    free(A_saved); free(tau_saved);
    if (cand->cungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c) != 0) {
        free(A_oq); free(tau_o); free(A_cq); free(tau_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Reconstruction: ||Q_oracle - Q_cand||_F / ||Q_oracle||_F */
    double norm_diff = 0.0, norm_Q = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double dr = (double)(__real__ A_oq[i*lda+j]) - (double)(__real__ A_cq[i*lda+j]);
            double di = (double)(__imag__ A_oq[i*lda+j]) - (double)(__imag__ A_cq[i*lda+j]);
            norm_diff += dr*dr + di*di;
            double qr = (double)(__real__ A_oq[i*lda+j]);
            double qi = (double)(__imag__ A_oq[i*lda+j]);
            norm_Q += qr*qr + qi*qi;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_Q < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_Q));

    free(A_oq); free(tau_o); free(A_cq); free(tau_c);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SORMQR — apply Q (from QR factorization) to a matrix C from the left
 * ========================================================================= */
static fb_judge_status_t run_sormqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgeqrf || !oracle->sormqr || !cand->sgeqrf || !cand->sormqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    /* Use oracle SGEQRF to produce Householder vectors H and tau.
     * Both oracle and cand receive this same H+tau so we isolate SORMQR. */
    float *H = clone_matrix_f32(A_orig, m, n, lda);
    float *tau = (float *)malloc((size_t)k * sizeof(float));
    if (!H || !tau) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->sgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, H, lda, tau) != 0) {
        free(H); free(tau); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Build C matrix (m x n_c, ldc columns): use tc->B if available, else I */
    int ldc = lda;
    int n_c = n;
    size_t C_sz = (size_t)m * (size_t)ldc;
    float *C_template = (float *)calloc(C_sz, sizeof(float));
    if (!C_template) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (tc->B && tc->B_elems >= C_sz) {
        memcpy(C_template, tc->B, C_sz * sizeof(float));
    } else {
        /* Identity block: I_min where min = min(m,n_c) */
        int id_sz = m < n_c ? m : n_c;
        for (int i = 0; i < id_sz; i++)
            C_template[i * ldc + i] = 1.0f;
    }

    /* Oracle SORMQR */
    float *H_o = clone_matrix_f32(H, m, n, lda);
    float *tau_o = (float *)malloc((size_t)k * sizeof(float));
    float *C_o   = (float *)malloc(C_sz * sizeof(float));
    if (!H_o || !tau_o || !C_o) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_o, tau, (size_t)k * sizeof(float));
    memcpy(C_o, C_template, C_sz * sizeof(float));
    if (oracle->sormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                       m, n_c, k, H_o, lda, tau_o, C_o, ldc) != 0) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Candidate SORMQR */
    float *H_c   = clone_matrix_f32(H, m, n, lda);
    float *tau_c = (float *)malloc((size_t)k * sizeof(float));
    float *C_c   = (float *)malloc(C_sz * sizeof(float));
    if (!H_c || !tau_c || !C_c) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(float));
            memcpy(tau_c, tau,        (size_t)k*sizeof(float));
            memcpy(C_c,   C_template, C_sz*sizeof(float));
            (void)cand->sormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(float));
            memcpy(tau_c, tau,        (size_t)k*sizeof(float));
            memcpy(C_c,   C_template, C_sz*sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(float));
    memcpy(tau_c, tau,        (size_t)k*sizeof(float));
    memcpy(C_c,   C_template, C_sz*sizeof(float));
    if (cand->sormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                     m, n_c, k, H_c, lda, tau_c, C_c, ldc) != 0) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Reconstruction: ||C_oracle - C_cand||_F / ||C_oracle||_F */
    double norm_diff = 0.0, norm_C = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_c; j++) {
            double diff = (double)C_o[i*ldc+j] - (double)C_c[i*ldc+j];
            norm_diff += diff * diff;
            double co = (double)C_o[i*ldc+j];
            norm_C += co * co;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_C < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_C));

    free(H); free(tau); free(C_template);
    free(H_o); free(tau_o); free(C_o);
    free(H_c); free(tau_c); free(C_c);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * STRTRI — invert a single-precision upper triangular matrix in-place
 * ========================================================================= */
static fb_judge_status_t run_strtri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->strtri || !cand->strtri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;
    if (n <= 0 || !A_orig ||
            tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Oracle: invert upper-triangular, non-unit diagonal */
    float *A_oi = clone_matrix_f32(A_orig, n, n, lda);
    if (!A_oi) return FB_JUDGE_ERR_ALLOC;
    if (oracle->strtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_oi, lda) != 0) {
        free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    /* Candidate: invert */
    float *A_ci = clone_matrix_f32(A_orig, n, n, lda);
    if (!A_ci) { free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(float));
            (void)cand->strtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->strtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(float));
    if (cand->strtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda) != 0) {
        free(A_oi); free(A_ci); mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    /* Reconstruction: ||A_oi - A_ci||_F / ||A_oi||_F (upper triangle only) */
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double diff = (double)A_oi[i*lda+j] - (double)A_ci[i*lda+j];
            norm_diff += diff * diff;
            double ao = (double)A_oi[i*lda+j];
            norm_inv += ao * ao;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));

    free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}


/* =========================================================================
 * Dispatch
 * ========================================================================= */

/**
 * Dispatch table for FACTORIZATION archetype operations.
 * Indexed by op_id; covers LU, Cholesky, QR variants.
 */
static const fb_factorization_runner_fn fb_factorization_dispatch[] = {
    [FB_OP_SGETRF] = run_sgetrf,
    [FB_OP_DGETRF] = run_dgetrf,
    [FB_OP_CGETRF] = run_cgetrf,
    [FB_OP_ZGETRF] = run_zgetrf,
    [FB_OP_SPOTRF] = run_spotrf,
    [FB_OP_DPOTRF] = run_dpotrf,
    [FB_OP_CPOTRF] = run_cpotrf,
    [FB_OP_ZPOTRF] = run_zpotrf,
    [FB_OP_SGEQRF] = run_sgeqrf,
    [FB_OP_DGEQRF] = run_dgeqrf,
    [FB_OP_CGEQRF] = run_cgeqrf,
    [FB_OP_ZGEQRF] = run_zgeqrf,
    [FB_OP_SORGQR] = run_sorgqr,
    [FB_OP_CUNGQR] = run_cungqr,
    [FB_OP_SORMQR] = run_sormqr,
    [FB_OP_STRTRI] = run_strtri,
};

#define FB_FACTORIZATION_DISPATCH_SIZE \
    (sizeof(fb_factorization_dispatch) / sizeof(fb_factorization_dispatch[0]))

/* =========================================================================
 * Public entry point
 * ========================================================================= */

fb_judge_status_t fb_judge_run_factorization_case(
    const fb_backend_vtable_t       *oracle,
    const fb_backend_vtable_t       *cand,
    const fb_corpus_case_t          *tc,
    fb_judge_factorization_result_t *result_out,
    uint64_t                        *elapsed_ns_out)
{
    if (!oracle || !cand || !tc || !result_out)
        return FB_JUDGE_ERR_INVALID_OP;

    memset(result_out, 0, sizeof(*result_out));

    uint32_t op = tc->meta.op_id;
    if (op >= FB_FACTORIZATION_DISPATCH_SIZE || !fb_factorization_dispatch[op])
        return FB_JUDGE_ERR_NOT_IMPL;

    return fb_factorization_dispatch[op](oracle, cand, tc, result_out, elapsed_ns_out);
}