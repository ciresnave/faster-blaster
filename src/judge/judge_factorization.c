/**
 * @file judge_factorization.c
 * @brief FB_JUDGE_FACTORIZATION archetype evaluator — LU, Cholesky, QR, Hessenberg.
 *
 * Implements oracle-vs-candidate comparison for matrix factorizations with
 * two primary metrics: reconstruction residual and (for QR) orthogonality.
 *
 * Phase 3 coverage:
 *   LU (GETRF):      SGETRF, DGETRF
 *   Cholesky (POTRF): SPOTRF, DPOTRF
 *   QR (GEQRF):      SGEQRF, DGEQRF
 *   Hessenberg (GEHRD): SGEHRD, DGEHRD, CGEHRD, ZGEHRD
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
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Constants
 * ========================================================================= */

#define FB_FACTORIZATION_WARMUP_RUNS   2
#define FB_FACTORIZATION_TIMING_RUNS   5
#define FB_BAND_FACT_CALL_NOT_IMPL INT_MIN
#define FB_BAND_FACT_CALL_ALLOC_FAILURE (INT_MIN + 1)
#define FB_FACTORIZATION_CALL_ALLOC_FAILURE (INT_MIN + 2)

/* =========================================================================
 * Private utilities
 * ========================================================================= */

/** Clone an m × n matrix (row-major, leading dimension lda). */
static float *clone_matrix_f32(const float *A, int m, int n, int lda)
{
    (void)n;
    size_t sz = (size_t)m * (size_t)lda * sizeof(float);
    float *cpy = (float *)malloc(sz);
    if (!cpy) return NULL;
    memcpy(cpy, A, sz);
    return cpy;
}

/** Clone an m × n matrix (double precision). */
static double *clone_matrix_f64(const double *A, int m, int n, int lda)
{
    (void)n;
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

static fb_judge_status_t run_sgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out);
static fb_judge_status_t run_dgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out);
static fb_judge_status_t run_cgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out);
static fb_judge_status_t run_zgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out);

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
    if (!oracle->sgeqrf || !cand->sgeqrf || !cand->sorgqr) {
        return run_sgeqrf_fortran_fallback(oracle, cand, tc, res, ns_out);
    }

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
    if (!oracle->dgeqrf || !cand->dgeqrf || !cand->dorgqr) {
        return run_dgeqrf_fortran_fallback(oracle, cand, tc, res, ns_out);
    }

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

static int fb_factor_query_size_from_float(float query_size)
{
    return (query_size > 1.0f) ? (int)query_size : 1;
}

static int fb_factor_query_size_from_double(double query_size)
{
    return (query_size > 1.0) ? (int)query_size : 1;
}

static int fb_factor_query_size_from_cfloat(fb_complex_float_t query_size)
{
    float real_part = 0.0f;
    memcpy(&real_part, &query_size, sizeof(real_part));
    return (real_part > 1.0f) ? (int)real_part : 1;
}

static int fb_factor_query_size_from_cdouble(fb_complex_double_t query_size)
{
    double real_part = 0.0;
    memcpy(&real_part, &query_size, sizeof(real_part));
    return (real_part > 1.0) ? (int)real_part : 1;
}

typedef void (*fb_sgehrd_fortran_fn_t)(const int *n, const int *ilo,
                                       const int *ihi, float *a,
                                       const int *lda, float *tau,
                                       float *work, const int *lwork,
                                       int *info);
typedef void (*fb_dgehrd_fortran_fn_t)(const int *n, const int *ilo,
                                       const int *ihi, double *a,
                                       const int *lda, double *tau,
                                       double *work, const int *lwork,
                                       int *info);
typedef void (*fb_cgehrd_fortran_fn_t)(const int *n, const int *ilo,
                                       const int *ihi, fb_complex_float_t *a,
                                       const int *lda,
                                       fb_complex_float_t *tau,
                                       fb_complex_float_t *work,
                                       const int *lwork, int *info);
typedef void (*fb_zgehrd_fortran_fn_t)(const int *n, const int *ilo,
                                       const int *ihi, fb_complex_double_t *a,
                                       const int *lda,
                                       fb_complex_double_t *tau,
                                       fb_complex_double_t *work,
                                       const int *lwork, int *info);

static int call_sgehrd_backend(fb_sgehrd_fortran_fn_t fn, int n, int ilo,
                               int ihi, float *a, int lda, float *tau)
{
    float work_query = 0.0f;
    int lwork = -1;
    int info = 0;

    fn(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_float(work_query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_dgehrd_backend(fb_dgehrd_fortran_fn_t fn, int n, int ilo,
                               int ihi, double *a, int lda, double *tau)
{
    double work_query = 0.0;
    int lwork = -1;
    int info = 0;

    fn(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_double(work_query);
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_cgehrd_backend(fb_cgehrd_fortran_fn_t fn, int n, int ilo,
                               int ihi, fb_complex_float_t *a, int lda,
                               fb_complex_float_t *tau)
{
    fb_complex_float_t work_query = 0.0f;
    int lwork = -1;
    int info = 0;

    fn(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_cfloat(work_query);
    fb_complex_float_t *work =
        (fb_complex_float_t *)malloc((size_t)lwork * sizeof(fb_complex_float_t));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_zgehrd_backend(fb_zgehrd_fortran_fn_t fn, int n, int ilo,
                               int ihi, fb_complex_double_t *a, int lda,
                               fb_complex_double_t *tau)
{
    fb_complex_double_t work_query = 0.0;
    int lwork = -1;
    int info = 0;

    fn(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_cdouble(work_query);
    fb_complex_double_t *work =
        (fb_complex_double_t *)malloc((size_t)lwork * sizeof(fb_complex_double_t));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static double fb_gehrd_relerr_f32(const float *a_oracle, const float *a_cand,
                                  int n, int lda, const float *tau_oracle,
                                  const float *tau_cand, int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = (double)a_cand[i * lda + j] -
                          (double)a_oracle[i * lda + j];
            double ref = (double)a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = (double)tau_cand[i] - (double)tau_oracle[i];
        double ref = (double)tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_gehrd_relerr_f64(const double *a_oracle, const double *a_cand,
                                  int n, int lda, const double *tau_oracle,
                                  const double *tau_cand, int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = a_cand[i * lda + j] - a_oracle[i * lda + j];
            double ref = a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = tau_cand[i] - tau_oracle[i];
        double ref = tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_gehrd_relerr_cf32(const fb_complex_float_t *a_oracle,
                                   const fb_complex_float_t *a_cand,
                                   int n, int lda,
                                   const fb_complex_float_t *tau_oracle,
                                   const fb_complex_float_t *tau_cand,
                                   int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff_re = (double)__real__(a_cand[i * lda + j]) -
                             (double)__real__(a_oracle[i * lda + j]);
            double diff_im = (double)__imag__(a_cand[i * lda + j]) -
                             (double)__imag__(a_oracle[i * lda + j]);
            double ref_re = (double)__real__(a_oracle[i * lda + j]);
            double ref_im = (double)__imag__(a_oracle[i * lda + j]);
            diff2 += diff_re * diff_re + diff_im * diff_im;
            ref2 += ref_re * ref_re + ref_im * ref_im;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff_re = (double)__real__(tau_cand[i]) -
                         (double)__real__(tau_oracle[i]);
        double diff_im = (double)__imag__(tau_cand[i]) -
                         (double)__imag__(tau_oracle[i]);
        double ref_re = (double)__real__(tau_oracle[i]);
        double ref_im = (double)__imag__(tau_oracle[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }

    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_gehrd_relerr_cf64(const fb_complex_double_t *a_oracle,
                                   const fb_complex_double_t *a_cand,
                                   int n, int lda,
                                   const fb_complex_double_t *tau_oracle,
                                   const fb_complex_double_t *tau_cand,
                                   int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff_re = __real__(a_cand[i * lda + j]) -
                             __real__(a_oracle[i * lda + j]);
            double diff_im = __imag__(a_cand[i * lda + j]) -
                             __imag__(a_oracle[i * lda + j]);
            double ref_re = __real__(a_oracle[i * lda + j]);
            double ref_im = __imag__(a_oracle[i * lda + j]);
            diff2 += diff_re * diff_re + diff_im * diff_im;
            ref2 += ref_re * ref_re + ref_im * ref_im;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff_re = __real__(tau_cand[i]) - __real__(tau_oracle[i]);
        double diff_im = __imag__(tau_cand[i]) - __imag__(tau_oracle[i]);
        double ref_re = __real__(tau_oracle[i]);
        double ref_im = __imag__(tau_oracle[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }

    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static fb_judge_status_t run_sgehrd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGEHRD][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGEHRD][FB_CONV_FORTRAN];
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int ilo = 1;
    int ihi = n;
    int tau_len = (n > 1) ? (n - 1) : 0;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, n, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, n, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgehrd_backend((fb_sgehrd_fortran_fn_t)oracle_fn,
                                          n, ilo, ihi, a_oracle, lda,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)n * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, n, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)n * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            (void)call_sgehrd_backend((fb_sgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)n * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgehrd_backend((fb_sgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_sgehrd_backend((fb_sgehrd_fortran_fn_t)cand_fn, n,
                                        ilo, ihi, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)n * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_gehrd_relerr_f32(a_oracle, a_cand, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgehrd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGEHRD][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGEHRD][FB_CONV_FORTRAN];
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int ilo = 1;
    int ihi = n;
    int tau_len = (n > 1) ? (n - 1) : 0;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, n, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, n, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgehrd_backend((fb_dgehrd_fortran_fn_t)oracle_fn,
                                          n, ilo, ihi, a_oracle, lda,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)n * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, n, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)n * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            (void)call_dgehrd_backend((fb_dgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)n * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgehrd_backend((fb_dgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_dgehrd_backend((fb_dgehrd_fortran_fn_t)cand_fn, n,
                                        ilo, ihi, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)n * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_gehrd_relerr_f64(a_oracle, a_cand, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgehrd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_CGEHRD][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_CGEHRD][FB_CONV_FORTRAN];
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int ilo = 1;
    int ihi = n;
    int tau_len = (n > 1) ? (n - 1) : 0;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_float_t *a_orig = (const fb_complex_float_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *a_oracle = clone_matrix_cf32(a_orig, n, n, lda);
    fb_complex_float_t *a_cand = clone_matrix_cf32(a_orig, n, n, lda);
    fb_complex_float_t *tau_oracle =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    fb_complex_float_t *tau_cand =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_cgehrd_backend((fb_cgehrd_fortran_fn_t)oracle_fn,
                                          n, ilo, ihi, a_oracle, lda,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)n * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_float_t *a_tmp = clone_matrix_cf32(a_orig, n, n, lda);
        fb_complex_float_t *tau_tmp =
            (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            (void)call_cgehrd_backend((fb_cgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_cgehrd_backend((fb_cgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_cgehrd_backend((fb_cgehrd_fortran_fn_t)cand_fn, n,
                                        ilo, ihi, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)n * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_gehrd_relerr_cf32(a_oracle, a_cand, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgehrd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_ZGEHRD][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_ZGEHRD][FB_CONV_FORTRAN];
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int ilo = 1;
    int ihi = n;
    int tau_len = (n > 1) ? (n - 1) : 0;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_double_t *a_orig = (const fb_complex_double_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *a_oracle = clone_matrix_cf64(a_orig, n, n, lda);
    fb_complex_double_t *a_cand = clone_matrix_cf64(a_orig, n, n, lda);
    fb_complex_double_t *tau_oracle =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    fb_complex_double_t *tau_cand =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_zgehrd_backend((fb_zgehrd_fortran_fn_t)oracle_fn,
                                          n, ilo, ihi, a_oracle, lda,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)n * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_double_t *a_tmp = clone_matrix_cf64(a_orig, n, n, lda);
        fb_complex_double_t *tau_tmp =
            (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            (void)call_zgehrd_backend((fb_zgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_zgehrd_backend((fb_zgehrd_fortran_fn_t)cand_fn, n, ilo,
                                      ihi, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_zgehrd_backend((fb_zgehrd_fortran_fn_t)cand_fn, n,
                                        ilo, ihi, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)n * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_gehrd_relerr_cf64(a_oracle, a_cand, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

typedef void (*fb_sgeqlf_fortran_fn_t)(int *m, int *n, float *a, int *lda,
                                       float *tau, float *work, int *lwork,
                                       int *info);
typedef void (*fb_dgeqlf_fortran_fn_t)(int *m, int *n, double *a, int *lda,
                                       double *tau, double *work, int *lwork,
                                       int *info);
typedef void (*fb_cgeqlf_fortran_fn_t)(int *m, int *n,
                                       fb_complex_float_t *a, int *lda,
                                       fb_complex_float_t *tau,
                                       fb_complex_float_t *work, int *lwork,
                                       int *info);
typedef void (*fb_zgeqlf_fortran_fn_t)(int *m, int *n,
                                       fb_complex_double_t *a, int *lda,
                                       fb_complex_double_t *tau,
                                       fb_complex_double_t *work, int *lwork,
                                       int *info);

static int call_sgeqlf_backend(fb_sgeqlf_fortran_fn_t fn, int m, int n,
                               float *a, int lda, float *tau)
{
    float work_query = 0.0f;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_float(work_query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_dgeqlf_backend(fb_dgeqlf_fortran_fn_t fn, int m, int n,
                               double *a, int lda, double *tau)
{
    double work_query = 0.0;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_double(work_query);
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_cgeqlf_backend(fb_cgeqlf_fortran_fn_t fn, int m, int n,
                               fb_complex_float_t *a, int lda,
                               fb_complex_float_t *tau)
{
    fb_complex_float_t work_query = 0.0f;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_cfloat(work_query);
    fb_complex_float_t *work =
        (fb_complex_float_t *)malloc((size_t)lwork * sizeof(fb_complex_float_t));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_zgeqlf_backend(fb_zgeqlf_fortran_fn_t fn, int m, int n,
                               fb_complex_double_t *a, int lda,
                               fb_complex_double_t *tau)
{
    fb_complex_double_t work_query = 0.0;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_cdouble(work_query);
    fb_complex_double_t *work =
        (fb_complex_double_t *)malloc((size_t)lwork * sizeof(fb_complex_double_t));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}

static double fb_geqlf_relerr_f32(const float *a_oracle, const float *a_cand,
                                  int m, int n, int lda,
                                  const float *tau_oracle,
                                  const float *tau_cand, int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = (double)a_cand[i * lda + j] -
                          (double)a_oracle[i * lda + j];
            double ref = (double)a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = (double)tau_cand[i] - (double)tau_oracle[i];
        double ref = (double)tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_geqlf_relerr_f64(const double *a_oracle, const double *a_cand,
                                  int m, int n, int lda,
                                  const double *tau_oracle,
                                  const double *tau_cand, int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = a_cand[i * lda + j] - a_oracle[i * lda + j];
            double ref = a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = tau_cand[i] - tau_oracle[i];
        double ref = tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_geqlf_relerr_cf32(const fb_complex_float_t *a_oracle,
                                   const fb_complex_float_t *a_cand,
                                   int m, int n, int lda,
                                   const fb_complex_float_t *tau_oracle,
                                   const fb_complex_float_t *tau_cand,
                                   int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff_re = (double)__real__(a_cand[i * lda + j]) -
                             (double)__real__(a_oracle[i * lda + j]);
            double diff_im = (double)__imag__(a_cand[i * lda + j]) -
                             (double)__imag__(a_oracle[i * lda + j]);
            double ref_re = (double)__real__(a_oracle[i * lda + j]);
            double ref_im = (double)__imag__(a_oracle[i * lda + j]);
            diff2 += diff_re * diff_re + diff_im * diff_im;
            ref2 += ref_re * ref_re + ref_im * ref_im;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff_re = (double)__real__(tau_cand[i]) -
                         (double)__real__(tau_oracle[i]);
        double diff_im = (double)__imag__(tau_cand[i]) -
                         (double)__imag__(tau_oracle[i]);
        double ref_re = (double)__real__(tau_oracle[i]);
        double ref_im = (double)__imag__(tau_oracle[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }

    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_geqlf_relerr_cf64(const fb_complex_double_t *a_oracle,
                                   const fb_complex_double_t *a_cand,
                                   int m, int n, int lda,
                                   const fb_complex_double_t *tau_oracle,
                                   const fb_complex_double_t *tau_cand,
                                   int tau_len)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff_re = __real__(a_cand[i * lda + j]) -
                             __real__(a_oracle[i * lda + j]);
            double diff_im = __imag__(a_cand[i * lda + j]) -
                             __imag__(a_oracle[i * lda + j]);
            double ref_re = __real__(a_oracle[i * lda + j]);
            double ref_im = __imag__(a_oracle[i * lda + j]);
            diff2 += diff_re * diff_re + diff_im * diff_im;
            ref2 += ref_re * ref_re + ref_im * ref_im;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff_re = __real__(tau_cand[i]) - __real__(tau_oracle[i]);
        double diff_im = __imag__(tau_cand[i]) - __imag__(tau_oracle[i]);
        double ref_re = __real__(tau_oracle[i]);
        double ref_im = __imag__(tau_oracle[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }

    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static fb_judge_status_t run_sgeqlf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGEQLF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGEQLF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeqlf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGEQLF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGEQLF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgelqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGELQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGELQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgelqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGELQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGELQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeqlf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_CGEQLF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_CGEQLF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_float_t *a_orig = (const fb_complex_float_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *a_oracle = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *a_cand = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *tau_oracle =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    fb_complex_float_t *tau_cand =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_float_t *a_tmp = clone_matrix_cf32(a_orig, m, n, lda);
        fb_complex_float_t *tau_tmp =
            (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf32(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeqlf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_ZGEQLF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_ZGEQLF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_double_t *a_orig = (const fb_complex_double_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *a_oracle = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *a_cand = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *tau_oracle =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    fb_complex_double_t *tau_cand =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_double_t *a_tmp = clone_matrix_cf64(a_orig, m, n, lda);
        fb_complex_double_t *tau_tmp =
            (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf64(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgerqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGERQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGERQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgerqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGERQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGERQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgerqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_CGERQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_CGERQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_float_t *a_orig = (const fb_complex_float_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *a_oracle = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *a_cand = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *tau_oracle =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    fb_complex_float_t *tau_cand =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_float_t *a_tmp = clone_matrix_cf32(a_orig, m, n, lda);
        fb_complex_float_t *tau_tmp =
            (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf32(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgerqf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_ZGERQF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_ZGERQF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_double_t *a_orig = (const fb_complex_double_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *a_oracle = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *a_cand = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *tau_oracle =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    fb_complex_double_t *tau_cand =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_double_t *a_tmp = clone_matrix_cf64(a_orig, m, n, lda);
        fb_complex_double_t *tau_tmp =
            (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf64(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGEQRF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGEQRF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_sgeqlf_backend((fb_sgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGEQRF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGEQRF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_dgeqlf_backend((fb_dgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_CGEQRF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_CGEQRF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_float_t *a_orig = (const fb_complex_float_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *a_oracle = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *a_cand = clone_matrix_cf32(a_orig, m, n, lda);
    fb_complex_float_t *tau_oracle =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    fb_complex_float_t *tau_cand =
        (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_float_t *a_tmp = clone_matrix_cf32(a_orig, m, n, lda);
        fb_complex_float_t *tau_tmp =
            (fb_complex_float_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_float_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_float_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_cgeqlf_backend((fb_cgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf32(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeqrf_fortran_fallback(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_ZGEQRF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_ZGEQRF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const fb_complex_double_t *a_orig = (const fb_complex_double_t *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *a_oracle = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *a_cand = clone_matrix_cf64(a_orig, m, n, lda);
    fb_complex_double_t *tau_oracle =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    fb_complex_double_t *tau_cand =
        (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fb_complex_double_t *a_tmp = clone_matrix_cf64(a_orig, m, n, lda);
        fb_complex_double_t *tau_tmp =
            (fb_complex_double_t *)calloc((size_t)tau_alloc, sizeof(fb_complex_double_t));
        if (!a_tmp || !tau_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(a_tmp); free(tau_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig,
                   (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
            memset(tau_tmp, 0,
                   (size_t)tau_alloc * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp);
    }

    int info_cand = call_zgeqlf_backend((fb_zgeqlf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_CF64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_CF64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqlf_relerr_cf64(a_oracle, a_cand, m, n, lda,
                                            tau_oracle, tau_cand, tau_len));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    return FB_JUDGE_OK;
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

static int fb_band_test_lower_width(int n)
{
    return n > 1 ? 1 : 0;
}

static int fb_band_test_upper_width(int n)
{
    return n > 1 ? 1 : 0;
}

static double fb_cf32_abs(fb_complex_float_t value)
{
    double real_part = (double)__real__(value);
    double imag_part = (double)__imag__(value);
    return sqrt(real_part * real_part + imag_part * imag_part);
}

static double fb_cf64_abs(fb_complex_double_t value)
{
    double real_part = __real__(value);
    double imag_part = __imag__(value);
    return sqrt(real_part * real_part + imag_part * imag_part);
}

static void fb_fill_band_dense_f32(float *A, int n, int lda, int kl, int ku)
{
    memset(A, 0, (size_t)n * (size_t)lda * sizeof(*A));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            A[(size_t)i * (size_t)lda + (size_t)j] = (float)((j > i) ? -scale : scale);
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fabs((double)A[(size_t)i * (size_t)lda + (size_t)j]);
        }
        A[(size_t)i * (size_t)lda + (size_t)i] = (float)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

static void fb_fill_band_dense_f64(double *A, int n, int lda, int kl, int ku)
{
    memset(A, 0, (size_t)n * (size_t)lda * sizeof(*A));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            A[(size_t)i * (size_t)lda + (size_t)j] = (j > i) ? -scale : scale;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fabs(A[(size_t)i * (size_t)lda + (size_t)j]);
        }
        A[(size_t)i * (size_t)lda + (size_t)i] = row_sum + 8.0 + 0.0625 * (double)(i + 1);
    }
}

static void fb_fill_band_dense_cf32(fb_complex_float_t *A, int n, int lda, int kl, int ku)
{
    memset(A, 0, (size_t)n * (size_t)lda * sizeof(*A));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            float scale = 0.125f / (float)(abs(j - i));
            fb_complex_float_t value = 0.0f;
            __real__ value = (j > i) ? -scale : scale;
            __imag__ value = (i > j) ? 0.03125f * scale : -0.03125f * scale;
            A[(size_t)i * (size_t)lda + (size_t)j] = value;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fb_cf32_abs(A[(size_t)i * (size_t)lda + (size_t)j]);
        }
        A[(size_t)i * (size_t)lda + (size_t)i] = (fb_complex_float_t)(float)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

static void fb_fill_band_dense_cf64(fb_complex_double_t *A, int n, int lda, int kl, int ku)
{
    memset(A, 0, (size_t)n * (size_t)lda * sizeof(*A));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            fb_complex_double_t value = 0.0;
            __real__ value = (j > i) ? -scale : scale;
            __imag__ value = (i > j) ? 0.03125 * scale : -0.03125 * scale;
            A[(size_t)i * (size_t)lda + (size_t)j] = value;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fb_cf64_abs(A[(size_t)i * (size_t)lda + (size_t)j]);
        }
        A[(size_t)i * (size_t)lda + (size_t)i] = (fb_complex_double_t)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

#define FB_DEFINE_PACK_BAND_ROW_MAJOR(name, scalar_t) \
static void name(const scalar_t *A, int n, int lda, int kl, int ku, scalar_t *ab, int ldab) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    memset(ab, 0, (size_t)band_rows * (size_t)ldab * sizeof(*ab)); \
    for (int j = 0; j < n; j++) { \
        int i0 = j - ku; \
        int i1 = j + kl; \
        if (i0 < 0) i0 = 0; \
        if (i1 >= n) i1 = n - 1; \
        for (int i = i0; i <= i1; i++) { \
            int band_row = kl + ku + i - j; \
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] = A[(size_t)i * (size_t)lda + (size_t)j]; \
        } \
    } \
}

#define FB_DEFINE_PACK_BAND_COL_MAJOR(name, scalar_t) \
static void name(const scalar_t *A, int n, int lda, int kl, int ku, scalar_t *ab, int ldab) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    memset(ab, 0, (size_t)band_rows * (size_t)n * sizeof(*ab)); \
    for (int j = 0; j < n; j++) { \
        int i0 = j - ku; \
        int i1 = j + kl; \
        if (i0 < 0) i0 = 0; \
        if (i1 >= n) i1 = n - 1; \
        for (int i = i0; i <= i1; i++) { \
            int band_row = kl + ku + i - j; \
            ab[(size_t)band_row + (size_t)j * (size_t)ldab] = A[(size_t)i * (size_t)lda + (size_t)j]; \
        } \
    } \
}

#define FB_DEFINE_COL_TO_ROW_COPY(name, scalar_t) \
static void name(const scalar_t *src, int rows, int cols, scalar_t *dst) \
{ \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) { \
            dst[(size_t)i * (size_t)cols + (size_t)j] = src[(size_t)i + (size_t)j * (size_t)rows]; \
        } \
    } \
}

FB_DEFINE_PACK_BAND_ROW_MAJOR(fb_pack_band_row_major_f32, float)
FB_DEFINE_PACK_BAND_ROW_MAJOR(fb_pack_band_row_major_f64, double)
FB_DEFINE_PACK_BAND_ROW_MAJOR(fb_pack_band_row_major_cf32, fb_complex_float_t)
FB_DEFINE_PACK_BAND_ROW_MAJOR(fb_pack_band_row_major_cf64, fb_complex_double_t)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_f32, float)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_f64, double)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_cf32, fb_complex_float_t)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_cf64, fb_complex_double_t)
FB_DEFINE_COL_TO_ROW_COPY(fb_copy_col_to_row_f32, float)
FB_DEFINE_COL_TO_ROW_COPY(fb_copy_col_to_row_f64, double)
FB_DEFINE_COL_TO_ROW_COPY(fb_copy_col_to_row_cf32, fb_complex_float_t)
FB_DEFINE_COL_TO_ROW_COPY(fb_copy_col_to_row_cf64, fb_complex_double_t)

typedef int (*fb_sgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              float *ab, int ldab, int *ipiv);
typedef int (*fb_dgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              double *ab, int ldab, int *ipiv);
typedef int (*fb_cgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              fb_complex_float_t *ab, int ldab, int *ipiv);
typedef int (*fb_zgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              fb_complex_double_t *ab, int ldab, int *ipiv);

typedef void (*fb_sgbtrf_fortran_fn_t)(const int *m, const int *n,
                                       const int *kl, const int *ku,
                                       float *ab, const int *ldab,
                                       int *ipiv, int *info);
typedef void (*fb_dgbtrf_fortran_fn_t)(const int *m, const int *n,
                                       const int *kl, const int *ku,
                                       double *ab, const int *ldab,
                                       int *ipiv, int *info);
typedef void (*fb_cgbtrf_fortran_fn_t)(const int *m, const int *n,
                                       const int *kl, const int *ku,
                                       fb_complex_float_t *ab,
                                       const int *ldab, int *ipiv,
                                       int *info);
typedef void (*fb_zgbtrf_fortran_fn_t)(const int *m, const int *n,
                                       const int *kl, const int *ku,
                                       fb_complex_double_t *ab,
                                       const int *ldab, int *ipiv,
                                       int *info);

#define FB_DEFINE_GBTRF_CALLER(name, cblas_t, fortran_t, scalar_t, pack_row_fn, pack_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, const scalar_t *A_dense, int n, int kl, int ku, scalar_t *lu_row, int *ipiv) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        pack_row_fn(A_dense, n, n, kl, ku, lu_row, n); \
        return ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, n, n, kl, ku, lu_row, n, ipiv); \
    } \
    if (fortran_fn != NULL) { \
        scalar_t *lu_col = (scalar_t *)malloc((size_t)band_rows * (size_t)n * sizeof(scalar_t)); \
        if (!lu_col) return FB_BAND_FACT_CALL_ALLOC_FAILURE; \
        int n_ = n, kl_ = kl, ku_ = ku, ldab_ = band_rows, info = 0; \
        pack_col_fn(A_dense, n, n, kl, ku, lu_col, ldab_); \
        ((fortran_t)fortran_fn)(&n_, &n_, &kl_, &ku_, lu_col, &ldab_, ipiv, &info); \
        if (info == 0) col_to_row_fn(lu_col, band_rows, n, lu_row); \
        free(lu_col); \
        return info; \
    } \
    return FB_BAND_FACT_CALL_NOT_IMPL; \
}

FB_DEFINE_GBTRF_CALLER(fb_call_sgbtrf, fb_sgbtrf_fn_t, fb_sgbtrf_fortran_fn_t, float,
                       fb_pack_band_row_major_f32, fb_pack_band_col_major_f32, fb_copy_col_to_row_f32)
FB_DEFINE_GBTRF_CALLER(fb_call_dgbtrf, fb_dgbtrf_fn_t, fb_dgbtrf_fortran_fn_t, double,
                       fb_pack_band_row_major_f64, fb_pack_band_col_major_f64, fb_copy_col_to_row_f64)
FB_DEFINE_GBTRF_CALLER(fb_call_cgbtrf, fb_cgbtrf_fn_t, fb_cgbtrf_fortran_fn_t, fb_complex_float_t,
                       fb_pack_band_row_major_cf32, fb_pack_band_col_major_cf32, fb_copy_col_to_row_cf32)
FB_DEFINE_GBTRF_CALLER(fb_call_zgbtrf, fb_zgbtrf_fn_t, fb_zgbtrf_fortran_fn_t, fb_complex_double_t,
                       fb_pack_band_row_major_cf64, fb_pack_band_col_major_cf64, fb_copy_col_to_row_cf64)

static bool fb_ipiv_equal(const int *lhs, const int *rhs, int n)
{
    for (int i = 0; i < n; i++) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

typedef void (*fb_sgeqpf_fortran_fn_t)(const int *m, const int *n, float *a,
                                       const int *lda, int *jpvt, float *tau,
                                       float *work, int *info);
typedef void (*fb_dgeqpf_fortran_fn_t)(const int *m, const int *n, double *a,
                                       const int *lda, int *jpvt, double *tau,
                                       double *work, int *info);
typedef void (*fb_sgeqp3_fortran_fn_t)(const int *m, const int *n, float *a,
                                       const int *lda, int *jpvt, float *tau,
                                       float *work, const int *lwork,
                                       int *info);
typedef void (*fb_dgeqp3_fortran_fn_t)(const int *m, const int *n, double *a,
                                       const int *lda, int *jpvt, double *tau,
                                       double *work, const int *lwork,
                                       int *info);

static int call_sgeqpf_backend(fb_sgeqpf_fortran_fn_t fn, int m, int n,
                               float *a, int lda, int *jpvt, float *tau)
{
    size_t work_count = (n > 0) ? ((size_t)3 * (size_t)n + 1u) : 1u;
    float *work = (float *)calloc(work_count, sizeof(float));
    int info = 0;

    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;
    fn(&m, &n, a, &lda, jpvt, tau, work, &info);
    free(work);
    return info;
}

static int call_dgeqpf_backend(fb_dgeqpf_fortran_fn_t fn, int m, int n,
                               double *a, int lda, int *jpvt, double *tau)
{
    size_t work_count = (n > 0) ? ((size_t)3 * (size_t)n + 1u) : 1u;
    double *work = (double *)calloc(work_count, sizeof(double));
    int info = 0;

    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;
    fn(&m, &n, a, &lda, jpvt, tau, work, &info);
    free(work);
    return info;
}

static int call_sgeqp3_backend(fb_sgeqp3_fortran_fn_t fn, int m, int n,
                               float *a, int lda, int *jpvt, float *tau)
{
    float work_query = 0.0f;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, jpvt, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_float(work_query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
    free(work);
    return info;
}

static int call_dgeqp3_backend(fb_dgeqp3_fortran_fn_t fn, int m, int n,
                               double *a, int lda, int *jpvt, double *tau)
{
    double work_query = 0.0;
    int lwork = -1;
    int info = 0;

    fn(&m, &n, a, &lda, jpvt, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = fb_factor_query_size_from_double(work_query);
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    if (!work) return FB_FACTORIZATION_CALL_ALLOC_FAILURE;

    fn(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
    free(work);
    return info;
}

static double fb_geqp3_relerr_f32(const float *a_oracle, const float *a_cand,
                                  int m, int n, int lda,
                                  const float *tau_oracle,
                                  const float *tau_cand, int tau_len,
                                  const int *jpvt_oracle,
                                  const int *jpvt_cand)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = (double)a_cand[i * lda + j] -
                          (double)a_oracle[i * lda + j];
            double ref = (double)a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = (double)tau_cand[i] - (double)tau_oracle[i];
        double ref = (double)tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (!fb_ipiv_equal(jpvt_oracle, jpvt_cand, n)) return 1.0;
    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_geqp3_relerr_f64(const double *a_oracle, const double *a_cand,
                                  int m, int n, int lda,
                                  const double *tau_oracle,
                                  const double *tau_cand, int tau_len,
                                  const int *jpvt_oracle,
                                  const int *jpvt_cand)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double diff = a_cand[i * lda + j] - a_oracle[i * lda + j];
            double ref = a_oracle[i * lda + j];
            diff2 += diff * diff;
            ref2 += ref * ref;
        }
    }
    for (int i = 0; i < tau_len; i++) {
        double diff = tau_cand[i] - tau_oracle[i];
        double ref = tau_oracle[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (!fb_ipiv_equal(jpvt_oracle, jpvt_cand, n)) return 1.0;
    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static fb_judge_status_t run_sgeqpf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGEQPF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGEQPF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    int *jpvt_oracle = (int *)calloc((size_t)n, sizeof(int));
    int *jpvt_cand = (int *)calloc((size_t)n, sizeof(int));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand || !jpvt_oracle || !jpvt_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqpf_backend((fb_sgeqpf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, jpvt_oracle,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        int *jpvt_tmp = (int *)calloc((size_t)n, sizeof(int));
        if (!a_tmp || !tau_tmp || !jpvt_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(jpvt_oracle); free(jpvt_cand);
            free(a_tmp); free(tau_tmp); free(jpvt_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            (void)call_sgeqpf_backend((fb_sgeqpf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqpf_backend((fb_sgeqpf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp); free(jpvt_tmp);
    }

    int info_cand = call_sgeqpf_backend((fb_sgeqpf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, jpvt_cand,
                                        tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqp3_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len,
                                           jpvt_oracle, jpvt_cand));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    free(jpvt_oracle); free(jpvt_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeqpf(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGEQPF][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGEQPF][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    int *jpvt_oracle = (int *)calloc((size_t)n, sizeof(int));
    int *jpvt_cand = (int *)calloc((size_t)n, sizeof(int));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand || !jpvt_oracle || !jpvt_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqpf_backend((fb_dgeqpf_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, jpvt_oracle,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        int *jpvt_tmp = (int *)calloc((size_t)n, sizeof(int));
        if (!a_tmp || !tau_tmp || !jpvt_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(jpvt_oracle); free(jpvt_cand);
            free(a_tmp); free(tau_tmp); free(jpvt_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            (void)call_dgeqpf_backend((fb_dgeqpf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqpf_backend((fb_dgeqpf_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp); free(jpvt_tmp);
    }

    int info_cand = call_dgeqpf_backend((fb_dgeqpf_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, jpvt_cand,
                                        tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqp3_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len,
                                           jpvt_oracle, jpvt_cand));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    free(jpvt_oracle); free(jpvt_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgeqp3(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_SGEQP3][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_SGEQP3][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const float *a_orig = (const float *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    float *a_oracle = clone_matrix_f32(a_orig, m, n, lda);
    float *a_cand = clone_matrix_f32(a_orig, m, n, lda);
    float *tau_oracle = (float *)calloc((size_t)tau_alloc, sizeof(float));
    float *tau_cand = (float *)calloc((size_t)tau_alloc, sizeof(float));
    int *jpvt_oracle = (int *)calloc((size_t)n, sizeof(int));
    int *jpvt_cand = (int *)calloc((size_t)n, sizeof(int));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand || !jpvt_oracle || !jpvt_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_sgeqp3_backend((fb_sgeqp3_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, jpvt_oracle,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        float *a_tmp = clone_matrix_f32(a_orig, m, n, lda);
        float *tau_tmp = (float *)calloc((size_t)tau_alloc, sizeof(float));
        int *jpvt_tmp = (int *)calloc((size_t)n, sizeof(int));
        if (!a_tmp || !tau_tmp || !jpvt_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(jpvt_oracle); free(jpvt_cand);
            free(a_tmp); free(tau_tmp); free(jpvt_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            (void)call_sgeqp3_backend((fb_sgeqp3_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(float));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(float));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_sgeqp3_backend((fb_sgeqp3_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp); free(jpvt_tmp);
    }

    int info_cand = call_sgeqp3_backend((fb_sgeqp3_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, jpvt_cand,
                                        tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F32) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F32))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqp3_relerr_f32(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len,
                                           jpvt_oracle, jpvt_cand));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    free(jpvt_oracle); free(jpvt_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeqp3(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    fb_generic_fn oracle_fn = oracle->ext_ops[FB_OP_DGEQP3][FB_CONV_FORTRAN];
    fb_generic_fn cand_fn = cand->ext_ops[FB_OP_DGEQP3][FB_CONV_FORTRAN];
    int m = (int)tc->m;
    int n = (int)tc->n;
    int lda = (int)tc->lda;
    int tau_len = (m < n) ? m : n;
    int tau_alloc = (tau_len > 0) ? tau_len : 1;
    const double *a_orig = (const double *)tc->A;

    if (!oracle_fn || !cand_fn) return FB_JUDGE_ERR_NOT_IMPL;
    if (m <= 0 || n <= 0 || lda < n || !a_orig || tc->A_elems < (size_t)m * (size_t)lda) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    double *a_oracle = clone_matrix_f64(a_orig, m, n, lda);
    double *a_cand = clone_matrix_f64(a_orig, m, n, lda);
    double *tau_oracle = (double *)calloc((size_t)tau_alloc, sizeof(double));
    double *tau_cand = (double *)calloc((size_t)tau_alloc, sizeof(double));
    int *jpvt_oracle = (int *)calloc((size_t)n, sizeof(int));
    int *jpvt_cand = (int *)calloc((size_t)n, sizeof(int));
    if (!a_oracle || !a_cand || !tau_oracle || !tau_cand || !jpvt_oracle || !jpvt_cand) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }

    int info_oracle = call_dgeqp3_backend((fb_dgeqp3_fortran_fn_t)oracle_fn,
                                          m, n, a_oracle, lda, jpvt_oracle,
                                          tau_oracle);
    if (info_oracle == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_oracle != 0 ||
        fb_judge_has_nan_inf(a_oracle, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_oracle, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        double *a_tmp = clone_matrix_f64(a_orig, m, n, lda);
        double *tau_tmp = (double *)calloc((size_t)tau_alloc, sizeof(double));
        int *jpvt_tmp = (int *)calloc((size_t)n, sizeof(int));
        if (!a_tmp || !tau_tmp || !jpvt_tmp) {
            free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
            free(jpvt_oracle); free(jpvt_cand);
            free(a_tmp); free(tau_tmp); free(jpvt_tmp);
            return FB_JUDGE_ERR_ALLOC;
        }
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            (void)call_dgeqp3_backend((fb_dgeqp3_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(a_tmp, a_orig, (size_t)m * (size_t)lda * sizeof(double));
            memset(tau_tmp, 0, (size_t)tau_alloc * sizeof(double));
            memset(jpvt_tmp, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            (void)call_dgeqp3_backend((fb_dgeqp3_fortran_fn_t)cand_fn,
                                      m, n, a_tmp, lda, jpvt_tmp, tau_tmp);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
        free(a_tmp); free(tau_tmp); free(jpvt_tmp);
    }

    int info_cand = call_dgeqp3_backend((fb_dgeqp3_fortran_fn_t)cand_fn,
                                        m, n, a_cand, lda, jpvt_cand,
                                        tau_cand);
    if (info_cand == FB_FACTORIZATION_CALL_ALLOC_FAILURE) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info_cand != 0 ||
        fb_judge_has_nan_inf(a_cand, (size_t)m * (size_t)lda, FB_DTYPE_F64) ||
        (tau_len > 0 && fb_judge_has_nan_inf(tau_cand, (size_t)tau_len, FB_DTYPE_F64))) {
        free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
        free(jpvt_oracle); free(jpvt_cand);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    result_from_relerr(&res->reconstruction,
                       fb_geqp3_relerr_f64(a_oracle, a_cand, m, n, lda,
                                           tau_oracle, tau_cand, tau_len,
                                           jpvt_oracle, jpvt_cand));
    res->orthogonality = (fb_judge_case_result_t){
        .digits = 16.0,
        .relative_error = 0.0,
        .is_fatal = false,
        .is_oracle_fatal = false
    };

    free(a_oracle); free(a_cand); free(tau_oracle); free(tau_cand);
    free(jpvt_oracle); free(jpvt_cand);
    return FB_JUDGE_OK;
}

static double fb_band_buffer_relerr_f32(const float *oracle_lu, const float *cand_lu, int count)
{
    double diff2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = (double)cand_lu[i] - (double)oracle_lu[i];
        double ref = (double)oracle_lu[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }
    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_band_buffer_relerr_f64(const double *oracle_lu, const double *cand_lu, int count)
{
    double diff2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = cand_lu[i] - oracle_lu[i];
        double ref = oracle_lu[i];
        diff2 += diff * diff;
        ref2 += ref * ref;
    }
    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_band_buffer_relerr_cf32(const fb_complex_float_t *oracle_lu,
                                         const fb_complex_float_t *cand_lu,
                                         int count)
{
    double diff2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < count; i++) {
        double diff_re = (double)__real__(cand_lu[i]) - (double)__real__(oracle_lu[i]);
        double diff_im = (double)__imag__(cand_lu[i]) - (double)__imag__(oracle_lu[i]);
        double ref_re = (double)__real__(oracle_lu[i]);
        double ref_im = (double)__imag__(oracle_lu[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }
    if (ref2 < (double)FLT_EPSILON) return (diff2 < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

static double fb_band_buffer_relerr_cf64(const fb_complex_double_t *oracle_lu,
                                         const fb_complex_double_t *cand_lu,
                                         int count)
{
    double diff2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < count; i++) {
        double diff_re = __real__(cand_lu[i]) - __real__(oracle_lu[i]);
        double diff_im = __imag__(cand_lu[i]) - __imag__(oracle_lu[i]);
        double ref_re = __real__(oracle_lu[i]);
        double ref_im = __imag__(oracle_lu[i]);
        diff2 += diff_re * diff_re + diff_im * diff_im;
        ref2 += ref_re * ref_re + ref_im * ref_im;
    }
    if (ref2 < DBL_EPSILON) return (diff2 < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(diff2 / ref2);
}

#define FB_DEFINE_GBTRF_RUNNER(name, op_id, call_fn, scalar_t, fill_fn, relerr_fn) \
static fb_judge_status_t name(\
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,\
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,\
    uint64_t *ns_out)\
{\
    fb_generic_fn oracle_cblas = oracle->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_fortran = oracle->ext_ops[op_id][FB_CONV_FORTRAN];\
    fb_generic_fn cand_cblas = cand->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn cand_fortran = cand->ext_ops[op_id][FB_CONV_FORTRAN];\
    if ((oracle_cblas == NULL && oracle_fortran == NULL) || (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;\
    int n = (int)tc->n;\
    if (n <= 0) { mark_oracle_fatal(res); return FB_JUDGE_OK; }\
    int kl = fb_band_test_lower_width(n);\
    int ku = fb_band_test_upper_width(n);\
    int band_rows = 2 * kl + ku + 1;\
    size_t dense_sz = (size_t)n * (size_t)n * sizeof(scalar_t);\
    size_t band_sz = (size_t)band_rows * (size_t)n * sizeof(scalar_t);\
    scalar_t *Aorig = (scalar_t *)malloc(dense_sz);\
    scalar_t *LUo = (scalar_t *)malloc(band_sz);\
    scalar_t *LUc = (scalar_t *)malloc(band_sz);\
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));\
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));\
    if (!Aorig || !LUo || !LUc || !ipiv_o || !ipiv_c) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_ALLOC;\
    }\
    fill_fn(Aorig, n, n, kl, ku);\
    int info_o = call_fn(oracle_cblas, oracle_fortran, Aorig, n, kl, ku, LUo, ipiv_o);\
    if (info_o == FB_BAND_FACT_CALL_ALLOC_FAILURE) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_ALLOC;\
    }\
    if (info_o == FB_BAND_FACT_CALL_NOT_IMPL) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_NOT_IMPL;\
    }\
    if (info_o != 0) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); mark_oracle_fatal(res); return FB_JUDGE_OK;\
    }\
    if (ns_out) {\
        uint64_t best = UINT64_MAX;\
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {\
            int info_w = call_fn(cand_cblas, cand_fortran, Aorig, n, kl, ku, LUc, ipiv_c);\
            if (info_w == FB_BAND_FACT_CALL_ALLOC_FAILURE) { free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_ALLOC; }\
        }\
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {\
            uint64_t t0 = fb_judge_time_ns();\
            int info_t = call_fn(cand_cblas, cand_fortran, Aorig, n, kl, ku, LUc, ipiv_c);\
            if (info_t == FB_BAND_FACT_CALL_ALLOC_FAILURE) { free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_ALLOC; }\
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;\
        }\
        *ns_out = best;\
    }\
    int info_c = call_fn(cand_cblas, cand_fortran, Aorig, n, kl, ku, LUc, ipiv_c);\
    if (info_c == FB_BAND_FACT_CALL_ALLOC_FAILURE) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_ALLOC;\
    }\
    if (info_c == FB_BAND_FACT_CALL_NOT_IMPL) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); return FB_JUDGE_ERR_NOT_IMPL;\
    }\
    if (info_c != 0) {\
        free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c); mark_cand_fatal(res); return FB_JUDGE_OK;\
    }\
    double relerr = relerr_fn(LUo, LUc, band_rows * n);\
    if (!fb_ipiv_equal(ipiv_o, ipiv_c, n)) relerr = 1.0;\
    result_from_relerr(&res->reconstruction, relerr);\
    res->orthogonality.digits = 16.0;\
    res->orthogonality.relative_error = 0.0;\
    res->orthogonality.is_fatal = false;\
    res->orthogonality.is_oracle_fatal = false;\
    free(Aorig); free(LUo); free(LUc); free(ipiv_o); free(ipiv_c);\
    return FB_JUDGE_OK;\
}

FB_DEFINE_GBTRF_RUNNER(run_sgbtrf, FB_OP_SGBTRF, fb_call_sgbtrf, float, fb_fill_band_dense_f32, fb_band_buffer_relerr_f32)
FB_DEFINE_GBTRF_RUNNER(run_dgbtrf, FB_OP_DGBTRF, fb_call_dgbtrf, double, fb_fill_band_dense_f64, fb_band_buffer_relerr_f64)
FB_DEFINE_GBTRF_RUNNER(run_cgbtrf, FB_OP_CGBTRF, fb_call_cgbtrf, fb_complex_float_t, fb_fill_band_dense_cf32, fb_band_buffer_relerr_cf32)
FB_DEFINE_GBTRF_RUNNER(run_zgbtrf, FB_OP_ZGBTRF, fb_call_zgbtrf, fb_complex_double_t, fb_fill_band_dense_cf64, fb_band_buffer_relerr_cf64)

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
    if (!oracle->cgeqrf || !cand->cgeqrf || !cand->cungqr) {
        return run_cgeqrf_fortran_fallback(oracle, cand, tc, res, ns_out);
    }

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
    if (!oracle->zgeqrf || !cand->zgeqrf || !cand->zungqr) {
        return run_zgeqrf_fortran_fallback(oracle, cand, tc, res, ns_out);
    }

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
 * DORGQR — generate Q from double-precision QR factorization
 * ========================================================================= */
static fb_judge_status_t run_dorgqr(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_factorization_result_t *res,
                                    uint64_t *ns_out) {
  if (!oracle->dgeqrf || !oracle->dorgqr || !cand->dgeqrf || !cand->dorgqr)
    return FB_JUDGE_ERR_NOT_IMPL;
  int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
  const double *A_orig = (const double *)tc->A;
  if (m <= 0 || n <= 0 || !A_orig) {
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  int k = m < n ? m : n;

  /* Oracle path: DGEQRF then DORGQR */
  double *A_oq = clone_matrix_f64(A_orig, m, n, lda);
  double *tau_o = (double *)malloc((size_t)k * sizeof(double));
  if (!A_oq || !tau_o) {
    free(A_oq);
    free(tau_o);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (oracle->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oq, lda, tau_o) != 0) {
    free(A_oq);
    free(tau_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  if (oracle->dorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_oq, lda, tau_o) != 0) {
    free(A_oq);
    free(tau_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  /* Candidate path: DGEQRF, then timed DORGQR */
  double *A_cq = clone_matrix_f64(A_orig, m, n, lda);
  double *tau_c = (double *)malloc((size_t)k * sizeof(double));
  if (!A_cq || !tau_c) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (cand->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cq, lda, tau_c) != 0) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }
  double *A_saved = clone_matrix_f64(A_cq, m, n, lda);
  double *tau_saved = (double *)malloc((size_t)k * sizeof(double));
  if (!A_saved || !tau_saved) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    free(A_saved);
    free(tau_saved);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(tau_saved, tau_c, (size_t)k * sizeof(double));

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
      memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(double));
      memcpy(tau_c, tau_saved, (size_t)k * sizeof(double));
      (void)cand->dorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
    }
    for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
      memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(double));
      memcpy(tau_c, tau_saved, (size_t)k * sizeof(double));
      uint64_t t0 = fb_judge_time_ns();
      (void)cand->dorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }
  memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(double));
  memcpy(tau_c, tau_saved, (size_t)k * sizeof(double));
  free(A_saved);
  free(tau_saved);
  if (cand->dorgqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c) != 0) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }

  /* Reconstruction: ||Q_oracle - Q_cand||_F / ||Q_oracle||_F */
  double norm_diff = 0.0, norm_Q = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      double diff = A_oq[i * lda + j] - A_cq[i * lda + j];
      norm_diff += diff * diff;
      norm_Q += A_oq[i * lda + j] * A_oq[i * lda + j];
    }
  }
  result_from_relerr(&res->reconstruction,
                     (norm_Q < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_Q));

  /* Orthogonality of Q_cand: ||Q^T Q - I||_F / sqrt(k) */
  double ortho_err = 0.0;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double qtq = 0.0;
      for (int l = 0; l < m; l++)
        qtq += A_cq[l * lda + i] * A_cq[l * lda + j];
      double delta = qtq - (i == j ? 1.0 : 0.0);
      ortho_err += delta * delta;
    }
  }
  result_from_relerr(&res->orthogonality, sqrt(ortho_err) / sqrt((double)k));

  free(A_oq);
  free(tau_o);
  free(A_cq);
  free(tau_c);
  return FB_JUDGE_OK;
}

/* =========================================================================
 * ZUNGQR — generate Q from double-complex QR factorization
 * ========================================================================= */
static fb_judge_status_t run_zungqr(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_factorization_result_t *res,
                                    uint64_t *ns_out) {
  if (!oracle->zgeqrf || !oracle->zungqr || !cand->zgeqrf || !cand->zungqr)
    return FB_JUDGE_ERR_NOT_IMPL;
  int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
  const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
  if (m <= 0 || n <= 0 || !A_orig || tc->A_elems < (size_t)m * (size_t)lda) {
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  int k = m < n ? m : n;

  /* Oracle path: ZGEQRF then ZUNGQR */
  fb_complex_double_t *A_oq = clone_matrix_cf64(A_orig, m, n, lda);
  fb_complex_double_t *tau_o =
      (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
  if (!A_oq || !tau_o) {
    free(A_oq);
    free(tau_o);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (oracle->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_oq, lda, tau_o) != 0) {
    free(A_oq);
    free(tau_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  if (oracle->zungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_oq, lda, tau_o) != 0) {
    free(A_oq);
    free(tau_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }
  /* Guard against NaN/Inf in oracle Q (can occur on extreme-scale inputs). */
  for (int qi = 0; qi < m && !res->reconstruction.is_oracle_fatal; qi++) {
    for (int qj = 0; qj < k && !res->reconstruction.is_oracle_fatal; qj++) {
      double qr = __real__ A_oq[qi * lda + qj];
      double qi_ = __imag__ A_oq[qi * lda + qj];
      if (qr != qr || qi_ != qi_ || qr * 0.0 != 0.0 || qi_ * 0.0 != 0.0) {
        free(A_oq); free(tau_o); mark_oracle_fatal(res); return FB_JUDGE_OK;
      }
    }
  }

  /* Candidate path */
  fb_complex_double_t *A_cq = clone_matrix_cf64(A_orig, m, n, lda);
  fb_complex_double_t *tau_c =
      (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
  if (!A_cq || !tau_c) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (cand->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, A_cq, lda, tau_c) != 0) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }
  fb_complex_double_t *A_saved = clone_matrix_cf64(A_cq, m, n, lda);
  fb_complex_double_t *tau_saved =
      (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
  if (!A_saved || !tau_saved) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    free(A_saved);
    free(tau_saved);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(tau_saved, tau_c, (size_t)k * sizeof(fb_complex_double_t));

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
      memcpy(A_cq, A_saved,
             (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
      memcpy(tau_c, tau_saved, (size_t)k * sizeof(fb_complex_double_t));
      (void)cand->zungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
    }
    for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
      memcpy(A_cq, A_saved,
             (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
      memcpy(tau_c, tau_saved, (size_t)k * sizeof(fb_complex_double_t));
      uint64_t t0 = fb_judge_time_ns();
      (void)cand->zungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c);
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }
  memcpy(A_cq, A_saved, (size_t)m * (size_t)lda * sizeof(fb_complex_double_t));
  memcpy(tau_c, tau_saved, (size_t)k * sizeof(fb_complex_double_t));
  free(A_saved);
  free(tau_saved);
  if (cand->zungqr(FB_LAYOUT_ROW_MAJOR, m, k, k, A_cq, lda, tau_c) != 0) {
    free(A_oq);
    free(tau_o);
    free(A_cq);
    free(tau_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }

  /* Reconstruction: ||Q_oracle - Q_cand||_F / ||Q_oracle||_F */
  double norm_diff = 0.0, norm_Q = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      double dr = __real__ A_oq[i * lda + j] - __real__ A_cq[i * lda + j];
      double di = __imag__ A_oq[i * lda + j] - __imag__ A_cq[i * lda + j];
      norm_diff += dr * dr + di * di;
      double qr = __real__ A_oq[i * lda + j];
      double qi = __imag__ A_oq[i * lda + j];
      norm_Q += qr * qr + qi * qi;
    }
  }
  result_from_relerr(&res->reconstruction,
                     (norm_Q < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_Q));

  free(A_oq);
  free(tau_o);
  free(A_cq);
  free(tau_c);
  return FB_JUDGE_OK;
}

/* =========================================================================
 * Dispatch
 * ========================================================================= */

/* =========================================================================
 * DORMQR — apply Q from double-precision QR to a matrix (left, no-trans)
 * ========================================================================= */
static fb_judge_status_t run_dormqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgeqrf || !oracle->dormqr || !cand->dgeqrf || !cand->dormqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    double *H = clone_matrix_f64(A_orig, m, n, lda);
    double *tau = (double *)malloc((size_t)k * sizeof(double));
    if (!H || !tau) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, H, lda, tau) != 0) {
        free(H); free(tau); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    int ldc = lda, n_c = n;
    size_t C_sz = (size_t)m * (size_t)ldc;
    double *C_template = (double *)calloc(C_sz, sizeof(double));
    if (!C_template) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (tc->B && tc->B_elems >= C_sz) {
        memcpy(C_template, tc->B, C_sz * sizeof(double));
    } else {
        int id_sz = m < n_c ? m : n_c;
        for (int i = 0; i < id_sz; i++) C_template[i * ldc + i] = 1.0;
    }

    double *H_o = clone_matrix_f64(H, m, n, lda);
    double *tau_o = (double *)malloc((size_t)k * sizeof(double));
    double *C_o   = (double *)malloc(C_sz * sizeof(double));
    if (!H_o || !tau_o || !C_o) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_o, tau, (size_t)k * sizeof(double));
    memcpy(C_o, C_template, C_sz * sizeof(double));
    if (oracle->dormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                       m, n_c, k, H_o, lda, tau_o, C_o, ldc) != 0) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    double *H_c   = clone_matrix_f64(H, m, n, lda);
    double *tau_c = (double *)malloc((size_t)k * sizeof(double));
    double *C_c   = (double *)malloc(C_sz * sizeof(double));
    if (!H_c || !tau_c || !C_c) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(H_c, H, (size_t)m*(size_t)lda*sizeof(double));
            memcpy(tau_c, tau, (size_t)k*sizeof(double));
            memcpy(C_c, C_template, C_sz*sizeof(double));
            (void)cand->dormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(H_c, H, (size_t)m*(size_t)lda*sizeof(double));
            memcpy(tau_c, tau, (size_t)k*sizeof(double));
            memcpy(C_c, C_template, C_sz*sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(H_c, H, (size_t)m*(size_t)lda*sizeof(double));
    memcpy(tau_c, tau, (size_t)k*sizeof(double));
    memcpy(C_c, C_template, C_sz*sizeof(double));
    if (cand->dormqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                     m, n_c, k, H_c, lda, tau_c, C_c, ldc) != 0) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double norm_diff = 0.0, norm_C = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_c; j++) {
            double diff = C_o[i*ldc+j] - C_c[i*ldc+j];
            norm_diff += diff * diff;
            double co = C_o[i*ldc+j];
            norm_C += co * co;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_C < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_C));
    free(H); free(tau); free(C_template);
    free(H_o); free(tau_o); free(C_o);
    free(H_c); free(tau_c); free(C_c);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CUNMQR — apply Q (from complex single-precision QR) to a matrix C
 * ========================================================================= */
static fb_judge_status_t run_cunmqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgeqrf || !oracle->cunmqr || !cand->cgeqrf || !cand->cunmqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    fb_complex_float_t *H =
        clone_matrix_cf32(A_orig, m, n, lda);
    fb_complex_float_t *tau =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    if (!H || !tau) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, H, lda, tau) != 0) {
        free(H); free(tau); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    int ldc = lda, n_c = n;
    size_t C_sz = (size_t)m * (size_t)ldc;
    fb_complex_float_t *C_template =
        (fb_complex_float_t *)calloc(C_sz, sizeof(fb_complex_float_t));
    if (!C_template) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (tc->B && tc->B_elems >= C_sz) {
        memcpy(C_template, tc->B, C_sz * sizeof(fb_complex_float_t));
    } else {
        int id_sz = m < n_c ? m : n_c;
        for (int i = 0; i < id_sz; i++) {
            __real__ C_template[i * ldc + i] = 1.0f;
            __imag__ C_template[i * ldc + i] = 0.0f;
        }
    }

    fb_complex_float_t *H_o   = clone_matrix_cf32(H, m, n, lda);
    fb_complex_float_t *tau_o =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    fb_complex_float_t *C_o   =
        (fb_complex_float_t *)malloc(C_sz * sizeof(fb_complex_float_t));
    if (!H_o || !tau_o || !C_o) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_o, tau, (size_t)k * sizeof(fb_complex_float_t));
    memcpy(C_o, C_template, C_sz * sizeof(fb_complex_float_t));
    if (oracle->cunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                       m, n_c, k, H_o, lda, tau_o, C_o, ldc) != 0) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_float_t *H_c   = clone_matrix_cf32(H, m, n, lda);
    fb_complex_float_t *tau_c =
        (fb_complex_float_t *)malloc((size_t)k * sizeof(fb_complex_float_t));
    fb_complex_float_t *C_c   =
        (fb_complex_float_t *)malloc(C_sz * sizeof(fb_complex_float_t));
    if (!H_c || !tau_c || !C_c) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_float_t));
            memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_float_t));
            (void)cand->cunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
            memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_float_t));
            memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_float_t));
    memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_float_t));
    memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_float_t));
    if (cand->cunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                     m, n_c, k, H_c, lda, tau_c, C_c, ldc) != 0) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double norm_diff = 0.0, norm_C = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_c; j++) {
            double dr = (double)(__real__ C_o[i*ldc+j]) - (double)(__real__ C_c[i*ldc+j]);
            double di = (double)(__imag__ C_o[i*ldc+j]) - (double)(__imag__ C_c[i*ldc+j]);
            norm_diff += dr*dr + di*di;
            double cr = (double)(__real__ C_o[i*ldc+j]);
            double ci = (double)(__imag__ C_o[i*ldc+j]);
            norm_C += cr*cr + ci*ci;
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
 * ZUNMQR — apply Q (from complex double-precision QR) to a matrix C
 * ========================================================================= */
static fb_judge_status_t run_zunmqr(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgeqrf || !oracle->zunmqr || !cand->zgeqrf || !cand->zunmqr)
        return FB_JUDGE_ERR_NOT_IMPL;
    int m = (int)tc->m, n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    if (m <= 0 || n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    int k = m < n ? m : n;

    fb_complex_double_t *H =
        clone_matrix_cf64(A_orig, m, n, lda);
    fb_complex_double_t *tau =
        (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
    if (!H || !tau) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zgeqrf(FB_LAYOUT_ROW_MAJOR, m, n, H, lda, tau) != 0) {
        free(H); free(tau); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* Skip extreme-scale cases where zgeqrf produces Inf/NaN tau
     * (occurs for |A|~1e100 inputs where intermediate norms overflow). */
    {
        int has_bad = 0;
        for (int i = 0; i < k; i++) {
            double tr = __real__(tau[i]), ti = __imag__(tau[i]);
            if (isinf(tr) || isinf(ti) || isnan(tr) || isnan(ti)) { has_bad = 1; break; }
        }
        if (has_bad) { free(H); free(tau); mark_oracle_fatal(res); return FB_JUDGE_OK; }
    }

    int ldc = lda, n_c = n;
    size_t C_sz = (size_t)m * (size_t)ldc;
    fb_complex_double_t *C_template =
        (fb_complex_double_t *)calloc(C_sz, sizeof(fb_complex_double_t));
    if (!C_template) { free(H); free(tau); return FB_JUDGE_ERR_ALLOC; }
    if (tc->B && tc->B_elems >= C_sz) {
        memcpy(C_template, tc->B, C_sz * sizeof(fb_complex_double_t));
    } else {
        int id_sz = m < n_c ? m : n_c;
        for (int i = 0; i < id_sz; i++) {
            C_template[i * ldc + i] = 1.0;  /* double _Complex = 1.0 → 1+0i */
        }
    }

    fb_complex_double_t *H_o   = clone_matrix_cf64(H, m, n, lda);
    fb_complex_double_t *tau_o =
        (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
    fb_complex_double_t *C_o   =
        (fb_complex_double_t *)malloc(C_sz * sizeof(fb_complex_double_t));
    if (!H_o || !tau_o || !C_o) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(tau_o, tau, (size_t)k * sizeof(fb_complex_double_t));
    memcpy(C_o, C_template, C_sz * sizeof(fb_complex_double_t));
    if (oracle->zunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                       m, n_c, k, H_o, lda, tau_o, C_o, ldc) != 0) {
        free(H); free(tau); free(C_template); free(H_o); free(tau_o); free(C_o);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }

    fb_complex_double_t *H_c   = clone_matrix_cf64(H, m, n, lda);
    fb_complex_double_t *tau_c =
        (fb_complex_double_t *)malloc((size_t)k * sizeof(fb_complex_double_t));
    fb_complex_double_t *C_c   =
        (fb_complex_double_t *)malloc(C_sz * sizeof(fb_complex_double_t));
    if (!H_c || !tau_c || !C_c) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
            memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_double_t));
            memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_double_t));
            (void)cand->zunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
            memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_double_t));
            memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                               m, n_c, k, H_c, lda, tau_c, C_c, ldc);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(H_c,   H,          (size_t)m*(size_t)lda*sizeof(fb_complex_double_t));
    memcpy(tau_c, tau,        (size_t)k*sizeof(fb_complex_double_t));
    memcpy(C_c,   C_template, C_sz*sizeof(fb_complex_double_t));
    if (cand->zunmqr(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS,
                     m, n_c, k, H_c, lda, tau_c, C_c, ldc) != 0) {
        free(H); free(tau); free(C_template);
        free(H_o); free(tau_o); free(C_o);
        free(H_c); free(tau_c); free(C_c);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }

    double norm_diff = 0.0, norm_C = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_c; j++) {
            double dr = (double)(__real__ C_o[i*ldc+j]) - (double)(__real__ C_c[i*ldc+j]);
            double di = (double)(__imag__ C_o[i*ldc+j]) - (double)(__imag__ C_c[i*ldc+j]);
            norm_diff += dr*dr + di*di;
            double cr = (double)(__real__ C_o[i*ldc+j]);
            double ci = (double)(__imag__ C_o[i*ldc+j]);
            norm_C += cr*cr + ci*ci;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_C < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_C));

    free(H); free(tau); free(C_template);
    free(H_o); free(tau_o); free(C_o);
    free(H_c); free(tau_c); free(C_c);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DTRTRI — invert double-precision upper triangular matrix in-place
 * ========================================================================= */
static fb_judge_status_t run_dtrtri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dtrtri || !cand->dtrtri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double *)tc->A;
    if (n <= 0 || !A_orig || tc->A_elems < (size_t)n * (size_t)lda) {
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *A_oi = clone_matrix_f64(A_orig, n, n, lda);
    if (!A_oi) return FB_JUDGE_ERR_ALLOC;
    if (oracle->dtrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_oi, lda) != 0) {
        free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *A_ci = clone_matrix_f64(A_orig, n, n, lda);
    if (!A_ci) { free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(double));
            (void)cand->dtrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dtrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(double));
    if (cand->dtrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda) != 0) {
        free(A_oi); free(A_ci); mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double diff = A_oi[i*lda+j] - A_ci[i*lda+j];
            norm_diff += diff * diff;
            double ao = A_oi[i*lda+j];
            norm_inv += ao * ao;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CTRTRI — invert complex single-precision upper triangular matrix in-place
 * ========================================================================= */
static fb_judge_status_t run_ctrtri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->ctrtri || !cand->ctrtri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *A_oi = clone_matrix_cf32(A_orig, n, n, lda);
    if (!A_oi) return FB_JUDGE_ERR_ALLOC;
    if (oracle->ctrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_oi, lda) != 0) {
        free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_ci = clone_matrix_cf32(A_orig, n, n, lda);
    if (!A_ci) { free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            (void)cand->ctrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->ctrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    if (cand->ctrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda) != 0) {
        free(A_oi); free(A_ci); mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    /* Treat complex elements as float pairs to avoid <complex.h> incompatibility */
    const float *A_oi_r = (const float *)A_oi;
    const float *A_ci_r = (const float *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = (double)A_oi_r[base]   - (double)A_ci_r[base];
            double di = (double)A_oi_r[base+1] - (double)A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = (double)A_oi_r[base];
            double ai = (double)A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZTRTRI — invert complex double-precision upper triangular matrix in-place
 * ========================================================================= */
static fb_judge_status_t run_ztrtri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->ztrtri || !cand->ztrtri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *A_oi = clone_matrix_cf64(A_orig, n, n, lda);
    if (!A_oi) return FB_JUDGE_ERR_ALLOC;
    if (oracle->ztrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_oi, lda) != 0) {
        free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_ci = clone_matrix_cf64(A_orig, n, n, lda);
    if (!A_ci) { free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            (void)cand->ztrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->ztrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_orig, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
    if (cand->ztrtri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NON_UNIT, n, A_ci, lda) != 0) {
        free(A_oi); free(A_ci); mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    /* Treat complex elements as double pairs to avoid <complex.h> incompatibility */
    const double *A_oi_r = (const double *)A_oi;
    const double *A_ci_r = (const double *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = A_oi_r[base]   - A_ci_r[base];
            double di = A_oi_r[base+1] - A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = A_oi_r[base];
            double ai = A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGETRI — invert single-precision matrix from LU factors (after sgetrf)
 * ========================================================================= */
static fb_judge_status_t run_sgetri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgetri || !cand->sgetri || !oracle->sgetrf)
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const float *A_orig = (const float *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    /* Pre-factor with oracle SGETRF -> LU factors + ipiv */
    float *A_lu = clone_matrix_f32(A_orig, n, n, lda);
    if (!A_lu) return FB_JUDGE_ERR_ALLOC;
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A_lu); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->sgetrf(FB_LAYOUT_ROW_MAJOR, n, n, A_lu, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* Oracle: invert from LU */
    float *A_oi = clone_matrix_f32(A_lu, n, n, lda);
    if (!A_oi) { free(A_lu); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->sgetri(FB_LAYOUT_ROW_MAJOR, n, A_oi, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* Candidate: invert from same LU (with optional timing) */
    float *A_ci = clone_matrix_f32(A_lu, n, n, lda);
    if (!A_ci) { free(A_lu); free(ipiv); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(float));
            (void)cand->sgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(float));
    if (cand->sgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = (double)A_oi[i*lda + j] - (double)A_ci[i*lda + j];
            norm_diff += d * d;
            double a = (double)A_oi[i*lda + j];
            norm_inv += a * a;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_lu); free(ipiv); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGETRI — invert double-precision matrix from LU factors (after dgetrf)
 * ========================================================================= */
static fb_judge_status_t run_dgetri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgetri || !cand->dgetri || !oracle->dgetrf)
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const double *A_orig = (const double *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    double *A_lu = clone_matrix_f64(A_orig, n, n, lda);
    if (!A_lu) return FB_JUDGE_ERR_ALLOC;
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A_lu); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dgetrf(FB_LAYOUT_ROW_MAJOR, n, n, A_lu, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *A_oi = clone_matrix_f64(A_lu, n, n, lda);
    if (!A_oi) { free(A_lu); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dgetri(FB_LAYOUT_ROW_MAJOR, n, A_oi, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *A_ci = clone_matrix_f64(A_lu, n, n, lda);
    if (!A_ci) { free(A_lu); free(ipiv); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(double));
            (void)cand->dgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(double));
    if (cand->dgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = A_oi[i*lda + j] - A_ci[i*lda + j];
            norm_diff += d * d;
            double a = A_oi[i*lda + j];
            norm_inv += a * a;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_lu); free(ipiv); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGETRI — invert complex single-precision matrix from LU factors
 * ========================================================================= */
static fb_judge_status_t run_cgetri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgetri || !cand->cgetri || !oracle->cgetrf)
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_float_t *A_orig = (const fb_complex_float_t *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_float_t *A_lu = clone_matrix_cf32(A_orig, n, n, lda);
    if (!A_lu) return FB_JUDGE_ERR_ALLOC;
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A_lu); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgetrf(FB_LAYOUT_ROW_MAJOR, n, n, A_lu, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_oi = clone_matrix_cf32(A_lu, n, n, lda);
    if (!A_oi) { free(A_lu); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cgetri(FB_LAYOUT_ROW_MAJOR, n, A_oi, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_ci = clone_matrix_cf32(A_lu, n, n, lda);
    if (!A_ci) { free(A_lu); free(ipiv); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            (void)cand->cgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    if (cand->cgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    /* Treat complex elements as float pairs to avoid <complex.h> incompatibility */
    const float *A_oi_r = (const float *)A_oi;
    const float *A_ci_r = (const float *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = (double)A_oi_r[base]   - (double)A_ci_r[base];
            double di = (double)A_oi_r[base+1] - (double)A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = (double)A_oi_r[base];
            double ai = (double)A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_lu); free(ipiv); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGETRI — invert complex double-precision matrix from LU factors
 * ========================================================================= */
static fb_judge_status_t run_zgetri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgetri || !cand->zgetri || !oracle->zgetrf)
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    const fb_complex_double_t *A_orig = (const fb_complex_double_t *)tc->A;
    if (n <= 0 || !A_orig) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *A_lu = clone_matrix_cf64(A_orig, n, n, lda);
    if (!A_lu) return FB_JUDGE_ERR_ALLOC;
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A_lu); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zgetrf(FB_LAYOUT_ROW_MAJOR, n, n, A_lu, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_oi = clone_matrix_cf64(A_lu, n, n, lda);
    if (!A_oi) { free(A_lu); free(ipiv); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zgetri(FB_LAYOUT_ROW_MAJOR, n, A_oi, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi);
        mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_ci = clone_matrix_cf64(A_lu, n, n, lda);
    if (!A_ci) { free(A_lu); free(ipiv); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            (void)cand->zgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_lu, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
    if (cand->zgetri(FB_LAYOUT_ROW_MAJOR, n, A_ci, lda, ipiv) != 0) {
        free(A_lu); free(ipiv); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    /* Treat complex elements as double pairs to avoid <complex.h> incompatibility */
    const double *A_oi_r = (const double *)A_oi;
    const double *A_ci_r = (const double *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = A_oi_r[base]   - A_ci_r[base];
            double di = A_oi_r[base+1] - A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = A_oi_r[base];
            double ai = A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_lu); free(ipiv); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SPOTRI — invert single-precision SPD matrix from Cholesky factor (spotrf)
 * ========================================================================= */
static fb_judge_status_t run_spotri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->spotri || !cand->spotri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    if (n <= 0) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    /* Build a synthetic upper triangular Cholesky factor: diagonal entries
     * forced > 0 (required for a valid Cholesky factor), upper off-diagonal
     * scaled from tc->A to give non-trivial but well-conditioned input. */
    float *A_chol = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
    if (!A_chol) return FB_JUDGE_ERR_ALLOC;
    const float *A_orig = (const float *)tc->A;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++)
            A_chol[i*lda + j] = A_orig ? A_orig[i*lda + j] * 0.05f : 0.0f;
        A_chol[i*lda + i] = (float)(i + 2);  /* positive diagonal */
    }
    /* Oracle: compute inverse of (A_chol^T * A_chol) via POTRI */
    float *A_oi = clone_matrix_f32(A_chol, n, n, lda);
    if (!A_oi) { free(A_chol); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->spotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_oi, lda) != 0) {
        free(A_chol); free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    /* Candidate: compute same inverse (with optional timing) */
    float *A_ci = clone_matrix_f32(A_chol, n, n, lda);
    if (!A_ci) { free(A_chol); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(float));
            (void)cand->spotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->spotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(float));
    if (cand->spotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda) != 0) {
        free(A_chol); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double d = (double)A_oi[i*lda + j] - (double)A_ci[i*lda + j];
            norm_diff += d * d;
            double a = (double)A_oi[i*lda + j];
            norm_inv += a * a;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_chol); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DPOTRI — invert double-precision SPD matrix from Cholesky factor (dpotrf)
 * ========================================================================= */
static fb_judge_status_t run_dpotri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dpotri || !cand->dpotri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    if (n <= 0) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    /* Synthetic upper triangular Cholesky factor with forced positive diagonal */
    double *A_chol = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
    if (!A_chol) return FB_JUDGE_ERR_ALLOC;
    const double *A_orig = (const double *)tc->A;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++)
            A_chol[i*lda + j] = A_orig ? A_orig[i*lda + j] * 0.05 : 0.0;
        A_chol[i*lda + i] = (double)(i + 2);
    }
    double *A_oi = clone_matrix_f64(A_chol, n, n, lda);
    if (!A_oi) { free(A_chol); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->dpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_oi, lda) != 0) {
        free(A_chol); free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    double *A_ci = clone_matrix_f64(A_chol, n, n, lda);
    if (!A_ci) { free(A_chol); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(double));
            (void)cand->dpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(double));
    if (cand->dpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda) != 0) {
        free(A_chol); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double d = A_oi[i*lda + j] - A_ci[i*lda + j];
            norm_diff += d * d;
            double a = A_oi[i*lda + j];
            norm_inv += a * a;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_chol); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CPOTRI — invert complex single-precision HPD matrix from Cholesky factor
 * ========================================================================= */
static fb_judge_status_t run_cpotri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cpotri || !cand->cpotri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    if (n <= 0) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    /* Synthetic upper triangular Cholesky factor: real positive diagonal,
     * small complex off-diagonals to keep the matrix well-conditioned. */
    fb_complex_float_t *A_chol =
        (fb_complex_float_t *)calloc((size_t)n * (size_t)lda,
                                     sizeof(fb_complex_float_t));
    if (!A_chol) return FB_JUDGE_ERR_ALLOC;
    const float *A_orig_r = tc->A ? (const float *)tc->A : NULL;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int base = 2 * (i*lda + j);
            /* small off-diagonal complex elements */
            ((float *)A_chol)[base]   = A_orig_r ? A_orig_r[base]   * 0.05f : 0.0f;
            ((float *)A_chol)[base+1] = A_orig_r ? A_orig_r[base+1] * 0.05f : 0.0f;
        }
        int diag_base = 2 * (i*lda + i);
        ((float *)A_chol)[diag_base]   = (float)(i + 2);  /* real positive diagonal */
        ((float *)A_chol)[diag_base+1] = 0.0f;            /* zero imaginary part */
    }
    fb_complex_float_t *A_oi = clone_matrix_cf32(A_chol, n, n, lda);
    if (!A_oi) { free(A_chol); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->cpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_oi, lda) != 0) {
        free(A_chol); free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_float_t *A_ci = clone_matrix_cf32(A_chol, n, n, lda);
    if (!A_ci) { free(A_chol); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            (void)cand->cpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    if (cand->cpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda) != 0) {
        free(A_chol); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    const float *A_oi_r = (const float *)A_oi;
    const float *A_ci_r = (const float *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = (double)A_oi_r[base]   - (double)A_ci_r[base];
            double di = (double)A_oi_r[base+1] - (double)A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = (double)A_oi_r[base];
            double ai = (double)A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < (double)FLT_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_chol); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZPOTRI — invert complex double-precision HPD matrix from Cholesky factor
 * ========================================================================= */
static fb_judge_status_t run_zpotri(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zpotri || !cand->zpotri) return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, lda = (int)tc->lda;
    if (n <= 0) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    fb_complex_double_t *A_chol =
        (fb_complex_double_t *)calloc((size_t)n * (size_t)lda,
                                      sizeof(fb_complex_double_t));
    if (!A_chol) return FB_JUDGE_ERR_ALLOC;
    const double *A_orig_r = tc->A ? (const double *)tc->A : NULL;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int base = 2 * (i*lda + j);
            ((double *)A_chol)[base]   = A_orig_r ? A_orig_r[base]   * 0.05 : 0.0;
            ((double *)A_chol)[base+1] = A_orig_r ? A_orig_r[base+1] * 0.05 : 0.0;
        }
        int diag_base = 2 * (i*lda + i);
        ((double *)A_chol)[diag_base]   = (double)(i + 2);
        ((double *)A_chol)[diag_base+1] = 0.0;
    }
    fb_complex_double_t *A_oi = clone_matrix_cf64(A_chol, n, n, lda);
    if (!A_oi) { free(A_chol); return FB_JUDGE_ERR_ALLOC; }
    if (oracle->zpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_oi, lda) != 0) {
        free(A_chol); free(A_oi); mark_oracle_fatal(res); return FB_JUDGE_OK;
    }
    fb_complex_double_t *A_ci = clone_matrix_cf64(A_chol, n, n, lda);
    if (!A_ci) { free(A_chol); free(A_oi); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            (void)cand->zpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(A_ci, A_chol, (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
    if (cand->zpotri(FB_LAYOUT_ROW_MAJOR, FB_UPPER, n, A_ci, lda) != 0) {
        free(A_chol); free(A_oi); free(A_ci);
        mark_cand_fatal(res); return FB_JUDGE_OK;
    }
    double norm_diff = 0.0, norm_inv = 0.0;
    const double *A_oi_r = (const double *)A_oi;
    const double *A_ci_r = (const double *)A_ci;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int base = 2 * (i*lda + j);
            double dr = A_oi_r[base]   - A_ci_r[base];
            double di = A_oi_r[base+1] - A_ci_r[base+1];
            norm_diff += dr*dr + di*di;
            double ar = A_oi_r[base];
            double ai = A_oi_r[base+1];
            norm_inv += ar*ar + ai*ai;
        }
    }
    result_from_relerr(&res->reconstruction,
        (norm_inv < DBL_EPSILON) ? 0.0 : sqrt(norm_diff / norm_inv));
    free(A_chol); free(A_oi); free(A_ci);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * LDL^T Reconstruction (for SSYTRF / DSYTRF)
 *
 * Assumes lower‑triangle storage, no 2×2 blocks (diagonally‑dominant test
 * matrix never triggers Bunch–Kaufman 2×2 pivoting).
 *   D[k]    = Afact[k*lda+k]
 *   L[i][k] = Afact[i*lda+k]  for k < i,  L[i][i] = 1
 *   recon[i][j] = Σ_{k=0}^{min(i,j)} L[i][k]·D[k]·L[j][k]
 * ========================================================================= */

static double fb_ldlt_reconstruction_f32(const float *Aorig, const float *Afact,
                                         int n, int lda) {
  double norm_diff = 0.0, norm_A = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int kmax = (i < j) ? i : j;
      double sum = 0.0;
      for (int k = 0; k <= kmax; k++) {
        double lik = (k == i) ? 1.0 : (double)Afact[i * lda + k];
        double ljk = (k == j) ? 1.0 : (double)Afact[j * lda + k];
        double dk = (double)Afact[k * lda + k];
        sum += lik * dk * ljk;
      }
      double a_ij = (double)Aorig[i * lda + j];
      double diff = sum - a_ij;
      norm_diff += diff * diff;
      norm_A += a_ij * a_ij;
    }
  }
  if (norm_A < (double)FLT_EPSILON)
    return 0.0;
  return sqrt(norm_diff / norm_A);
}

static double fb_ldlt_reconstruction_f64(const double *Aorig,
                                         const double *Afact, int n, int lda) {
  double norm_diff = 0.0, norm_A = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int kmax = (i < j) ? i : j;
      double sum = 0.0;
      for (int k = 0; k <= kmax; k++) {
        double lik = (k == i) ? 1.0 : Afact[i * lda + k];
        double ljk = (k == j) ? 1.0 : Afact[j * lda + k];
        double dk = Afact[k * lda + k];
        sum += lik * dk * ljk;
      }
      double a_ij = Aorig[i * lda + j];
      double diff = sum - a_ij;
      norm_diff += diff * diff;
      norm_A += a_ij * a_ij;
    }
  }
  if (norm_A < DBL_EPSILON)
    return 0.0;
  return sqrt(norm_diff / norm_A);
}

static void fb_fill_sym_indef_cf32(fb_complex_float_t *Aorig, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(Aorig[i * lda + i]) = (float)(n + 2 * (i + 1));
        __imag__(Aorig[i * lda + i]) = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float re = 0.5f / (float)(j - i + 1);
            float im = 0.05f * (float)(i + 1) / (float)(j + 1);
            __real__(Aorig[i * lda + j]) = re;
            __imag__(Aorig[i * lda + j]) = im;
            __real__(Aorig[j * lda + i]) = re;
            __imag__(Aorig[j * lda + i]) = im;
        }
    }
}

static void fb_fill_sym_indef_cf64(fb_complex_double_t *Aorig, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(Aorig[i * lda + i]) = (double)(n + 2 * (i + 1));
        __imag__(Aorig[i * lda + i]) = 0.0;
        for (int j = i + 1; j < n; j++) {
            double re = 0.5 / (double)(j - i + 1);
            double im = 0.05 * (double)(i + 1) / (double)(j + 1);
            __real__(Aorig[i * lda + j]) = re;
            __imag__(Aorig[i * lda + j]) = im;
            __real__(Aorig[j * lda + i]) = re;
            __imag__(Aorig[j * lda + i]) = im;
        }
    }
}

static void fb_fill_herm_indef_cf32(fb_complex_float_t *Aorig, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(Aorig[i * lda + i]) = (float)(n + 2 * (i + 1));
        __imag__(Aorig[i * lda + i]) = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float re = 0.5f / (float)(j - i + 1);
            float im = 0.05f * (float)(i + 1) / (float)(j + 1);
            __real__(Aorig[i * lda + j]) = re;
            __imag__(Aorig[i * lda + j]) = im;
            __real__(Aorig[j * lda + i]) = re;
            __imag__(Aorig[j * lda + i]) = -im;
        }
    }
}

static void fb_fill_herm_indef_cf64(fb_complex_double_t *Aorig, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(Aorig[i * lda + i]) = (double)(n + 2 * (i + 1));
        __imag__(Aorig[i * lda + i]) = 0.0;
        for (int j = i + 1; j < n; j++) {
            double re = 0.5 / (double)(j - i + 1);
            double im = 0.05 * (double)(i + 1) / (double)(j + 1);
            __real__(Aorig[i * lda + j]) = re;
            __imag__(Aorig[i * lda + j]) = im;
            __real__(Aorig[j * lda + i]) = re;
            __imag__(Aorig[j * lda + i]) = -im;
        }
    }
}

static double fb_ldlt_reconstruction_cf32(const fb_complex_float_t *Aorig,
                                                                                    const fb_complex_float_t *Afact,
                                                                                    int n, int lda,
                                                                                    int hermitian) {
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = (i < j) ? i : j;
            double sum_re = 0.0, sum_im = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double lik_re = (k == i) ? 1.0 : (double)__real__(Afact[i * lda + k]);
                double lik_im = (k == i) ? 0.0 : (double)__imag__(Afact[i * lda + k]);
                double ljk_re = (k == j) ? 1.0 : (double)__real__(Afact[j * lda + k]);
                double ljk_im = (k == j) ? 0.0 : (double)__imag__(Afact[j * lda + k]);
                double dk_re = (double)__real__(Afact[k * lda + k]);
                double dk_im = (double)__imag__(Afact[k * lda + k]);
                if (hermitian) ljk_im = -ljk_im;

                double tmp_re = lik_re * dk_re - lik_im * dk_im;
                double tmp_im = lik_re * dk_im + lik_im * dk_re;
                sum_re += tmp_re * ljk_re - tmp_im * ljk_im;
                sum_im += tmp_re * ljk_im + tmp_im * ljk_re;
            }
            double a_re = (double)__real__(Aorig[i * lda + j]);
            double a_im = (double)__imag__(Aorig[i * lda + j]);
            double diff_re = sum_re - a_re;
            double diff_im = sum_im - a_im;
            norm_diff += diff_re * diff_re + diff_im * diff_im;
            norm_A += a_re * a_re + a_im * a_im;
        }
    }
    if (norm_A < (double)FLT_EPSILON)
        return (norm_diff < (double)FLT_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_A);
}

static double fb_ldlt_reconstruction_cf64(const fb_complex_double_t *Aorig,
                                                                                    const fb_complex_double_t *Afact,
                                                                                    int n, int lda,
                                                                                    int hermitian) {
    double norm_diff = 0.0, norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int kmax = (i < j) ? i : j;
            double sum_re = 0.0, sum_im = 0.0;
            for (int k = 0; k <= kmax; k++) {
                double lik_re = (k == i) ? 1.0 : __real__(Afact[i * lda + k]);
                double lik_im = (k == i) ? 0.0 : __imag__(Afact[i * lda + k]);
                double ljk_re = (k == j) ? 1.0 : __real__(Afact[j * lda + k]);
                double ljk_im = (k == j) ? 0.0 : __imag__(Afact[j * lda + k]);
                double dk_re = __real__(Afact[k * lda + k]);
                double dk_im = __imag__(Afact[k * lda + k]);
                if (hermitian) ljk_im = -ljk_im;

                double tmp_re = lik_re * dk_re - lik_im * dk_im;
                double tmp_im = lik_re * dk_im + lik_im * dk_re;
                sum_re += tmp_re * ljk_re - tmp_im * ljk_im;
                sum_im += tmp_re * ljk_im + tmp_im * ljk_re;
            }
            double a_re = __real__(Aorig[i * lda + j]);
            double a_im = __imag__(Aorig[i * lda + j]);
            double diff_re = sum_re - a_re;
            double diff_im = sum_im - a_im;
            norm_diff += diff_re * diff_re + diff_im * diff_im;
            norm_A += a_re * a_re + a_im * a_im;
        }
    }
    if (norm_A < DBL_EPSILON)
        return (norm_diff < DBL_EPSILON) ? 0.0 : 1.0;
    return sqrt(norm_diff / norm_A);
}

/* =========================================================================
 * Symmetric Indefinite Factorization (SSYTRF, DSYTRF)
 *
 * Test matrix: diagonally‑dominant symmetric, A[i][i] = n+2*(i+1),
 * A[i][j] = 0.5/(|i‑j|+1) for i≠j.  Strictly diagonally dominant ⇒
 * Bunch–Kaufman always picks 1×1 pivots, so L·D·L^T reconstruction is
 * straightforward (no 2×2 block parsing).
 * ========================================================================= */

static fb_judge_status_t run_ssytrf(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_factorization_result_t *res,
                                    uint64_t *ns_out) {
  typedef int (*fb_ssytrf_fn_t)(int layout, char uplo, int n, float *a, int lda,
                                int *ipiv);
  if (!oracle->ssytrf || !cand->ssytrf)
    return FB_JUDGE_ERR_NOT_IMPL;

  int n = (int)tc->n;
  int lda = (tc->lda > 0) ? (int)tc->lda : n;
  if (n <= 0) {
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  size_t Asz = (size_t)n * (size_t)lda * sizeof(float);

  /* Build diagonally‑dominant symmetric matrix. */
  float *Aorig = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
  if (!Aorig)
    return FB_JUDGE_ERR_ALLOC;
  for (int i = 0; i < n; i++) {
    Aorig[i * lda + i] = (float)(n + 2 * (i + 1));
    for (int j = i + 1; j < n; j++) {
      float v = 0.5f / (float)(j - i + 1);
      Aorig[i * lda + j] = v;
      Aorig[j * lda + i] = v;
    }
  }

  /* Oracle */
  float *Ao = (float *)malloc(Asz);
  int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
  if (!Ao || !ipiv_o) {
    free(Aorig);
    free(Ao);
    free(ipiv_o);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(Ao, Aorig, Asz);
  int info_o = ((fb_ssytrf_fn_t)oracle->ssytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ao,
                                                lda, ipiv_o);
  free(Ao);
  if (info_o != 0) {
    free(Aorig);
    free(ipiv_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  /* Candidate */
  float *Ac = (float *)malloc(Asz);
  int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
  if (!Ac || !ipiv_c) {
    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_ERR_ALLOC;
  }

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
      memcpy(Ac, Aorig, Asz);
      (void)((fb_ssytrf_fn_t)cand->ssytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac, lda,
                                           ipiv_c);
    }
    for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
      memcpy(Ac, Aorig, Asz);
      uint64_t t0 = fb_judge_time_ns();
      (void)((fb_ssytrf_fn_t)cand->ssytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac, lda,
                                           ipiv_c);
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }

  memcpy(Ac, Aorig, Asz);
  int info_c = ((fb_ssytrf_fn_t)cand->ssytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                              lda, ipiv_c);
  if (info_c != 0) {
    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }

  double recon_err = fb_ldlt_reconstruction_f32(Aorig, Ac, n, lda);
  result_from_relerr(&res->reconstruction, recon_err);
  res->orthogonality = (fb_judge_case_result_t){.digits = 16};

  free(Aorig);
  free(ipiv_o);
  free(Ac);
  free(ipiv_c);
  return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsytrf(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_factorization_result_t *res,
                                    uint64_t *ns_out) {
  typedef int (*fb_dsytrf_fn_t)(int layout, char uplo, int n, double *a,
                                int lda, int *ipiv);
  if (!oracle->dsytrf || !cand->dsytrf)
    return FB_JUDGE_ERR_NOT_IMPL;

  int n = (int)tc->n;
  int lda = (tc->lda > 0) ? (int)tc->lda : n;
  if (n <= 0) {
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  size_t Asz = (size_t)n * (size_t)lda * sizeof(double);

  double *Aorig = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
  if (!Aorig)
    return FB_JUDGE_ERR_ALLOC;
  for (int i = 0; i < n; i++) {
    Aorig[i * lda + i] = (double)(n + 2 * (i + 1));
    for (int j = i + 1; j < n; j++) {
      double v = 0.5 / (double)(j - i + 1);
      Aorig[i * lda + j] = v;
      Aorig[j * lda + i] = v;
    }
  }

  double *Ao = (double *)malloc(Asz);
  int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
  if (!Ao || !ipiv_o) {
    free(Aorig);
    free(Ao);
    free(ipiv_o);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(Ao, Aorig, Asz);
  int info_o = ((fb_dsytrf_fn_t)oracle->dsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ao,
                                                lda, ipiv_o);
  free(Ao);
  if (info_o != 0) {
    free(Aorig);
    free(ipiv_o);
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  double *Ac = (double *)malloc(Asz);
  int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
  if (!Ac || !ipiv_c) {
    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_ERR_ALLOC;
  }

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
      memcpy(Ac, Aorig, Asz);
      (void)((fb_dsytrf_fn_t)cand->dsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac, lda,
                                           ipiv_c);
    }
    for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
      memcpy(Ac, Aorig, Asz);
      uint64_t t0 = fb_judge_time_ns();
      (void)((fb_dsytrf_fn_t)cand->dsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac, lda,
                                           ipiv_c);
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }

  memcpy(Ac, Aorig, Asz);
  int info_c = ((fb_dsytrf_fn_t)cand->dsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                              lda, ipiv_c);
  if (info_c != 0) {
    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    mark_cand_fatal(res);
    return FB_JUDGE_OK;
  }

  double recon_err = fb_ldlt_reconstruction_f64(Aorig, Ac, n, lda);
  result_from_relerr(&res->reconstruction, recon_err);
  res->orthogonality = (fb_judge_case_result_t){.digits = 16};

  free(Aorig);
  free(ipiv_o);
  free(Ac);
  free(ipiv_c);
  return FB_JUDGE_OK;
}

static fb_judge_status_t run_csytrf(const fb_backend_vtable_t *oracle,
                                                                        const fb_backend_vtable_t *cand,
                                                                        const fb_corpus_case_t *tc,
                                                                        fb_judge_factorization_result_t *res,
                                                                        uint64_t *ns_out) {
    typedef int (*fb_csytrf_fn_t)(int layout, char uplo, int n,
                                                                fb_complex_float_t *a, int lda, int *ipiv);
    if (!oracle->csytrf || !cand->csytrf)
        return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n;
    int lda = (tc->lda > 0) ? (int)tc->lda : n;
    if (n <= 0) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    fb_complex_float_t *Aorig =
            (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    if (!Aorig)
        return FB_JUDGE_ERR_ALLOC;
    fb_fill_sym_indef_cf32(Aorig, n, lda);

    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    if (!Ao || !ipiv_o) {
        free(Aorig);
        free(Ao);
        free(ipiv_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    int info_o = ((fb_csytrf_fn_t)oracle->csytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n,
                                                                                                Ao, lda, ipiv_o);
    free(Ao);
    if (info_o != 0) {
        free(Aorig);
        free(ipiv_o);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Ac || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(Ac, Aorig, Asz);
            (void)((fb_csytrf_fn_t)cand->csytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(Ac, Aorig, Asz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_csytrf_fn_t)cand->csytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }

    memcpy(Ac, Aorig, Asz);
    int info_c = ((fb_csytrf_fn_t)cand->csytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                            lda, ipiv_c);
    if (info_c != 0) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    double recon_err = fb_ldlt_reconstruction_cf32(Aorig, Ac, n, lda, 0);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsytrf(const fb_backend_vtable_t *oracle,
                                                                        const fb_backend_vtable_t *cand,
                                                                        const fb_corpus_case_t *tc,
                                                                        fb_judge_factorization_result_t *res,
                                                                        uint64_t *ns_out) {
    typedef int (*fb_zsytrf_fn_t)(int layout, char uplo, int n,
                                                                fb_complex_double_t *a, int lda, int *ipiv);
    if (!oracle->zsytrf || !cand->zsytrf)
        return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n;
    int lda = (tc->lda > 0) ? (int)tc->lda : n;
    if (n <= 0) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    fb_complex_double_t *Aorig =
            (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    if (!Aorig)
        return FB_JUDGE_ERR_ALLOC;
    fb_fill_sym_indef_cf64(Aorig, n, lda);

    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    if (!Ao || !ipiv_o) {
        free(Aorig);
        free(Ao);
        free(ipiv_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    int info_o = ((fb_zsytrf_fn_t)oracle->zsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n,
                                                                                                Ao, lda, ipiv_o);
    free(Ao);
    if (info_o != 0) {
        free(Aorig);
        free(ipiv_o);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Ac || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(Ac, Aorig, Asz);
            (void)((fb_zsytrf_fn_t)cand->zsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(Ac, Aorig, Asz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_zsytrf_fn_t)cand->zsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }

    memcpy(Ac, Aorig, Asz);
    int info_c = ((fb_zsytrf_fn_t)cand->zsytrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                            lda, ipiv_c);
    if (info_c != 0) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    double recon_err = fb_ldlt_reconstruction_cf64(Aorig, Ac, n, lda, 0);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_chetrf(const fb_backend_vtable_t *oracle,
                                                                        const fb_backend_vtable_t *cand,
                                                                        const fb_corpus_case_t *tc,
                                                                        fb_judge_factorization_result_t *res,
                                                                        uint64_t *ns_out) {
    typedef int (*fb_chetrf_fn_t)(int layout, char uplo, int n,
                                                                fb_complex_float_t *a, int lda, int *ipiv);
    if (!oracle->chetrf || !cand->chetrf)
        return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n;
    int lda = (tc->lda > 0) ? (int)tc->lda : n;
    if (n <= 0) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    fb_complex_float_t *Aorig =
            (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    if (!Aorig)
        return FB_JUDGE_ERR_ALLOC;
    fb_fill_herm_indef_cf32(Aorig, n, lda);

    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    if (!Ao || !ipiv_o) {
        free(Aorig);
        free(Ao);
        free(ipiv_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    int info_o = ((fb_chetrf_fn_t)oracle->chetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n,
                                                                                                Ao, lda, ipiv_o);
    free(Ao);
    if (info_o != 0) {
        free(Aorig);
        free(ipiv_o);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Ac || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(Ac, Aorig, Asz);
            (void)((fb_chetrf_fn_t)cand->chetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(Ac, Aorig, Asz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_chetrf_fn_t)cand->chetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }

    memcpy(Ac, Aorig, Asz);
    int info_c = ((fb_chetrf_fn_t)cand->chetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                            lda, ipiv_c);
    if (info_c != 0) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    double recon_err = fb_ldlt_reconstruction_cf32(Aorig, Ac, n, lda, 1);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zhetrf(const fb_backend_vtable_t *oracle,
                                                                        const fb_backend_vtable_t *cand,
                                                                        const fb_corpus_case_t *tc,
                                                                        fb_judge_factorization_result_t *res,
                                                                        uint64_t *ns_out) {
    typedef int (*fb_zhetrf_fn_t)(int layout, char uplo, int n,
                                                                fb_complex_double_t *a, int lda, int *ipiv);
    if (!oracle->zhetrf || !cand->zhetrf)
        return FB_JUDGE_ERR_NOT_IMPL;

    int n = (int)tc->n;
    int lda = (tc->lda > 0) ? (int)tc->lda : n;
    if (n <= 0) {
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    fb_complex_double_t *Aorig =
            (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    if (!Aorig)
        return FB_JUDGE_ERR_ALLOC;
    fb_fill_herm_indef_cf64(Aorig, n, lda);

    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    if (!Ao || !ipiv_o) {
        free(Aorig);
        free(Ao);
        free(ipiv_o);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    int info_o = ((fb_zhetrf_fn_t)oracle->zhetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n,
                                                                                                Ao, lda, ipiv_o);
    free(Ao);
    if (info_o != 0) {
        free(Aorig);
        free(ipiv_o);
        mark_oracle_fatal(res);
        return FB_JUDGE_OK;
    }

    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Ac || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_FACTORIZATION_WARMUP_RUNS; w++) {
            memcpy(Ac, Aorig, Asz);
            (void)((fb_zhetrf_fn_t)cand->zhetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
        }
        for (int t = 0; t < FB_FACTORIZATION_TIMING_RUNS; t++) {
            memcpy(Ac, Aorig, Asz);
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_zhetrf_fn_t)cand->zhetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                     lda, ipiv_c);
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }

    memcpy(Ac, Aorig, Asz);
    int info_c = ((fb_zhetrf_fn_t)cand->zhetrf)(FB_LAYOUT_ROW_MAJOR, 'L', n, Ac,
                                                                                            lda, ipiv_c);
    if (info_c != 0) {
        free(Aorig);
        free(ipiv_o);
        free(Ac);
        free(ipiv_c);
        mark_cand_fatal(res);
        return FB_JUDGE_OK;
    }

    double recon_err = fb_ldlt_reconstruction_cf64(Aorig, Ac, n, lda, 1);
    result_from_relerr(&res->reconstruction, recon_err);
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    free(Aorig);
    free(ipiv_o);
    free(Ac);
    free(ipiv_c);
    return FB_JUDGE_OK;
}

/**
 * Dispatch table for FACTORIZATION archetype operations.
 * Indexed by op_id; covers LU, Cholesky, QR variants.
 */
static const fb_factorization_runner_fn fb_factorization_dispatch[] = {
    [FB_OP_SGETRF] = run_sgetrf, [FB_OP_DGETRF] = run_dgetrf,
    [FB_OP_CGETRF] = run_cgetrf, [FB_OP_ZGETRF] = run_zgetrf,
    [FB_OP_SGBTRF] = run_sgbtrf, [FB_OP_DGBTRF] = run_dgbtrf,
    [FB_OP_CGBTRF] = run_cgbtrf, [FB_OP_ZGBTRF] = run_zgbtrf,
    [FB_OP_SPOTRF] = run_spotrf, [FB_OP_DPOTRF] = run_dpotrf,
    [FB_OP_CPOTRF] = run_cpotrf, [FB_OP_ZPOTRF] = run_zpotrf,
    [FB_OP_SGEQRF] = run_sgeqrf, [FB_OP_DGEQRF] = run_dgeqrf,
    [FB_OP_CGEQRF] = run_cgeqrf, [FB_OP_ZGEQRF] = run_zgeqrf,
    [FB_OP_SGEHRD] = run_sgehrd, [FB_OP_DGEHRD] = run_dgehrd,
    [FB_OP_CGEHRD] = run_cgehrd, [FB_OP_ZGEHRD] = run_zgehrd,
    [FB_OP_SGELQF] = run_sgelqf, [FB_OP_DGELQF] = run_dgelqf,
    [FB_OP_SGEQLF] = run_sgeqlf, [FB_OP_DGEQLF] = run_dgeqlf,
    [FB_OP_CGEQLF] = run_cgeqlf, [FB_OP_ZGEQLF] = run_zgeqlf,
    [FB_OP_SGERQF] = run_sgerqf, [FB_OP_DGERQF] = run_dgerqf,
    [FB_OP_CGERQF] = run_cgerqf, [FB_OP_ZGERQF] = run_zgerqf,
    [FB_OP_SGEQPF] = run_sgeqpf, [FB_OP_DGEQPF] = run_dgeqpf,
    [FB_OP_SGEQP3] = run_sgeqp3, [FB_OP_DGEQP3] = run_dgeqp3,
    [FB_OP_SORGQR] = run_sorgqr, [FB_OP_DORGQR] = run_dorgqr,
    [FB_OP_CUNGQR] = run_cungqr, [FB_OP_ZUNGQR] = run_zungqr,
    [FB_OP_SORMQR] = run_sormqr, [FB_OP_DORMQR] = run_dormqr,
    [FB_OP_CUNMQR] = run_cunmqr, [FB_OP_ZUNMQR] = run_zunmqr,
    [FB_OP_STRTRI] = run_strtri, [FB_OP_DTRTRI] = run_dtrtri,
    [FB_OP_CTRTRI] = run_ctrtri, [FB_OP_ZTRTRI] = run_ztrtri,
    [FB_OP_SGETRI] = run_sgetri, [FB_OP_DGETRI] = run_dgetri,
    [FB_OP_CGETRI] = run_cgetri, [FB_OP_ZGETRI] = run_zgetri,
    [FB_OP_SPOTRI] = run_spotri, [FB_OP_DPOTRI] = run_dpotri,
    [FB_OP_CPOTRI] = run_cpotri, [FB_OP_ZPOTRI] = run_zpotri,
    [FB_OP_SSYTRF] = run_ssytrf, [FB_OP_DSYTRF] = run_dsytrf,
    [FB_OP_CSYTRF] = run_csytrf, [FB_OP_ZSYTRF] = run_zsytrf,
    [FB_OP_CHETRF] = run_chetrf, [FB_OP_ZHETRF] = run_zhetrf,
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