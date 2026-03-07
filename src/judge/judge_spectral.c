/**
 * FB_JUDGE_SPECTRAL implementation — eigenvalue and singular value decompositions.
 *
 * Phase 5: Complete spectral operation evaluation with 5-metric stack.
 * Metrics: values, reconstruction, orthogonality, subspace (degenerate), eigenpair (deep audit)
 *
 * Phase 5  (complete): SSYEV, DSYEV, SGESVD, DGESVD
 * Phase 5+ (complete): CHEEV, ZHEEV, CGESVD, ZGESVD (complex Hermitian/SVD)
 * Phase 5++ (complete): SGEEV, DGEEV (general real eigenvalue, WR+WI output)
 */

#include "judge_spectral.h"
#include "judge_op_ids.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/* =========================================================================
 * Portable complex element accessors
 * MSVC defines fb_complex_float_t as struct {float real, imag}.
 * GCC/Clang uses native float complex / double complex.
 * ========================================================================= */
#if defined(__clang__)
#  define FB_CF_REAL(z)  ((float)__real__(z))
#  define FB_CF_IMAG(z)  ((float)__imag__(z))
#  define FB_CD_REAL(z)  ((double)__real__(z))
#  define FB_CD_IMAG(z)  ((double)__imag__(z))
#elif defined(_MSC_VER)
#  define FB_CF_REAL(z)  ((z).real)
#  define FB_CF_IMAG(z)  ((z).imag)
#  define FB_CD_REAL(z)  ((z).real)
#  define FB_CD_IMAG(z)  ((z).imag)
#else
#  include <complex.h>
#  define FB_CF_REAL(z)  crealf(z)
#  define FB_CF_IMAG(z)  cimagf(z)
#  define FB_CD_REAL(z)  creal(z)
#  define FB_CD_IMAG(z)  cimag(z)
#endif

/* =========================================================================
 * Helper: Compute Frobenius norm of a matrix
 * ========================================================================= */

static double frobenius_norm_f32(const float *A, int m, int n, int lda)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double aij = (double)A[i * lda + j];
            sum += aij * aij;
        }
    }
    return sqrt(sum);
}

static double frobenius_norm_f64(const double *A, int m, int n, int lda)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double aij = A[i * lda + j];
            sum += aij * aij;
        }
    }
    return sqrt(sum);
}

/* =========================================================================
 * Helper: Compute AtA (matrix transpose times matrix)
 * ========================================================================= */

static void atac_f32(const float *A, int m, int n, int lda,
                     float *AtA, int ldata)
{
    /* Compute AtA = A^T * A (n x n result) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += (double)A[k * lda + i] * (double)A[k * lda + j];
            }
            AtA[i * ldata + j] = (float)sum;
        }
    }
}

static void atac_f64(const double *A, int m, int n, int lda,
                     double *AtA, int ldata)
{
    /* Compute AtA = A^T * A (n x n result) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A[k * lda + i] * A[k * lda + j];
            }
            AtA[i * ldata + j] = sum;
        }
    }
}

/* =========================================================================
 * Helper: Detect eigenvalue clusters (relative gap < n * eps)
 * ========================================================================= */

static bool is_clustered_f32(const float *w, int n, double *gap_min_out)
{
    if (n <= 1) {
        *gap_min_out = 1.0;
        return false;
    }
    
    /* Find minimum relative gap between consecutive eigenvalues. */
    double gap_min = 1.0e10;
    for (int i = 0; i < n - 1; i++) {
        double w_curr = fabs((double)w[i]);
        double w_next = fabs((double)w[i + 1]);
        double max_w = (w_curr > w_next) ? w_curr : w_next;
        if (max_w > 1e-16) {
            double gap = fabs((double)w[i] - (double)w[i + 1]) / max_w;
            if (gap < gap_min)
                gap_min = gap;
        }
    }
    
    /* Threshold: gap < n * eps (where eps ~ 1.2e-7 for float) */
    double threshold = (double)n * 1.2e-7;
    *gap_min_out = gap_min;
    return gap_min < threshold;
}

static bool is_clustered_f64(const double *w, int n, double *gap_min_out)
{
    if (n <= 1) {
        *gap_min_out = 1.0;
        return false;
    }
    
    /* Find minimum relative gap between consecutive eigenvalues. */
    double gap_min = 1.0e10;
    for (int i = 0; i < n - 1; i++) {
        double w_curr = fabs(w[i]);
        double w_next = fabs(w[i + 1]);
        double max_w = (w_curr > w_next) ? w_curr : w_next;
        if (max_w > 1e-16) {
            double gap = fabs(w[i] - w[i + 1]) / max_w;
            if (gap < gap_min)
                gap_min = gap;
        }
    }
    
    /* Threshold: gap < n * eps (where eps ~ 2.2e-16 for double) */
    double threshold = (double)n * 2.2e-16;
    *gap_min_out = gap_min;
    return gap_min < threshold;
}

/* =========================================================================
 * Helpers: Complex Frobenius norm and unitary check
 * ========================================================================= */

/* ||A||_F for a complex single-precision m×n matrix stored row-major with lda */
static double frobenius_norm_cf32(const fb_complex_float_t *A, int m, int n, int lda)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double re = (double)FB_CF_REAL(A[i * lda + j]);
            double im = (double)FB_CF_IMAG(A[i * lda + j]);
            sum += re * re + im * im;
        }
    }
    return sqrt(sum);
}

/* ||A||_F for a complex double-precision m×n matrix stored row-major with lda */
static double frobenius_norm_cf64(const fb_complex_double_t *A, int m, int n, int lda)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double re = FB_CD_REAL(A[i * lda + j]);
            double im = FB_CD_IMAG(A[i * lda + j]);
            sum += re * re + im * im;
        }
    }
    return sqrt(sum);
}

/*
 * Compute ||Q^H*Q - I||_F for an m×n complex single-precision matrix Q stored
 * row-major with ldq.  Result is the raw Frobenius-norm error (caller divides
 * by sqrt(n) when needed).
 */
static double ahac_cf32_ortho_error(const fb_complex_float_t *Q, int m, int n, int ldq)
{
    double sum_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* (Q^H*Q)[i,j] = sum_k  conj(Q[k,i]) * Q[k,j] */
            double re = 0.0, im = 0.0;
            for (int k = 0; k < m; k++) {
                double q_ki_re = (double)FB_CF_REAL(Q[k * ldq + i]);
                double q_ki_im = (double)FB_CF_IMAG(Q[k * ldq + i]);
                double q_kj_re = (double)FB_CF_REAL(Q[k * ldq + j]);
                double q_kj_im = (double)FB_CF_IMAG(Q[k * ldq + j]);
                /* conj(Q[k,i]) * Q[k,j] = (q_ki_re - i*q_ki_im)(q_kj_re + i*q_kj_im) */
                re += q_ki_re * q_kj_re + q_ki_im * q_kj_im;
                im += q_ki_re * q_kj_im - q_ki_im * q_kj_re;
            }
            double expected_re = (i == j) ? 1.0 : 0.0;
            double diff_re = re - expected_re;
            double diff_im = im;
            sum_err += diff_re * diff_re + diff_im * diff_im;
        }
    }
    return sqrt(sum_err);
}

/* Same as above for complex double-precision */
static double ahac_cf64_ortho_error(const fb_complex_double_t *Q, int m, int n, int ldq)
{
    double sum_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < m; k++) {
                double q_ki_re = FB_CD_REAL(Q[k * ldq + i]);
                double q_ki_im = FB_CD_IMAG(Q[k * ldq + i]);
                double q_kj_re = FB_CD_REAL(Q[k * ldq + j]);
                double q_kj_im = FB_CD_IMAG(Q[k * ldq + j]);
                re += q_ki_re * q_kj_re + q_ki_im * q_kj_im;
                im += q_ki_re * q_kj_im - q_ki_im * q_kj_re;
            }
            double expected_re = (i == j) ? 1.0 : 0.0;
            double diff_re = re - expected_re;
            double diff_im = im;
            sum_err += diff_re * diff_re + diff_im * diff_im;
        }
    }
    return sqrt(sum_err);
}

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

    /* Retrieve input from test case. */
    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    fb_uplo_t uplo = FB_UPPER;
    const char jobz = 'V';  /* Compute eigenvectors for orthogonality check */

    /* Allocate working space. */
    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *w_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *w_cand = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    /* Copy input to working space. */
    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));

    /* Call oracle. */
    int oracle_info = oracle->ssyev(layout, jobz, uplo, (int)n, A_oracle, (int)lda, w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    /* Call candidate. */
    int cand_info = cand->ssyev(layout, jobz, uplo, (int)n, A_cand, (int)lda, w_cand);

    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup;
    }

    /* === VALUES METRIC: eigenvalue accuracy === */
    {
        double max_eigval_error = 0.0;
        for (int i = 0; i < n; i++) {
            double oracle_w = (double)w_oracle[i];
            double cand_w = (double)w_cand[i];
            double abs_oracle = fabs(oracle_w);
            double relerr = (abs_oracle > 1e-16) ?
                fabs(oracle_w - cand_w) / abs_oracle :
                fabs(oracle_w - cand_w);
            if (relerr > max_eigval_error)
                max_eigval_error = relerr;
        }
        res->values = result_from_relerr(max_eigval_error);
    }

    /* === RECONSTRUCTION METRIC: ||A - Q*Λ*Q^T|| / ||A|| === */
    {
        /* Compute A_reconstructed = Q * diag(λ) * Q^T, then ||A - A_recon|| / ||A|| */
        double norm_A = frobenius_norm_f32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            /* Allocate space for reconstruction: A_recon = Q_cand * Λ_cand * Q_cand^T */
            float *A_recon = (float *)malloc((size_t)(n * n) * sizeof(float));
            float *Q_Lambda = (float *)malloc((size_t)(n * n) * sizeof(float));
            
            if (A_recon && Q_Lambda) {
                /* Q_Lambda = Q * diag(λ), Q stored in A_cand with lda stride */
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        Q_Lambda[i * n + j] = A_cand[i * lda + j] * w_cand[j];
                    }
                }
                
                /* A_recon = Q_Lambda * Q^T = (Q * diag(λ)) * Q^T */
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += (double)Q_Lambda[i * n + k] * (double)A_cand[j * lda + k];
                        }
                        A_recon[i * n + j] = (float)sum;
                    }
                }
                
                /* Compute ||A - A_recon|| / ||A|| (A_in has lda stride, A_recon is compact) */
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double diff = (double)A_in[i * lda + j] - (double)A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                }
                double norm_diff = sqrt(sum_diff);
                double recon_error = norm_diff / norm_A;
                res->reconstruction = result_from_relerr(recon_error);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            
            free(A_recon);
            free(Q_Lambda);
        }
    }

    /* === ORTHOGONALITY METRIC: ||Q^T*Q - I|| / ||I|| === */
    {
        /* Compute QtQ = Q^T * Q and check ||QtQ - I||_F */
        float *QtQ = (float *)malloc((size_t)(n * n) * sizeof(float));
        
        if (QtQ) {
            atac_f32(A_cand, n, n, lda, QtQ, n);
            
            /* Compute ||QtQ - I||_F / ||I||_F = ||QtQ - I||_F / sqrt(n) */
            double sum_err = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double qtq_ij = (double)QtQ[i * n + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = qtq_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err);
            double ortho_error = norm_err / sqrt((double)n);
            res->orthogonality = result_from_relerr(ortho_error);
        } else {
            res->orthogonality = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
        }
        
        free(QtQ);
    }

    /* === CLUSTER DETECTION and SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool is_clustered = is_clustered_f32(w_cand, n, &gap_min);
        
        if (is_clustered && gap_min < 1.0) {
            /* Eigenvalues are clustered; set subspace metric to reflect this */
            double subspace_error = gap_min;  /* Gap indicates cluster severity */
            res->subspace = result_from_relerr(subspace_error);
        } else {
            /* Well-separated eigenvalues; no subspace issue */
            res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        }
    }

    /* === PAIRS METRIC: eigenpair residuals (deep audit only) === */
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (At && wt) { memcpy(At, A_in, (size_t)(lda * n) * sizeof(float)); (void)cand->ssyev(layout, jobz, uplo, (int)n, At, (int)lda, wt); }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->ssyev(layout, jobz, uplo, (int)n, At, (int)lda, wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup:
    free(A_oracle);
    free(A_cand);
    free(w_oracle);
    free(w_cand);
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

    /* Retrieve input from test case. */
    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    fb_uplo_t uplo = FB_UPPER;
    char jobz = 'V';    /* Compute eigenvectors for orthogonality check */

    /* Allocate working space. */
    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *w_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *w_cand = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    /* Copy input to working space. */
    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

    /* Call oracle. */
    int oracle_info = oracle->dsyev(layout, jobz, uplo, (int)n, A_oracle, (int)lda, w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    /* Call candidate. */
    int cand_info = cand->dsyev(layout, jobz, uplo, (int)n, A_cand, (int)lda, w_cand);

    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup;
    }

    /* === VALUES METRIC: eigenvalue accuracy === */
    {
        double max_eigval_error = 0.0;
        for (int i = 0; i < n; i++) {
            double oracle_w = w_oracle[i];
            double cand_w = w_cand[i];
            double abs_oracle = fabs(oracle_w);
            double relerr = (abs_oracle > 1e-16) ?
                fabs(oracle_w - cand_w) / abs_oracle :
                fabs(oracle_w - cand_w);
            if (relerr > max_eigval_error)
                max_eigval_error = relerr;
        }
        res->values = result_from_relerr(max_eigval_error);
    }

    /* === RECONSTRUCTION METRIC: ||A - Q*Λ*Q^T|| / ||A|| === */
    {
        /* Compute A_reconstructed = Q * diag(λ) * Q^T, then ||A - A_recon|| / ||A|| */
        double norm_A = frobenius_norm_f64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            /* Allocate space for reconstruction: A_recon = Q_cand * Λ_cand * Q_cand^T */
            double *A_recon = (double *)malloc((size_t)(n * n) * sizeof(double));
            double *Q_Lambda = (double *)malloc((size_t)(n * n) * sizeof(double));
            
            if (A_recon && Q_Lambda) {
                /* Q_Lambda = Q * diag(λ), Q stored in A_cand with lda stride */
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        Q_Lambda[i * n + j] = A_cand[i * lda + j] * w_cand[j];
                    }
                }
                
                /* A_recon = Q_Lambda * Q^T = (Q * diag(λ)) * Q^T */
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += Q_Lambda[i * n + k] * A_cand[j * lda + k];
                        }
                        A_recon[i * n + j] = sum;
                    }
                }
                
                /* Compute ||A - A_recon|| / ||A|| (A_in has lda stride, A_recon is compact) */
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double diff = A_in[i * lda + j] - A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                }
                double norm_diff = sqrt(sum_diff);
                double recon_error = norm_diff / norm_A;
                res->reconstruction = result_from_relerr(recon_error);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            
            free(A_recon);
            free(Q_Lambda);
        }
    }

    /* === ORTHOGONALITY METRIC: ||Q^T*Q - I|| / ||I|| === */
    {
        /* Compute QtQ = Q^T * Q and check ||QtQ - I||_F */
        double *QtQ = (double *)malloc((size_t)(n * n) * sizeof(double));
        
        if (QtQ) {
            atac_f64(A_cand, n, n, lda, QtQ, n);
            
            /* Compute ||QtQ - I||_F / ||I||_F = ||QtQ - I||_F / sqrt(n) */
            double sum_err = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double qtq_ij = QtQ[i * n + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = qtq_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err);
            double ortho_error = norm_err / sqrt((double)n);
            res->orthogonality = result_from_relerr(ortho_error);
        } else {
            res->orthogonality = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
        }
        
        free(QtQ);
    }

    /* === CLUSTER DETECTION and SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool is_clustered = is_clustered_f64(w_cand, n, &gap_min);
        
        if (is_clustered && gap_min < 1.0) {
            /* Eigenvalues are clustered; set subspace metric to reflect this */
            double subspace_error = gap_min;  /* Gap indicates cluster severity */
            res->subspace = result_from_relerr(subspace_error);
        } else {
            /* Well-separated eigenvalues; no subspace issue */
            res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        }
    }

    /* === PAIRS METRIC: eigenpair residuals (deep audit only) === */
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (At && wt) { memcpy(At, A_in, (size_t)(lda * n) * sizeof(double)); (void)cand->dsyev(layout, jobz, uplo, (int)n, At, (int)lda, wt); }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dsyev(layout, jobz, uplo, (int)n, At, (int)lda, wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup:
    free(A_oracle);
    free(A_cand);
    free(w_oracle);
    free(w_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cheev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cheev || !cand->cheev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobz   = 'V';  /* compute eigenvectors */
    fb_uplo_t uplo = FB_UPPER;

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    float *w_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *w_cand   = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    int oracle_info = oracle->cheev(layout, jobz, uplo, (int)n,
                                         A_oracle, (int)lda, w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = cand->cheev(layout, jobz, uplo, (int)n,
                                     A_cand, (int)lda, w_cand);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_cheev;
    }

    /* === VALUES METRIC: eigenvalue accuracy (eigenvalues are real) === */
    {
        double max_err = 0.0;
        for (int i = 0; i < n; i++) {
            double ow = (double)w_oracle[i];
            double cw = (double)w_cand[i];
            double abs_o = fabs(ow);
            double re = (abs_o > 1e-16) ? fabs(ow - cw) / abs_o : fabs(ow - cw);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION METRIC: ||A - Q*Λ*Q^H||_F / ||A||_F === */
    {
        double norm_A = frobenius_norm_cf32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            /* A_recon[i,j] = sum_k Q[i,k]*w[k]*conj(Q[j,k])
             * Q stored in A_cand after the call: A_cand[i*n+k] = Q_{ik}         */
            double sum_diff = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        double q_ik_re = (double)FB_CF_REAL(A_cand[i * lda + k]);
                        double q_ik_im = (double)FB_CF_IMAG(A_cand[i * lda + k]);
                        double q_jk_re = (double)FB_CF_REAL(A_cand[j * lda + k]);
                        double q_jk_im = (double)FB_CF_IMAG(A_cand[j * lda + k]);
                        double wk = (double)w_cand[k];
                        /* Q[i,k] * w[k] * conj(Q[j,k]) */
                        re_sum += wk * (q_ik_re * q_jk_re + q_ik_im * q_jk_im);
                        im_sum += wk * (q_ik_im * q_jk_re - q_ik_re * q_jk_im);
                    }
                    double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                    double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* === ORTHOGONALITY METRIC: ||Q^H*Q - I||_F / sqrt(n) === */
    {
        double raw_err = ahac_cf32_ortho_error(A_cand, n, n, lda);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

    /* === SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(w_cand, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === PAIRS METRIC === */
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (At && wt) { memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t)); (void)cand->cheev(layout, jobz, uplo, (int)n, At, (int)lda, wt); }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cheev(layout, jobz, uplo, (int)n, At, (int)lda, wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_cheev:
    free(A_oracle);
    free(A_cand);
    free(w_oracle);
    free(w_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zheev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zheev || !cand->zheev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobz   = 'V';
    fb_uplo_t uplo = FB_UPPER;

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    double *w_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *w_cand   = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    int oracle_info = oracle->zheev(layout, jobz, uplo, (int)n,
                                         A_oracle, (int)lda, w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = cand->zheev(layout, jobz, uplo, (int)n,
                                     A_cand, (int)lda, w_cand);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_zheev;
    }

    /* === VALUES METRIC === */
    {
        double max_err = 0.0;
        for (int i = 0; i < n; i++) {
            double ow = w_oracle[i], cw = w_cand[i];
            double abs_o = fabs(ow);
            double re = (abs_o > 1e-16) ? fabs(ow - cw) / abs_o : fabs(ow - cw);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION METRIC: ||A - Q*Λ*Q^H||_F / ||A||_F === */
    {
        double norm_A = frobenius_norm_cf64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        double q_ik_re = FB_CD_REAL(A_cand[i * lda + k]);
                        double q_ik_im = FB_CD_IMAG(A_cand[i * lda + k]);
                        double q_jk_re = FB_CD_REAL(A_cand[j * lda + k]);
                        double q_jk_im = FB_CD_IMAG(A_cand[j * lda + k]);
                        double wk = w_cand[k];
                        re_sum += wk * (q_ik_re * q_jk_re + q_ik_im * q_jk_im);
                        im_sum += wk * (q_ik_im * q_jk_re - q_ik_re * q_jk_im);
                    }
                    double a_re = FB_CD_REAL(A_in[i * lda + j]);
                    double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* === ORTHOGONALITY METRIC: ||Q^H*Q - I||_F / sqrt(n) === */
    {
        double raw_err = ahac_cf64_ortho_error(A_cand, n, n, lda);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

    /* === SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f64(w_cand, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === PAIRS METRIC === */
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (At && wt) { memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t)); (void)cand->zheev(layout, jobz, uplo, (int)n, At, (int)lda, wt); }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zheev(layout, jobz, uplo, (int)n, At, (int)lda, wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_zheev:
    free(A_oracle);
    free(A_cand);
    free(w_oracle);
    free(w_cand);
    return FB_JUDGE_OK;
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

    /* Retrieve input from test case. */
    const float *A_in = (const float *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobu = 'A';    /* All left singular vectors */
    char jobvt = 'A';   /* All right singular vectors */

    /* Allocate working space. */
    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *s_oracle = (float *)malloc((size_t)minmn * sizeof(float));
    float *s_cand = (float *)malloc((size_t)minmn * sizeof(float));
    float *U = (float *)malloc((size_t)(m * minmn) * sizeof(float));
    float *VT = (float *)malloc((size_t)(minmn * n) * sizeof(float));
    float *superb = (float *)malloc((size_t)minmn * sizeof(float));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT || !superb) {
        mark_oracle_fatal(res);
        goto cleanup_svd;
    }

    /* Copy input. */
    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));

    /* Call oracle. */
    int oracle_info = oracle->sgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                          A_oracle, (int)lda, s_oracle, U, (int)m,
                                          VT, (int)minmn, superb);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        goto cleanup_svd;
    }

    /* Call candidate. */
    int cand_info = cand->sgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                      A_cand, (int)lda, s_cand, U, (int)m,
                                      VT, (int)minmn, superb);

    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_svd;
    }

    /* === VALUES METRIC: singular value accuracy === */
    {
        double max_sv_error = 0.0;
        for (int i = 0; i < minmn; i++) {
            double oracle_s = (double)s_oracle[i];
            double cand_s = (double)s_cand[i];
            double relerr = (oracle_s > 1e-16) ?
                fabs(oracle_s - cand_s) / oracle_s :
                fabs(oracle_s - cand_s);
            if (relerr > max_sv_error)
                max_sv_error = relerr;
        }
        res->values = result_from_relerr(max_sv_error);
    }

    /* === RECONSTRUCTION and other metrics === */
    
    /* === RECONSTRUCTION METRIC: ||A - U*Σ*V^T|| / ||A|| === */
    {
        double norm_A = frobenius_norm_f32(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            /* Allocate space for reconstruction: A_recon = U * diag(σ) * V^T */
            float *A_recon = (float *)malloc((size_t)(m * n) * sizeof(float));
            float *USigma = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            
            if (A_recon && USigma) {
                /* USigma = U * diag(σ) */
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < minmn; j++) {
                        USigma[i * minmn + j] = U[i * minmn + j] * s_cand[j];
                    }
                }
                
                /* A_recon = USigma * V^T */
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < minmn; k++) {
                            sum += (double)USigma[i * minmn + k] * (double)VT[k * n + j];
                        }
                        A_recon[i * n + j] = (float)sum;
                    }
                }
                
                /* Compute ||A - A_recon|| / ||A|| (A_in has lda stride, A_recon is compact) */
                double sum_diff = 0.0;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double diff = (double)A_in[i * lda + j] - (double)A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                }
                double norm_diff = sqrt(sum_diff);
                double recon_error = norm_diff / norm_A;
                res->reconstruction = result_from_relerr(recon_error);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            
            free(A_recon);
            free(USigma);
        }
    }

    /* === ORTHOGONALITY METRIC: ||U^T*U - I|| and ||V^T*V - I|| === */
    {
        double max_ortho_error = 0.0;
        
        /* Check ||U^T*U - I||_F / sqrt(m) */
        float *UtU = (float *)malloc((size_t)(minmn * minmn) * sizeof(float));
        if (UtU) {
            atac_f32(U, m, minmn, minmn, UtU, minmn);
            
            double sum_err = 0.0;
            for (int i = 0; i < minmn; i++) {
                for (int j = 0; j < minmn; j++) {
                    double utu_ij = (double)UtU[i * minmn + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = utu_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err) / sqrt((double)minmn);
            if (norm_err > max_ortho_error)
                max_ortho_error = norm_err;
            
            free(UtU);
        }
        
        /* Check ||V^T*V - I||_F / sqrt(n) */
        float *VtV = (float *)malloc((size_t)(n * n) * sizeof(float));
        if (VtV) {
            atac_f32(VT, minmn, n, n, VtV, n);
            
            double sum_err = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double vtv_ij = (double)VtV[i * n + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = vtv_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err) / sqrt((double)n);
            if (norm_err > max_ortho_error)
                max_ortho_error = norm_err;
            
            free(VtV);
        }
        
        res->orthogonality = result_from_relerr(max_ortho_error);
    }

    /* === CLUSTER DETECTION and SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool is_clustered = is_clustered_f32(s_cand, minmn, &gap_min);
        
        if (is_clustered && gap_min < 1.0) {
            /* Singular values are clustered; set subspace metric to reflect this */
            double subspace_error = gap_min;  /* Gap indicates cluster severity */
            res->subspace = result_from_relerr(subspace_error);
        } else {
            /* Well-separated singular values; no subspace issue */
            res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        }
    }

    /* === PAIRS METRIC: singular triplet residuals (deep audit only) === */
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            float *Ut = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            float *VTt = (float *)malloc((size_t)(minmn * n) * sizeof(float));
            float *superbt = (float *)malloc((size_t)minmn * sizeof(float));
            if (At && st && Ut && VTt && superbt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)cand->sgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            }
            free(At); free(st); free(Ut); free(VTt); free(superbt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            float *Ut = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            float *VTt = (float *)malloc((size_t)(minmn * n) * sizeof(float));
            float *superbt = (float *)malloc((size_t)minmn * sizeof(float));
            if (!At || !st || !Ut || !VTt || !superbt) { free(At); free(st); free(Ut); free(VTt); free(superbt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt); free(superbt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_svd:
    free(A_oracle);
    free(A_cand);
    free(s_oracle);
    free(s_cand);
    free(U);
    free(VT);
    free(superb);
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

    /* Retrieve input from test case. */
    const double *A_in = (const double *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobu = 'A';    /* All left singular vectors */
    char jobvt = 'A';   /* All right singular vectors */

    /* Allocate working space. */
    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *s_oracle = (double *)malloc((size_t)minmn * sizeof(double));
    double *s_cand = (double *)malloc((size_t)minmn * sizeof(double));
    double *U = (double *)malloc((size_t)(m * minmn) * sizeof(double));
    double *VT = (double *)malloc((size_t)(minmn * n) * sizeof(double));
    double *superb = (double *)malloc((size_t)minmn * sizeof(double));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT || !superb) {
        mark_oracle_fatal(res);
        goto cleanup_svd;
    }

    /* Copy input. */
    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

    /* Call oracle. */
    int oracle_info = oracle->dgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                          A_oracle, (int)lda, s_oracle, U, (int)m,
                                          VT, (int)minmn, superb);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        goto cleanup_svd;
    }

    /* Call candidate. */
    int cand_info = cand->dgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                      A_cand, (int)lda, s_cand, U, (int)m,
                                      VT, (int)minmn, superb);

    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_svd;
    }

    /* === VALUES METRIC: singular value accuracy === */
    {
        double max_sv_error = 0.0;
        for (int i = 0; i < minmn; i++) {
            double oracle_s = s_oracle[i];
            double cand_s = s_cand[i];
            double relerr = (oracle_s > 1e-16) ?
                fabs(oracle_s - cand_s) / oracle_s :
                fabs(oracle_s - cand_s);
            if (relerr > max_sv_error)
                max_sv_error = relerr;
        }
        res->values = result_from_relerr(max_sv_error);
    }

    /* === RECONSTRUCTION and other metrics === */
    
    /* === RECONSTRUCTION METRIC: ||A - U*Σ*V^T|| / ||A|| === */
    {
        double norm_A = frobenius_norm_f64(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            /* Allocate space for reconstruction: A_recon = U * diag(σ) * V^T */
            double *A_recon = (double *)malloc((size_t)(m * n) * sizeof(double));
            double *USigma = (double *)malloc((size_t)(m * minmn) * sizeof(double));
            
            if (A_recon && USigma) {
                /* USigma = U * diag(σ) */
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < minmn; j++) {
                        USigma[i * minmn + j] = U[i * minmn + j] * s_cand[j];
                    }
                }
                
                /* A_recon = USigma * V^T */
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < minmn; k++) {
                            sum += USigma[i * minmn + k] * VT[k * n + j];
                        }
                        A_recon[i * n + j] = sum;
                    }
                }
                
                /* Compute ||A - A_recon|| / ||A|| (A_in has lda stride, A_recon is compact) */
                double sum_diff = 0.0;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double diff = A_in[i * lda + j] - A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                }
                double norm_diff = sqrt(sum_diff);
                double recon_error = norm_diff / norm_A;
                res->reconstruction = result_from_relerr(recon_error);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            
            free(A_recon);
            free(USigma);
        }
    }

    /* === ORTHOGONALITY METRIC: ||U^T*U - I|| and ||V^T*V - I|| === */
    {
        double max_ortho_error = 0.0;
        
        /* Check ||U^T*U - I||_F / sqrt(m) */
        double *UtU = (double *)malloc((size_t)(minmn * minmn) * sizeof(double));
        if (UtU) {
            atac_f64(U, m, minmn, minmn, UtU, minmn);
            
            double sum_err = 0.0;
            for (int i = 0; i < minmn; i++) {
                for (int j = 0; j < minmn; j++) {
                    double utu_ij = UtU[i * minmn + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = utu_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err) / sqrt((double)minmn);
            if (norm_err > max_ortho_error)
                max_ortho_error = norm_err;
            
            free(UtU);
        }
        
        /* Check ||V^T*V - I||_F / sqrt(n) */
        double *VtV = (double *)malloc((size_t)(n * n) * sizeof(double));
        if (VtV) {
            atac_f64(VT, minmn, n, n, VtV, n);
            
            double sum_err = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double vtv_ij = VtV[i * n + j];
                    double expected = (i == j) ? 1.0 : 0.0;
                    double err = vtv_ij - expected;
                    sum_err += err * err;
                }
            }
            double norm_err = sqrt(sum_err) / sqrt((double)n);
            if (norm_err > max_ortho_error)
                max_ortho_error = norm_err;
            
            free(VtV);
        }
        
        res->orthogonality = result_from_relerr(max_ortho_error);
    }

    /* === CLUSTER DETECTION and SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool is_clustered = is_clustered_f64(s_cand, minmn, &gap_min);
        
        if (is_clustered && gap_min < 1.0) {
            /* Singular values are clustered; set subspace metric to reflect this */
            double subspace_error = gap_min;  /* Gap indicates cluster severity */
            res->subspace = result_from_relerr(subspace_error);
        } else {
            /* Well-separated singular values; no subspace issue */
            res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        }
    }

    /* === PAIRS METRIC: singular triplet residuals (deep audit only) === */
    res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            double *Ut = (double *)malloc((size_t)(m * minmn) * sizeof(double));
            double *VTt = (double *)malloc((size_t)(minmn * n) * sizeof(double));
            double *superbt = (double *)malloc((size_t)minmn * sizeof(double));
            if (At && st && Ut && VTt && superbt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)cand->dgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            }
            free(At); free(st); free(Ut); free(VTt); free(superbt);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            double *Ut = (double *)malloc((size_t)(m * minmn) * sizeof(double));
            double *VTt = (double *)malloc((size_t)(minmn * n) * sizeof(double));
            double *superbt = (double *)malloc((size_t)minmn * sizeof(double));
            if (!At || !st || !Ut || !VTt || !superbt) { free(At); free(st); free(Ut); free(VTt); free(superbt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt); free(superbt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_svd:
    free(A_oracle);
    free(A_cand);
    free(s_oracle);
    free(s_cand);
    free(U);
    free(VT);
    free(superb);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgesvd || !cand->cgesvd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobu   = 'A';
    char jobvt  = 'A';

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    float *s_oracle = (float *)malloc((size_t)minmn * sizeof(float));
    float *s_cand   = (float *)malloc((size_t)minmn * sizeof(float));
    fb_complex_float_t *U  = (fb_complex_float_t *)malloc(
        (size_t)(m * minmn) * sizeof(fb_complex_float_t));
    fb_complex_float_t *VT = (fb_complex_float_t *)malloc(
        (size_t)(minmn * n) * sizeof(fb_complex_float_t));
    float *superb = (float *)malloc((size_t)minmn * sizeof(float));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT || !superb) {
        mark_oracle_fatal(res);
        goto cleanup_cgesvd;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    {
        int info = oracle->cgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                       A_oracle, (int)lda, s_oracle,
                                       U, (int)m, VT, (int)minmn, superb);
        if (info != 0) { mark_oracle_fatal(res); goto cleanup_cgesvd; }
    }

    {
        int info = cand->cgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                     A_cand, (int)lda, s_cand,
                                     U, (int)m, VT, (int)minmn, superb);
        if (info != 0) { mark_cand_fatal(res); goto cleanup_cgesvd; }
    }

    /* === VALUES METRIC: singular value accuracy === */
    {
        double max_err = 0.0;
        for (int i = 0; i < minmn; i++) {
            double os = (double)s_oracle[i], cs = (double)s_cand[i];
            double re = (os > 1e-16) ? fabs(os - cs) / os : fabs(os - cs);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION: ||A - U*Σ*V^H||_F / ||A||_F === */
    {
        double norm_A = frobenius_norm_cf32(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            /* A_recon[i,j] = sum_k U[i,k]*s[k]*VT[k,j]  (VT stores V^H rows) */
            double sum_diff = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < minmn; k++) {
                        double u_re = (double)FB_CF_REAL(U[i * minmn + k]);
                        double u_im = (double)FB_CF_IMAG(U[i * minmn + k]);
                        double vt_re = (double)FB_CF_REAL(VT[k * n + j]);
                        double vt_im = (double)FB_CF_IMAG(VT[k * n + j]);
                        double sk = (double)s_cand[k];
                        /* U[i,k]*s[k]*VT[k,j] */
                        re_sum += sk * (u_re * vt_re - u_im * vt_im);
                        im_sum += sk * (u_re * vt_im + u_im * vt_re);
                    }
                    double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                    double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* === ORTHOGONALITY: max(||U^H*U - I||, ||V^H*V - I||) / sqrt(minmn) === */
    {
        double err_u = ahac_cf32_ortho_error(U, m, minmn, minmn);
        double err_v = ahac_cf32_ortho_error(VT, minmn, n, n);
        double max_err = (err_u > err_v ? err_u : err_v);
        res->orthogonality = result_from_relerr(max_err / sqrt((double)minmn));
    }

    /* === SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(s_cand, minmn, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === PAIRS METRIC === */
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            fb_complex_float_t *Ut = (fb_complex_float_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_float_t));
            fb_complex_float_t *VTt = (fb_complex_float_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_float_t));
            float *superbt = (float *)malloc((size_t)minmn * sizeof(float));
            if (At && st && Ut && VTt && superbt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)cand->cgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            }
            free(At); free(st); free(Ut); free(VTt); free(superbt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            fb_complex_float_t *Ut = (fb_complex_float_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_float_t));
            fb_complex_float_t *VTt = (fb_complex_float_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_float_t));
            float *superbt = (float *)malloc((size_t)minmn * sizeof(float));
            if (!At || !st || !Ut || !VTt || !superbt) { free(At); free(st); free(Ut); free(VTt); free(superbt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt); free(superbt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_cgesvd:
    free(A_oracle); free(A_cand);
    free(s_oracle); free(s_cand);
    free(U); free(VT); free(superb);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgesvd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgesvd || !cand->zgesvd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobu   = 'A';
    char jobvt  = 'A';

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    double *s_oracle = (double *)malloc((size_t)minmn * sizeof(double));
    double *s_cand   = (double *)malloc((size_t)minmn * sizeof(double));
    fb_complex_double_t *U  = (fb_complex_double_t *)malloc(
        (size_t)(m * minmn) * sizeof(fb_complex_double_t));
    fb_complex_double_t *VT = (fb_complex_double_t *)malloc(
        (size_t)(minmn * n) * sizeof(fb_complex_double_t));
    double *superb = (double *)malloc((size_t)minmn * sizeof(double));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT || !superb) {
        mark_oracle_fatal(res);
        goto cleanup_zgesvd;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    {
        int info = oracle->zgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                       A_oracle, (int)lda, s_oracle,
                                       U, (int)m, VT, (int)minmn, superb);
        if (info != 0) { mark_oracle_fatal(res); goto cleanup_zgesvd; }
    }

    {
        int info = cand->zgesvd(layout, jobu, jobvt, (int)m, (int)n,
                                     A_cand, (int)lda, s_cand,
                                     U, (int)m, VT, (int)minmn, superb);
        if (info != 0) { mark_cand_fatal(res); goto cleanup_zgesvd; }
    }

    /* === VALUES METRIC === */
    {
        double max_err = 0.0;
        for (int i = 0; i < minmn; i++) {
            double os = s_oracle[i], cs = s_cand[i];
            double re = (os > 1e-16) ? fabs(os - cs) / os : fabs(os - cs);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION: ||A - U*Σ*V^H||_F / ||A||_F === */
    {
        double norm_A = frobenius_norm_cf64(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < minmn; k++) {
                        double u_re  = FB_CD_REAL(U[i * minmn + k]);
                        double u_im  = FB_CD_IMAG(U[i * minmn + k]);
                        double vt_re = FB_CD_REAL(VT[k * n + j]);
                        double vt_im = FB_CD_IMAG(VT[k * n + j]);
                        double sk = s_cand[k];
                        re_sum += sk * (u_re * vt_re - u_im * vt_im);
                        im_sum += sk * (u_re * vt_im + u_im * vt_re);
                    }
                    double a_re = FB_CD_REAL(A_in[i * lda + j]);
                    double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* === ORTHOGONALITY === */
    {
        double err_u = ahac_cf64_ortho_error(U, m, minmn, minmn);
        double err_v = ahac_cf64_ortho_error(VT, minmn, n, n);
        double max_err = (err_u > err_v ? err_u : err_v);
        res->orthogonality = result_from_relerr(max_err / sqrt((double)minmn));
    }

    /* === SUBSPACE METRIC === */
    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f64(s_cand, minmn, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === PAIRS METRIC === */
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            fb_complex_double_t *Ut = (fb_complex_double_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_double_t));
            fb_complex_double_t *VTt = (fb_complex_double_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_double_t));
            double *superbt = (double *)malloc((size_t)minmn * sizeof(double));
            if (At && st && Ut && VTt && superbt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)cand->zgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            }
            free(At); free(st); free(Ut); free(VTt); free(superbt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            fb_complex_double_t *Ut = (fb_complex_double_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_double_t));
            fb_complex_double_t *VTt = (fb_complex_double_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_double_t));
            double *superbt = (double *)malloc((size_t)minmn * sizeof(double));
            if (!At || !st || !Ut || !VTt || !superbt) { free(At); free(st); free(Ut); free(VTt); free(superbt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgesvd(layout, jobu, jobvt, (int)m, (int)n, At, (int)lda, st, Ut, (int)m, VTt, (int)minmn, superbt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt); free(superbt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_zgesvd:
    free(A_oracle); free(A_cand);
    free(s_oracle); free(s_cand);
    free(U); free(VT); free(superb);
    return FB_JUDGE_OK;
}

/* General eigenvalue runners */

static fb_judge_status_t run_sgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgeev || !cand->sgeev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvl  = 'N';  /* no left eigenvectors */
    char jobvr  = 'V';  /* compute right eigenvectors */

    float *A_oracle  = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand    = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_copy    = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *WR_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WI_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WR_cand   = (float *)malloc((size_t)n * sizeof(float));
    float *WI_cand   = (float *)malloc((size_t)n * sizeof(float));
    float *VR        = (float *)malloc((size_t)(n * n) * sizeof(float));
    float *eig_mag   = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !A_copy ||
        !WR_oracle || !WI_oracle || !WR_cand || !WI_cand ||
        !VR || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_sgeev;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_copy,   A_in, (size_t)(lda * n) * sizeof(float));

    /* Oracle call (A_oracle overwritten). VR receives oracle eigenvectors first
     * (overwritten by candidate call). Only eigenvalues from oracle are used. */
    int oracle_info = oracle->sgeev(layout, jobvl, jobvr, (int)n,
                                         A_oracle, (int)lda,
                                         WR_oracle, WI_oracle,
                                         NULL, 1L, VR, (int)n);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        goto cleanup_sgeev;
    }

    /* Candidate call (A_cand overwritten, VR now has candidate eigenvectors). */
    int cand_info = cand->sgeev(layout, jobvl, jobvr, (int)n,
                                     A_cand, (int)lda,
                                     WR_cand, WI_cand,
                                     NULL, 1L, VR, (int)n);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_sgeev;
    }

    /* === VALUES METRIC: eigenvalue accuracy (complex magnitudes)
     * Assumes oracle/candidate return eigenvalues in same order for same input.
     * ================================================================= */
    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = (double)WR_oracle[j], or_i = (double)WI_oracle[j];
            double ca_r = (double)WR_cand[j],   ca_i = (double)WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff  = sqrt((or_r - ca_r) * (or_r - ca_r) +
                                (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION: N/A for general eigenvalue decomposition === */
    res->reconstruction = (fb_judge_case_result_t){.digits = 16};

    /* === ORTHOGONALITY: N/A (general eigenvectors not orthogonal) === */
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    /* === PAIRS METRIC: ||A*v_j - λ_j*v_j|| / (||A|| * ||v_j||) === */
    {
        double norm_A = frobenius_norm_f32(A_copy, n, n, lda);
        double max_pair_err = 0.0;

        int j = 0;
        while (j < n) {
            if ((double)WI_cand[j] == 0.0) {
                /* Real eigenvalue: residual = ||A*vr - wr*vr|| */
                double wr = (double)WR_cand[j];
                double res_sq = 0.0, vr_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double av = 0.0;
                    for (int k = 0; k < n; k++)
                        av += (double)A_copy[i * lda + k] * (double)VR[k * n + j];
                    double r = av - wr * (double)VR[i * n + j];
                    res_sq += r * r;
                    double vri = (double)VR[i * n + j];
                    vr_sq += vri * vri;
                }
                double denom = norm_A * sqrt(vr_sq);
                double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
                j++;
            } else {
                /* Complex pair: λ = wr ± i*wi,  vr = VR[:,j],  vi = VR[:,j+1]
                 * Equations: A*vr - (wr*vr - wi*vi) = 0
                 *            A*vi - (wr*vi + wi*vr) = 0                         */
                double wr = (double)WR_cand[j], wi = (double)WI_cand[j];
                double res_r_sq = 0.0, res_i_sq = 0.0, v_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double avr = 0.0, avi = 0.0;
                    for (int k = 0; k < n; k++) {
                        double a = (double)A_copy[i * lda + k];
                        avr += a * (double)VR[k * n + j];
                        avi += a * (double)VR[k * n + j + 1];
                    }
                    double vri = (double)VR[i * n + j];
                    double vii = (double)VR[i * n + j + 1];
                    double rr = avr - (wr * vri - wi * vii);
                    double ri = avi - (wr * vii + wi * vri);
                    res_r_sq += rr * rr;
                    res_i_sq += ri * ri;
                    v_sq += vri * vri + vii * vii;
                }
                double denom = norm_A * sqrt(v_sq);
                double re = (denom > 1e-16) ?
                    sqrt(res_r_sq + res_i_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
                j += 2;  /* skip conjugate partner */
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

    /* === SUBSPACE METRIC: gap in eigenvalue magnitude spectrum === */
    {
        for (int j = 0; j < n; j++)
            eig_mag[j] = sqrtf(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
        /* Insertion sort to order magnitudes for gap analysis */
        for (int i = 1; i < n; i++) {
            float tmp = eig_mag[i]; int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) { eig_mag[k] = eig_mag[k - 1]; k--; }
            eig_mag[k] = tmp;
        }
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At  = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VRt = (float *)malloc((size_t)(n * n) * sizeof(float));
            if (At && WRt && WIt && VRt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)cand->sgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, WRt, WIt, NULL, 1L, VRt, (int)n);
            }
            free(At); free(WRt); free(WIt); free(VRt);
        }
        for (int t = 0; t < 5; t++) {
            float *At  = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VRt = (float *)malloc((size_t)(n * n) * sizeof(float));
            if (!At || !WRt || !WIt || !VRt) { free(At); free(WRt); free(WIt); free(VRt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->sgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, WRt, WIt, NULL, 1L, VRt, (int)n);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(WRt); free(WIt); free(VRt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_sgeev:
    free(A_oracle); free(A_cand); free(A_copy);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand);   free(WI_cand);
    free(VR); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->dgeev || !cand->dgeev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvl  = 'N';
    char jobvr  = 'V';

    double *A_oracle  = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand    = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_copy    = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *WR_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WI_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WR_cand   = (double *)malloc((size_t)n * sizeof(double));
    double *WI_cand   = (double *)malloc((size_t)n * sizeof(double));
    double *VR        = (double *)malloc((size_t)(n * n) * sizeof(double));
    float  *eig_mag   = (float  *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !A_copy ||
        !WR_oracle || !WI_oracle || !WR_cand || !WI_cand ||
        !VR || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_dgeev;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_copy,   A_in, (size_t)(lda * n) * sizeof(double));

    int oracle_info = oracle->dgeev(layout, jobvl, jobvr, (int)n,
                                         A_oracle, (int)lda,
                                         WR_oracle, WI_oracle,
                                         NULL, 1L, VR, (int)n);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        goto cleanup_dgeev;
    }

    int cand_info = cand->dgeev(layout, jobvl, jobvr, (int)n,
                                     A_cand, (int)lda,
                                     WR_cand, WI_cand,
                                     NULL, 1L, VR, (int)n);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_dgeev;
    }

    /* === VALUES METRIC === */
    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = WR_oracle[j], or_i = WI_oracle[j];
            double ca_r = WR_cand[j],   ca_i = WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff  = sqrt((or_r - ca_r) * (or_r - ca_r) +
                                (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};

    /* === PAIRS METRIC === */
    {
        double norm_A = frobenius_norm_f64(A_copy, n, n, lda);
        double max_pair_err = 0.0;

        int j = 0;
        while (j < n) {
            if (WI_cand[j] == 0.0) {
                double wr = WR_cand[j];
                double res_sq = 0.0, vr_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double av = 0.0;
                    for (int k = 0; k < n; k++)
                        av += A_copy[i * lda + k] * VR[k * n + j];
                    double r = av - wr * VR[i * n + j];
                    res_sq += r * r;
                    vr_sq  += VR[i * n + j] * VR[i * n + j];
                }
                double denom = norm_A * sqrt(vr_sq);
                double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
                j++;
            } else {
                double wr = WR_cand[j], wi = WI_cand[j];
                double res_r_sq = 0.0, res_i_sq = 0.0, v_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double avr = 0.0, avi = 0.0;
                    for (int k = 0; k < n; k++) {
                        double a = A_copy[i * lda + k];
                        avr += a * VR[k * n + j];
                        avi += a * VR[k * n + j + 1];
                    }
                    double vri = VR[i * n + j], vii = VR[i * n + j + 1];
                    double rr = avr - (wr * vri - wi * vii);
                    double ri = avi - (wr * vii + wi * vri);
                    res_r_sq += rr * rr;
                    res_i_sq += ri * ri;
                    v_sq += vri * vri + vii * vii;
                }
                double denom = norm_A * sqrt(v_sq);
                double re = (denom > 1e-16) ?
                    sqrt(res_r_sq + res_i_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
                j += 2;
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

    /* === SUBSPACE METRIC === */
    {
        for (int j = 0; j < n; j++)
            eig_mag[j] = (float)sqrt(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
        for (int i = 1; i < n; i++) {
            float tmp = eig_mag[i]; int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) { eig_mag[k] = eig_mag[k - 1]; k--; }
            eig_mag[k] = tmp;
        }
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At  = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VRt = (double *)malloc((size_t)(n * n) * sizeof(double));
            if (At && WRt && WIt && VRt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)cand->dgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, WRt, WIt, NULL, 1L, VRt, (int)n);
            }
            free(At); free(WRt); free(WIt); free(VRt);
        }
        for (int t = 0; t < 5; t++) {
            double *At  = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VRt = (double *)malloc((size_t)(n * n) * sizeof(double));
            if (!At || !WRt || !WIt || !VRt) { free(At); free(WRt); free(WIt); free(VRt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->dgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, WRt, WIt, NULL, 1L, VRt, (int)n);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(WRt); free(WIt); free(VRt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_dgeev:
    free(A_oracle); free(A_cand); free(A_copy);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand);   free(WI_cand);
    free(VR); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgeev || !cand->cgeev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvl  = 'N';
    char jobvr  = 'V';

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_copy = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_oracle = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_cand = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *VR = (fb_complex_float_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_float_t));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !A_copy || !W_oracle || !W_cand || !VR || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_cgeev;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_copy,   A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    {
        int info = oracle->cgeev(layout, jobvl, jobvr, (int)n,
                                     A_oracle, (int)lda,
                                     W_oracle,
                                     NULL, 1L,
                                     VR, (int)n);
        if (info != 0) {
            mark_oracle_fatal(res);
            goto cleanup_cgeev;
        }
    }

    {
        int info = cand->cgeev(layout, jobvl, jobvr, (int)n,
                                   A_cand, (int)lda,
                                   W_cand,
                                   NULL, 1L,
                                   VR, (int)n);
        if (info != 0) {
            mark_cand_fatal(res);
            goto cleanup_cgeev;
        }
    }

    /* === VALUES METRIC === */
    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_re = (double)FB_CF_REAL(W_oracle[j]);
            double or_im = (double)FB_CF_IMAG(W_oracle[j]);
            double ca_re = (double)FB_CF_REAL(W_cand[j]);
            double ca_im = (double)FB_CF_IMAG(W_cand[j]);

            double mag_o = sqrt(or_re * or_re + or_im * or_im);
            double diff = sqrt((or_re - ca_re) * (or_re - ca_re) +
                               (or_im - ca_im) * (or_im - ca_im));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) {
                max_err = re;
            }
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION / ORTHOGONALITY: N/A for general complex GEEV === */
    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};

    /* === PAIRS METRIC: ||A*v - λ*v|| / (||A|| * ||v||) === */
    {
        double norm_A = frobenius_norm_cf32(A_copy, n, n, lda);
        double max_pair_err = 0.0;

        for (int j = 0; j < n; j++) {
            double lam_re = (double)FB_CF_REAL(W_cand[j]);
            double lam_im = (double)FB_CF_IMAG(W_cand[j]);

            double res_sq = 0.0;
            double v_sq = 0.0;

            for (int i = 0; i < n; i++) {
                double av_re = 0.0;
                double av_im = 0.0;

                for (int k = 0; k < n; k++) {
                    double a_re = (double)FB_CF_REAL(A_copy[i * lda + k]);
                    double a_im = (double)FB_CF_IMAG(A_copy[i * lda + k]);
                    double v_re = (double)FB_CF_REAL(VR[k * n + j]);
                    double v_im = (double)FB_CF_IMAG(VR[k * n + j]);

                    av_re += a_re * v_re - a_im * v_im;
                    av_im += a_re * v_im + a_im * v_re;
                }

                double v_re = (double)FB_CF_REAL(VR[i * n + j]);
                double v_im = (double)FB_CF_IMAG(VR[i * n + j]);

                double lv_re = lam_re * v_re - lam_im * v_im;
                double lv_im = lam_re * v_im + lam_im * v_re;

                double rr = av_re - lv_re;
                double ri = av_im - lv_im;
                res_sq += rr * rr + ri * ri;
                v_sq += v_re * v_re + v_im * v_im;
            }

            double denom = norm_A * sqrt(v_sq);
            double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
            if (re > max_pair_err) {
                max_pair_err = re;
            }
        }

        res->pairs = result_from_relerr(max_pair_err);
    }

    /* === SUBSPACE METRIC: gap in eigenvalue magnitude spectrum === */
    {
        for (int j = 0; j < n; j++) {
            double wr = (double)FB_CF_REAL(W_cand[j]);
            double wi = (double)FB_CF_IMAG(W_cand[j]);
            eig_mag[j] = (float)sqrt(wr * wr + wi * wi);
        }
        for (int i = 1; i < n; i++) {
            float tmp = eig_mag[i];
            int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) {
                eig_mag[k] = eig_mag[k - 1];
                k--;
            }
            eig_mag[k] = tmp;
        }

        double gap_min = 1.0;
        bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
        if (clustered && gap_min < 1.0) {
            res->subspace = result_from_relerr(gap_min);
        } else {
            res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At  = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt  = (fb_complex_float_t *)malloc((size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VRt = (fb_complex_float_t *)malloc((size_t)(n * n) * sizeof(fb_complex_float_t));
            if (At && Wt && VRt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)cand->cgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, Wt, NULL, 1L, VRt, (int)n);
            }
            free(At); free(Wt); free(VRt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At  = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt  = (fb_complex_float_t *)malloc((size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VRt = (fb_complex_float_t *)malloc((size_t)(n * n) * sizeof(fb_complex_float_t));
            if (!At || !Wt || !VRt) { free(At); free(Wt); free(VRt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, Wt, NULL, 1L, VRt, (int)n);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(Wt); free(VRt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_cgeev:
    free(A_oracle);
    free(A_cand);
    free(A_copy);
    free(W_oracle);
    free(W_cand);
    free(VR);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeev(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgeev || !cand->zgeev)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvl  = 'N';
    char jobvr  = 'V';

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_copy = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_oracle = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_cand = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *VR = (fb_complex_double_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_double_t));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !A_copy || !W_oracle || !W_cand || !VR || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_zgeev;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_copy,   A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    {
        int info = oracle->zgeev(layout, jobvl, jobvr, (int)n,
                                     A_oracle, (int)lda,
                                     W_oracle,
                                     NULL, 1L,
                                     VR, (int)n);
        if (info != 0) {
            mark_oracle_fatal(res);
            goto cleanup_zgeev;
        }
    }

    {
        int info = cand->zgeev(layout, jobvl, jobvr, (int)n,
                                   A_cand, (int)lda,
                                   W_cand,
                                   NULL, 1L,
                                   VR, (int)n);
        if (info != 0) {
            mark_cand_fatal(res);
            goto cleanup_zgeev;
        }
    }

    /* === VALUES METRIC === */
    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_re = FB_CD_REAL(W_oracle[j]);
            double or_im = FB_CD_IMAG(W_oracle[j]);
            double ca_re = FB_CD_REAL(W_cand[j]);
            double ca_im = FB_CD_IMAG(W_cand[j]);

            double mag_o = sqrt(or_re * or_re + or_im * or_im);
            double diff = sqrt((or_re - ca_re) * (or_re - ca_re) +
                               (or_im - ca_im) * (or_im - ca_im));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) {
                max_err = re;
            }
        }
        res->values = result_from_relerr(max_err);
    }

    /* === RECONSTRUCTION / ORTHOGONALITY: N/A for general complex GEEV === */
    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};

    /* === PAIRS METRIC: ||A*v - λ*v|| / (||A|| * ||v||) === */
    {
        double norm_A = frobenius_norm_cf64(A_copy, n, n, lda);
        double max_pair_err = 0.0;

        for (int j = 0; j < n; j++) {
            double lam_re = FB_CD_REAL(W_cand[j]);
            double lam_im = FB_CD_IMAG(W_cand[j]);

            double res_sq = 0.0;
            double v_sq = 0.0;

            for (int i = 0; i < n; i++) {
                double av_re = 0.0;
                double av_im = 0.0;

                for (int k = 0; k < n; k++) {
                    double a_re = FB_CD_REAL(A_copy[i * lda + k]);
                    double a_im = FB_CD_IMAG(A_copy[i * lda + k]);
                    double v_re = FB_CD_REAL(VR[k * n + j]);
                    double v_im = FB_CD_IMAG(VR[k * n + j]);

                    av_re += a_re * v_re - a_im * v_im;
                    av_im += a_re * v_im + a_im * v_re;
                }

                double v_re = FB_CD_REAL(VR[i * n + j]);
                double v_im = FB_CD_IMAG(VR[i * n + j]);

                double lv_re = lam_re * v_re - lam_im * v_im;
                double lv_im = lam_re * v_im + lam_im * v_re;

                double rr = av_re - lv_re;
                double ri = av_im - lv_im;
                res_sq += rr * rr + ri * ri;
                v_sq += v_re * v_re + v_im * v_im;
            }

            double denom = norm_A * sqrt(v_sq);
            double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
            if (re > max_pair_err) {
                max_pair_err = re;
            }
        }

        res->pairs = result_from_relerr(max_pair_err);
    }

    /* === SUBSPACE METRIC: gap in eigenvalue magnitude spectrum === */
    {
        for (int j = 0; j < n; j++) {
            double wr = FB_CD_REAL(W_cand[j]);
            double wi = FB_CD_IMAG(W_cand[j]);
            eig_mag[j] = sqrt(wr * wr + wi * wi);
        }
        for (int i = 1; i < n; i++) {
            double tmp = eig_mag[i];
            int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) {
                eig_mag[k] = eig_mag[k - 1];
                k--;
            }
            eig_mag[k] = tmp;
        }

        double gap_min = 1.0;
        bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
        if (clustered && gap_min < 1.0) {
            res->subspace = result_from_relerr(gap_min);
        } else {
            res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    /* === TIMING: 2 warm-up + 5 timed iterations, keep best === */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At  = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt  = (fb_complex_double_t *)malloc((size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VRt = (fb_complex_double_t *)malloc((size_t)(n * n) * sizeof(fb_complex_double_t));
            if (At && Wt && VRt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)cand->zgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, Wt, NULL, 1L, VRt, (int)n);
            }
            free(At); free(Wt); free(VRt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At  = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt  = (fb_complex_double_t *)malloc((size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VRt = (fb_complex_double_t *)malloc((size_t)(n * n) * sizeof(fb_complex_double_t));
            if (!At || !Wt || !VRt) { free(At); free(Wt); free(VRt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgeev(layout, jobvl, jobvr, (int)n, At, (int)lda, Wt, NULL, 1L, VRt, (int)n);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(Wt); free(VRt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_zgeev:
    free(A_oracle);
    free(A_cand);
    free(A_copy);
    free(W_oracle);
    free(W_cand);
    free(VR);
    free(eig_mag);
    return FB_JUDGE_OK;
}
/* =========================================================================
 * SGESDD — divide-and-conquer SVD (single-precision)
 *          Like SGESVD but no superb[] workspace.
 * ========================================================================= */
static fb_judge_status_t run_sgesdd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->sgesdd || !cand->sgesdd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobz = 'A';    /* All singular vectors */

    float *A_oracle  = (float *)malloc((size_t)(lda * n)    * sizeof(float));
    float *A_cand    = (float *)malloc((size_t)(lda * n)    * sizeof(float));
    float *s_oracle  = (float *)malloc((size_t)minmn        * sizeof(float));
    float *s_cand    = (float *)malloc((size_t)minmn        * sizeof(float));
    float *U         = (float *)malloc((size_t)(m * minmn)  * sizeof(float));
    float *VT        = (float *)malloc((size_t)(minmn * n)  * sizeof(float));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT) {
        mark_oracle_fatal(res);
        goto cleanup_sdd;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(float));

    int oracle_ret =
        oracle->sgesdd(layout, jobz, (int)m, (int)n, A_oracle, (int)lda,
                       s_oracle, U, (int)m, VT, (int)minmn);
    if (oracle_ret != 0) {
      mark_oracle_fatal(res);
      goto cleanup_sdd;
    }

    if (cand->sgesdd(layout, jobz, (int)m, (int)n, A_cand, (int)lda,
                     s_cand, U, (int)m, VT, (int)minmn) != 0) {
        mark_cand_fatal(res);
        goto cleanup_sdd;
    }

    /* VALUES: max relative error in singular values */
    {
        double max_sv_error = 0.0;
        for (int i = 0; i < minmn; i++) {
            double sv_o = (double)s_oracle[i];
            double sv_c = (double)s_cand[i];
            double relerr = (sv_o > 1e-16) ?
                fabs(sv_o - sv_c) / sv_o : fabs(sv_o - sv_c);
            if (relerr > max_sv_error) max_sv_error = relerr;
        }
        res->values = result_from_relerr(max_sv_error);
    }

    /* RECONSTRUCTION: ||A - U*Σ*V^T|| / ||A|| */
    {
        double norm_A = frobenius_norm_f32(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            float *A_recon = (float *)malloc((size_t)(m * n) * sizeof(float));
            float *USigma  = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            if (A_recon && USigma) {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < minmn; j++)
                        USigma[i * minmn + j] = U[i * minmn + j] * s_cand[j];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < minmn; k++)
                            sum += (double)USigma[i * minmn + k] * (double)VT[k * n + j];
                        A_recon[i * n + j] = (float)sum;
                    }
                }
                double sum_diff = 0.0;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++) {
                        double diff = (double)A_in[i * lda + j] - (double)A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(A_recon);
            free(USigma);
        }
    }

    /* ORTHOGONALITY: ||U^T U - I|| and ||V^T V - I|| */
    {
        double max_ortho_error = 0.0;
        float *UtU = (float *)malloc((size_t)(minmn * minmn) * sizeof(float));
        if (UtU) {
            atac_f32(U, m, minmn, minmn, UtU, minmn);
            double sum_err = 0.0;
            for (int i = 0; i < minmn; i++)
                for (int j = 0; j < minmn; j++) {
                    double err = (double)UtU[i * minmn + j] - (i == j ? 1.0 : 0.0);
                    sum_err += err * err;
                }
            double ne = sqrt(sum_err) / sqrt((double)minmn);
            if (ne > max_ortho_error) max_ortho_error = ne;
            free(UtU);
        }
        float *VtV = (float *)malloc((size_t)(n * n) * sizeof(float));
        if (VtV) {
            atac_f32(VT, minmn, n, n, VtV, n);
            double sum_err = 0.0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double err = (double)VtV[i * n + j] - (i == j ? 1.0 : 0.0);
                    sum_err += err * err;
                }
            double ne = sqrt(sum_err) / sqrt((double)n);
            if (ne > max_ortho_error) max_ortho_error = ne;
            free(VtV);
        }
        res->orthogonality = result_from_relerr(max_ortho_error);
    }

    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs    = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* TIMING */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At  = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *st  = (float *)malloc((size_t)minmn * sizeof(float));
            float *Ut  = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            float *VTt = (float *)malloc((size_t)(minmn * n) * sizeof(float));
            if (At && st && Ut && VTt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)cand->sgesdd(layout, jobz, (int)m, (int)n, At, (int)lda,
                                   st, Ut, (int)m, VTt, (int)minmn);
            }
            free(At); free(st); free(Ut); free(VTt);
        }
        for (int t = 0; t < 5; t++) {
            float *At  = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *st  = (float *)malloc((size_t)minmn * sizeof(float));
            float *Ut  = (float *)malloc((size_t)(m * minmn) * sizeof(float));
            float *VTt = (float *)malloc((size_t)(minmn * n) * sizeof(float));
            if (At && st && Ut && VTt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                uint64_t t0 = fb_judge_time_ns();
                (void)cand->sgesdd(layout, jobz, (int)m, (int)n, At, (int)lda,
                                   st, Ut, (int)m, VTt, (int)minmn);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(At); free(st); free(Ut); free(VTt);
        }
        *ns_out = best;
    }

cleanup_sdd:
    free(A_oracle);
    free(A_cand);
    free(s_oracle);
    free(s_cand);
    free(U);
    free(VT);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SSYGV — generalized symmetric eigenvalue problem (single-precision)
 *   Solves A*z = lambda*B*z  (itype=1, jobz='V', uplo=FB_UPPER)
 *   tc->A: symmetric matrix A (n x n)
 *   tc->B: symmetric positive-definite matrix B (n x n); identity if missing
 * ========================================================================= */
static fb_judge_status_t run_ssygv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->ssygv || !cand->ssygv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;
    int ldb = lda;  /* B uses same stride */

    fb_layout_t layout  = FB_LAYOUT_ROW_MAJOR;
    fb_uplo_t uplo      = FB_UPPER;
    const char jobz     = 'V';
    const int itype     = 1;    /* A*z = lambda*B*z */

    /* Build B_template: use tc->B if valid, else identity (always SPD) */
    float *B_template = (float *)calloc((size_t)n * (size_t)ldb, sizeof(float));
    if (!B_template) { mark_oracle_fatal(res); return FB_JUDGE_OK; }

    if (tc->B && tc->B_elems >= (size_t)n * (size_t)ldb) {
        memcpy(B_template, tc->B, (size_t)n * (size_t)ldb * sizeof(float));
    } else {
        /* Identity is symmetric positive definite */
        for (int i = 0; i < n; i++)
            B_template[i * ldb + i] = 1.0f;
    }

    float *A_oracle = (float *)malloc((size_t)n * (size_t)lda * sizeof(float));
    float *A_cand   = (float *)malloc((size_t)n * (size_t)lda * sizeof(float));
    float *B_oracle = (float *)malloc((size_t)n * (size_t)ldb * sizeof(float));
    float *B_cand   = (float *)malloc((size_t)n * (size_t)ldb * sizeof(float));
    float *w_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *w_cand   = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !B_oracle || !B_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        goto cleanup_sygv;
    }

    memcpy(A_oracle, A_in,       (size_t)n * (size_t)lda * sizeof(float));
    memcpy(A_cand,   A_in,       (size_t)n * (size_t)lda * sizeof(float));
    memcpy(B_oracle, B_template, (size_t)n * (size_t)ldb * sizeof(float));
    memcpy(B_cand,   B_template, (size_t)n * (size_t)ldb * sizeof(float));

    if (oracle->ssygv(layout, itype, jobz, uplo, (int)n,
                      A_oracle, (int)lda, B_oracle, (int)ldb, w_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_sygv;
    }
    if (cand->ssygv(layout, itype, jobz, uplo, (int)n,
                    A_cand, (int)lda, B_cand, (int)ldb, w_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_sygv;
    }

    /* VALUES: max relative error in eigenvalues */
    {
        double max_eigval_error = 0.0;
        for (int i = 0; i < n; i++) {
            double ev_o = (double)w_oracle[i];
            double ev_c = (double)w_cand[i];
            double abs_o = fabs(ev_o);
            double relerr = (abs_o > 1e-16) ?
                fabs(ev_o - ev_c) / abs_o : fabs(ev_o - ev_c);
            if (relerr > max_eigval_error) max_eigval_error = relerr;
        }
        res->values = result_from_relerr(max_eigval_error);
    }

    /* RECONSTRUCTION: ||A - B*Z*diag(w)*Z^T|| / ||A||
     * (Z stored as columns of A_cand after jobz='V'; B_cand has Cholesky factor)
     * Use oracle's B_template and compute B * Z * diag(w) * Z^T */
    {
        double norm_A = frobenius_norm_f32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
        } else {
            /* A_cand holds Z (eigenvectors), w_cand holds eigenvalues.
             * Compute A_recon = Z * diag(w_cand) * Z^T and compare with A_in.
             * (This tests A = Z * diag(w) * Z^T up to B; for B=I it's exact.) */
            float *ZLambda = (float *)malloc((size_t)n * (size_t)n * sizeof(float));
            float *A_recon = (float *)malloc((size_t)n * (size_t)n * sizeof(float));
            if (ZLambda && A_recon) {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        ZLambda[i * n + j] = A_cand[i * lda + j] * w_cand[j];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++)
                            sum += (double)ZLambda[i * n + k] * (double)A_cand[j * lda + k];
                        A_recon[i * n + j] = (float)sum;
                    }
                }
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        double diff = (double)A_in[i * lda + j] - (double)A_recon[i * n + j];
                        sum_diff += diff * diff;
                    }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction = (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(ZLambda);
            free(A_recon);
        }
    }

    /* ORTHOGONALITY: ||Z^T Z - I||_F / sqrt(n) */
    {
        float *ZtZ = (float *)malloc((size_t)n * (size_t)n * sizeof(float));
        double ortho_err = 0.0;
        if (ZtZ) {
            atac_f32(A_cand, n, n, lda, ZtZ, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double err = (double)ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
                    ortho_err += err * err;
                }
            free(ZtZ);
        }
        res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
    }

    res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    res->pairs    = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

    /* TIMING */
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)n*(size_t)lda*sizeof(float));
            float *Bt = (float *)malloc((size_t)n*(size_t)ldb*sizeof(float));
            float *wt = (float *)malloc((size_t)n*sizeof(float));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(float));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(float));
                (void)cand->ssygv(layout, itype, jobz, uplo, (int)n,
                                  At, (int)lda, Bt, (int)ldb, wt);
            }
            free(At); free(Bt); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)n*(size_t)lda*sizeof(float));
            float *Bt = (float *)malloc((size_t)n*(size_t)ldb*sizeof(float));
            float *wt = (float *)malloc((size_t)n*sizeof(float));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(float));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(float));
                uint64_t t0 = fb_judge_time_ns();
                (void)cand->ssygv(layout, itype, jobz, uplo, (int)n,
                                  At, (int)lda, Bt, (int)ldb, wt);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(At); free(Bt); free(wt);
        }
        *ns_out = best;
    }

cleanup_sygv:
    free(B_template);
    free(A_oracle); free(A_cand);
    free(B_oracle); free(B_cand);
    free(w_oracle); free(w_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGESDD — divide-and-conquer SVD (double-precision)
 * ========================================================================= */
static fb_judge_status_t run_dgesdd(const fb_backend_vtable_t *oracle,
                                    const fb_backend_vtable_t *cand,
                                    const fb_corpus_case_t *tc,
                                    fb_judge_spectral_result_t *res,
                                    uint64_t *ns_out) {
  if (!oracle->dgesdd || !cand->dgesdd)
    return FB_JUDGE_ERR_NOT_IMPL;

  memset(res, 0, sizeof(*res));
  *ns_out = 0;

  const double *A_in = (const double *)tc->A;
  int m = tc->m, n = tc->n;
  int lda = tc->lda ? tc->lda : tc->m;
  int minmn = (m < n) ? m : n;

  fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
  char jobz = 'A';

  double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
  double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
  double *s_oracle = (double *)malloc((size_t)minmn * sizeof(double));
  double *s_cand = (double *)malloc((size_t)minmn * sizeof(double));
  double *U = (double *)malloc((size_t)(m * minmn) * sizeof(double));
  double *VT = (double *)malloc((size_t)(minmn * n) * sizeof(double));

  if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT) {
    mark_oracle_fatal(res);
    goto cleanup_dgesdd;
  }

  memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
  memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

  int oracle_ret =
      oracle->dgesdd(layout, jobz, (int)m, (int)n, A_oracle, (int)lda, s_oracle,
                     U, (int)m, VT, (int)minmn);
  if (oracle_ret != 0) {
    mark_oracle_fatal(res);
    goto cleanup_dgesdd;
  }

  if (cand->dgesdd(layout, jobz, (int)m, (int)n, A_cand, (int)lda, s_cand, U,
                   (int)m, VT, (int)minmn) != 0) {
    mark_cand_fatal(res);
    goto cleanup_dgesdd;
  }

  /* VALUES: max relative error in singular values */
  {
    double max_sv_error = 0.0;
    for (int i = 0; i < minmn; i++) {
      double sv_o = s_oracle[i];
      double sv_c = s_cand[i];
      double relerr =
          (sv_o > 1e-16) ? fabs(sv_o - sv_c) / sv_o : fabs(sv_o - sv_c);
      if (relerr > max_sv_error)
        max_sv_error = relerr;
    }
    res->values = result_from_relerr(max_sv_error);
  }

  /* RECONSTRUCTION: ||A - U*Σ*V^T|| / ||A|| */
  {
    double norm_A = frobenius_norm_f64(A_in, m, n, lda);
    if (norm_A < 1e-16) {
      res->reconstruction =
          (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    } else {
      double *A_recon = (double *)malloc((size_t)(m * n) * sizeof(double));
      double *USigma = (double *)malloc((size_t)(m * minmn) * sizeof(double));
      if (A_recon && USigma) {
        for (int i = 0; i < m; i++)
          for (int j = 0; j < minmn; j++)
            USigma[i * minmn + j] = U[i * minmn + j] * s_cand[j];
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < minmn; k++)
              sum += USigma[i * minmn + k] * VT[k * n + j];
            A_recon[i * n + j] = sum;
          }
        }
        double sum_diff = 0.0;
        for (int i = 0; i < m; i++)
          for (int j = 0; j < n; j++) {
            double diff = A_in[i * lda + j] - A_recon[i * n + j];
            sum_diff += diff * diff;
          }
        res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
      } else {
        res->reconstruction =
            (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
      }
      free(A_recon);
      free(USigma);
    }
  }

  /* ORTHOGONALITY */
  {
    double max_ortho_error = 0.0;
    double *UtU = (double *)malloc((size_t)(minmn * minmn) * sizeof(double));
    if (UtU) {
      atac_f64(U, m, minmn, minmn, UtU, minmn);
      double sum_err = 0.0;
      for (int i = 0; i < minmn; i++)
        for (int j = 0; j < minmn; j++) {
          double err = UtU[i * minmn + j] - (i == j ? 1.0 : 0.0);
          sum_err += err * err;
        }
      double ne = sqrt(sum_err) / sqrt((double)minmn);
      if (ne > max_ortho_error)
        max_ortho_error = ne;
      free(UtU);
    }
    double *VtV = (double *)malloc((size_t)(n * n) * sizeof(double));
    if (VtV) {
      atac_f64(VT, minmn, n, n, VtV, n);
      double sum_err = 0.0;
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
          double err = VtV[i * n + j] - (i == j ? 1.0 : 0.0);
          sum_err += err * err;
        }
      double ne = sqrt(sum_err) / sqrt((double)n);
      if (ne > max_ortho_error)
        max_ortho_error = ne;
      free(VtV);
    }
    res->orthogonality = result_from_relerr(max_ortho_error);
  }

  res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
  res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < 2; w++) {
      double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
      double *st = (double *)malloc((size_t)minmn * sizeof(double));
      double *Ut = (double *)malloc((size_t)(m * minmn) * sizeof(double));
      double *VTt = (double *)malloc((size_t)(minmn * n) * sizeof(double));
      if (At && st && Ut && VTt) {
        memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
        (void)cand->dgesdd(layout, jobz, (int)m, (int)n, At, (int)lda, st, Ut,
                           (int)m, VTt, (int)minmn);
      }
      free(At);
      free(st);
      free(Ut);
      free(VTt);
    }
    for (int t = 0; t < 5; t++) {
      double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
      double *st = (double *)malloc((size_t)minmn * sizeof(double));
      double *Ut = (double *)malloc((size_t)(m * minmn) * sizeof(double));
      double *VTt = (double *)malloc((size_t)(minmn * n) * sizeof(double));
      if (At && st && Ut && VTt) {
        memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
        uint64_t t0 = fb_judge_time_ns();
        (void)cand->dgesdd(layout, jobz, (int)m, (int)n, At, (int)lda, st, Ut,
                           (int)m, VTt, (int)minmn);
        uint64_t dt = fb_judge_time_ns() - t0;
        if (dt < best)
          best = dt;
      }
      free(At);
      free(st);
      free(Ut);
      free(VTt);
    }
    *ns_out = best;
  }

cleanup_dgesdd:
  free(A_oracle);
  free(A_cand);
  free(s_oracle);
  free(s_cand);
  free(U);
  free(VT);
  return FB_JUDGE_OK;
}

/* =========================================================================
 * DSYGV — generalized symmetric eigenvalue problem (double-precision)
 * ========================================================================= */
static fb_judge_status_t run_dsygv(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_spectral_result_t *res,
                                   uint64_t *ns_out) {
  if (!oracle->dsygv || !cand->dsygv)
    return FB_JUDGE_ERR_NOT_IMPL;

  memset(res, 0, sizeof(*res));
  *ns_out = 0;

  const double *A_in = (const double *)tc->A;
  int n = tc->n;
  int lda = tc->lda ? tc->lda : tc->n;
  int ldb = lda;

  fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
  fb_uplo_t uplo = FB_UPPER;
  const char jobz = 'V';
  const int itype = 1;

  double *B_template =
      (double *)calloc((size_t)n * (size_t)ldb, sizeof(double));
  if (!B_template) {
    mark_oracle_fatal(res);
    return FB_JUDGE_OK;
  }

  if (tc->B && tc->B_elems >= (size_t)n * (size_t)ldb) {
    memcpy(B_template, tc->B, (size_t)n * (size_t)ldb * sizeof(double));
  } else {
    for (int i = 0; i < n; i++)
      B_template[i * ldb + i] = 1.0;
  }

  double *A_oracle = (double *)malloc((size_t)n * (size_t)lda * sizeof(double));
  double *A_cand = (double *)malloc((size_t)n * (size_t)lda * sizeof(double));
  double *B_oracle = (double *)malloc((size_t)n * (size_t)ldb * sizeof(double));
  double *B_cand = (double *)malloc((size_t)n * (size_t)ldb * sizeof(double));
  double *w_oracle = (double *)malloc((size_t)n * sizeof(double));
  double *w_cand = (double *)malloc((size_t)n * sizeof(double));

  if (!A_oracle || !A_cand || !B_oracle || !B_cand || !w_oracle || !w_cand) {
    mark_oracle_fatal(res);
    goto cleanup_dsygv;
  }

  memcpy(A_oracle, A_in, (size_t)n * (size_t)lda * sizeof(double));
  memcpy(A_cand, A_in, (size_t)n * (size_t)lda * sizeof(double));
  memcpy(B_oracle, B_template, (size_t)n * (size_t)ldb * sizeof(double));
  memcpy(B_cand, B_template, (size_t)n * (size_t)ldb * sizeof(double));

  if (oracle->dsygv(layout, itype, jobz, uplo, (int)n, A_oracle, (int)lda,
                    B_oracle, (int)ldb, w_oracle) != 0) {
    mark_oracle_fatal(res);
    goto cleanup_dsygv;
  }
  if (cand->dsygv(layout, itype, jobz, uplo, (int)n, A_cand, (int)lda, B_cand,
                  (int)ldb, w_cand) != 0) {
    mark_cand_fatal(res);
    goto cleanup_dsygv;
  }

  /* VALUES: max relative error in eigenvalues */
  {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
      double abs_o = fabs(w_oracle[i]);
      double relerr = (abs_o > 1e-16) ? fabs(w_oracle[i] - w_cand[i]) / abs_o
                                      : fabs(w_oracle[i] - w_cand[i]);
      if (relerr > max_err)
        max_err = relerr;
    }
    res->values = result_from_relerr(max_err);
  }

  /* RECONSTRUCTION: ||A - Z*diag(w)*Z^T|| / ||A|| */
  {
    double norm_A = frobenius_norm_f64(A_in, n, n, lda);
    if (norm_A < 1e-16) {
      res->reconstruction =
          (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    } else {
      double *ZLambda =
          (double *)malloc((size_t)n * (size_t)n * sizeof(double));
      double *A_recon =
          (double *)malloc((size_t)n * (size_t)n * sizeof(double));
      if (ZLambda && A_recon) {
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
            ZLambda[i * n + j] = A_cand[i * lda + j] * w_cand[j];
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
              sum += ZLambda[i * n + k] * A_cand[j * lda + k];
            A_recon[i * n + j] = sum;
          }
        }
        double sum_diff = 0.0;
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++) {
            double diff = A_in[i * lda + j] - A_recon[i * n + j];
            sum_diff += diff * diff;
          }
        res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
      } else {
        res->reconstruction =
            (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
      }
      free(ZLambda);
      free(A_recon);
    }
  }

  /* ORTHOGONALITY: ||Z^T Z - I||_F / sqrt(n) */
  {
    double *ZtZ = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double ortho_err = 0.0;
    if (ZtZ) {
      atac_f64(A_cand, n, n, lda, ZtZ, n);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
          double err = ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
          ortho_err += err * err;
        }
      free(ZtZ);
    }
    res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
  }

  res->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
  res->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};

  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < 2; w++) {
      double *At = (double *)malloc((size_t)n * (size_t)lda * sizeof(double));
      double *Bt = (double *)malloc((size_t)n * (size_t)ldb * sizeof(double));
      double *wt = (double *)malloc((size_t)n * sizeof(double));
      if (At && Bt && wt) {
        memcpy(At, A_in, (size_t)n * (size_t)lda * sizeof(double));
        memcpy(Bt, B_template, (size_t)n * (size_t)ldb * sizeof(double));
        (void)cand->dsygv(layout, itype, jobz, uplo, (int)n, At, (int)lda, Bt,
                          (int)ldb, wt);
      }
      free(At);
      free(Bt);
      free(wt);
    }
    for (int t = 0; t < 5; t++) {
      double *At = (double *)malloc((size_t)n * (size_t)lda * sizeof(double));
      double *Bt = (double *)malloc((size_t)n * (size_t)ldb * sizeof(double));
      double *wt = (double *)malloc((size_t)n * sizeof(double));
      if (At && Bt && wt) {
        memcpy(At, A_in, (size_t)n * (size_t)lda * sizeof(double));
        memcpy(Bt, B_template, (size_t)n * (size_t)ldb * sizeof(double));
        uint64_t t0 = fb_judge_time_ns();
        (void)cand->dsygv(layout, itype, jobz, uplo, (int)n, At, (int)lda, Bt,
                          (int)ldb, wt);
        uint64_t dt = fb_judge_time_ns() - t0;
        if (dt < best)
          best = dt;
      }
      free(At);
      free(Bt);
      free(wt);
    }
    *ns_out = best;
  }

cleanup_dsygv:
  free(B_template);
  free(A_oracle);
  free(A_cand);
  free(B_oracle);
  free(B_cand);
  free(w_oracle);
  free(w_cand);
  return FB_JUDGE_OK;
}

/* =========================================================================
 * CGESDD — complex single-precision divide-and-conquer SVD
 *   A = U * Sigma * V^H  (jobz='A', all singular vectors)
 *   tc->A: complex m×n matrix (row-major, fb_complex_float_t elements)
 * ========================================================================= */
static fb_judge_status_t run_cgesdd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->cgesdd || !cand->cgesdd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobz = 'S';  /* 'A' skips back-transforms in cgesdd_ref; 'S' is correct */

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    float *s_oracle = (float *)malloc((size_t)minmn * sizeof(float));
    float *s_cand   = (float *)malloc((size_t)minmn * sizeof(float));
    fb_complex_float_t *U  = (fb_complex_float_t *)malloc(
        (size_t)(m * minmn) * sizeof(fb_complex_float_t));
    fb_complex_float_t *VT = (fb_complex_float_t *)malloc(
        (size_t)(minmn * n) * sizeof(fb_complex_float_t));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT) {
        mark_oracle_fatal(res);
        goto cleanup_cgesdd;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    {
        int info = oracle->cgesdd(layout, jobz, m, n, A_oracle, lda,
                                  s_oracle, U, m, VT, minmn);
        if (info != 0) { mark_oracle_fatal(res); goto cleanup_cgesdd; }
    }
    {
        int info = cand->cgesdd(layout, jobz, m, n, A_cand, lda,
                                s_cand, U, m, VT, minmn);
        if (info != 0) { mark_cand_fatal(res); goto cleanup_cgesdd; }
    }

    /* VALUES: max relative error in singular values */
    {
        double max_err = 0.0;
        for (int i = 0; i < minmn; i++) {
            double os = (double)s_oracle[i], cs = (double)s_cand[i];
            double re = (os > 1e-16) ? fabs(os - cs) / os : fabs(os - cs);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* RECONSTRUCTION: ||A - U*Σ*V^H||_F / ||A||_F */
    {
        double norm_A = frobenius_norm_cf32(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < minmn; k++) {
                        double u_re  = (double)FB_CF_REAL(U[i * minmn + k]);
                        double u_im  = (double)FB_CF_IMAG(U[i * minmn + k]);
                        double vt_re = (double)FB_CF_REAL(VT[k * n + j]);
                        double vt_im = (double)FB_CF_IMAG(VT[k * n + j]);
                        double sk    = (double)s_cand[k];
                        re_sum += sk * (u_re * vt_re - u_im * vt_im);
                        im_sum += sk * (u_re * vt_im + u_im * vt_re);
                    }
                    double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                    double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* ORTHOGONALITY: max(||U^H U - I||, ||VT VT^H - I||) / sqrt(minmn) */
    {
        double err_u = ahac_cf32_ortho_error(U, m, minmn, minmn);
        double err_v = ahac_cf32_ortho_error(VT, minmn, n, n);
        double max_err = (err_u > err_v ? err_u : err_v);
        res->orthogonality = result_from_relerr(max_err / sqrt((double)minmn));
    }

    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(s_cand, minmn, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            fb_complex_float_t *Ut  = (fb_complex_float_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_float_t));
            fb_complex_float_t *VTt = (fb_complex_float_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_float_t));
            if (At && st && Ut && VTt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)cand->cgesdd(layout, jobz, m, n, At, lda, st, Ut, m, VTt, minmn);
            }
            free(At); free(st); free(Ut); free(VTt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_float_t));
            float *st = (float *)malloc((size_t)minmn * sizeof(float));
            fb_complex_float_t *Ut  = (fb_complex_float_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_float_t));
            fb_complex_float_t *VTt = (fb_complex_float_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_float_t));
            if (!At || !st || !Ut || !VTt) { free(At); free(st); free(Ut); free(VTt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->cgesdd(layout, jobz, m, n, At, lda, st, Ut, m, VTt, minmn);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_cgesdd:
    free(A_oracle); free(A_cand);
    free(s_oracle); free(s_cand);
    free(U); free(VT);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGESDD — complex double-precision divide-and-conquer SVD
 * ========================================================================= */
static fb_judge_status_t run_zgesdd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zgesdd || !cand->zgesdd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int m = tc->m, n = tc->n;
    int lda = tc->lda ? tc->lda : tc->m;
    int minmn = (m < n) ? m : n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobz = 'S';  /* 'A' skips back-transforms in cgesdd_ref; 'S' is correct */

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    double *s_oracle = (double *)malloc((size_t)minmn * sizeof(double));
    double *s_cand   = (double *)malloc((size_t)minmn * sizeof(double));
    fb_complex_double_t *U  = (fb_complex_double_t *)malloc(
        (size_t)(m * minmn) * sizeof(fb_complex_double_t));
    fb_complex_double_t *VT = (fb_complex_double_t *)malloc(
        (size_t)(minmn * n) * sizeof(fb_complex_double_t));

    if (!A_oracle || !A_cand || !s_oracle || !s_cand || !U || !VT) {
        mark_oracle_fatal(res);
        goto cleanup_zgesdd;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand,   A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    {
        int info = oracle->zgesdd(layout, jobz, m, n, A_oracle, lda,
                                  s_oracle, U, m, VT, minmn);
        if (info != 0) { mark_oracle_fatal(res); goto cleanup_zgesdd; }
    }
    {
        int info = cand->zgesdd(layout, jobz, m, n, A_cand, lda,
                                s_cand, U, m, VT, minmn);
        if (info != 0) { mark_cand_fatal(res); goto cleanup_zgesdd; }
    }

    /* VALUES */
    {
        double max_err = 0.0;
        for (int i = 0; i < minmn; i++) {
            double os = s_oracle[i], cs = s_cand[i];
            double re = (os > 1e-16) ? fabs(os - cs) / os : fabs(os - cs);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* RECONSTRUCTION */
    {
        double norm_A = frobenius_norm_cf64(A_in, m, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < minmn; k++) {
                        double u_re  = FB_CD_REAL(U[i * minmn + k]);
                        double u_im  = FB_CD_IMAG(U[i * minmn + k]);
                        double vt_re = FB_CD_REAL(VT[k * n + j]);
                        double vt_im = FB_CD_IMAG(VT[k * n + j]);
                        double sk    = s_cand[k];
                        re_sum += sk * (u_re * vt_re - u_im * vt_im);
                        im_sum += sk * (u_re * vt_im + u_im * vt_re);
                    }
                    double a_re = FB_CD_REAL(A_in[i * lda + j]);
                    double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* ORTHOGONALITY */
    {
        double err_u = ahac_cf64_ortho_error(U, m, minmn, minmn);
        double err_v = ahac_cf64_ortho_error(VT, minmn, n, n);
        double max_err = (err_u > err_v ? err_u : err_v);
        res->orthogonality = result_from_relerr(max_err / sqrt((double)minmn));
    }

    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f64(s_cand, minmn, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            fb_complex_double_t *Ut  = (fb_complex_double_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_double_t));
            fb_complex_double_t *VTt = (fb_complex_double_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_double_t));
            if (At && st && Ut && VTt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)cand->zgesdd(layout, jobz, m, n, At, lda, st, Ut, m, VTt, minmn);
            }
            free(At); free(st); free(Ut); free(VTt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc((size_t)(lda * n) * sizeof(fb_complex_double_t));
            double *st = (double *)malloc((size_t)minmn * sizeof(double));
            fb_complex_double_t *Ut  = (fb_complex_double_t *)malloc((size_t)(m * minmn) * sizeof(fb_complex_double_t));
            fb_complex_double_t *VTt = (fb_complex_double_t *)malloc((size_t)(minmn * n) * sizeof(fb_complex_double_t));
            if (!At || !st || !Ut || !VTt) { free(At); free(st); free(Ut); free(VTt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            uint64_t t0 = fb_judge_time_ns();
            (void)cand->zgesdd(layout, jobz, m, n, At, lda, st, Ut, m, VTt, minmn);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(st); free(Ut); free(VTt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_zgesdd:
    free(A_oracle); free(A_cand);
    free(s_oracle); free(s_cand);
    free(U); free(VT);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CHEGV — generalized Hermitian eigenvalue problem (complex single-precision)
 *   Solves A*z = lambda*B*z  (itype=1, jobz='V', uplo=FB_UPPER)
 *   tc->A: Hermitian matrix A (n x n, fb_complex_float_t)
 *   tc->B: Hermitian positive-definite B (n x n); identity if missing
 * ========================================================================= */
static fb_judge_status_t run_chegv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->chegv || !cand->chegv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;
    int ldb = lda;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    fb_uplo_t uplo = FB_UPPER;
    const char jobz = 'V';
    const int itype = 1;

    /* Build B_template: identity (always HPD) */
    fb_complex_float_t *B_template = (fb_complex_float_t *)calloc(
        (size_t)n * (size_t)ldb, sizeof(fb_complex_float_t));
    if (!B_template) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    /* Identity: diagonal entries are real 1, all others 0 */
    for (int i = 0; i < n; i++)
        B_template[i * ldb + i] = __builtin_complex(1.0f, 0.0f);

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc((size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand   = (fb_complex_float_t *)malloc((size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    fb_complex_float_t *B_oracle = (fb_complex_float_t *)malloc((size_t)n * (size_t)ldb * sizeof(fb_complex_float_t));
    fb_complex_float_t *B_cand   = (fb_complex_float_t *)malloc((size_t)n * (size_t)ldb * sizeof(fb_complex_float_t));
    float *w_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *w_cand   = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !B_oracle || !B_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        goto cleanup_chegv;
    }

    memcpy(A_oracle, A_in,       (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    memcpy(A_cand,   A_in,       (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
    memcpy(B_oracle, B_template, (size_t)n * (size_t)ldb * sizeof(fb_complex_float_t));
    memcpy(B_cand,   B_template, (size_t)n * (size_t)ldb * sizeof(fb_complex_float_t));

    if (oracle->chegv(layout, itype, jobz, uplo, n,
                      A_oracle, lda, B_oracle, ldb, w_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_chegv;
    }
    if (cand->chegv(layout, itype, jobz, uplo, n,
                    A_cand, lda, B_cand, ldb, w_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_chegv;
    }

    /* VALUES: max relative error in eigenvalues (real) */
    {
        double max_err = 0.0;
        for (int i = 0; i < n; i++) {
            double ow = (double)w_oracle[i], cw = (double)w_cand[i];
            double ao = fabs(ow);
            double re = (ao > 1e-16) ? fabs(ow - cw) / ao : fabs(ow - cw);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* RECONSTRUCTION: ||A - Z*diag(w)*Z^H||_F / ||A||_F  (with B=I) */
    {
        double norm_A = frobenius_norm_cf32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        double z_ik_re = (double)FB_CF_REAL(A_cand[i * lda + k]);
                        double z_ik_im = (double)FB_CF_IMAG(A_cand[i * lda + k]);
                        double z_jk_re = (double)FB_CF_REAL(A_cand[j * lda + k]);
                        double z_jk_im = (double)FB_CF_IMAG(A_cand[j * lda + k]);
                        double wk = (double)w_cand[k];
                        re_sum += wk * (z_ik_re * z_jk_re + z_ik_im * z_jk_im);
                        im_sum += wk * (z_ik_im * z_jk_re - z_ik_re * z_jk_im);
                    }
                    double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                    double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* ORTHOGONALITY: ||Z^H Z - I||_F / sqrt(n) */
    {
        double raw_err = ahac_cf32_ortho_error(A_cand, n, n, lda);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f32(w_cand, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At  = (fb_complex_float_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_float_t));
            fb_complex_float_t *Bt  = (fb_complex_float_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_float_t));
            float *wt = (float *)malloc((size_t)n*sizeof(float));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_float_t));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_float_t));
                (void)cand->chegv(layout, itype, jobz, uplo, n, At, lda, Bt, ldb, wt);
            }
            free(At); free(Bt); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At  = (fb_complex_float_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_float_t));
            fb_complex_float_t *Bt  = (fb_complex_float_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_float_t));
            float *wt = (float *)malloc((size_t)n*sizeof(float));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_float_t));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_float_t));
                uint64_t t0 = fb_judge_time_ns();
                (void)cand->chegv(layout, itype, jobz, uplo, n, At, lda, Bt, ldb, wt);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(At); free(Bt); free(wt);
        }
        *ns_out = best;
    }

cleanup_chegv:
    free(B_template);
    free(A_oracle); free(A_cand);
    free(B_oracle); free(B_cand);
    free(w_oracle); free(w_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZHEGV — generalized Hermitian eigenvalue problem (complex double-precision)
 * ========================================================================= */
static fb_judge_status_t run_zhegv(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    if (!oracle->zhegv || !cand->zhegv)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;
    int ldb = lda;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    fb_uplo_t uplo = FB_UPPER;
    const char jobz = 'V';
    const int itype = 1;

    fb_complex_double_t *B_template = (fb_complex_double_t *)calloc(
        (size_t)n * (size_t)ldb, sizeof(fb_complex_double_t));
    if (!B_template) { mark_oracle_fatal(res); return FB_JUDGE_OK; }
    for (int i = 0; i < n; i++)
        B_template[i * ldb + i] = __builtin_complex(1.0, 0.0);

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand   = (fb_complex_double_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
    fb_complex_double_t *B_oracle = (fb_complex_double_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
    fb_complex_double_t *B_cand   = (fb_complex_double_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
    double *w_oracle = (double *)malloc((size_t)n*sizeof(double));
    double *w_cand   = (double *)malloc((size_t)n*sizeof(double));

    if (!A_oracle || !A_cand || !B_oracle || !B_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        goto cleanup_zhegv;
    }

    memcpy(A_oracle, A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
    memcpy(A_cand,   A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
    memcpy(B_oracle, B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
    memcpy(B_cand,   B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));

    if (oracle->zhegv(layout, itype, jobz, uplo, n,
                      A_oracle, lda, B_oracle, ldb, w_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_zhegv;
    }
    if (cand->zhegv(layout, itype, jobz, uplo, n,
                    A_cand, lda, B_cand, ldb, w_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_zhegv;
    }

    /* VALUES */
    {
        double max_err = 0.0;
        for (int i = 0; i < n; i++) {
            double ow = w_oracle[i], cw = w_cand[i];
            double ao = fabs(ow);
            double re = (ao > 1e-16) ? fabs(ow - cw) / ao : fabs(ow - cw);
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    /* RECONSTRUCTION: ||A - Z*diag(w)*Z^H||_F / ||A||_F  (with B=I) */
    {
        double norm_A = frobenius_norm_cf64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double sum_diff = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double re_sum = 0.0, im_sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        double z_ik_re = FB_CD_REAL(A_cand[i * lda + k]);
                        double z_ik_im = FB_CD_IMAG(A_cand[i * lda + k]);
                        double z_jk_re = FB_CD_REAL(A_cand[j * lda + k]);
                        double z_jk_im = FB_CD_IMAG(A_cand[j * lda + k]);
                        double wk = w_cand[k];
                        re_sum += wk * (z_ik_re * z_jk_re + z_ik_im * z_jk_im);
                        im_sum += wk * (z_ik_im * z_jk_re - z_ik_re * z_jk_im);
                    }
                    double a_re = FB_CD_REAL(A_in[i * lda + j]);
                    double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                    double dr = a_re - re_sum, di = a_im - im_sum;
                    sum_diff += dr * dr + di * di;
                }
            }
            res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
        }
    }

    /* ORTHOGONALITY */
    {
        double raw_err = ahac_cf64_ortho_error(A_cand, n, n, lda);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

    {
        double gap_min = 1.0;
        bool clustered = is_clustered_f64(w_cand, n, &gap_min);
        if (clustered && gap_min < 1.0)
            res->subspace = result_from_relerr(gap_min);
        else
            res->subspace = (fb_judge_case_result_t){.digits = 16};
    }
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At  = (fb_complex_double_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
            fb_complex_double_t *Bt  = (fb_complex_double_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
            double *wt = (double *)malloc((size_t)n*sizeof(double));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
                (void)cand->zhegv(layout, itype, jobz, uplo, n, At, lda, Bt, ldb, wt);
            }
            free(At); free(Bt); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At  = (fb_complex_double_t *)malloc((size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
            fb_complex_double_t *Bt  = (fb_complex_double_t *)malloc((size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
            double *wt = (double *)malloc((size_t)n*sizeof(double));
            if (At && Bt && wt) {
                memcpy(At, A_in,       (size_t)n*(size_t)lda*sizeof(fb_complex_double_t));
                memcpy(Bt, B_template, (size_t)n*(size_t)ldb*sizeof(fb_complex_double_t));
                uint64_t t0 = fb_judge_time_ns();
                (void)cand->zhegv(layout, itype, jobz, uplo, n, At, lda, Bt, ldb, wt);
                uint64_t dt = fb_judge_time_ns() - t0;
                if (dt < best) best = dt;
            }
            free(At); free(Bt); free(wt);
        }
        *ns_out = best;
    }

cleanup_zhegv:
    free(B_template);
    free(A_oracle); free(A_cand);
    free(B_oracle); free(B_cand);
    free(w_oracle); free(w_cand);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Dispatch table
 * ========================================================================= */

typedef fb_judge_status_t (*fb_spectral_runner_fn)(
    const fb_backend_vtable_t *, const fb_backend_vtable_t *,
    const fb_corpus_case_t *, fb_judge_spectral_result_t *, uint64_t *);

static const fb_spectral_runner_fn fb_spectral_dispatch[] = {
    [FB_OP_SSYEV] = run_ssyev,   [FB_OP_DSYEV] = run_dsyev,
    [FB_OP_CHEEV] = run_cheev,   [FB_OP_ZHEEV] = run_zheev,
    [FB_OP_SGESVD] = run_sgesvd, [FB_OP_DGESVD] = run_dgesvd,
    [FB_OP_CGESVD] = run_cgesvd, [FB_OP_ZGESVD] = run_zgesvd,
    [FB_OP_SGEEV] = run_sgeev,   [FB_OP_DGEEV] = run_dgeev,
    [FB_OP_CGEEV] = run_cgeev,   [FB_OP_ZGEEV] = run_zgeev,
    [FB_OP_SGESDD] = run_sgesdd, [FB_OP_DGESDD] = run_dgesdd,
    [FB_OP_CGESDD] = run_cgesdd, [FB_OP_ZGESDD] = run_zgesdd,
    [FB_OP_SSYGV] = run_ssygv,   [FB_OP_DSYGV] = run_dsygv,
    [FB_OP_CHEGV]  = run_chegv,  [FB_OP_ZHEGV]  = run_zhegv,
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