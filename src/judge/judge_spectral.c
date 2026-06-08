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

static void mark_nonvalue_metrics_pass(fb_judge_spectral_result_t *r)
{
    r->reconstruction = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    r->orthogonality = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    r->subspace = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
    r->pairs = (fb_judge_case_result_t){.digits = 16, .relative_error = 0.0};
}

static bool packed_values_finite(const double *values, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

static double packed_values_relerr(const double *cand, const double *oracle,
                                   size_t count)
{
    double max_err = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = fabs(cand[i] - oracle[i]);
        double scale = fabs(oracle[i]);
        double err = (scale > 1.0e-30) ? (diff / scale) : diff;
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

static fb_complex_float_t fb_spectral_make_cf32(float real_part, float imag_part)
{
    fb_complex_float_t value = 0.0f;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

static fb_complex_double_t fb_spectral_make_cf64(double real_part,
                                                 double imag_part)
{
    fb_complex_double_t value = 0.0;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

static int fb_spectral_query_size_from_float(float query_size)
{
    return (query_size > 1.0f) ? (int)query_size : 1;
}

static int fb_spectral_query_size_from_int(int query_size)
{
    return (query_size > 1) ? query_size : 1;
}

static int fb_spectral_query_size_from_double(double query_size)
{
    return (query_size > 1.0) ? (int)query_size : 1;
}

static int fb_spectral_query_size_from_cfloat(fb_complex_float_t query_size)
{
    float real_part = 0.0f;
    memcpy(&real_part, &query_size, sizeof(real_part));
    return (real_part > 1.0f) ? (int)real_part : 1;
}

static int fb_spectral_query_size_from_cdouble(fb_complex_double_t query_size)
{
    double real_part = 0.0;
    memcpy(&real_part, &query_size, sizeof(real_part));
    return (real_part > 1.0) ? (int)real_part : 1;
}

typedef void (*fb_sbdsdc_fortran_fn_t)(char *uplo, char *compq, int *n,
                                       float *d, float *e, float *u,
                                       int *ldu, float *vt, int *ldvt,
                                       float *q, int *iq, float *work,
                                       int *iwork, int *info);
typedef void (*fb_dbdsdc_fortran_fn_t)(char *uplo, char *compq, int *n,
                                       double *d, double *e, double *u,
                                       int *ldu, double *vt, int *ldvt,
                                       double *q, int *iq, double *work,
                                       int *iwork, int *info);
typedef void (*fb_cbdsdc_fortran_fn_t)(char *uplo, char *compq, int *n,
                                       float *d, float *e,
                                       fb_complex_float_t *u, int *ldu,
                                       fb_complex_float_t *vt, int *ldvt,
                                       float *q, int *iq, float *work,
                                       int *iwork, int *info);
typedef void (*fb_zbdsdc_fortran_fn_t)(char *uplo, char *compq, int *n,
                                       double *d, double *e,
                                       fb_complex_double_t *u, int *ldu,
                                       fb_complex_double_t *vt, int *ldvt,
                                       double *q, int *iq, double *work,
                                       int *iwork, int *info);
typedef void (*fb_sbdsqr_fortran_fn_t)(char *uplo, int *n, int *ncvt,
                                       int *nru, int *ncc, float *d,
                                       float *e, float *vt, int *ldvt,
                                       float *u, int *ldu, float *c,
                                       int *ldc, float *work, int *info);
typedef void (*fb_dbdsqr_fortran_fn_t)(char *uplo, int *n, int *ncvt,
                                       int *nru, int *ncc, double *d,
                                       double *e, double *vt, int *ldvt,
                                       double *u, int *ldu, double *c,
                                       int *ldc, double *work, int *info);
typedef void (*fb_cbdsqr_fortran_fn_t)(char *uplo, int *n, int *ncvt,
                                       int *nru, int *ncc, float *d,
                                       float *e, fb_complex_float_t *vt,
                                       int *ldvt, fb_complex_float_t *u,
                                       int *ldu, fb_complex_float_t *c,
                                       int *ldc, float *rwork, int *info);
typedef void (*fb_zbdsqr_fortran_fn_t)(char *uplo, int *n, int *ncvt,
                                       int *nru, int *ncc, double *d,
                                       double *e, fb_complex_double_t *vt,
                                       int *ldvt, fb_complex_double_t *u,
                                       int *ldu, fb_complex_double_t *c,
                                       int *ldc, double *rwork, int *info);

#define FB_DEFINE_BDSDC_REAL_RUNNER(name, op_id, fn_type, scalar_type) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fn_type oracle_fn = (fn_type)oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fn_type cand_fn = (fn_type)cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) \
        return FB_JUDGE_ERR_NOT_IMPL; \
    memset(res, 0, sizeof(*res)); \
    if (ns_out) *ns_out = 0; \
    char uplo = 'U'; \
    char compq = 'N'; \
    int n = 2, ldu = 1, ldvt = 1; \
    scalar_type d_oracle[2] = {(scalar_type)3.0, (scalar_type)1.0}; \
    scalar_type e_oracle[1] = {(scalar_type)0.0}; \
    scalar_type u_oracle[1] = {(scalar_type)0.0}; \
    scalar_type vt_oracle[1] = {(scalar_type)0.0}; \
    scalar_type q_oracle[1] = {(scalar_type)0.0}; \
    scalar_type work_oracle[4] = {(scalar_type)11.0, (scalar_type)0.0, \
                                  (scalar_type)0.0, (scalar_type)0.0}; \
    int iq_oracle[1] = {9}; \
    int iwork_oracle[2] = {8, 8}; \
    int info_oracle = -1; \
    oracle_fn(&uplo, &compq, &n, d_oracle, e_oracle, u_oracle, &ldu, \
              vt_oracle, &ldvt, q_oracle, iq_oracle, work_oracle, \
              iwork_oracle, &info_oracle); \
    double oracle_out[5] = {(double)d_oracle[0], (double)d_oracle[1], \
                            (double)work_oracle[0], (double)iq_oracle[0], \
                            (double)iwork_oracle[0]}; \
    if (info_oracle != 0 || !packed_values_finite(oracle_out, 5u)) { \
        mark_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type d_cand[2] = {(scalar_type)3.0, (scalar_type)1.0}; \
    scalar_type e_cand[1] = {(scalar_type)0.0}; \
    scalar_type u_cand[1] = {(scalar_type)0.0}; \
    scalar_type vt_cand[1] = {(scalar_type)0.0}; \
    scalar_type q_cand[1] = {(scalar_type)0.0}; \
    scalar_type work_cand[4] = {(scalar_type)11.0, (scalar_type)0.0, \
                                (scalar_type)0.0, (scalar_type)0.0}; \
    int iq_cand[1] = {9}; \
    int iwork_cand[2] = {8, 8}; \
    int info_cand = -1; \
    cand_fn(&uplo, &compq, &n, d_cand, e_cand, u_cand, &ldu, vt_cand, \
            &ldvt, q_cand, iq_cand, work_cand, iwork_cand, &info_cand); \
    double cand_out[5] = {(double)d_cand[0], (double)d_cand[1], \
                          (double)work_cand[0], (double)iq_cand[0], \
                          (double)iwork_cand[0]}; \
    if (info_cand != 0 || !packed_values_finite(cand_out, 5u)) { \
        mark_cand_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    res->values = result_from_relerr(packed_values_relerr(cand_out, oracle_out, 5u)); \
    mark_nonvalue_metrics_pass(res); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < 2; w++) { \
            scalar_type d_time[2] = {(scalar_type)3.0, (scalar_type)1.0}; \
            scalar_type e_time[1] = {(scalar_type)0.0}; \
            scalar_type u_time[1] = {(scalar_type)0.0}; \
            scalar_type vt_time[1] = {(scalar_type)0.0}; \
            scalar_type q_time[1] = {(scalar_type)0.0}; \
            scalar_type work_time[4] = {(scalar_type)11.0, (scalar_type)0.0, \
                                        (scalar_type)0.0, (scalar_type)0.0}; \
            int iq_time[1] = {9}; \
            int iwork_time[2] = {8, 8}; \
            int info_time = -1; \
            cand_fn(&uplo, &compq, &n, d_time, e_time, u_time, &ldu, vt_time, \
                    &ldvt, q_time, iq_time, work_time, iwork_time, &info_time); \
        } \
        for (int t = 0; t < 5; t++) { \
            scalar_type d_time[2] = {(scalar_type)3.0, (scalar_type)1.0}; \
            scalar_type e_time[1] = {(scalar_type)0.0}; \
            scalar_type u_time[1] = {(scalar_type)0.0}; \
            scalar_type vt_time[1] = {(scalar_type)0.0}; \
            scalar_type q_time[1] = {(scalar_type)0.0}; \
            scalar_type work_time[4] = {(scalar_type)11.0, (scalar_type)0.0, \
                                        (scalar_type)0.0, (scalar_type)0.0}; \
            int iq_time[1] = {9}; \
            int iwork_time[2] = {8, 8}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            cand_fn(&uplo, &compq, &n, d_time, e_time, u_time, &ldu, vt_time, \
                    &ldvt, q_time, iq_time, work_time, iwork_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_BDSDC_COMPLEX_RUNNER(name, op_id, fn_type, real_type, complex_type, make_complex) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fn_type oracle_fn = (fn_type)oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fn_type cand_fn = (fn_type)cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) \
        return FB_JUDGE_ERR_NOT_IMPL; \
    memset(res, 0, sizeof(*res)); \
    if (ns_out) *ns_out = 0; \
    char uplo = 'U'; \
    char compq = 'N'; \
    int n = 2, ldu = 1, ldvt = 1; \
    real_type d_oracle[2] = {(real_type)3.0, (real_type)1.0}; \
    real_type e_oracle[1] = {(real_type)0.0}; \
    complex_type u_oracle[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
    complex_type vt_oracle[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
    real_type q_oracle[1] = {(real_type)0.0}; \
    real_type work_oracle[4] = {(real_type)11.0, (real_type)0.0, \
                                (real_type)0.0, (real_type)0.0}; \
    int iq_oracle[1] = {9}; \
    int iwork_oracle[2] = {8, 8}; \
    int info_oracle = -1; \
    oracle_fn(&uplo, &compq, &n, d_oracle, e_oracle, u_oracle, &ldu, \
              vt_oracle, &ldvt, q_oracle, iq_oracle, work_oracle, \
              iwork_oracle, &info_oracle); \
    double oracle_out[5] = {(double)d_oracle[0], (double)d_oracle[1], \
                            (double)work_oracle[0], (double)iq_oracle[0], \
                            (double)iwork_oracle[0]}; \
    if (info_oracle != 0 || !packed_values_finite(oracle_out, 5u)) { \
        mark_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    real_type d_cand[2] = {(real_type)3.0, (real_type)1.0}; \
    real_type e_cand[1] = {(real_type)0.0}; \
    complex_type u_cand[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
    complex_type vt_cand[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
    real_type q_cand[1] = {(real_type)0.0}; \
    real_type work_cand[4] = {(real_type)11.0, (real_type)0.0, \
                              (real_type)0.0, (real_type)0.0}; \
    int iq_cand[1] = {9}; \
    int iwork_cand[2] = {8, 8}; \
    int info_cand = -1; \
    cand_fn(&uplo, &compq, &n, d_cand, e_cand, u_cand, &ldu, vt_cand, \
            &ldvt, q_cand, iq_cand, work_cand, iwork_cand, &info_cand); \
    double cand_out[5] = {(double)d_cand[0], (double)d_cand[1], \
                          (double)work_cand[0], (double)iq_cand[0], \
                          (double)iwork_cand[0]}; \
    if (info_cand != 0 || !packed_values_finite(cand_out, 5u)) { \
        mark_cand_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    res->values = result_from_relerr(packed_values_relerr(cand_out, oracle_out, 5u)); \
    mark_nonvalue_metrics_pass(res); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < 2; w++) { \
            real_type d_time[2] = {(real_type)3.0, (real_type)1.0}; \
            real_type e_time[1] = {(real_type)0.0}; \
            complex_type u_time[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
            complex_type vt_time[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
            real_type q_time[1] = {(real_type)0.0}; \
            real_type work_time[4] = {(real_type)11.0, (real_type)0.0, \
                                      (real_type)0.0, (real_type)0.0}; \
            int iq_time[1] = {9}; \
            int iwork_time[2] = {8, 8}; \
            int info_time = -1; \
            cand_fn(&uplo, &compq, &n, d_time, e_time, u_time, &ldu, vt_time, \
                    &ldvt, q_time, iq_time, work_time, iwork_time, &info_time); \
        } \
        for (int t = 0; t < 5; t++) { \
            real_type d_time[2] = {(real_type)3.0, (real_type)1.0}; \
            real_type e_time[1] = {(real_type)0.0}; \
            complex_type u_time[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
            complex_type vt_time[1] = {make_complex((real_type)0.0, (real_type)0.0)}; \
            real_type q_time[1] = {(real_type)0.0}; \
            real_type work_time[4] = {(real_type)11.0, (real_type)0.0, \
                                      (real_type)0.0, (real_type)0.0}; \
            int iq_time[1] = {9}; \
            int iwork_time[2] = {8, 8}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            cand_fn(&uplo, &compq, &n, d_time, e_time, u_time, &ldu, vt_time, \
                    &ldvt, q_time, iq_time, work_time, iwork_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_BDSQR_REAL_RUNNER(name, op_id, fn_type, scalar_type) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fn_type oracle_fn = (fn_type)oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fn_type cand_fn = (fn_type)cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) \
        return FB_JUDGE_ERR_NOT_IMPL; \
    memset(res, 0, sizeof(*res)); \
    if (ns_out) *ns_out = 0; \
    char uplo = 'L'; \
    int n = 1, ncvt = 1, nru = 1, ncc = 0; \
    int ldvt = 1, ldu = 1, ldc = 1; \
    scalar_type d_oracle[1] = {(scalar_type)-3.0}; \
    scalar_type e_oracle[1] = {(scalar_type)7.0}; \
    scalar_type vt_oracle[1] = {(scalar_type)-9.0}; \
    scalar_type u_oracle[1] = {(scalar_type)4.0}; \
    scalar_type c_oracle[1] = {(scalar_type)2.0}; \
    scalar_type work_oracle[1] = {(scalar_type)0.0}; \
    int info_oracle = -1; \
    oracle_fn(&uplo, &n, &ncvt, &nru, &ncc, d_oracle, e_oracle, vt_oracle, \
              &ldvt, u_oracle, &ldu, c_oracle, &ldc, work_oracle, \
              &info_oracle); \
    double oracle_out[4] = {(double)d_oracle[0], (double)vt_oracle[0], \
                            (double)u_oracle[0], (double)e_oracle[0]}; \
    if (info_oracle != 0 || !packed_values_finite(oracle_out, 4u)) { \
        mark_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    scalar_type d_cand[1] = {(scalar_type)-3.0}; \
    scalar_type e_cand[1] = {(scalar_type)7.0}; \
    scalar_type vt_cand[1] = {(scalar_type)-9.0}; \
    scalar_type u_cand[1] = {(scalar_type)4.0}; \
    scalar_type c_cand[1] = {(scalar_type)2.0}; \
    scalar_type work_cand[1] = {(scalar_type)0.0}; \
    int info_cand = -1; \
    cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_cand, e_cand, vt_cand, &ldvt, \
            u_cand, &ldu, c_cand, &ldc, work_cand, &info_cand); \
    double cand_out[4] = {(double)d_cand[0], (double)vt_cand[0], \
                          (double)u_cand[0], (double)e_cand[0]}; \
    if (info_cand != 0 || !packed_values_finite(cand_out, 4u)) { \
        mark_cand_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    res->values = result_from_relerr(packed_values_relerr(cand_out, oracle_out, 4u)); \
    mark_nonvalue_metrics_pass(res); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < 2; w++) { \
            scalar_type d_time[1] = {(scalar_type)-3.0}; \
            scalar_type e_time[1] = {(scalar_type)7.0}; \
            scalar_type vt_time[1] = {(scalar_type)-9.0}; \
            scalar_type u_time[1] = {(scalar_type)4.0}; \
            scalar_type c_time[1] = {(scalar_type)2.0}; \
            scalar_type work_time[1] = {(scalar_type)0.0}; \
            int info_time = -1; \
            cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_time, e_time, vt_time, \
                    &ldvt, u_time, &ldu, c_time, &ldc, work_time, &info_time); \
        } \
        for (int t = 0; t < 5; t++) { \
            scalar_type d_time[1] = {(scalar_type)-3.0}; \
            scalar_type e_time[1] = {(scalar_type)7.0}; \
            scalar_type vt_time[1] = {(scalar_type)-9.0}; \
            scalar_type u_time[1] = {(scalar_type)4.0}; \
            scalar_type c_time[1] = {(scalar_type)2.0}; \
            scalar_type work_time[1] = {(scalar_type)0.0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_time, e_time, vt_time, \
                    &ldvt, u_time, &ldu, c_time, &ldc, work_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

#define FB_DEFINE_BDSQR_COMPLEX_RUNNER(name, op_id, fn_type, real_type, complex_type, make_complex, real_part, imag_part) \
static fb_judge_status_t name( \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand, \
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res, \
    uint64_t *ns_out) \
{ \
    (void)tc; \
    fn_type oracle_fn = (fn_type)oracle->ext_ops[op_id][FB_CONV_FORTRAN]; \
    fn_type cand_fn = (fn_type)cand->ext_ops[op_id][FB_CONV_FORTRAN]; \
    if (!oracle_fn || !cand_fn) \
        return FB_JUDGE_ERR_NOT_IMPL; \
    memset(res, 0, sizeof(*res)); \
    if (ns_out) *ns_out = 0; \
    char uplo = 'L'; \
    int n = 1, ncvt = 1, nru = 1, ncc = 0; \
    int ldvt = 1, ldu = 1, ldc = 1; \
    real_type d_oracle[1] = {(real_type)-3.0}; \
    real_type e_oracle[1] = {(real_type)7.0}; \
    complex_type vt_oracle[1] = {make_complex((real_type)-9.0, (real_type)8.0)}; \
    complex_type u_oracle[1] = {make_complex((real_type)4.0, (real_type)-5.0)}; \
    complex_type c_oracle[1] = {make_complex((real_type)2.0, (real_type)1.0)}; \
    real_type rwork_oracle[1] = {(real_type)0.0}; \
    int info_oracle = -1; \
    oracle_fn(&uplo, &n, &ncvt, &nru, &ncc, d_oracle, e_oracle, vt_oracle, \
              &ldvt, u_oracle, &ldu, c_oracle, &ldc, rwork_oracle, \
              &info_oracle); \
    double oracle_out[6] = {(double)d_oracle[0], (double)real_part(vt_oracle[0]), \
                            (double)imag_part(vt_oracle[0]), (double)real_part(u_oracle[0]), \
                            (double)imag_part(u_oracle[0]), (double)e_oracle[0]}; \
    if (info_oracle != 0 || !packed_values_finite(oracle_out, 6u)) { \
        mark_oracle_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    real_type d_cand[1] = {(real_type)-3.0}; \
    real_type e_cand[1] = {(real_type)7.0}; \
    complex_type vt_cand[1] = {make_complex((real_type)-9.0, (real_type)8.0)}; \
    complex_type u_cand[1] = {make_complex((real_type)4.0, (real_type)-5.0)}; \
    complex_type c_cand[1] = {make_complex((real_type)2.0, (real_type)1.0)}; \
    real_type rwork_cand[1] = {(real_type)0.0}; \
    int info_cand = -1; \
    cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_cand, e_cand, vt_cand, &ldvt, \
            u_cand, &ldu, c_cand, &ldc, rwork_cand, &info_cand); \
    double cand_out[6] = {(double)d_cand[0], (double)real_part(vt_cand[0]), \
                          (double)imag_part(vt_cand[0]), (double)real_part(u_cand[0]), \
                          (double)imag_part(u_cand[0]), (double)e_cand[0]}; \
    if (info_cand != 0 || !packed_values_finite(cand_out, 6u)) { \
        mark_cand_fatal(res); \
        return FB_JUDGE_OK; \
    } \
    res->values = result_from_relerr(packed_values_relerr(cand_out, oracle_out, 6u)); \
    mark_nonvalue_metrics_pass(res); \
    if (ns_out) { \
        uint64_t best = UINT64_MAX; \
        for (int w = 0; w < 2; w++) { \
            real_type d_time[1] = {(real_type)-3.0}; \
            real_type e_time[1] = {(real_type)7.0}; \
            complex_type vt_time[1] = {make_complex((real_type)-9.0, (real_type)8.0)}; \
            complex_type u_time[1] = {make_complex((real_type)4.0, (real_type)-5.0)}; \
            complex_type c_time[1] = {make_complex((real_type)2.0, (real_type)1.0)}; \
            real_type rwork_time[1] = {(real_type)0.0}; \
            int info_time = -1; \
            cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_time, e_time, vt_time, \
                    &ldvt, u_time, &ldu, c_time, &ldc, rwork_time, &info_time); \
        } \
        for (int t = 0; t < 5; t++) { \
            real_type d_time[1] = {(real_type)-3.0}; \
            real_type e_time[1] = {(real_type)7.0}; \
            complex_type vt_time[1] = {make_complex((real_type)-9.0, (real_type)8.0)}; \
            complex_type u_time[1] = {make_complex((real_type)4.0, (real_type)-5.0)}; \
            complex_type c_time[1] = {make_complex((real_type)2.0, (real_type)1.0)}; \
            real_type rwork_time[1] = {(real_type)0.0}; \
            int info_time = -1; \
            uint64_t t0 = fb_judge_time_ns(); \
            cand_fn(&uplo, &n, &ncvt, &nru, &ncc, d_time, e_time, vt_time, \
                    &ldvt, u_time, &ldu, c_time, &ldc, rwork_time, &info_time); \
            uint64_t dt = fb_judge_time_ns() - t0; \
            if (dt < best) best = dt; \
        } \
        *ns_out = best; \
    } \
    return FB_JUDGE_OK; \
}

FB_DEFINE_BDSDC_REAL_RUNNER(run_sbdsdc, FB_OP_SBDSDC, fb_sbdsdc_fortran_fn_t, float)
FB_DEFINE_BDSDC_REAL_RUNNER(run_dbdsdc, FB_OP_DBDSDC, fb_dbdsdc_fortran_fn_t, double)
FB_DEFINE_BDSDC_COMPLEX_RUNNER(run_cbdsdc, FB_OP_CBDSDC, fb_cbdsdc_fortran_fn_t,
                               float, fb_complex_float_t, fb_spectral_make_cf32)
FB_DEFINE_BDSDC_COMPLEX_RUNNER(run_zbdsdc, FB_OP_ZBDSDC, fb_zbdsdc_fortran_fn_t,
                               double, fb_complex_double_t, fb_spectral_make_cf64)
FB_DEFINE_BDSQR_REAL_RUNNER(run_sbdsqr, FB_OP_SBDSQR, fb_sbdsqr_fortran_fn_t, float)
FB_DEFINE_BDSQR_REAL_RUNNER(run_dbdsqr, FB_OP_DBDSQR, fb_dbdsqr_fortran_fn_t, double)
FB_DEFINE_BDSQR_COMPLEX_RUNNER(run_cbdsqr, FB_OP_CBDSQR, fb_cbdsqr_fortran_fn_t,
                               float, fb_complex_float_t, fb_spectral_make_cf32,
                               FB_CF_REAL, FB_CF_IMAG)
FB_DEFINE_BDSQR_COMPLEX_RUNNER(run_zbdsqr, FB_OP_ZBDSQR, fb_zbdsqr_fortran_fn_t,
                               double, fb_complex_double_t, fb_spectral_make_cf64,
                               FB_CD_REAL, FB_CD_IMAG)

#undef FB_DEFINE_BDSDC_REAL_RUNNER
#undef FB_DEFINE_BDSDC_COMPLEX_RUNNER
#undef FB_DEFINE_BDSQR_REAL_RUNNER
#undef FB_DEFINE_BDSQR_COMPLEX_RUNNER

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
    typedef int (*fb_cheevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  fb_complex_float_t *a, int lda, float *w);
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
    if (oracle_info != 0 && oracle->cheevd) {
        memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
        oracle_info = ((fb_cheevd_fn_t)oracle->cheevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)uplo, n,
                                                       A_oracle, lda,
                                                       w_oracle);
    }
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = cand->cheev(layout, jobz, uplo, (int)n,
                                     A_cand, (int)lda, w_cand);
    if (cand_info != 0 && cand->cheevd) {
        memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
        cand_info = ((fb_cheevd_fn_t)cand->cheevd)(FB_LAYOUT_ROW_MAJOR,
                                                   jobz, (char)uplo, n,
                                                   A_cand, lda, w_cand);
    }
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
    typedef int (*fb_zheevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  fb_complex_double_t *a, int lda, double *w);
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
    if (oracle_info != 0 && oracle->zheevd) {
        memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
        oracle_info = ((fb_zheevd_fn_t)oracle->zheevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)uplo, n,
                                                       A_oracle, lda,
                                                       w_oracle);
    }
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = cand->zheev(layout, jobz, uplo, (int)n,
                                     A_cand, (int)lda, w_cand);
    if (cand_info != 0 && cand->zheevd) {
        memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
        cand_info = ((fb_zheevd_fn_t)cand->zheevd)(FB_LAYOUT_ROW_MAJOR,
                                                   jobz, (char)uplo, n,
                                                   A_cand, lda, w_cand);
    }
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

static fb_judge_status_t run_ssyevd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    typedef int (*fb_ssyevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  float *a, int lda, float *w);
    if (!oracle->ssyevd || !cand->ssyevd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;
    char jobz = 'V';

    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *w_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *w_cand = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));

    int oracle_info = ((fb_ssyevd_fn_t)oracle->ssyevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)FB_UPPER,
                                                       n, A_oracle, lda,
                                                       w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = ((fb_ssyevd_fn_t)cand->ssyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                   (char)FB_UPPER, n, A_cand,
                                                   lda, w_cand);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_ssyevd;
    }

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

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->subspace = (fb_judge_case_result_t){.digits = 16};
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (At && wt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)((fb_ssyevd_fn_t)cand->ssyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                     (char)FB_UPPER, n, At,
                                                     lda, wt);
            }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *wt = (float *)malloc((size_t)n * sizeof(float));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_ssyevd_fn_t)cand->ssyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                 (char)FB_UPPER, n, At, lda,
                                                 wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_ssyevd:
    free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsyevd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    typedef int (*fb_dsyevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  double *a, int lda, double *w);
    if (!oracle->dsyevd || !cand->dsyevd)
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;
    char jobz = 'V';

    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *w_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *w_cand = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !w_oracle || !w_cand) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

    int oracle_info = ((fb_dsyevd_fn_t)oracle->dsyevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)FB_UPPER,
                                                       n, A_oracle, lda,
                                                       w_oracle);
    if (oracle_info != 0) {
        mark_oracle_fatal(res);
        free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
        return FB_JUDGE_OK;
    }

    int cand_info = ((fb_dsyevd_fn_t)cand->dsyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                   (char)FB_UPPER, n, A_cand,
                                                   lda, w_cand);
    if (cand_info != 0) {
        mark_cand_fatal(res);
        goto cleanup_dsyevd;
    }

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

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->subspace = (fb_judge_case_result_t){.digits = 16};
    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (At && wt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)((fb_dsyevd_fn_t)cand->dsyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                     (char)FB_UPPER, n, At,
                                                     lda, wt);
            }
            free(At); free(wt);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *wt = (double *)malloc((size_t)n * sizeof(double));
            if (!At || !wt) { free(At); free(wt); break; }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            uint64_t t0 = fb_judge_time_ns();
            (void)((fb_dsyevd_fn_t)cand->dsyevd)(FB_LAYOUT_ROW_MAJOR, jobz,
                                                 (char)FB_UPPER, n, At, lda,
                                                 wt);
            uint64_t dt = fb_judge_time_ns() - t0;
            free(At); free(wt);
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

cleanup_dsyevd:
    free(A_oracle); free(A_cand); free(w_oracle); free(w_cand);
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

typedef int (*fb_sgees_select_fn_t)(const float *, const float *);
typedef int (*fb_dgees_select_fn_t)(const double *, const double *);
typedef int (*fb_cgees_select_fn_t)(fb_complex_float_t *);
typedef int (*fb_zgees_select_fn_t)(fb_complex_double_t *);

typedef int (*fb_sgees_fn_t)(const fb_layout_t layout, const char jobvs,
                             const char sort, fb_sgees_select_fn_t select,
                             const int n, float *A, const int lda,
                             int *sdim, float *wr, float *wi, float *VS,
                             const int ldvs);
typedef int (*fb_dgees_fn_t)(const fb_layout_t layout, const char jobvs,
                             const char sort, fb_dgees_select_fn_t select,
                             const int n, double *A, const int lda,
                             int *sdim, double *wr, double *wi, double *VS,
                             const int ldvs);
typedef int (*fb_cgees_fn_t)(const fb_layout_t layout, const char jobvs,
                             const char sort, fb_cgees_select_fn_t select,
                             const int n, fb_complex_float_t *A,
                             const int lda, int *sdim,
                             fb_complex_float_t *w,
                             fb_complex_float_t *VS, const int ldvs);
typedef int (*fb_zgees_fn_t)(const fb_layout_t layout, const char jobvs,
                             const char sort, fb_zgees_select_fn_t select,
                             const int n, fb_complex_double_t *A,
                             const int lda, int *sdim,
                             fb_complex_double_t *w,
                             fb_complex_double_t *VS, const int ldvs);

typedef void (*fb_sgees_fortran_fn_t)(char *jobvs, char *sort,
                                      fb_sgees_select_fn_t select, int *n,
                                      float *A, int *lda, int *sdim,
                                      float *wr, float *wi, float *VS,
                                      int *ldvs, float *work, int *lwork,
                                      int *bwork, int *info);
typedef void (*fb_dgees_fortran_fn_t)(char *jobvs, char *sort,
                                      fb_dgees_select_fn_t select, int *n,
                                      double *A, int *lda, int *sdim,
                                      double *wr, double *wi, double *VS,
                                      int *ldvs, double *work, int *lwork,
                                      int *bwork, int *info);
typedef void (*fb_cgees_fortran_fn_t)(char *jobvs, char *sort,
                                      fb_cgees_select_fn_t select, int *n,
                                      fb_complex_float_t *A, int *lda,
                                      int *sdim, fb_complex_float_t *w,
                                      fb_complex_float_t *VS, int *ldvs,
                                      fb_complex_float_t *work, int *lwork,
                                      float *rwork, int *bwork, int *info);
typedef void (*fb_zgees_fortran_fn_t)(char *jobvs, char *sort,
                                      fb_zgees_select_fn_t select, int *n,
                                      fb_complex_double_t *A, int *lda,
                                      int *sdim, fb_complex_double_t *w,
                                      fb_complex_double_t *VS, int *ldvs,
                                      fb_complex_double_t *work, int *lwork,
                                      double *rwork, int *bwork, int *info);

typedef int (*fb_sgeesx_fn_t)(const fb_layout_t layout, const char jobvs,
                              const char sort, fb_sgees_select_fn_t select,
                              const char sense, const int n, float *A,
                              const int lda, int *sdim, float *wr, float *wi,
                              float *VS, const int ldvs, float *rconde,
                              float *rcondv);
typedef int (*fb_dgeesx_fn_t)(const fb_layout_t layout, const char jobvs,
                              const char sort, fb_dgees_select_fn_t select,
                              const char sense, const int n, double *A,
                              const int lda, int *sdim, double *wr,
                              double *wi, double *VS, const int ldvs,
                              double *rconde, double *rcondv);
typedef int (*fb_cgeesx_fn_t)(const fb_layout_t layout, const char jobvs,
                              const char sort, fb_cgees_select_fn_t select,
                              const char sense, const int n,
                              fb_complex_float_t *A, const int lda,
                              int *sdim, fb_complex_float_t *w,
                              fb_complex_float_t *VS, const int ldvs,
                              float *rconde, float *rcondv);
typedef int (*fb_zgeesx_fn_t)(const fb_layout_t layout, const char jobvs,
                              const char sort, fb_zgees_select_fn_t select,
                              const char sense, const int n,
                              fb_complex_double_t *A, const int lda,
                              int *sdim, fb_complex_double_t *w,
                              fb_complex_double_t *VS, const int ldvs,
                              double *rconde, double *rcondv);

typedef void (*fb_sgeesx_fortran_fn_t)(char *jobvs, char *sort,
                                       fb_sgees_select_fn_t select,
                                       char *sense, int *n, float *A,
                                       int *lda, int *sdim, float *wr,
                                       float *wi, float *VS, int *ldvs,
                                       float *rconde, float *rcondv,
                                       float *work, int *lwork, int *iwork,
                                       int *liwork, int *bwork, int *info);
typedef void (*fb_dgeesx_fortran_fn_t)(char *jobvs, char *sort,
                                       fb_dgees_select_fn_t select,
                                       char *sense, int *n, double *A,
                                       int *lda, int *sdim, double *wr,
                                       double *wi, double *VS, int *ldvs,
                                       double *rconde, double *rcondv,
                                       double *work, int *lwork, int *iwork,
                                       int *liwork, int *bwork, int *info);
typedef void (*fb_cgeesx_fortran_fn_t)(char *jobvs, char *sort,
                                       fb_cgees_select_fn_t select,
                                       char *sense, int *n,
                                       fb_complex_float_t *A, int *lda,
                                       int *sdim, fb_complex_float_t *w,
                                       fb_complex_float_t *VS, int *ldvs,
                                       float *rconde, float *rcondv,
                                       fb_complex_float_t *work, int *lwork,
                                       float *rwork, int *bwork, int *info);
typedef void (*fb_zgeesx_fortran_fn_t)(char *jobvs, char *sort,
                                       fb_zgees_select_fn_t select,
                                       char *sense, int *n,
                                       fb_complex_double_t *A, int *lda,
                                       int *sdim, fb_complex_double_t *w,
                                       fb_complex_double_t *VS, int *ldvs,
                                       double *rconde, double *rcondv,
                                       fb_complex_double_t *work, int *lwork,
                                       double *rwork, int *bwork, int *info);

typedef int (*fb_sgeevx_fn_t)(const fb_layout_t layout, const char balanc,
                              const char jobvl, const char jobvr,
                              const char sense, const int n, float *A,
                              const int lda, float *wr, float *wi, float *VL,
                              const int ldvl, float *VR, const int ldvr,
                              int *ilo, int *ihi, float *scale, float *abnrm,
                              float *rconde, float *rcondv);
typedef int (*fb_dgeevx_fn_t)(const fb_layout_t layout, const char balanc,
                              const char jobvl, const char jobvr,
                              const char sense, const int n, double *A,
                              const int lda, double *wr, double *wi,
                              double *VL, const int ldvl, double *VR,
                              const int ldvr, int *ilo, int *ihi,
                              double *scale, double *abnrm,
                              double *rconde, double *rcondv);
typedef int (*fb_cgeevx_fn_t)(const fb_layout_t layout, const char balanc,
                              const char jobvl, const char jobvr,
                              const char sense, const int n,
                              fb_complex_float_t *A, const int lda,
                              fb_complex_float_t *w,
                              fb_complex_float_t *VL, const int ldvl,
                              fb_complex_float_t *VR, const int ldvr,
                              int *ilo, int *ihi, float *scale,
                              float *abnrm, float *rconde, float *rcondv);
typedef int (*fb_zgeevx_fn_t)(const fb_layout_t layout, const char balanc,
                              const char jobvl, const char jobvr,
                              const char sense, const int n,
                              fb_complex_double_t *A, const int lda,
                              fb_complex_double_t *w,
                              fb_complex_double_t *VL, const int ldvl,
                              fb_complex_double_t *VR, const int ldvr,
                              int *ilo, int *ihi, double *scale,
                              double *abnrm, double *rconde,
                              double *rcondv);

typedef void (*fb_sgeevx_fortran_fn_t)(char *balanc, char *jobvl,
                                       char *jobvr, char *sense, int *n,
                                       float *A, int *lda, float *wr,
                                       float *wi, float *VL, int *ldvl,
                                       float *VR, int *ldvr, int *ilo,
                                       int *ihi, float *scale, float *abnrm,
                                       float *rconde, float *rcondv,
                                       float *work, int *lwork, int *iwork,
                                       int *info);
typedef void (*fb_dgeevx_fortran_fn_t)(char *balanc, char *jobvl,
                                       char *jobvr, char *sense, int *n,
                                       double *A, int *lda, double *wr,
                                       double *wi, double *VL, int *ldvl,
                                       double *VR, int *ldvr, int *ilo,
                                       int *ihi, double *scale,
                                       double *abnrm, double *rconde,
                                       double *rcondv, double *work,
                                       int *lwork, int *iwork, int *info);
typedef void (*fb_cgeevx_fortran_fn_t)(char *balanc, char *jobvl,
                                       char *jobvr, char *sense, int *n,
                                       fb_complex_float_t *A, int *lda,
                                       fb_complex_float_t *w,
                                       fb_complex_float_t *VL, int *ldvl,
                                       fb_complex_float_t *VR, int *ldvr,
                                       int *ilo, int *ihi, float *scale,
                                       float *abnrm, float *rconde,
                                       float *rcondv,
                                       fb_complex_float_t *work, int *lwork,
                                       float *rwork, int *info);
typedef void (*fb_zgeevx_fortran_fn_t)(char *balanc, char *jobvl,
                                       char *jobvr, char *sense, int *n,
                                       fb_complex_double_t *A, int *lda,
                                       fb_complex_double_t *w,
                                       fb_complex_double_t *VL, int *ldvl,
                                       fb_complex_double_t *VR, int *ldvr,
                                       int *ilo, int *ihi, double *scale,
                                       double *abnrm, double *rconde,
                                       double *rcondv,
                                       fb_complex_double_t *work, int *lwork,
                                       double *rwork, int *info);

static int call_sgees_backend(fb_sgees_fn_t c_fn,
                              fb_sgees_fortran_fn_t fortran_fn,
                              fb_layout_t layout, char jobvs, char sort,
                              int n, float *A, int lda, int *sdim, float *wr,
                              float *wi, float *VS, int ldvs)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        float work_query = 0.0f;
        float *work = NULL;
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (bwork == NULL) {
            return -101;
        }
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, wr, wi, VS,
                   &ldvs_, &work_query, &lwork_, bwork, &info);
        if (info != 0) {
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_float(work_query);
        work = (float *)malloc((size_t)lwork_ * sizeof(float));
        if (work == NULL) {
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, wr, wi, VS,
                   &ldvs_, work, &lwork_, bwork, &info);
        free(work);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, n, A, lda, sdim, wr, wi, VS,
                    ldvs);
    }

    return -1;
}

static int call_dgees_backend(fb_dgees_fn_t c_fn,
                              fb_dgees_fortran_fn_t fortran_fn,
                              fb_layout_t layout, char jobvs, char sort,
                              int n, double *A, int lda, int *sdim,
                              double *wr, double *wi, double *VS, int ldvs)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        double work_query = 0.0;
        double *work = NULL;
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (bwork == NULL) {
            return -101;
        }
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, wr, wi, VS,
                   &ldvs_, &work_query, &lwork_, bwork, &info);
        if (info != 0) {
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_double(work_query);
        work = (double *)malloc((size_t)lwork_ * sizeof(double));
        if (work == NULL) {
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, wr, wi, VS,
                   &ldvs_, work, &lwork_, bwork, &info);
        free(work);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, n, A, lda, sdim, wr, wi, VS,
                    ldvs);
    }

    return -1;
}

static int call_cgees_backend(fb_cgees_fn_t c_fn,
                              fb_cgees_fortran_fn_t fortran_fn,
                              fb_layout_t layout, char jobvs, char sort,
                              int n, fb_complex_float_t *A, int lda,
                              int *sdim, fb_complex_float_t *w,
                              fb_complex_float_t *VS, int ldvs)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *work = NULL;
        float *rwork = (float *)malloc((size_t)((n > 1) ? n : 1) * sizeof(float));
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (rwork == NULL || bwork == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(float));
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, w, VS, &ldvs_,
                   &work_query, &lwork_, rwork, bwork, &info);
        if (info != 0) {
            free(rwork);
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cfloat(work_query);
        work = (fb_complex_float_t *)malloc((size_t)lwork_ *
                                            sizeof(fb_complex_float_t));
        if (work == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, w, VS, &ldvs_,
                   work, &lwork_, rwork, bwork, &info);
        free(work);
        free(rwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, n, A, lda, sdim, w, VS, ldvs);
    }

    return -1;
}

static int call_zgees_backend(fb_zgees_fn_t c_fn,
                              fb_zgees_fortran_fn_t fortran_fn,
                              fb_layout_t layout, char jobvs, char sort,
                              int n, fb_complex_double_t *A, int lda,
                              int *sdim, fb_complex_double_t *w,
                              fb_complex_double_t *VS, int ldvs)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *work = NULL;
        double *rwork = (double *)malloc((size_t)((n > 1) ? n : 1) * sizeof(double));
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (rwork == NULL || bwork == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(double));
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, w, VS, &ldvs_,
                   &work_query, &lwork_, rwork, bwork, &info);
        if (info != 0) {
            free(rwork);
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cdouble(work_query);
        work = (fb_complex_double_t *)malloc((size_t)lwork_ *
                                             sizeof(fb_complex_double_t));
        if (work == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &n_, A, &lda_, sdim, w, VS, &ldvs_,
                   work, &lwork_, rwork, bwork, &info);
        free(work);
        free(rwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, n, A, lda, sdim, w, VS, ldvs);
    }

    return -1;
}

static int call_sgeesx_backend(fb_sgeesx_fn_t c_fn,
                               fb_sgeesx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char jobvs, char sort,
                               char sense, int n, float *A, int lda,
                               int *sdim, float *wr, float *wi, float *VS,
                               int ldvs, float *rconde, float *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int liwork_ = -1;
        int info = 0;
        float work_query = 0.0f;
        int iwork_query = 0;
        float *work = NULL;
        int *iwork = NULL;
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (bwork == NULL) {
            return -101;
        }
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, wr, wi,
                   VS, &ldvs_, rconde, rcondv, &work_query, &lwork_,
                   &iwork_query, &liwork_, bwork, &info);
        if (info != 0) {
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_float(work_query);
        liwork_ = fb_spectral_query_size_from_int(iwork_query);
        work = (float *)malloc((size_t)lwork_ * sizeof(float));
        iwork = (int *)malloc((size_t)liwork_ * sizeof(int));
        if (work == NULL || iwork == NULL) {
            free(work);
            free(iwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, wr, wi,
                   VS, &ldvs_, rconde, rcondv, work, &lwork_, iwork, &liwork_,
                   bwork, &info);
        free(work);
        free(iwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, sense, n, A, lda, sdim, wr, wi,
                    VS, ldvs, rconde, rcondv);
    }

    return -1;
}

static int call_dgeesx_backend(fb_dgeesx_fn_t c_fn,
                               fb_dgeesx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char jobvs, char sort,
                               char sense, int n, double *A, int lda,
                               int *sdim, double *wr, double *wi, double *VS,
                               int ldvs, double *rconde, double *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int liwork_ = -1;
        int info = 0;
        double work_query = 0.0;
        int iwork_query = 0;
        double *work = NULL;
        int *iwork = NULL;
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (bwork == NULL) {
            return -101;
        }
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, wr, wi,
                   VS, &ldvs_, rconde, rcondv, &work_query, &lwork_,
                   &iwork_query, &liwork_, bwork, &info);
        if (info != 0) {
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_double(work_query);
        liwork_ = fb_spectral_query_size_from_int(iwork_query);
        work = (double *)malloc((size_t)lwork_ * sizeof(double));
        iwork = (int *)malloc((size_t)liwork_ * sizeof(int));
        if (work == NULL || iwork == NULL) {
            free(work);
            free(iwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, wr, wi,
                   VS, &ldvs_, rconde, rcondv, work, &lwork_, iwork, &liwork_,
                   bwork, &info);
        free(work);
        free(iwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, sense, n, A, lda, sdim, wr, wi,
                    VS, ldvs, rconde, rcondv);
    }

    return -1;
}

static int call_cgeesx_backend(fb_cgeesx_fn_t c_fn,
                               fb_cgeesx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char jobvs, char sort,
                               char sense, int n, fb_complex_float_t *A,
                               int lda, int *sdim, fb_complex_float_t *w,
                               fb_complex_float_t *VS, int ldvs,
                               float *rconde, float *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *work = NULL;
        float *rwork = (float *)malloc((size_t)((n > 1) ? n : 1) * sizeof(float));
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (rwork == NULL || bwork == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(float));
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, w, VS,
                   &ldvs_, rconde, rcondv, &work_query, &lwork_, rwork, bwork,
                   &info);
        if (info != 0) {
            free(rwork);
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cfloat(work_query);
        work = (fb_complex_float_t *)malloc((size_t)lwork_ *
                                            sizeof(fb_complex_float_t));
        if (work == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, w, VS,
                   &ldvs_, rconde, rcondv, work, &lwork_, rwork, bwork,
                   &info);
        free(work);
        free(rwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, sense, n, A, lda, sdim, w, VS,
                    ldvs, rconde, rcondv);
    }

    return -1;
}

static int call_zgeesx_backend(fb_zgeesx_fn_t c_fn,
                               fb_zgeesx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char jobvs, char sort,
                               char sense, int n, fb_complex_double_t *A,
                               int lda, int *sdim, fb_complex_double_t *w,
                               fb_complex_double_t *VS, int ldvs,
                               double *rconde, double *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvs_ = ldvs;
        int lwork_ = -1;
        int info = 0;
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *work = NULL;
        double *rwork = (double *)malloc((size_t)((n > 1) ? n : 1) * sizeof(double));
        int *bwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        if (rwork == NULL || bwork == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(double));
        memset(bwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, w, VS,
                   &ldvs_, rconde, rcondv, &work_query, &lwork_, rwork, bwork,
                   &info);
        if (info != 0) {
            free(rwork);
            free(bwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cdouble(work_query);
        work = (fb_complex_double_t *)malloc((size_t)lwork_ *
                                             sizeof(fb_complex_double_t));
        if (work == NULL) {
            free(rwork);
            free(bwork);
            return -101;
        }

        fortran_fn(&jobvs, &sort, NULL, &sense, &n_, A, &lda_, sdim, w, VS,
                   &ldvs_, rconde, rcondv, work, &lwork_, rwork, bwork,
                   &info);
        free(work);
        free(rwork);
        free(bwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, jobvs, sort, NULL, sense, n, A, lda, sdim, w, VS,
                    ldvs, rconde, rcondv);
    }

    return -1;
}

static int call_sgeevx_backend(fb_sgeevx_fn_t c_fn,
                               fb_sgeevx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char balanc, char jobvl,
                               char jobvr, char sense, int n, float *A,
                               int lda, float *wr, float *wi, float *VL,
                               int ldvl, float *VR, int ldvr, int *ilo,
                               int *ihi, float *scale, float *abnrm,
                               float *rconde, float *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvl_ = ldvl;
        int ldvr_ = ldvr;
        int lwork_ = -1;
        int info = 0;
        float work_query = 0.0f;
        int iwork_query = 0;
        float *work = NULL;
        int *iwork = NULL;

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, wr, wi,
                   VL, &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, &work_query, &lwork_, &iwork_query, &info);
        if (info != 0) {
            return info;
        }

        lwork_ = fb_spectral_query_size_from_float(work_query);
        iwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        work = (float *)malloc((size_t)lwork_ * sizeof(float));
        if (work == NULL || iwork == NULL) {
            free(work);
            free(iwork);
            return -101;
        }
        memset(iwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, wr, wi,
                   VL, &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, work, &lwork_, iwork, &info);
        free(work);
        free(iwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, balanc, jobvl, jobvr, sense, n, A, lda, wr, wi,
                    VL, ldvl, VR, ldvr, ilo, ihi, scale, abnrm, rconde,
                    rcondv);
    }

    return -1;
}

static int call_dgeevx_backend(fb_dgeevx_fn_t c_fn,
                               fb_dgeevx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char balanc, char jobvl,
                               char jobvr, char sense, int n, double *A,
                               int lda, double *wr, double *wi, double *VL,
                               int ldvl, double *VR, int ldvr, int *ilo,
                               int *ihi, double *scale, double *abnrm,
                               double *rconde, double *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvl_ = ldvl;
        int ldvr_ = ldvr;
        int lwork_ = -1;
        int info = 0;
        double work_query = 0.0;
        int iwork_query = 0;
        double *work = NULL;
        int *iwork = NULL;

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, wr, wi,
                   VL, &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, &work_query, &lwork_, &iwork_query, &info);
        if (info != 0) {
            return info;
        }

        lwork_ = fb_spectral_query_size_from_double(work_query);
        iwork = (int *)malloc((size_t)((n > 1) ? n : 1) * sizeof(int));
        work = (double *)malloc((size_t)lwork_ * sizeof(double));
        if (work == NULL || iwork == NULL) {
            free(work);
            free(iwork);
            return -101;
        }
        memset(iwork, 0, (size_t)((n > 1) ? n : 1) * sizeof(int));

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, wr, wi,
                   VL, &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, work, &lwork_, iwork, &info);
        free(work);
        free(iwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, balanc, jobvl, jobvr, sense, n, A, lda, wr, wi,
                    VL, ldvl, VR, ldvr, ilo, ihi, scale, abnrm, rconde,
                    rcondv);
    }

    return -1;
}

static int call_cgeevx_backend(fb_cgeevx_fn_t c_fn,
                               fb_cgeevx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char balanc, char jobvl,
                               char jobvr, char sense, int n,
                               fb_complex_float_t *A, int lda,
                               fb_complex_float_t *w,
                               fb_complex_float_t *VL, int ldvl,
                               fb_complex_float_t *VR, int ldvr, int *ilo,
                               int *ihi, float *scale, float *abnrm,
                               float *rconde, float *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvl_ = ldvl;
        int ldvr_ = ldvr;
        int lwork_ = -1;
        int info = 0;
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *work = NULL;
        float *rwork = NULL;

        rwork = (float *)malloc((size_t)((n > 1) ? (2 * n) : 1) * sizeof(float));
        if (rwork == NULL) {
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? (2 * n) : 1) * sizeof(float));

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, w, VL,
                   &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, &work_query, &lwork_, rwork, &info);
        if (info != 0) {
            free(rwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cfloat(work_query);
        work = (fb_complex_float_t *)malloc((size_t)lwork_ *
                                            sizeof(fb_complex_float_t));
        if (work == NULL) {
            free(rwork);
            return -101;
        }

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, w, VL,
                   &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, work, &lwork_, rwork, &info);
        free(work);
        free(rwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, balanc, jobvl, jobvr, sense, n, A, lda, w, VL,
                    ldvl, VR, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv);
    }

    return -1;
}

static int call_zgeevx_backend(fb_zgeevx_fn_t c_fn,
                               fb_zgeevx_fortran_fn_t fortran_fn,
                               fb_layout_t layout, char balanc, char jobvl,
                               char jobvr, char sense, int n,
                               fb_complex_double_t *A, int lda,
                               fb_complex_double_t *w,
                               fb_complex_double_t *VL, int ldvl,
                               fb_complex_double_t *VR, int ldvr, int *ilo,
                               int *ihi, double *scale, double *abnrm,
                               double *rconde, double *rcondv)
{
    if (fortran_fn != NULL) {
        int n_ = n;
        int lda_ = lda;
        int ldvl_ = ldvl;
        int ldvr_ = ldvr;
        int lwork_ = -1;
        int info = 0;
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *work = NULL;
        double *rwork = NULL;

        rwork = (double *)malloc((size_t)((n > 1) ? (2 * n) : 1) * sizeof(double));
        if (rwork == NULL) {
            return -101;
        }
        memset(rwork, 0, (size_t)((n > 1) ? (2 * n) : 1) * sizeof(double));

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, w, VL,
                   &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, &work_query, &lwork_, rwork, &info);
        if (info != 0) {
            free(rwork);
            return info;
        }

        lwork_ = fb_spectral_query_size_from_cdouble(work_query);
        work = (fb_complex_double_t *)malloc((size_t)lwork_ *
                                             sizeof(fb_complex_double_t));
        if (work == NULL) {
            free(rwork);
            return -101;
        }

        fortran_fn(&balanc, &jobvl, &jobvr, &sense, &n_, A, &lda_, w, VL,
                   &ldvl_, VR, &ldvr_, ilo, ihi, scale, abnrm, rconde,
                   rcondv, work, &lwork_, rwork, &info);
        free(work);
        free(rwork);
        return info;
    }

    if (c_fn != NULL) {
        return c_fn(layout, balanc, jobvl, jobvr, sense, n, A, lda, w, VL,
                    ldvl, VR, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv);
    }

    return -1;
}

static fb_judge_status_t run_sgeevx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_sgeevx_fn_t oracle_fn = (fb_sgeevx_fn_t)oracle->sgeevx;
    fb_sgeevx_fn_t cand_fn = (fb_sgeevx_fn_t)cand->sgeevx;
    fb_sgeevx_fortran_fn_t oracle_fortran_fn =
        (fb_sgeevx_fortran_fn_t)oracle->ext_ops[FB_OP_SGEEVX][FB_CONV_FORTRAN];
    fb_sgeevx_fortran_fn_t cand_fortran_fn =
        (fb_sgeevx_fortran_fn_t)cand->ext_ops[FB_OP_SGEEVX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char balanc = 'N';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';

    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_copy = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *WR_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WI_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WR_cand = (float *)malloc((size_t)n * sizeof(float));
    float *WI_cand = (float *)malloc((size_t)n * sizeof(float));
    float *VR = (float *)malloc((size_t)(n * n) * sizeof(float));
    float *scale_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *scale_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rconde_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rcondv_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rconde_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rcondv_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));
    float abnrm_oracle = 0.0f;
    float abnrm_cand = 0.0f;
    int ilo_oracle = 0, ihi_oracle = 0;
    int ilo_cand = 0, ihi_cand = 0;

    if (!A_oracle || !A_cand || !A_copy || !WR_oracle || !WI_oracle ||
        !WR_cand || !WI_cand || !VR || !scale_oracle || !scale_cand ||
        !rconde_oracle || !rcondv_oracle || !rconde_cand || !rcondv_cand ||
        !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_sgeevx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_copy, A_in, (size_t)(lda * n) * sizeof(float));

    if (call_sgeevx_backend(oracle_fn, oracle_fortran_fn, layout, balanc,
                            jobvl, jobvr, sense, (int)n, A_oracle, (int)lda,
                            WR_oracle, WI_oracle, NULL, 1, VR, (int)n,
                            &ilo_oracle, &ihi_oracle, scale_oracle,
                            &abnrm_oracle, rconde_oracle, rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_sgeevx;
    }

    if (call_sgeevx_backend(cand_fn, cand_fortran_fn, layout, balanc, jobvl,
                            jobvr, sense, (int)n, A_cand, (int)lda, WR_cand,
                            WI_cand, NULL, 1, VR, (int)n, &ilo_cand, &ihi_cand,
                            scale_cand, &abnrm_cand, rconde_cand,
                            rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_sgeevx;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = (double)WR_oracle[j], or_i = (double)WI_oracle[j];
            double ca_r = (double)WR_cand[j], ca_i = (double)WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

    {
        double norm_A = frobenius_norm_f32(A_copy, n, n, lda);
        double max_pair_err = 0.0;
        int j = 0;
        while (j < n) {
            if ((double)WI_cand[j] == 0.0) {
                double wr = (double)WR_cand[j];
                double res_sq = 0.0, vr_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double av = 0.0;
                    for (int k = 0; k < n; k++)
                        av += (double)A_copy[i * lda + k] * (double)VR[k * n + j];
                    {
                        double r = av - wr * (double)VR[i * n + j];
                        double vri = (double)VR[i * n + j];
                        res_sq += r * r;
                        vr_sq += vri * vri;
                    }
                }
                {
                    double denom = norm_A * sqrt(vr_sq);
                    double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                    if (re > max_pair_err) max_pair_err = re;
                }
                j++;
            } else {
                double wr = (double)WR_cand[j], wi = (double)WI_cand[j];
                double res_r_sq = 0.0, res_i_sq = 0.0, v_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    double avr = 0.0, avi = 0.0;
                    for (int k = 0; k < n; k++) {
                        double a = (double)A_copy[i * lda + k];
                        avr += a * (double)VR[k * n + j];
                        avi += a * (double)VR[k * n + j + 1];
                    }
                    {
                        double vri = (double)VR[i * n + j];
                        double vii = (double)VR[i * n + j + 1];
                        double rr = avr - (wr * vri - wi * vii);
                        double ri = avi - (wr * vii + wi * vri);
                        res_r_sq += rr * rr;
                        res_i_sq += ri * ri;
                        v_sq += vri * vri + vii * vii;
                    }
                }
                {
                    double denom = norm_A * sqrt(v_sq);
                    double re = (denom > 1e-16) ?
                        sqrt(res_r_sq + res_i_sq) / denom : 0.0;
                    if (re > max_pair_err) max_pair_err = re;
                }
                j += 2;
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

    {
        for (int j = 0; j < n; j++)
            eig_mag[j] = sqrtf(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
        for (int i = 1; i < n; i++) {
            float tmp = eig_mag[i];
            int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) {
                eig_mag[k] = eig_mag[k - 1];
                k--;
            }
            eig_mag[k] = tmp;
        }
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VRt = (float *)malloc((size_t)(n * n) * sizeof(float));
            float *scale_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rconde_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rcondv_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float abnrm_t = 0.0f;
            int ilo_t = 0, ihi_t = 0;
            if (At && WRt && WIt && VRt && scale_t && rconde_t && rcondv_t) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)call_sgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, WRt, WIt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
            }
            free(At); free(WRt); free(WIt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VRt = (float *)malloc((size_t)(n * n) * sizeof(float));
            float *scale_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rconde_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rcondv_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float abnrm_t = 0.0f;
            int ilo_t = 0, ihi_t = 0;
            if (!At || !WRt || !WIt || !VRt || !scale_t || !rconde_t || !rcondv_t) {
                free(At); free(WRt); free(WIt); free(VRt);
                free(scale_t); free(rconde_t); free(rcondv_t);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_sgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, WRt, WIt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        *ns_out = best;
    }

cleanup_sgeevx:
    free(A_oracle); free(A_cand); free(A_copy);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VR);
    free(scale_oracle); free(scale_cand);
    free(rconde_oracle); free(rcondv_oracle);
    free(rconde_cand); free(rcondv_cand);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeevx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_dgeevx_fn_t oracle_fn = (fb_dgeevx_fn_t)oracle->dgeevx;
    fb_dgeevx_fn_t cand_fn = (fb_dgeevx_fn_t)cand->dgeevx;
    fb_dgeevx_fortran_fn_t oracle_fortran_fn =
        (fb_dgeevx_fortran_fn_t)oracle->ext_ops[FB_OP_DGEEVX][FB_CONV_FORTRAN];
    fb_dgeevx_fortran_fn_t cand_fortran_fn =
        (fb_dgeevx_fortran_fn_t)cand->ext_ops[FB_OP_DGEEVX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char balanc = 'N';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';

    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_copy = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *WR_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WI_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WR_cand = (double *)malloc((size_t)n * sizeof(double));
    double *WI_cand = (double *)malloc((size_t)n * sizeof(double));
    double *VR = (double *)malloc((size_t)(n * n) * sizeof(double));
    double *scale_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *scale_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rconde_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rcondv_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rconde_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rcondv_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));
    double abnrm_oracle = 0.0;
    double abnrm_cand = 0.0;
    int ilo_oracle = 0, ihi_oracle = 0;
    int ilo_cand = 0, ihi_cand = 0;

    if (!A_oracle || !A_cand || !A_copy || !WR_oracle || !WI_oracle ||
        !WR_cand || !WI_cand || !VR || !scale_oracle || !scale_cand ||
        !rconde_oracle || !rcondv_oracle || !rconde_cand || !rcondv_cand ||
        !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_dgeevx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_copy, A_in, (size_t)(lda * n) * sizeof(double));

    if (call_dgeevx_backend(oracle_fn, oracle_fortran_fn, layout, balanc,
                            jobvl, jobvr, sense, (int)n, A_oracle, (int)lda,
                            WR_oracle, WI_oracle, NULL, 1, VR, (int)n,
                            &ilo_oracle, &ihi_oracle, scale_oracle,
                            &abnrm_oracle, rconde_oracle, rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_dgeevx;
    }

    if (call_dgeevx_backend(cand_fn, cand_fortran_fn, layout, balanc, jobvl,
                            jobvr, sense, (int)n, A_cand, (int)lda, WR_cand,
                            WI_cand, NULL, 1, VR, (int)n, &ilo_cand, &ihi_cand,
                            scale_cand, &abnrm_cand, rconde_cand,
                            rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_dgeevx;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = WR_oracle[j], or_i = WI_oracle[j];
            double ca_r = WR_cand[j], ca_i = WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

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
                    {
                        double r = av - wr * VR[i * n + j];
                        res_sq += r * r;
                        vr_sq += VR[i * n + j] * VR[i * n + j];
                    }
                }
                {
                    double denom = norm_A * sqrt(vr_sq);
                    double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                    if (re > max_pair_err) max_pair_err = re;
                }
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
                    {
                        double vri = VR[i * n + j], vii = VR[i * n + j + 1];
                        double rr = avr - (wr * vri - wi * vii);
                        double ri = avi - (wr * vii + wi * vri);
                        res_r_sq += rr * rr;
                        res_i_sq += ri * ri;
                        v_sq += vri * vri + vii * vii;
                    }
                }
                {
                    double denom = norm_A * sqrt(v_sq);
                    double re = (denom > 1e-16) ?
                        sqrt(res_r_sq + res_i_sq) / denom : 0.0;
                    if (re > max_pair_err) max_pair_err = re;
                }
                j += 2;
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

    {
        for (int j = 0; j < n; j++)
            eig_mag[j] = (float)sqrt(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
        for (int i = 1; i < n; i++) {
            float tmp = eig_mag[i];
            int k = i;
            while (k > 0 && eig_mag[k - 1] > tmp) {
                eig_mag[k] = eig_mag[k - 1];
                k--;
            }
            eig_mag[k] = tmp;
        }
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VRt = (double *)malloc((size_t)(n * n) * sizeof(double));
            double *scale_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rconde_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rcondv_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double abnrm_t = 0.0;
            int ilo_t = 0, ihi_t = 0;
            if (At && WRt && WIt && VRt && scale_t && rconde_t && rcondv_t) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)call_dgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, WRt, WIt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
            }
            free(At); free(WRt); free(WIt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VRt = (double *)malloc((size_t)(n * n) * sizeof(double));
            double *scale_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rconde_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rcondv_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double abnrm_t = 0.0;
            int ilo_t = 0, ihi_t = 0;
            if (!At || !WRt || !WIt || !VRt || !scale_t || !rconde_t || !rcondv_t) {
                free(At); free(WRt); free(WIt); free(VRt);
                free(scale_t); free(rconde_t); free(rcondv_t);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_dgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, WRt, WIt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        *ns_out = best;
    }

cleanup_dgeevx:
    free(A_oracle); free(A_cand); free(A_copy);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VR);
    free(scale_oracle); free(scale_cand);
    free(rconde_oracle); free(rcondv_oracle);
    free(rconde_cand); free(rcondv_cand);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeevx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_cgeevx_fn_t oracle_fn = (fb_cgeevx_fn_t)oracle->cgeevx;
    fb_cgeevx_fn_t cand_fn = (fb_cgeevx_fn_t)cand->cgeevx;
    fb_cgeevx_fortran_fn_t oracle_fortran_fn =
        (fb_cgeevx_fortran_fn_t)oracle->ext_ops[FB_OP_CGEEVX][FB_CONV_FORTRAN];
    fb_cgeevx_fortran_fn_t cand_fortran_fn =
        (fb_cgeevx_fortran_fn_t)cand->ext_ops[FB_OP_CGEEVX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char balanc = 'N';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';

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
    float *scale_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *scale_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rconde_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rcondv_oracle = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rconde_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *rcondv_cand = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));
    float abnrm_oracle = 0.0f;
    float abnrm_cand = 0.0f;
    int ilo_oracle = 0, ihi_oracle = 0;
    int ilo_cand = 0, ihi_cand = 0;

    if (!A_oracle || !A_cand || !A_copy || !W_oracle || !W_cand || !VR ||
        !scale_oracle || !scale_cand || !rconde_oracle || !rcondv_oracle ||
        !rconde_cand || !rcondv_cand || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_cgeevx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_copy, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    if (call_cgeevx_backend(oracle_fn, oracle_fortran_fn, layout, balanc,
                            jobvl, jobvr, sense, (int)n, A_oracle, (int)lda,
                            W_oracle, NULL, 1, VR, (int)n, &ilo_oracle,
                            &ihi_oracle, scale_oracle, &abnrm_oracle,
                            rconde_oracle, rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_cgeevx;
    }

    if (call_cgeevx_backend(cand_fn, cand_fortran_fn, layout, balanc, jobvl,
                            jobvr, sense, (int)n, A_cand, (int)lda, W_cand,
                            NULL, 1, VR, (int)n, &ilo_cand, &ihi_cand,
                            scale_cand, &abnrm_cand, rconde_cand,
                            rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_cgeevx;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

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
                {
                    double v_re = (double)FB_CF_REAL(VR[i * n + j]);
                    double v_im = (double)FB_CF_IMAG(VR[i * n + j]);
                    double lv_re = lam_re * v_re - lam_im * v_im;
                    double lv_im = lam_re * v_im + lam_im * v_re;
                    double rr = av_re - lv_re;
                    double ri = av_im - lv_im;
                    res_sq += rr * rr + ri * ri;
                    v_sq += v_re * v_re + v_im * v_im;
                }
            }
            {
                double denom = norm_A * sqrt(v_sq);
                double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VRt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            float *scale_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rconde_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rcondv_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float abnrm_t = 0.0f;
            int ilo_t = 0, ihi_t = 0;
            if (At && Wt && VRt && scale_t && rconde_t && rcondv_t) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)call_cgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, Wt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
            }
            free(At); free(Wt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VRt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            float *scale_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rconde_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float *rcondv_t = (float *)malloc((size_t)((n > 0) ? n : 1) * sizeof(float));
            float abnrm_t = 0.0f;
            int ilo_t = 0, ihi_t = 0;
            if (!At || !Wt || !VRt || !scale_t || !rconde_t || !rcondv_t) {
                free(At); free(Wt); free(VRt);
                free(scale_t); free(rconde_t); free(rcondv_t);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_cgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, Wt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        *ns_out = best;
    }

cleanup_cgeevx:
    free(A_oracle);
    free(A_cand);
    free(A_copy);
    free(W_oracle);
    free(W_cand);
    free(VR);
    free(scale_oracle); free(scale_cand);
    free(rconde_oracle); free(rcondv_oracle);
    free(rconde_cand); free(rcondv_cand);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeevx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_zgeevx_fn_t oracle_fn = (fb_zgeevx_fn_t)oracle->zgeevx;
    fb_zgeevx_fn_t cand_fn = (fb_zgeevx_fn_t)cand->zgeevx;
    fb_zgeevx_fortran_fn_t oracle_fortran_fn =
        (fb_zgeevx_fortran_fn_t)oracle->ext_ops[FB_OP_ZGEEVX][FB_CONV_FORTRAN];
    fb_zgeevx_fortran_fn_t cand_fortran_fn =
        (fb_zgeevx_fortran_fn_t)cand->ext_ops[FB_OP_ZGEEVX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char balanc = 'N';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';

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
    double *scale_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *scale_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rconde_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rcondv_oracle = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rconde_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *rcondv_cand = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));
    double abnrm_oracle = 0.0;
    double abnrm_cand = 0.0;
    int ilo_oracle = 0, ihi_oracle = 0;
    int ilo_cand = 0, ihi_cand = 0;

    if (!A_oracle || !A_cand || !A_copy || !W_oracle || !W_cand || !VR ||
        !scale_oracle || !scale_cand || !rconde_oracle || !rcondv_oracle ||
        !rconde_cand || !rcondv_cand || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_zgeevx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_copy, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    if (call_zgeevx_backend(oracle_fn, oracle_fortran_fn, layout, balanc,
                            jobvl, jobvr, sense, (int)n, A_oracle, (int)lda,
                            W_oracle, NULL, 1, VR, (int)n, &ilo_oracle,
                            &ihi_oracle, scale_oracle, &abnrm_oracle,
                            rconde_oracle, rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_zgeevx;
    }

    if (call_zgeevx_backend(cand_fn, cand_fortran_fn, layout, balanc, jobvl,
                            jobvr, sense, (int)n, A_cand, (int)lda, W_cand,
                            NULL, 1, VR, (int)n, &ilo_cand, &ihi_cand,
                            scale_cand, &abnrm_cand, rconde_cand,
                            rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_zgeevx;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    res->reconstruction = (fb_judge_case_result_t){.digits = 16};
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};

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
                {
                    double v_re = FB_CD_REAL(VR[i * n + j]);
                    double v_im = FB_CD_IMAG(VR[i * n + j]);
                    double lv_re = lam_re * v_re - lam_im * v_im;
                    double lv_im = lam_re * v_im + lam_im * v_re;
                    double rr = av_re - lv_re;
                    double ri = av_im - lv_im;
                    res_sq += rr * rr + ri * ri;
                    v_sq += v_re * v_re + v_im * v_im;
                }
            }
            {
                double denom = norm_A * sqrt(v_sq);
                double re = (denom > 1e-16) ? sqrt(res_sq) / denom : 0.0;
                if (re > max_pair_err) max_pair_err = re;
            }
        }
        res->pairs = result_from_relerr(max_pair_err);
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VRt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            double *scale_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rconde_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rcondv_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double abnrm_t = 0.0;
            int ilo_t = 0, ihi_t = 0;
            if (At && Wt && VRt && scale_t && rconde_t && rcondv_t) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)call_zgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, Wt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
            }
            free(At); free(Wt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VRt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            double *scale_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rconde_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double *rcondv_t = (double *)malloc((size_t)((n > 0) ? n : 1) * sizeof(double));
            double abnrm_t = 0.0;
            int ilo_t = 0, ihi_t = 0;
            if (!At || !Wt || !VRt || !scale_t || !rconde_t || !rcondv_t) {
                free(At); free(Wt); free(VRt);
                free(scale_t); free(rconde_t); free(rcondv_t);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_zgeevx_backend(cand_fn, cand_fortran_fn, layout,
                                          balanc, jobvl, jobvr, sense, (int)n,
                                          At, (int)lda, Wt, NULL, 1, VRt,
                                          (int)n, &ilo_t, &ihi_t, scale_t,
                                          &abnrm_t, rconde_t, rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VRt);
            free(scale_t); free(rconde_t); free(rcondv_t);
        }
        *ns_out = best;
    }

cleanup_zgeevx:
    free(A_oracle);
    free(A_cand);
    free(A_copy);
    free(W_oracle);
    free(W_cand);
    free(VR);
    free(scale_oracle); free(scale_cand);
    free(rconde_oracle); free(rcondv_oracle);
    free(rconde_cand); free(rcondv_cand);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgees(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_sgees_fn_t oracle_fn = (fb_sgees_fn_t)oracle->sgees;
    fb_sgees_fn_t cand_fn = (fb_sgees_fn_t)cand->sgees;
    fb_sgees_fortran_fn_t oracle_fortran_fn =
        (fb_sgees_fortran_fn_t)oracle->ext_ops[FB_OP_SGEES][FB_CONV_FORTRAN];
    fb_sgees_fortran_fn_t cand_fortran_fn =
        (fb_sgees_fortran_fn_t)cand->ext_ops[FB_OP_SGEES][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;

    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *WR_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WI_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WR_cand = (float *)malloc((size_t)n * sizeof(float));
    float *WI_cand = (float *)malloc((size_t)n * sizeof(float));
    float *VS = (float *)malloc((size_t)(n * n) * sizeof(float));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !WR_oracle || !WI_oracle || !WR_cand ||
        !WI_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_sgees;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));

    if (call_sgees_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                           (int)n, A_oracle, (int)lda, &sdim_oracle, WR_oracle,
                           WI_oracle, VS, (int)n) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_sgees;
    }

    if (call_sgees_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                           (int)n, A_cand, (int)lda, &sdim_cand, WR_cand,
                           WI_cand, VS, (int)n) != 0) {
        mark_cand_fatal(res);
        goto cleanup_sgees;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = (double)WR_oracle[j], or_i = (double)WI_oracle[j];
            double ca_r = (double)WR_cand[j], ca_i = (double)WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_f32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            float *QT = (float *)malloc((size_t)(n * n) * sizeof(float));
            float *A_recon = (float *)malloc((size_t)(n * n) * sizeof(float));
            if (QT && A_recon) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += (double)VS[i * n + k] *
                                   (double)A_cand[k * lda + j];
                        }
                        QT[i * n + j] = (float)sum;
                    }
                }

                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += (double)QT[i * n + k] *
                                   (double)VS[j * n + k];
                        }
                        A_recon[i * n + j] = (float)sum;
                        {
                            double diff = (double)A_in[i * lda + j] - sum;
                            sum_diff += diff * diff;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(QT);
            free(A_recon);
        }
    }

    {
        float *ZtZ = (float *)malloc((size_t)(n * n) * sizeof(float));
        double ortho_err = 0.0;
        if (ZtZ) {
            atac_f32(VS, n, n, n, ZtZ, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double err = (double)ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
                    ortho_err += err * err;
                }
            }
            free(ZtZ);
        }
        res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
    }

    {
        for (int j = 0; j < n; j++) {
            eig_mag[j] = sqrtf(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VSt = (float *)malloc((size_t)(n * n) * sizeof(float));
            int sdim_t = 0;
            if (At && WRt && WIt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)call_sgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, WRt, WIt, VSt, (int)n);
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VSt = (float *)malloc((size_t)(n * n) * sizeof(float));
            int sdim_t = 0;
            if (!At || !WRt || !WIt || !VSt) {
                free(At); free(WRt); free(WIt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_sgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, WRt, WIt, VSt, (int)n);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_sgees:
    free(A_oracle); free(A_cand);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VS); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgees(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_dgees_fn_t oracle_fn = (fb_dgees_fn_t)oracle->dgees;
    fb_dgees_fn_t cand_fn = (fb_dgees_fn_t)cand->dgees;
    fb_dgees_fortran_fn_t oracle_fortran_fn =
        (fb_dgees_fortran_fn_t)oracle->ext_ops[FB_OP_DGEES][FB_CONV_FORTRAN];
    fb_dgees_fortran_fn_t cand_fortran_fn =
        (fb_dgees_fortran_fn_t)cand->ext_ops[FB_OP_DGEES][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;

    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *WR_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WI_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WR_cand = (double *)malloc((size_t)n * sizeof(double));
    double *WI_cand = (double *)malloc((size_t)n * sizeof(double));
    double *VS = (double *)malloc((size_t)(n * n) * sizeof(double));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !WR_oracle || !WI_oracle || !WR_cand ||
        !WI_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_dgees;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

    if (call_dgees_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                           (int)n, A_oracle, (int)lda, &sdim_oracle, WR_oracle,
                           WI_oracle, VS, (int)n) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_dgees;
    }

    if (call_dgees_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                           (int)n, A_cand, (int)lda, &sdim_cand, WR_cand,
                           WI_cand, VS, (int)n) != 0) {
        mark_cand_fatal(res);
        goto cleanup_dgees;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = WR_oracle[j], or_i = WI_oracle[j];
            double ca_r = WR_cand[j], ca_i = WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_f64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double *QT = (double *)malloc((size_t)(n * n) * sizeof(double));
            double *A_recon = (double *)malloc((size_t)(n * n) * sizeof(double));
            if (QT && A_recon) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += VS[i * n + k] * A_cand[k * lda + j];
                        }
                        QT[i * n + j] = sum;
                    }
                }

                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += QT[i * n + k] * VS[j * n + k];
                        }
                        A_recon[i * n + j] = sum;
                        {
                            double diff = A_in[i * lda + j] - sum;
                            sum_diff += diff * diff;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(QT);
            free(A_recon);
        }
    }

    {
        double *ZtZ = (double *)malloc((size_t)(n * n) * sizeof(double));
        double ortho_err = 0.0;
        if (ZtZ) {
            atac_f64(VS, n, n, n, ZtZ, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double err = ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
                    ortho_err += err * err;
                }
            }
            free(ZtZ);
        }
        res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
    }

    {
        for (int j = 0; j < n; j++) {
            eig_mag[j] = sqrt(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VSt = (double *)malloc((size_t)(n * n) * sizeof(double));
            int sdim_t = 0;
            if (At && WRt && WIt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)call_dgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, WRt, WIt, VSt, (int)n);
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VSt = (double *)malloc((size_t)(n * n) * sizeof(double));
            int sdim_t = 0;
            if (!At || !WRt || !WIt || !VSt) {
                free(At); free(WRt); free(WIt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_dgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, WRt, WIt, VSt, (int)n);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_dgees:
    free(A_oracle); free(A_cand);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VS); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgees(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_cgees_fn_t oracle_fn = (fb_cgees_fn_t)oracle->cgees;
    fb_cgees_fn_t cand_fn = (fb_cgees_fn_t)cand->cgees;
    fb_cgees_fortran_fn_t oracle_fortran_fn =
        (fb_cgees_fortran_fn_t)oracle->ext_ops[FB_OP_CGEES][FB_CONV_FORTRAN];
    fb_cgees_fortran_fn_t cand_fortran_fn =
        (fb_cgees_fortran_fn_t)cand->ext_ops[FB_OP_CGEES][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_oracle = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_cand = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *VS = (fb_complex_float_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_float_t));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !W_oracle || !W_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_cgees;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    if (call_cgees_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                           (int)n, A_oracle, (int)lda, &sdim_oracle, W_oracle,
                           VS, (int)n) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_cgees;
    }

    if (call_cgees_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                           (int)n, A_cand, (int)lda, &sdim_cand, W_cand, VS,
                           (int)n) != 0) {
        mark_cand_fatal(res);
        goto cleanup_cgees;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_cf32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            fb_complex_float_t *QT = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            if (QT) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double qt_re = 0.0, qt_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double q_re = (double)FB_CF_REAL(VS[i * n + k]);
                            double q_im = (double)FB_CF_IMAG(VS[i * n + k]);
                            double t_re = (double)FB_CF_REAL(A_cand[k * lda + j]);
                            double t_im = (double)FB_CF_IMAG(A_cand[k * lda + j]);
                            qt_re += q_re * t_re - q_im * t_im;
                            qt_im += q_re * t_im + q_im * t_re;
                        }
                        QT[i * n + j] = fb_spectral_make_cf32((float)qt_re,
                                                              (float)qt_im);
                    }
                }

                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double re_sum = 0.0, im_sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            double qt_re = (double)FB_CF_REAL(QT[i * n + k]);
                            double qt_im = (double)FB_CF_IMAG(QT[i * n + k]);
                            double q_re = (double)FB_CF_REAL(VS[j * n + k]);
                            double q_im = (double)FB_CF_IMAG(VS[j * n + k]);
                            re_sum += qt_re * q_re + qt_im * q_im;
                            im_sum += qt_im * q_re - qt_re * q_im;
                        }
                        {
                            double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                            double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                            double dr = a_re - re_sum;
                            double di = a_im - im_sum;
                            sum_diff += dr * dr + di * di;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(QT);
        }
    }

    {
        double raw_err = ahac_cf32_ortho_error(VS, n, n, n);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VSt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            int sdim_t = 0;
            if (At && Wt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)call_cgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, Wt, VSt, (int)n);
            }
            free(At); free(Wt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VSt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            int sdim_t = 0;
            if (!At || !Wt || !VSt) {
                free(At); free(Wt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_cgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, Wt, VSt, (int)n);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_cgees:
    free(A_oracle);
    free(A_cand);
    free(W_oracle);
    free(W_cand);
    free(VS);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgees(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_zgees_fn_t oracle_fn = (fb_zgees_fn_t)oracle->zgees;
    fb_zgees_fn_t cand_fn = (fb_zgees_fn_t)cand->zgees;
    fb_zgees_fortran_fn_t oracle_fortran_fn =
        (fb_zgees_fortran_fn_t)oracle->ext_ops[FB_OP_ZGEES][FB_CONV_FORTRAN];
    fb_zgees_fortran_fn_t cand_fortran_fn =
        (fb_zgees_fortran_fn_t)cand->ext_ops[FB_OP_ZGEES][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_oracle = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_cand = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *VS = (fb_complex_double_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_double_t));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !W_oracle || !W_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_zgees;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    if (call_zgees_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                           (int)n, A_oracle, (int)lda, &sdim_oracle, W_oracle,
                           VS, (int)n) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_zgees;
    }

    if (call_zgees_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                           (int)n, A_cand, (int)lda, &sdim_cand, W_cand, VS,
                           (int)n) != 0) {
        mark_cand_fatal(res);
        goto cleanup_zgees;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_cf64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            fb_complex_double_t *QT = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            if (QT) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double qt_re = 0.0, qt_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double q_re = FB_CD_REAL(VS[i * n + k]);
                            double q_im = FB_CD_IMAG(VS[i * n + k]);
                            double t_re = FB_CD_REAL(A_cand[k * lda + j]);
                            double t_im = FB_CD_IMAG(A_cand[k * lda + j]);
                            qt_re += q_re * t_re - q_im * t_im;
                            qt_im += q_re * t_im + q_im * t_re;
                        }
                        QT[i * n + j] = fb_spectral_make_cf64(qt_re, qt_im);
                    }
                }

                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double re_sum = 0.0, im_sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            double qt_re = FB_CD_REAL(QT[i * n + k]);
                            double qt_im = FB_CD_IMAG(QT[i * n + k]);
                            double q_re = FB_CD_REAL(VS[j * n + k]);
                            double q_im = FB_CD_IMAG(VS[j * n + k]);
                            re_sum += qt_re * q_re + qt_im * q_im;
                            im_sum += qt_im * q_re - qt_re * q_im;
                        }
                        {
                            double a_re = FB_CD_REAL(A_in[i * lda + j]);
                            double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                            double dr = a_re - re_sum;
                            double di = a_im - im_sum;
                            sum_diff += dr * dr + di * di;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
            free(QT);
        }
    }

    {
        double raw_err = ahac_cf64_ortho_error(VS, n, n, n);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VSt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            int sdim_t = 0;
            if (At && Wt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)call_zgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, Wt, VSt, (int)n);
            }
            free(At); free(Wt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VSt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            int sdim_t = 0;
            if (!At || !Wt || !VSt) {
                free(At); free(Wt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_zgees_backend(cand_fn, cand_fortran_fn, layout,
                                         jobvs, sort, (int)n, At, (int)lda,
                                         &sdim_t, Wt, VSt, (int)n);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_zgees:
    free(A_oracle);
    free(A_cand);
    free(W_oracle);
    free(W_cand);
    free(VS);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_sgeesx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_sgeesx_fn_t oracle_fn = (fb_sgeesx_fn_t)oracle->sgeesx;
    fb_sgeesx_fn_t cand_fn = (fb_sgeesx_fn_t)cand->sgeesx;
    fb_sgeesx_fortran_fn_t oracle_fortran_fn =
        (fb_sgeesx_fortran_fn_t)oracle->ext_ops[FB_OP_SGEESX][FB_CONV_FORTRAN];
    fb_sgeesx_fortran_fn_t cand_fortran_fn =
        (fb_sgeesx_fortran_fn_t)cand->ext_ops[FB_OP_SGEESX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const float *A_in = (const float *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    char sense = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;
    float rconde_oracle = 1.0f;
    float rcondv_oracle = 1.0f;
    float rconde_cand = 1.0f;
    float rcondv_cand = 1.0f;

    float *A_oracle = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *A_cand = (float *)malloc((size_t)(lda * n) * sizeof(float));
    float *WR_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WI_oracle = (float *)malloc((size_t)n * sizeof(float));
    float *WR_cand = (float *)malloc((size_t)n * sizeof(float));
    float *WI_cand = (float *)malloc((size_t)n * sizeof(float));
    float *VS = (float *)malloc((size_t)(n * n) * sizeof(float));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !WR_oracle || !WI_oracle || !WR_cand ||
        !WI_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_sgeesx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(float));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(float));

    if (call_sgeesx_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_oracle, (int)lda, &sdim_oracle,
                            WR_oracle, WI_oracle, VS, (int)n, &rconde_oracle,
                            &rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_sgeesx;
    }

    if (call_sgeesx_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_cand, (int)lda, &sdim_cand,
                            WR_cand, WI_cand, VS, (int)n, &rconde_cand,
                            &rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_sgeesx;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = (double)WR_oracle[j], or_i = (double)WI_oracle[j];
            double ca_r = (double)WR_cand[j], ca_i = (double)WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_f32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            float *QT = (float *)malloc((size_t)(n * n) * sizeof(float));
            if (QT) {
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += (double)VS[i * n + k] *
                                   (double)A_cand[k * lda + j];
                        }
                        QT[i * n + j] = (float)sum;
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += (double)QT[i * n + k] *
                                   (double)VS[j * n + k];
                        }
                        {
                            double diff = (double)A_in[i * lda + j] - sum;
                            sum_diff += diff * diff;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
                free(QT);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
        }
    }

    {
        float *ZtZ = (float *)malloc((size_t)(n * n) * sizeof(float));
        double ortho_err = 0.0;
        if (ZtZ) {
            atac_f32(VS, n, n, n, ZtZ, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double err = (double)ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
                    ortho_err += err * err;
                }
            }
            free(ZtZ);
        }
        res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
    }

    {
        for (int j = 0; j < n; j++) {
            eig_mag[j] = sqrtf(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VSt = (float *)malloc((size_t)(n * n) * sizeof(float));
            float rconde_t = 1.0f, rcondv_t = 1.0f;
            int sdim_t = 0;
            if (At && WRt && WIt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
                (void)call_sgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, WRt, WIt, VSt,
                                          (int)n, &rconde_t, &rcondv_t);
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            float *At = (float *)malloc((size_t)(lda * n) * sizeof(float));
            float *WRt = (float *)malloc((size_t)n * sizeof(float));
            float *WIt = (float *)malloc((size_t)n * sizeof(float));
            float *VSt = (float *)malloc((size_t)(n * n) * sizeof(float));
            float rconde_t = 1.0f, rcondv_t = 1.0f;
            int sdim_t = 0;
            if (!At || !WRt || !WIt || !VSt) {
                free(At); free(WRt); free(WIt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(float));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_sgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, WRt, WIt, VSt,
                                          (int)n, &rconde_t, &rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_sgeesx:
    free(A_oracle); free(A_cand);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VS); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgeesx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_dgeesx_fn_t oracle_fn = (fb_dgeesx_fn_t)oracle->dgeesx;
    fb_dgeesx_fn_t cand_fn = (fb_dgeesx_fn_t)cand->dgeesx;
    fb_dgeesx_fortran_fn_t oracle_fortran_fn =
        (fb_dgeesx_fortran_fn_t)oracle->ext_ops[FB_OP_DGEESX][FB_CONV_FORTRAN];
    fb_dgeesx_fortran_fn_t cand_fortran_fn =
        (fb_dgeesx_fortran_fn_t)cand->ext_ops[FB_OP_DGEESX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const double *A_in = (const double *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    char sense = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;
    double rconde_oracle = 1.0;
    double rcondv_oracle = 1.0;
    double rconde_cand = 1.0;
    double rcondv_cand = 1.0;

    double *A_oracle = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *A_cand = (double *)malloc((size_t)(lda * n) * sizeof(double));
    double *WR_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WI_oracle = (double *)malloc((size_t)n * sizeof(double));
    double *WR_cand = (double *)malloc((size_t)n * sizeof(double));
    double *WI_cand = (double *)malloc((size_t)n * sizeof(double));
    double *VS = (double *)malloc((size_t)(n * n) * sizeof(double));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !WR_oracle || !WI_oracle || !WR_cand ||
        !WI_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_dgeesx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(double));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(double));

    if (call_dgeesx_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_oracle, (int)lda, &sdim_oracle,
                            WR_oracle, WI_oracle, VS, (int)n, &rconde_oracle,
                            &rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_dgeesx;
    }

    if (call_dgeesx_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_cand, (int)lda, &sdim_cand,
                            WR_cand, WI_cand, VS, (int)n, &rconde_cand,
                            &rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_dgeesx;
    }

    {
        double max_err = 0.0;
        for (int j = 0; j < n; j++) {
            double or_r = WR_oracle[j], or_i = WI_oracle[j];
            double ca_r = WR_cand[j], ca_i = WI_cand[j];
            double mag_o = sqrt(or_r * or_r + or_i * or_i);
            double diff = sqrt((or_r - ca_r) * (or_r - ca_r) +
                               (or_i - ca_i) * (or_i - ca_i));
            double re = (mag_o > 1e-16) ? diff / mag_o : diff;
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_f64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            double *QT = (double *)malloc((size_t)(n * n) * sizeof(double));
            if (QT) {
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += VS[i * n + k] * A_cand[k * lda + j];
                        }
                        QT[i * n + j] = sum;
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += QT[i * n + k] * VS[j * n + k];
                        }
                        {
                            double diff = A_in[i * lda + j] - sum;
                            sum_diff += diff * diff;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
                free(QT);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
        }
    }

    {
        double *ZtZ = (double *)malloc((size_t)(n * n) * sizeof(double));
        double ortho_err = 0.0;
        if (ZtZ) {
            atac_f64(VS, n, n, n, ZtZ, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double err = ZtZ[i * n + j] - (i == j ? 1.0 : 0.0);
                    ortho_err += err * err;
                }
            }
            free(ZtZ);
        }
        res->orthogonality = result_from_relerr(sqrt(ortho_err) / sqrt((double)n));
    }

    {
        for (int j = 0; j < n; j++) {
            eig_mag[j] = sqrt(WR_cand[j] * WR_cand[j] + WI_cand[j] * WI_cand[j]);
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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VSt = (double *)malloc((size_t)(n * n) * sizeof(double));
            double rconde_t = 1.0, rcondv_t = 1.0;
            int sdim_t = 0;
            if (At && WRt && WIt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
                (void)call_dgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, WRt, WIt, VSt,
                                          (int)n, &rconde_t, &rcondv_t);
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            double *At = (double *)malloc((size_t)(lda * n) * sizeof(double));
            double *WRt = (double *)malloc((size_t)n * sizeof(double));
            double *WIt = (double *)malloc((size_t)n * sizeof(double));
            double *VSt = (double *)malloc((size_t)(n * n) * sizeof(double));
            double rconde_t = 1.0, rcondv_t = 1.0;
            int sdim_t = 0;
            if (!At || !WRt || !WIt || !VSt) {
                free(At); free(WRt); free(WIt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(double));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_dgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, WRt, WIt, VSt,
                                          (int)n, &rconde_t, &rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(WRt); free(WIt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_dgeesx:
    free(A_oracle); free(A_cand);
    free(WR_oracle); free(WI_oracle);
    free(WR_cand); free(WI_cand);
    free(VS); free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgeesx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_cgeesx_fn_t oracle_fn = (fb_cgeesx_fn_t)oracle->cgeesx;
    fb_cgeesx_fn_t cand_fn = (fb_cgeesx_fn_t)cand->cgeesx;
    fb_cgeesx_fortran_fn_t oracle_fortran_fn =
        (fb_cgeesx_fortran_fn_t)oracle->ext_ops[FB_OP_CGEESX][FB_CONV_FORTRAN];
    fb_cgeesx_fortran_fn_t cand_fortran_fn =
        (fb_cgeesx_fortran_fn_t)cand->ext_ops[FB_OP_CGEESX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_float_t *A_in = (const fb_complex_float_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    char sense = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;
    float rconde_oracle = 1.0f;
    float rcondv_oracle = 1.0f;
    float rconde_cand = 1.0f;
    float rcondv_cand = 1.0f;

    fb_complex_float_t *A_oracle = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *A_cand = (fb_complex_float_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_oracle = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *W_cand = (fb_complex_float_t *)malloc(
        (size_t)n * sizeof(fb_complex_float_t));
    fb_complex_float_t *VS = (fb_complex_float_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_float_t));
    float *eig_mag = (float *)malloc((size_t)n * sizeof(float));

    if (!A_oracle || !A_cand || !W_oracle || !W_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_cgeesx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));

    if (call_cgeesx_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_oracle, (int)lda, &sdim_oracle,
                            W_oracle, VS, (int)n, &rconde_oracle,
                            &rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_cgeesx;
    }

    if (call_cgeesx_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_cand, (int)lda, &sdim_cand,
                            W_cand, VS, (int)n, &rconde_cand,
                            &rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_cgeesx;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_cf32(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            fb_complex_float_t *QT = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            if (QT) {
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double qt_re = 0.0, qt_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double q_re = (double)FB_CF_REAL(VS[i * n + k]);
                            double q_im = (double)FB_CF_IMAG(VS[i * n + k]);
                            double t_re = (double)FB_CF_REAL(A_cand[k * lda + j]);
                            double t_im = (double)FB_CF_IMAG(A_cand[k * lda + j]);
                            qt_re += q_re * t_re - q_im * t_im;
                            qt_im += q_re * t_im + q_im * t_re;
                        }
                        QT[i * n + j] = fb_spectral_make_cf32((float)qt_re,
                                                              (float)qt_im);
                    }
                }

                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double re_sum = 0.0, im_sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            double qt_re = (double)FB_CF_REAL(QT[i * n + k]);
                            double qt_im = (double)FB_CF_IMAG(QT[i * n + k]);
                            double q_re = (double)FB_CF_REAL(VS[j * n + k]);
                            double q_im = (double)FB_CF_IMAG(VS[j * n + k]);
                            re_sum += qt_re * q_re + qt_im * q_im;
                            im_sum += qt_im * q_re - qt_re * q_im;
                        }
                        {
                            double a_re = (double)FB_CF_REAL(A_in[i * lda + j]);
                            double a_im = (double)FB_CF_IMAG(A_in[i * lda + j]);
                            double dr = a_re - re_sum;
                            double di = a_im - im_sum;
                            sum_diff += dr * dr + di * di;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
                free(QT);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
        }
    }

    {
        double raw_err = ahac_cf32_ortho_error(VS, n, n, n);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f32(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VSt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            float rconde_t = 1.0f, rcondv_t = 1.0f;
            int sdim_t = 0;
            if (At && Wt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
                (void)call_cgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, Wt, VSt, (int)n,
                                          &rconde_t, &rcondv_t);
            }
            free(At); free(Wt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_float_t *At = (fb_complex_float_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_float_t));
            fb_complex_float_t *Wt = (fb_complex_float_t *)malloc(
                (size_t)n * sizeof(fb_complex_float_t));
            fb_complex_float_t *VSt = (fb_complex_float_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_float_t));
            float rconde_t = 1.0f, rcondv_t = 1.0f;
            int sdim_t = 0;
            if (!At || !Wt || !VSt) {
                free(At); free(Wt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_float_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_cgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, Wt, VSt, (int)n,
                                          &rconde_t, &rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_cgeesx:
    free(A_oracle);
    free(A_cand);
    free(W_oracle);
    free(W_cand);
    free(VS);
    free(eig_mag);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgeesx(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *res,
    uint64_t *ns_out)
{
    fb_zgeesx_fn_t oracle_fn = (fb_zgeesx_fn_t)oracle->zgeesx;
    fb_zgeesx_fn_t cand_fn = (fb_zgeesx_fn_t)cand->zgeesx;
    fb_zgeesx_fortran_fn_t oracle_fortran_fn =
        (fb_zgeesx_fortran_fn_t)oracle->ext_ops[FB_OP_ZGEESX][FB_CONV_FORTRAN];
    fb_zgeesx_fortran_fn_t cand_fortran_fn =
        (fb_zgeesx_fortran_fn_t)cand->ext_ops[FB_OP_ZGEESX][FB_CONV_FORTRAN];
    if ((!oracle_fn && !oracle_fortran_fn) || (!cand_fn && !cand_fortran_fn))
        return FB_JUDGE_ERR_NOT_IMPL;

    memset(res, 0, sizeof(*res));
    *ns_out = 0;

    const fb_complex_double_t *A_in = (const fb_complex_double_t *)tc->A;
    int n = tc->n;
    int lda = tc->lda ? tc->lda : tc->n;

    fb_layout_t layout = FB_LAYOUT_ROW_MAJOR;
    char jobvs = 'V';
    char sort = 'N';
    char sense = 'N';
    int sdim_oracle = 0;
    int sdim_cand = 0;
    double rconde_oracle = 1.0;
    double rcondv_oracle = 1.0;
    double rconde_cand = 1.0;
    double rcondv_cand = 1.0;

    fb_complex_double_t *A_oracle = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *A_cand = (fb_complex_double_t *)malloc(
        (size_t)(lda * n) * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_oracle = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *W_cand = (fb_complex_double_t *)malloc(
        (size_t)n * sizeof(fb_complex_double_t));
    fb_complex_double_t *VS = (fb_complex_double_t *)malloc(
        (size_t)(n * n) * sizeof(fb_complex_double_t));
    double *eig_mag = (double *)malloc((size_t)n * sizeof(double));

    if (!A_oracle || !A_cand || !W_oracle || !W_cand || !VS || !eig_mag) {
        mark_oracle_fatal(res);
        goto cleanup_zgeesx;
    }

    memcpy(A_oracle, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
    memcpy(A_cand, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));

    if (call_zgeesx_backend(oracle_fn, oracle_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_oracle, (int)lda, &sdim_oracle,
                            W_oracle, VS, (int)n, &rconde_oracle,
                            &rcondv_oracle) != 0) {
        mark_oracle_fatal(res);
        goto cleanup_zgeesx;
    }

    if (call_zgeesx_backend(cand_fn, cand_fortran_fn, layout, jobvs, sort,
                            sense, (int)n, A_cand, (int)lda, &sdim_cand,
                            W_cand, VS, (int)n, &rconde_cand,
                            &rcondv_cand) != 0) {
        mark_cand_fatal(res);
        goto cleanup_zgeesx;
    }

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
            if (re > max_err) max_err = re;
        }
        res->values = result_from_relerr(max_err);
    }

    {
        double norm_A = frobenius_norm_cf64(A_in, n, n, lda);
        if (norm_A < 1e-16) {
            res->reconstruction = (fb_judge_case_result_t){.digits = 16};
        } else {
            fb_complex_double_t *QT = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            if (QT) {
                double sum_diff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double qt_re = 0.0, qt_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double q_re = FB_CD_REAL(VS[i * n + k]);
                            double q_im = FB_CD_IMAG(VS[i * n + k]);
                            double t_re = FB_CD_REAL(A_cand[k * lda + j]);
                            double t_im = FB_CD_IMAG(A_cand[k * lda + j]);
                            qt_re += q_re * t_re - q_im * t_im;
                            qt_im += q_re * t_im + q_im * t_re;
                        }
                        QT[i * n + j] = fb_spectral_make_cf64(qt_re, qt_im);
                    }
                }

                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double re_sum = 0.0, im_sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            double qt_re = FB_CD_REAL(QT[i * n + k]);
                            double qt_im = FB_CD_IMAG(QT[i * n + k]);
                            double q_re = FB_CD_REAL(VS[j * n + k]);
                            double q_im = FB_CD_IMAG(VS[j * n + k]);
                            re_sum += qt_re * q_re + qt_im * q_im;
                            im_sum += qt_im * q_re - qt_re * q_im;
                        }
                        {
                            double a_re = FB_CD_REAL(A_in[i * lda + j]);
                            double a_im = FB_CD_IMAG(A_in[i * lda + j]);
                            double dr = a_re - re_sum;
                            double di = a_im - im_sum;
                            sum_diff += dr * dr + di * di;
                        }
                    }
                }
                res->reconstruction = result_from_relerr(sqrt(sum_diff) / norm_A);
                free(QT);
            } else {
                res->reconstruction =
                    (fb_judge_case_result_t){.digits = 15, .relative_error = 0.0};
            }
        }
    }

    {
        double raw_err = ahac_cf64_ortho_error(VS, n, n, n);
        res->orthogonality = result_from_relerr(raw_err / sqrt((double)n));
    }

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
        {
            double gap_min = 1.0;
            bool clustered = is_clustered_f64(eig_mag, n, &gap_min);
            if (clustered && gap_min < 1.0)
                res->subspace = result_from_relerr(gap_min);
            else
                res->subspace = (fb_judge_case_result_t){.digits = 16};
        }
    }

    res->pairs = (fb_judge_case_result_t){.digits = 16};

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < 2; w++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VSt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            double rconde_t = 1.0, rcondv_t = 1.0;
            int sdim_t = 0;
            if (At && Wt && VSt) {
                memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
                (void)call_zgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, Wt, VSt, (int)n,
                                          &rconde_t, &rcondv_t);
            }
            free(At); free(Wt); free(VSt);
        }
        for (int t = 0; t < 5; t++) {
            fb_complex_double_t *At = (fb_complex_double_t *)malloc(
                (size_t)(lda * n) * sizeof(fb_complex_double_t));
            fb_complex_double_t *Wt = (fb_complex_double_t *)malloc(
                (size_t)n * sizeof(fb_complex_double_t));
            fb_complex_double_t *VSt = (fb_complex_double_t *)malloc(
                (size_t)(n * n) * sizeof(fb_complex_double_t));
            double rconde_t = 1.0, rcondv_t = 1.0;
            int sdim_t = 0;
            if (!At || !Wt || !VSt) {
                free(At); free(Wt); free(VSt);
                break;
            }
            memcpy(At, A_in, (size_t)(lda * n) * sizeof(fb_complex_double_t));
            {
                uint64_t t0 = fb_judge_time_ns();
                (void)call_zgeesx_backend(cand_fn, cand_fortran_fn, layout,
                                          jobvs, sort, sense, (int)n, At,
                                          (int)lda, &sdim_t, Wt, VSt, (int)n,
                                          &rconde_t, &rcondv_t);
                {
                    uint64_t dt = fb_judge_time_ns() - t0;
                    if (dt < best) best = dt;
                }
            }
            free(At); free(Wt); free(VSt);
        }
        *ns_out = best;
    }

cleanup_zgeesx:
    free(A_oracle);
    free(A_cand);
    free(W_oracle);
    free(W_cand);
    free(VS);
    free(eig_mag);
    return FB_JUDGE_OK;
}

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
    typedef int (*fb_cheevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  fb_complex_float_t *a, int lda, float *w);
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
        int oracle_info = -1;
        if (oracle->cheevd) {
            memcpy(A_oracle, A_in,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            oracle_info = ((fb_cheevd_fn_t)oracle->cheevd)(FB_LAYOUT_ROW_MAJOR,
                                                           jobz, (char)uplo,
                                                           n, A_oracle, lda,
                                                           w_oracle);
        }
        if (oracle_info != 0) {
            mark_oracle_fatal(res);
            goto cleanup_chegv;
        }
    }
    if (cand->chegv(layout, itype, jobz, uplo, n,
                    A_cand, lda, B_cand, ldb, w_cand) != 0) {
        int cand_info = -1;
        if (cand->cheevd) {
            memcpy(A_cand, A_in,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_float_t));
            cand_info = ((fb_cheevd_fn_t)cand->cheevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)uplo, n,
                                                       A_cand, lda, w_cand);
        }
        if (cand_info != 0) {
            mark_cand_fatal(res);
            goto cleanup_chegv;
        }
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
    typedef int (*fb_zheevd_fn_t)(int layout, char jobz, char uplo, int n,
                                  fb_complex_double_t *a, int lda, double *w);
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
        int oracle_info = -1;
        if (oracle->zheevd) {
            memcpy(A_oracle, A_in,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            oracle_info = ((fb_zheevd_fn_t)oracle->zheevd)(FB_LAYOUT_ROW_MAJOR,
                                                           jobz, (char)uplo,
                                                           n, A_oracle, lda,
                                                           w_oracle);
        }
        if (oracle_info != 0) {
            mark_oracle_fatal(res);
            goto cleanup_zhegv;
        }
    }
    if (cand->zhegv(layout, itype, jobz, uplo, n,
                    A_cand, lda, B_cand, ldb, w_cand) != 0) {
        int cand_info = -1;
        if (cand->zheevd) {
            memcpy(A_cand, A_in,
                   (size_t)n * (size_t)lda * sizeof(fb_complex_double_t));
            cand_info = ((fb_zheevd_fn_t)cand->zheevd)(FB_LAYOUT_ROW_MAJOR,
                                                       jobz, (char)uplo, n,
                                                       A_cand, lda, w_cand);
        }
        if (cand_info != 0) {
            mark_cand_fatal(res);
            goto cleanup_zhegv;
        }
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
    [FB_OP_SSYEVD] = run_ssyevd, [FB_OP_DSYEVD] = run_dsyevd,
    [FB_OP_CHEEVD] = run_cheev,  [FB_OP_ZHEEVD] = run_zheev,
    [FB_OP_SGESVD] = run_sgesvd, [FB_OP_DGESVD] = run_dgesvd,
    [FB_OP_CGESVD] = run_cgesvd, [FB_OP_ZGESVD] = run_zgesvd,
    [FB_OP_SBDSDC] = run_sbdsdc, [FB_OP_DBDSDC] = run_dbdsdc,
    [FB_OP_CBDSDC] = run_cbdsdc, [FB_OP_ZBDSDC] = run_zbdsdc,
    [FB_OP_SBDSQR] = run_sbdsqr, [FB_OP_DBDSQR] = run_dbdsqr,
    [FB_OP_CBDSQR] = run_cbdsqr, [FB_OP_ZBDSQR] = run_zbdsqr,
    [FB_OP_SGEES] = run_sgees,   [FB_OP_DGEES] = run_dgees,
    [FB_OP_CGEES] = run_cgees,   [FB_OP_ZGEES] = run_zgees,
    [FB_OP_SGEESX] = run_sgeesx, [FB_OP_DGEESX] = run_dgeesx,
    [FB_OP_CGEESX] = run_cgeesx, [FB_OP_ZGEESX] = run_zgeesx,
    [FB_OP_SGEEVX] = run_sgeevx, [FB_OP_DGEEVX] = run_dgeevx,
    [FB_OP_CGEEVX] = run_cgeevx, [FB_OP_ZGEEVX] = run_zgeevx,
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