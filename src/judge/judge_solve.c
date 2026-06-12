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
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>

#define FB_SOLVE_WARMUP_RUNS  2
#define FB_SOLVE_TIMING_RUNS  5
#define FB_BAND_CALL_NOT_IMPL INT_MIN
#define FB_BAND_CALL_ALLOC_FAILURE (INT_MIN + 1)

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

typedef struct {
    double scale;
    double ssq;
} fb_scaled_sumsq_t;

static void fb_scaled_sumsq_init(fb_scaled_sumsq_t *acc)
{
    acc->scale = 0.0;
    acc->ssq = 1.0;
}

static void fb_scaled_sumsq_add(fb_scaled_sumsq_t *acc, double absx)
{
    if (absx <= 0.0) {
        return;
    }
    if (acc->scale < absx) {
        double ratio = (acc->scale > 0.0) ? (acc->scale / absx) : 0.0;
        acc->ssq = 1.0 + acc->ssq * ratio * ratio;
        acc->scale = absx;
    } else {
        double ratio = absx / acc->scale;
        acc->ssq += ratio * ratio;
    }
}

static double fb_scaled_sumsq_norm(const fb_scaled_sumsq_t *acc)
{
    return (acc->scale > 0.0) ? (acc->scale * sqrt(acc->ssq)) : 0.0;
}

static double fb_safe_positive_mul_add(double a, double b, double c)
{
    double product;

    if (a <= 0.0 || b <= 0.0) {
        return c;
    }

    {
        int exp_a = 0;
        int exp_b = 0;
        double mant_a = frexp(a, &exp_a);
        double mant_b = frexp(b, &exp_b);
        int product_exp = exp_a + exp_b;
        double product_mant = mant_a * mant_b;

        if (product_mant == 0.0) {
            product = 0.0;
        } else {
            if (fabs(product_mant) >= 1.0) {
                product_mant *= 0.5;
                ++product_exp;
            }
            if (product_exp > DBL_MAX_EXP) {
                product = INFINITY;
            } else if (product_exp < DBL_MIN_EXP - DBL_MANT_DIG) {
                product = 0.0;
            } else {
                product = ldexp(product_mant, product_exp);
            }
        }
    }

    if (isinf(product)) {
        return INFINITY;
    }

    {
        double max_term = fmax(product, c);
        if (max_term == 0.0) {
            return 0.0;
        }
        return max_term * (1.0 + fmin(product, c) / max_term);
    }
}

/**
 * Backward error η = ‖b − A·x‖_F / (‖A‖_F · ‖x‖_F + ‖b‖_F).
 * A: m×n row-major lda; b: m×nrhs ldb; x: n×nrhs ldx.
 */
static double bwerr_f32(const float  *A, int m, int n, int lda,
                         const float  *b, int ldb,
                         const float  *x, int ldx, int nrhs)
{
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij = (double)b[i*ldb + j];
            if (!isfinite(rij)) return NAN;
            for (int k = 0; k < n; k++)
            {
                double av = (double)A[i*lda + k];
                double xv = (double)x[k*ldx + j];
                if (!isfinite(av) || !isfinite(xv)) return NAN;
                rij -= av * xv;
            }
            if (!isfinite(rij)) return NAN;
            fb_scaled_sumsq_add(&r_acc, fabs(rij));
            fb_scaled_sumsq_add(&b_acc, fabs((double)b[i*ldb+j]));
        }
        for (int i = 0; i < n; i++) {
            double v = (double)x[i*ldx+j];
            if (!isfinite(v)) return NAN;
            fb_scaled_sumsq_add(&x_acc, fabs(v));
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            double v = (double)A[i*lda+k];
            if (!isfinite(v)) return NAN;
            fb_scaled_sumsq_add(&a_acc, fabs(v));
        }
    double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                            fb_scaled_sumsq_norm(&x_acc),
                                            fb_scaled_sumsq_norm(&b_acc));
    if (denom < (double)FLT_EPSILON)
        return (fb_scaled_sumsq_norm(&r_acc) < (double)FLT_EPSILON*(double)FLT_EPSILON) ? 0.0 : 1.0;
    if (isinf(denom)) return 0.0;
    return fb_scaled_sumsq_norm(&r_acc) / denom;
}

static double bwerr_f64(const double *A, int m, int n, int lda,
                         const double *b, int ldb,
                         const double *x, int ldx, int nrhs)
{
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij = b[i*ldb+j];
            if (!isfinite(rij)) return NAN;
            for (int k = 0; k < n; k++)
            {
                double av = A[i*lda+k];
                double xv = x[k*ldx+j];
                if (!isfinite(av) || !isfinite(xv)) return NAN;
                rij -= av * xv;
            }
            if (!isfinite(rij)) return NAN;
            fb_scaled_sumsq_add(&r_acc, fabs(rij));
            fb_scaled_sumsq_add(&b_acc, fabs(b[i*ldb+j]));
        }
        for (int i = 0; i < n; i++) {
            double v = x[i*ldx+j];
            if (!isfinite(v)) return NAN;
            fb_scaled_sumsq_add(&x_acc, fabs(v));
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            double v = A[i*lda+k];
            if (!isfinite(v)) return NAN;
            fb_scaled_sumsq_add(&a_acc, fabs(v));
        }
    double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                            fb_scaled_sumsq_norm(&x_acc),
                                            fb_scaled_sumsq_norm(&b_acc));
    if (denom < DBL_EPSILON) return (fb_scaled_sumsq_norm(&r_acc) < DBL_EPSILON*DBL_EPSILON) ? 0.0 : 1.0;
    if (isinf(denom)) return 0.0;
    return fb_scaled_sumsq_norm(&r_acc) / denom;
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

static double bwerr_cf32(const fb_complex_float_t *A, int m, int n, int lda,
                         const fb_complex_float_t *b, int ldb,
                         const fb_complex_float_t *x, int ldx, int nrhs);
static double bwerr_cf64(const fb_complex_double_t *A, int m, int n, int lda,
                         const fb_complex_double_t *b, int ldb,
                         const fb_complex_double_t *x, int ldx, int nrhs);

static void fb_build_band_dense_f32(
    const float *src, float *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            dst[(size_t)i * (size_t)n + (size_t)j] = (float)((j > i) ? -scale : scale);
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fabs((double)dst[(size_t)i * (size_t)n + (size_t)j]);
        }
        dst[(size_t)i * (size_t)n + (size_t)i] = (float)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

static void fb_build_band_dense_f64(
    const double *src, double *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            dst[(size_t)i * (size_t)n + (size_t)j] = (j > i) ? -scale : scale;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fabs(dst[(size_t)i * (size_t)n + (size_t)j]);
        }
        dst[(size_t)i * (size_t)n + (size_t)i] = row_sum + 8.0 + 0.0625 * (double)(i + 1);
    }
}

static void fb_build_band_dense_cf32(
    const fb_complex_float_t *src, fb_complex_float_t *dst,
    int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            float scale = 0.125f / (float)(abs(j - i));
            float real_part = (j > i) ? -scale : scale;
            float imag_part = (i > j) ? 0.03125f * scale : -0.03125f * scale;
            fb_complex_float_t value = 0.0f;
            __real__ value = real_part;
            __imag__ value = imag_part;
            dst[(size_t)i * (size_t)n + (size_t)j] = value;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fb_cf32_abs(dst[(size_t)i * (size_t)n + (size_t)j]);
        }
        dst[(size_t)i * (size_t)n + (size_t)i] = (fb_complex_float_t)(float)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

static void fb_build_band_dense_cf64(
    const fb_complex_double_t *src, fb_complex_double_t *dst,
    int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j == i) continue;
            double scale = 0.125 / (double)(abs(j - i));
            double real_part = (j > i) ? -scale : scale;
            double imag_part = (i > j) ? 0.03125 * scale : -0.03125 * scale;
            fb_complex_double_t value = 0.0;
            __real__ value = real_part;
            __imag__ value = imag_part;
            dst[(size_t)i * (size_t)n + (size_t)j] = value;
        }
    }
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j0 = i - kl;
        int j1 = i + ku;
        if (j0 < 0) j0 = 0;
        if (j1 >= n) j1 = n - 1;
        for (int j = j0; j <= j1; j++) {
            if (j != i) row_sum += fb_cf64_abs(dst[(size_t)i * (size_t)n + (size_t)j]);
        }
        dst[(size_t)i * (size_t)n + (size_t)i] = (fb_complex_double_t)(row_sum + 8.0 + 0.0625 * (double)(i + 1));
    }
}

static void fb_build_rhs_f32(const float *A, int n, int lda, int nrhs, float *b)
{
    for (int rhs = 0; rhs < nrhs; rhs++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                double xk = 0.25 * (double)(rhs + 1) + 0.03125 * (double)(k + 1);
                sum += (double)A[(size_t)i * (size_t)lda + (size_t)k] * xk;
            }
            b[(size_t)i * (size_t)nrhs + (size_t)rhs] = (float)sum;
        }
    }
}

static void fb_build_rhs_f64(const double *A, int n, int lda, int nrhs, double *b)
{
    for (int rhs = 0; rhs < nrhs; rhs++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                double xk = 0.25 * (double)(rhs + 1) + 0.03125 * (double)(k + 1);
                sum += A[(size_t)i * (size_t)lda + (size_t)k] * xk;
            }
            b[(size_t)i * (size_t)nrhs + (size_t)rhs] = sum;
        }
    }
}

[[maybe_unused]] static void fb_build_rhs_cf32(const fb_complex_float_t *A, int n, int lda, int nrhs, fb_complex_float_t *b)
{
    for (int rhs = 0; rhs < nrhs; rhs++) {
        for (int i = 0; i < n; i++) {
            double sum_re = 0.0;
            double sum_im = 0.0;
            for (int k = 0; k < n; k++) {
                double ar = (double)__real__(A[(size_t)i * (size_t)lda + (size_t)k]);
                double ai = (double)__imag__(A[(size_t)i * (size_t)lda + (size_t)k]);
                double xr = 0.25 * (double)(rhs + 1) + 0.03125 * (double)(k + 1);
                double xi = -0.0625 * (double)(rhs + 1) + 0.015625 * (double)(k + 1);
                sum_re += ar * xr - ai * xi;
                sum_im += ar * xi + ai * xr;
            }
            fb_complex_float_t value = 0.0f;
            __real__ value = (float)sum_re;
            __imag__ value = (float)sum_im;
            b[(size_t)i * (size_t)nrhs + (size_t)rhs] = value;
        }
    }
}

[[maybe_unused]] static void fb_build_rhs_cf64(const fb_complex_double_t *A, int n, int lda, int nrhs, fb_complex_double_t *b)
{
    for (int rhs = 0; rhs < nrhs; rhs++) {
        for (int i = 0; i < n; i++) {
            double sum_re = 0.0;
            double sum_im = 0.0;
            for (int k = 0; k < n; k++) {
                double ar = __real__(A[(size_t)i * (size_t)lda + (size_t)k]);
                double ai = __imag__(A[(size_t)i * (size_t)lda + (size_t)k]);
                double xr = 0.25 * (double)(rhs + 1) + 0.03125 * (double)(k + 1);
                double xi = -0.0625 * (double)(rhs + 1) + 0.015625 * (double)(k + 1);
                sum_re += ar * xr - ai * xi;
                sum_im += ar * xi + ai * xr;
            }
            fb_complex_double_t value = 0.0;
            __real__ value = sum_re;
            __imag__ value = sum_im;
            b[(size_t)i * (size_t)nrhs + (size_t)rhs] = value;
        }
    }
}

static void fb_build_gbsvx_dense_f32(const float *src, float *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        dst[(size_t)i * (size_t)n + (size_t)i] = 4.0f + 0.5f * (float)i;
        if (kl > 0 && i > 0) dst[(size_t)i * (size_t)n + (size_t)(i - 1)] = 1.0f + 0.0625f * (float)i;
        if (ku > 0 && i + 1 < n) dst[(size_t)i * (size_t)n + (size_t)(i + 1)] = 2.0f + 0.0625f * (float)i;
    }
}

static void fb_build_gbsvx_dense_f64(const double *src, double *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        dst[(size_t)i * (size_t)n + (size_t)i] = 4.0 + 0.5 * (double)i;
        if (kl > 0 && i > 0) dst[(size_t)i * (size_t)n + (size_t)(i - 1)] = 1.0 + 0.0625 * (double)i;
        if (ku > 0 && i + 1 < n) dst[(size_t)i * (size_t)n + (size_t)(i + 1)] = 2.0 + 0.0625 * (double)i;
    }
}

[[maybe_unused]] static void fb_build_gbsvx_dense_cf32(const fb_complex_float_t *src, fb_complex_float_t *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        dst[(size_t)i * (size_t)n + (size_t)i] = (fb_complex_float_t)(4.0f + 0.5f * (float)i);
        if (kl > 0 && i > 0) dst[(size_t)i * (size_t)n + (size_t)(i - 1)] = (fb_complex_float_t)(1.0f + 0.0625f * (float)i);
        if (ku > 0 && i + 1 < n) dst[(size_t)i * (size_t)n + (size_t)(i + 1)] = (fb_complex_float_t)(2.0f + 0.0625f * (float)i);
    }
}

[[maybe_unused]] static void fb_build_gbsvx_dense_cf64(const fb_complex_double_t *src, fb_complex_double_t *dst, int n, int lda, int kl, int ku)
{
    (void)src;
    (void)lda;
    memset(dst, 0, (size_t)n * (size_t)n * sizeof(*dst));
    for (int i = 0; i < n; i++) {
        dst[(size_t)i * (size_t)n + (size_t)i] = (fb_complex_double_t)(4.0 + 0.5 * (double)i);
        if (kl > 0 && i > 0) dst[(size_t)i * (size_t)n + (size_t)(i - 1)] = (fb_complex_double_t)(1.0 + 0.0625 * (double)i);
        if (ku > 0 && i + 1 < n) dst[(size_t)i * (size_t)n + (size_t)(i + 1)] = (fb_complex_double_t)(2.0 + 0.0625 * (double)i);
    }
}

static void fb_pack_band_row_major_f32(
    const float *A, int n, int lda, int kl, int ku, float *ab, int ldab)
{
    int band_rows = 2 * kl + ku + 1;
    memset(ab, 0, (size_t)band_rows * (size_t)ldab * sizeof(*ab));
    for (int j = 0; j < n; j++) {
        int i0 = j - ku;
        int i1 = j + kl;
        if (i0 < 0) i0 = 0;
        if (i1 >= n) i1 = n - 1;
        for (int i = i0; i <= i1; i++) {
            int band_row = kl + ku + i - j;
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] =
                A[(size_t)i * (size_t)lda + (size_t)j];
        }
    }
}

static void fb_pack_band_row_major_f64(
    const double *A, int n, int lda, int kl, int ku, double *ab, int ldab)
{
    int band_rows = 2 * kl + ku + 1;
    memset(ab, 0, (size_t)band_rows * (size_t)ldab * sizeof(*ab));
    for (int j = 0; j < n; j++) {
        int i0 = j - ku;
        int i1 = j + kl;
        if (i0 < 0) i0 = 0;
        if (i1 >= n) i1 = n - 1;
        for (int i = i0; i <= i1; i++) {
            int band_row = kl + ku + i - j;
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] =
                A[(size_t)i * (size_t)lda + (size_t)j];
        }
    }
}

static void fb_pack_band_row_major_cf32(
    const fb_complex_float_t *A, int n, int lda, int kl, int ku,
    fb_complex_float_t *ab, int ldab)
{
    int band_rows = 2 * kl + ku + 1;
    memset(ab, 0, (size_t)band_rows * (size_t)ldab * sizeof(*ab));
    for (int j = 0; j < n; j++) {
        int i0 = j - ku;
        int i1 = j + kl;
        if (i0 < 0) i0 = 0;
        if (i1 >= n) i1 = n - 1;
        for (int i = i0; i <= i1; i++) {
            int band_row = kl + ku + i - j;
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] =
                A[(size_t)i * (size_t)lda + (size_t)j];
        }
    }
}

static void fb_pack_band_row_major_cf64(
    const fb_complex_double_t *A, int n, int lda, int kl, int ku,
    fb_complex_double_t *ab, int ldab)
{
    int band_rows = 2 * kl + ku + 1;
    memset(ab, 0, (size_t)band_rows * (size_t)ldab * sizeof(*ab));
    for (int j = 0; j < n; j++) {
        int i0 = j - ku;
        int i1 = j + kl;
        if (i0 < 0) i0 = 0;
        if (i1 >= n) i1 = n - 1;
        for (int i = i0; i <= i1; i++) {
            int band_row = kl + ku + i - j;
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] =
                A[(size_t)i * (size_t)lda + (size_t)j];
        }
    }
}

#define FB_DEFINE_PACK_COMPACT_BAND_ROW_MAJOR(name, scalar_t) \
static void name(const scalar_t *A, int n, int lda, int kl, int ku, scalar_t *ab, int ldab) \
{ \
    int compact_rows = kl + ku + 1; \
    memset(ab, 0, (size_t)compact_rows * (size_t)ldab * sizeof(*ab)); \
    for (int j = 0; j < n; j++) { \
        int i0 = j - ku; \
        int i1 = j + kl; \
        if (i0 < 0) i0 = 0; \
        if (i1 >= n) i1 = n - 1; \
        for (int i = i0; i <= i1; i++) { \
            int band_row = ku + i - j; \
            ab[(size_t)band_row * (size_t)ldab + (size_t)j] = A[(size_t)i * (size_t)lda + (size_t)j]; \
        } \
    } \
}

#define FB_DEFINE_ROW_COL_COPY_FUNCS(row_to_col_name, col_to_row_name, scalar_t) \
static void row_to_col_name(const scalar_t *src, int rows, int cols, scalar_t *dst) \
{ \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) { \
            dst[(size_t)i + (size_t)j * (size_t)rows] = src[(size_t)i * (size_t)cols + (size_t)j]; \
        } \
    } \
} \
static void col_to_row_name(const scalar_t *src, int rows, int cols, scalar_t *dst) \
{ \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) { \
            dst[(size_t)i * (size_t)cols + (size_t)j] = src[(size_t)i + (size_t)j * (size_t)rows]; \
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
            ab[(size_t)band_row + (size_t)j * (size_t)ldab] = \
                A[(size_t)i * (size_t)lda + (size_t)j]; \
        } \
    } \
}

#define FB_DEFINE_PACK_COMPACT_BAND_COL_MAJOR(name, scalar_t) \
static void name(const scalar_t *A, int n, int lda, int kl, int ku, scalar_t *ab, int ldab) \
{ \
    int compact_rows = kl + ku + 1; \
    memset(ab, 0, (size_t)compact_rows * (size_t)n * sizeof(*ab)); \
    for (int j = 0; j < n; j++) { \
        int i0 = j - ku; \
        int i1 = j + kl; \
        if (i0 < 0) i0 = 0; \
        if (i1 >= n) i1 = n - 1; \
        for (int i = i0; i <= i1; i++) { \
            int band_row = ku + i - j; \
            ab[(size_t)band_row + (size_t)j * (size_t)ldab] = A[(size_t)i * (size_t)lda + (size_t)j]; \
        } \
    } \
}

FB_DEFINE_PACK_COMPACT_BAND_ROW_MAJOR(fb_pack_compact_band_row_major_f32, float)
FB_DEFINE_PACK_COMPACT_BAND_ROW_MAJOR(fb_pack_compact_band_row_major_f64, double)
FB_DEFINE_PACK_COMPACT_BAND_ROW_MAJOR(fb_pack_compact_band_row_major_cf32, fb_complex_float_t)
FB_DEFINE_PACK_COMPACT_BAND_ROW_MAJOR(fb_pack_compact_band_row_major_cf64, fb_complex_double_t)

FB_DEFINE_ROW_COL_COPY_FUNCS(fb_copy_row_to_col_f32, fb_copy_col_to_row_f32, float)
FB_DEFINE_ROW_COL_COPY_FUNCS(fb_copy_row_to_col_f64, fb_copy_col_to_row_f64, double)
FB_DEFINE_ROW_COL_COPY_FUNCS(fb_copy_row_to_col_cf32, fb_copy_col_to_row_cf32, fb_complex_float_t)
FB_DEFINE_ROW_COL_COPY_FUNCS(fb_copy_row_to_col_cf64, fb_copy_col_to_row_cf64, fb_complex_double_t)

FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_f32, float)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_f64, double)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_cf32, fb_complex_float_t)
FB_DEFINE_PACK_BAND_COL_MAJOR(fb_pack_band_col_major_cf64, fb_complex_double_t)
[[maybe_unused]] FB_DEFINE_PACK_COMPACT_BAND_COL_MAJOR(fb_pack_compact_band_col_major_f32, float)
[[maybe_unused]] FB_DEFINE_PACK_COMPACT_BAND_COL_MAJOR(fb_pack_compact_band_col_major_f64, double)
FB_DEFINE_PACK_COMPACT_BAND_COL_MAJOR(fb_pack_compact_band_col_major_cf32, fb_complex_float_t)
FB_DEFINE_PACK_COMPACT_BAND_COL_MAJOR(fb_pack_compact_band_col_major_cf64, fb_complex_double_t)

typedef int (*fb_sgbsv_fn_t)(int layout, int n, int kl, int ku, int nrhs,
                             float *ab, int ldab, int *ipiv, float *b,
                             int ldb);
typedef int (*fb_dgbsv_fn_t)(int layout, int n, int kl, int ku, int nrhs,
                             double *ab, int ldab, int *ipiv, double *b,
                             int ldb);
typedef int (*fb_cgbsv_fn_t)(int layout, int n, int kl, int ku, int nrhs,
                             fb_complex_float_t *ab, int ldab, int *ipiv,
                             fb_complex_float_t *b, int ldb);
typedef int (*fb_zgbsv_fn_t)(int layout, int n, int kl, int ku, int nrhs,
                             fb_complex_double_t *ab, int ldab, int *ipiv,
                             fb_complex_double_t *b, int ldb);

typedef int (*fb_sgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              float *ab, int ldab, int *ipiv);
typedef int (*fb_dgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              double *ab, int ldab, int *ipiv);
typedef int (*fb_cgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              fb_complex_float_t *ab, int ldab, int *ipiv);
typedef int (*fb_zgbtrf_fn_t)(int layout, int m, int n, int kl, int ku,
                              fb_complex_double_t *ab, int ldab, int *ipiv);

typedef int (*fb_sgbtrs_fn_t)(int layout, char trans, int n, int kl, int ku,
                              int nrhs, const float *ab, int ldab,
                              const int *ipiv, float *b, int ldb);
typedef int (*fb_dgbtrs_fn_t)(int layout, char trans, int n, int kl, int ku,
                              int nrhs, const double *ab, int ldab,
                              const int *ipiv, double *b, int ldb);
typedef int (*fb_cgbtrs_fn_t)(int layout, char trans, int n, int kl, int ku,
                              int nrhs, const fb_complex_float_t *ab,
                              int ldab, const int *ipiv,
                              fb_complex_float_t *b, int ldb);
typedef int (*fb_zgbtrs_fn_t)(int layout, char trans, int n, int kl, int ku,
                              int nrhs, const fb_complex_double_t *ab,
                              int ldab, const int *ipiv,
                              fb_complex_double_t *b, int ldb);

typedef int (*fb_sgbsvx_fn_t)(int layout, char fact, char trans, int n,
                              int kl, int ku, int nrhs, float *ab, int ldab,
                              float *afb, int ldafb, int *ipiv, char *equed,
                              float *r, float *c, float *b, int ldb,
                              float *x, int ldx, float *rcond, float *ferr,
                              float *berr, float *rpivot);
typedef int (*fb_dgbsvx_fn_t)(int layout, char fact, char trans, int n,
                              int kl, int ku, int nrhs, double *ab, int ldab,
                              double *afb, int ldafb, int *ipiv, char *equed,
                              double *r, double *c, double *b, int ldb,
                              double *x, int ldx, double *rcond,
                              double *ferr, double *berr, double *rpivot);
typedef int (*fb_cgbsvx_fn_t)(int layout, char fact, char trans, int n,
                              int kl, int ku, int nrhs,
                              fb_complex_float_t *ab, int ldab,
                              fb_complex_float_t *afb, int ldafb, int *ipiv,
                              char *equed, float *r, float *c,
                              fb_complex_float_t *b, int ldb,
                              fb_complex_float_t *x, int ldx, float *rcond,
                              float *ferr, float *berr, float *rpivot);
typedef int (*fb_zgbsvx_fn_t)(int layout, char fact, char trans, int n,
                              int kl, int ku, int nrhs,
                              fb_complex_double_t *ab, int ldab,
                              fb_complex_double_t *afb, int ldafb, int *ipiv,
                              char *equed, double *r, double *c,
                              fb_complex_double_t *b, int ldb,
                              fb_complex_double_t *x, int ldx, double *rcond,
                              double *ferr, double *berr, double *rpivot);

typedef void (*fb_sgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                      float *ab, int *ldab, int *ipiv,
                                      float *b, int *ldb, int *info);
typedef void (*fb_dgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                      double *ab, int *ldab, int *ipiv,
                                      double *b, int *ldb, int *info);
typedef void (*fb_cgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                      fb_complex_float_t *ab, int *ldab,
                                      int *ipiv, fb_complex_float_t *b,
                                      int *ldb, int *info);
typedef void (*fb_zgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                      fb_complex_double_t *ab, int *ldab,
                                      int *ipiv, fb_complex_double_t *b,
                                      int *ldb, int *info);

typedef void (*fb_psgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                       float *ab, int *iab, int *jab,
                                       int *descab, int *ipiv, float *b,
                                       int *ib, int *jb, int *descb,
                                       int *info);
typedef void (*fb_pdgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                       double *ab, int *iab, int *jab,
                                       int *descab, int *ipiv, double *b,
                                       int *ib, int *jb, int *descb,
                                       int *info);
typedef void (*fb_pcgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                       fb_complex_float_t *ab, int *iab,
                                       int *jab, int *descab, int *ipiv,
                                       fb_complex_float_t *b, int *ib,
                                       int *jb, int *descb, int *info);
typedef void (*fb_pzgbsv_fortran_fn_t)(int *n, int *kl, int *ku, int *nrhs,
                                       fb_complex_double_t *ab, int *iab,
                                       int *jab, int *descab, int *ipiv,
                                       fb_complex_double_t *b, int *ib,
                                       int *jb, int *descb, int *info);

typedef void (*fb_psgesv_fortran_fn_t)(int *n, int *nrhs, float *a, int *ia,
                                       int *ja, int *desca, int *ipiv,
                                       float *b, int *ib, int *jb,
                                       int *descb, int *info);
typedef void (*fb_pdgesv_fortran_fn_t)(int *n, int *nrhs, double *a, int *ia,
                                       int *ja, int *desca, int *ipiv,
                                       double *b, int *ib, int *jb,
                                       int *descb, int *info);
typedef void (*fb_pcgesv_fortran_fn_t)(int *n, int *nrhs,
                                       fb_complex_float_t *a, int *ia,
                                       int *ja, int *desca, int *ipiv,
                                       fb_complex_float_t *b, int *ib,
                                       int *jb, int *descb, int *info);
typedef void (*fb_pzgesv_fortran_fn_t)(int *n, int *nrhs,
                                       fb_complex_double_t *a, int *ia,
                                       int *ja, int *desca, int *ipiv,
                                       fb_complex_double_t *b, int *ib,
                                       int *jb, int *descb, int *info);

typedef void (*fb_psgesvx_fortran_fn_t)(const char *fact, const char *trans,
                                        int *n, int *nrhs, float *a, int *ia,
                                        int *ja, int *desca, float *af,
                                        int *iaf, int *jaf, int *descaf,
                                        int *ipiv, char *equed, float *r,
                                        float *c, float *b, int *ib, int *jb,
                                        int *descb, float *x, int *ix,
                                        int *jx, int *descx, float *rcond,
                                        float *ferr, float *berr, float *work,
                                        int *iwork, int *info);
typedef void (*fb_pdgesvx_fortran_fn_t)(const char *fact, const char *trans,
                                        int *n, int *nrhs, double *a, int *ia,
                                        int *ja, int *desca, double *af,
                                        int *iaf, int *jaf, int *descaf,
                                        int *ipiv, char *equed, double *r,
                                        double *c, double *b, int *ib, int *jb,
                                        int *descb, double *x, int *ix,
                                        int *jx, int *descx, double *rcond,
                                        double *ferr, double *berr,
                                        double *work, int *iwork, int *info);
typedef void (*fb_pcgesvx_fortran_fn_t)(const char *fact, const char *trans,
                                        int *n, int *nrhs,
                                        fb_complex_float_t *a, int *ia,
                                        int *ja, int *desca,
                                        fb_complex_float_t *af, int *iaf,
                                        int *jaf, int *descaf, int *ipiv,
                                        char *equed, float *r, float *c,
                                        fb_complex_float_t *b, int *ib,
                                        int *jb, int *descb,
                                        fb_complex_float_t *x, int *ix,
                                        int *jx, int *descx, float *rcond,
                                        float *ferr, float *berr,
                                        fb_complex_float_t *work,
                                        float *rwork, int *info);
typedef void (*fb_pzgesvx_fortran_fn_t)(const char *fact, const char *trans,
                                        int *n, int *nrhs,
                                        fb_complex_double_t *a, int *ia,
                                        int *ja, int *desca,
                                        fb_complex_double_t *af, int *iaf,
                                        int *jaf, int *descaf, int *ipiv,
                                        char *equed, double *r, double *c,
                                        fb_complex_double_t *b, int *ib,
                                        int *jb, int *descb,
                                        fb_complex_double_t *x, int *ix,
                                        int *jx, int *descx, double *rcond,
                                        double *ferr, double *berr,
                                        fb_complex_double_t *work,
                                        double *rwork, int *info);

typedef void (*fb_psposv_fortran_fn_t)(const char *uplo, int *n, int *nrhs,
                                       float *a, int *ia, int *ja,
                                       int *desca, float *b, int *ib,
                                       int *jb, int *descb, int *info);
typedef void (*fb_pdposv_fortran_fn_t)(const char *uplo, int *n, int *nrhs,
                                       double *a, int *ia, int *ja,
                                       int *desca, double *b, int *ib,
                                       int *jb, int *descb, int *info);
typedef void (*fb_pcposv_fortran_fn_t)(const char *uplo, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *ia,
                                       int *ja, int *desca,
                                       fb_complex_float_t *b, int *ib,
                                       int *jb, int *descb, int *info);
typedef void (*fb_pzposv_fortran_fn_t)(const char *uplo, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *ia,
                                       int *ja, int *desca,
                                       fb_complex_double_t *b, int *ib,
                                       int *jb, int *descb, int *info);

typedef void (*fb_psposvx_fortran_fn_t)(const char *fact, const char *uplo,
                                        int *n, int *nrhs, float *a, int *ia,
                                        int *ja, int *desca, float *af,
                                        int *iaf, int *jaf, int *descaf,
                                        char *equed, float *s, float *b,
                                        int *ib, int *jb, int *descb,
                                        float *x, int *ix, int *jx,
                                        int *descx, float *rcond, float *ferr,
                                        float *berr, float *work, int *lwork,
                                        int *iwork, int *liwork, int *info);
typedef void (*fb_pdposvx_fortran_fn_t)(const char *fact, const char *uplo,
                                        int *n, int *nrhs, double *a, int *ia,
                                        int *ja, int *desca, double *af,
                                        int *iaf, int *jaf, int *descaf,
                                        char *equed, double *s, double *b,
                                        int *ib, int *jb, int *descb,
                                        double *x, int *ix, int *jx,
                                        int *descx, double *rcond,
                                        double *ferr, double *berr,
                                        double *work, int *lwork, int *iwork,
                                        int *liwork, int *info);
typedef void (*fb_pcposvx_fortran_fn_t)(const char *fact, const char *uplo,
                                        int *n, int *nrhs,
                                        fb_complex_float_t *a, int *ia,
                                        int *ja, int *desca,
                                        fb_complex_float_t *af, int *iaf,
                                        int *jaf, int *descaf, char *equed,
                                        float *s, fb_complex_float_t *b,
                                        int *ib, int *jb, int *descb,
                                        fb_complex_float_t *x, int *ix,
                                        int *jx, int *descx, float *rcond,
                                        float *ferr, float *berr,
                                        fb_complex_float_t *work, int *lwork,
                                        float *rwork, int *lrwork, int *info);
typedef void (*fb_pzposvx_fortran_fn_t)(const char *fact, const char *uplo,
                                        int *n, int *nrhs,
                                        fb_complex_double_t *a, int *ia,
                                        int *ja, int *desca,
                                        fb_complex_double_t *af, int *iaf,
                                        int *jaf, int *descaf, char *equed,
                                        double *s, fb_complex_double_t *b,
                                        int *ib, int *jb, int *descb,
                                        fb_complex_double_t *x, int *ix,
                                        int *jx, int *descx, double *rcond,
                                        double *ferr, double *berr,
                                        fb_complex_double_t *work, int *lwork,
                                        double *rwork, int *lrwork, int *info);

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

typedef void (*fb_sgbtrs_fortran_fn_t)(const char *trans, const int *n,
                                       const int *kl, const int *ku,
                                       const int *nrhs, const float *ab,
                                       const int *ldab, const int *ipiv,
                                       float *b, const int *ldb, int *info);
typedef void (*fb_dgbtrs_fortran_fn_t)(const char *trans, const int *n,
                                       const int *kl, const int *ku,
                                       const int *nrhs, const double *ab,
                                       const int *ldab, const int *ipiv,
                                       double *b, const int *ldb, int *info);
typedef void (*fb_cgbtrs_fortran_fn_t)(const char *trans, const int *n,
                                       const int *kl, const int *ku,
                                       const int *nrhs,
                                       const fb_complex_float_t *ab,
                                       const int *ldab, const int *ipiv,
                                       fb_complex_float_t *b,
                                       const int *ldb, int *info);
typedef void (*fb_zgbtrs_fortran_fn_t)(const char *trans, const int *n,
                                       const int *kl, const int *ku,
                                       const int *nrhs,
                                       const fb_complex_double_t *ab,
                                       const int *ldab, const int *ipiv,
                                       fb_complex_double_t *b,
                                       const int *ldb, int *info);

typedef void (*fb_sgbsvx_fortran_fn_t)(char *fact, char *trans, int *n,
                                       int *kl, int *ku, int *nrhs,
                                       float *ab, int *ldab, float *afb,
                                       int *ldafb, int *ipiv, char *equed,
                                       float *r, float *c, float *b,
                                       int *ldb, float *x, int *ldx,
                                       float *rcond, float *ferr,
                                       float *berr, float *work,
                                       int *iwork, int *info);
typedef void (*fb_dgbsvx_fortran_fn_t)(char *fact, char *trans, int *n,
                                       int *kl, int *ku, int *nrhs,
                                       double *ab, int *ldab, double *afb,
                                       int *ldafb, int *ipiv, char *equed,
                                       double *r, double *c, double *b,
                                       int *ldb, double *x, int *ldx,
                                       double *rcond, double *ferr,
                                       double *berr, double *work,
                                       int *iwork, int *info);
typedef void (*fb_cgbsvx_fortran_fn_t)(char *fact, char *trans, int *n,
                                       int *kl, int *ku, int *nrhs,
                                       fb_complex_float_t *ab, int *ldab,
                                       fb_complex_float_t *afb, int *ldafb,
                                       int *ipiv, char *equed, float *r,
                                       float *c, fb_complex_float_t *b,
                                       int *ldb, fb_complex_float_t *x,
                                       int *ldx, float *rcond, float *ferr,
                                       float *berr,
                                       fb_complex_float_t *work,
                                       float *rwork, int *info);
typedef void (*fb_zgbsvx_fortran_fn_t)(char *fact, char *trans, int *n,
                                       int *kl, int *ku, int *nrhs,
                                       fb_complex_double_t *ab, int *ldab,
                                       fb_complex_double_t *afb, int *ldafb,
                                       int *ipiv, char *equed, double *r,
                                       double *c, fb_complex_double_t *b,
                                       int *ldb, fb_complex_double_t *x,
                                       int *ldx, double *rcond,
                                       double *ferr, double *berr,
                                       fb_complex_double_t *work,
                                       double *rwork, int *info);

#define FB_DEFINE_GBSV_CALLER(name, cblas_t, fortran_t, scalar_t, pack_row_fn, pack_col_fn, row_to_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *A_dense, int n, int kl, int ku, int nrhs, \
                scalar_t *ab_work, scalar_t *b_work, int *ipiv) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        pack_row_fn(A_dense, n, n, kl, ku, ab_work, n); \
        return ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, n, kl, ku, nrhs, ab_work, n, ipiv, b_work, nrhs); \
    } \
    if (fortran_fn != NULL) { \
        int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs; \
        int ldab_ = band_rows, ldb_ = n, info = 0; \
        scalar_t *b_col = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        if (!b_col) return FB_BAND_CALL_ALLOC_FAILURE; \
        pack_col_fn(A_dense, n, n, kl, ku, ab_work, ldab_); \
        row_to_col_fn(b_work, n, nrhs, b_col); \
        ((fortran_t)fortran_fn)(&n_, &kl_, &ku_, &nrhs_, ab_work, &ldab_, ipiv, b_col, &ldb_, &info); \
        if (info == 0) col_to_row_fn(b_col, n, nrhs, b_work); \
        free(b_col); \
        return info; \
    } \
    return FB_BAND_CALL_NOT_IMPL; \
}

#define FB_DEFINE_GBTRF_CALLER(name, cblas_t, fortran_t, scalar_t, pack_row_fn, pack_col_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *A_dense, int n, int kl, int ku, \
                scalar_t *lu_work, int *ipiv, fb_conv_t *used_conv) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        pack_row_fn(A_dense, n, n, kl, ku, lu_work, n); \
        if (used_conv) *used_conv = FB_CONV_CBLAS; \
        return ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, n, n, kl, ku, lu_work, n, ipiv); \
    } \
    if (fortran_fn != NULL) { \
        int n_ = n, kl_ = kl, ku_ = ku; \
        int ldab_ = band_rows, info = 0; \
        pack_col_fn(A_dense, n, n, kl, ku, lu_work, ldab_); \
        if (used_conv) *used_conv = FB_CONV_FORTRAN; \
        ((fortran_t)fortran_fn)(&n_, &n_, &kl_, &ku_, lu_work, &ldab_, ipiv, &info); \
        return info; \
    } \
    if (used_conv) *used_conv = FB_CONV_COUNT; \
    return FB_BAND_CALL_NOT_IMPL; \
}

#define FB_DEFINE_GBTRS_CALLER(name, cblas_t, fortran_t, scalar_t, row_to_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *lu_in, fb_conv_t lu_conv, int n, int kl, \
                int ku, int nrhs, const int *ipiv, scalar_t *b_work) \
{ \
    int band_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        const scalar_t *lu_arg = lu_in; \
        scalar_t *lu_row = NULL; \
        if (lu_conv == FB_CONV_FORTRAN) { \
            lu_row = (scalar_t *)malloc((size_t)band_rows * (size_t)n * sizeof(scalar_t)); \
            if (!lu_row) return FB_BAND_CALL_ALLOC_FAILURE; \
            col_to_row_fn(lu_in, band_rows, n, lu_row); \
            lu_arg = lu_row; \
        } \
        int info = ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, 'N', n, kl, ku, nrhs, lu_arg, n, ipiv, b_work, nrhs); \
        free(lu_row); \
        return info; \
    } \
    if (fortran_fn != NULL) { \
        const scalar_t *lu_arg = lu_in; \
        scalar_t *lu_col = NULL; \
        scalar_t *b_col = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        if (!b_col) return FB_BAND_CALL_ALLOC_FAILURE; \
        if (lu_conv == FB_CONV_CBLAS) { \
            lu_col = (scalar_t *)malloc((size_t)band_rows * (size_t)n * sizeof(scalar_t)); \
            if (!lu_col) { free(b_col); return FB_BAND_CALL_ALLOC_FAILURE; } \
            row_to_col_fn(lu_in, band_rows, n, lu_col); \
            lu_arg = lu_col; \
        } \
        row_to_col_fn(b_work, n, nrhs, b_col); \
        char trans = 'N'; \
        int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs; \
        int ldab_ = band_rows, ldb_ = n, info = 0; \
        ((fortran_t)fortran_fn)(&trans, &n_, &kl_, &ku_, &nrhs_, lu_arg, &ldab_, ipiv, b_col, &ldb_, &info); \
        if (info == 0) col_to_row_fn(b_col, n, nrhs, b_work); \
        free(lu_col); \
        free(b_col); \
        return info; \
    } \
    return FB_BAND_CALL_NOT_IMPL; \
}

#define FB_DEFINE_GBSVX_REAL_CALLER(name, cblas_t, fortran_t, scalar_t, pack_row_fn, pack_col_fn, row_to_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *A_dense, int n, int kl, int ku, int nrhs, \
                const scalar_t *b_in, scalar_t *x_out) \
{ \
    int compact_rows = kl + ku + 1; \
    int fact_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        scalar_t *ab = (scalar_t *)malloc((size_t)compact_rows * (size_t)n * sizeof(scalar_t)); \
        scalar_t *afb = (scalar_t *)malloc((size_t)fact_rows * (size_t)n * sizeof(scalar_t)); \
        int *ipiv = (int *)malloc((size_t)n * sizeof(int)); \
        scalar_t *r = (scalar_t *)malloc((size_t)n * sizeof(scalar_t)); \
        scalar_t *c = (scalar_t *)malloc((size_t)n * sizeof(scalar_t)); \
        scalar_t *b = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *ferr = (scalar_t *)malloc((size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *berr = (scalar_t *)malloc((size_t)nrhs * sizeof(scalar_t)); \
        scalar_t rpvgrw = 0; \
        scalar_t rcond = 0; \
        char fact = 'N', trans = 'N', equed = 'N'; \
        if (!ab || !afb || !ipiv || !r || !c || !b || !ferr || !berr) { \
            free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(ferr); free(berr); return FB_BAND_CALL_ALLOC_FAILURE; \
        } \
        pack_row_fn(A_dense, n, n, kl, ku, ab, n); \
        memcpy(b, b_in, (size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        int info = ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, fact, trans, n, kl, ku, nrhs, ab, n, afb, n, ipiv, &equed, r, c, b, nrhs, x_out, nrhs, &rcond, ferr, berr, &rpvgrw); \
        free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(ferr); free(berr); \
        return info; \
    } \
    if (fortran_fn != NULL) { \
        scalar_t *ab = (scalar_t *)malloc((size_t)fact_rows * (size_t)n * sizeof(scalar_t)); \
        scalar_t *afb = (scalar_t *)malloc((size_t)fact_rows * (size_t)n * sizeof(scalar_t)); \
        int *ipiv = (int *)malloc((size_t)n * sizeof(int)); \
        scalar_t *r = (scalar_t *)malloc((size_t)n * sizeof(scalar_t)); \
        scalar_t *c = (scalar_t *)malloc((size_t)n * sizeof(scalar_t)); \
        scalar_t *b = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *x = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *ferr = (scalar_t *)malloc((size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *berr = (scalar_t *)malloc((size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *work = (scalar_t *)malloc((size_t)3 * (size_t)n * sizeof(scalar_t)); \
        int *iwork = (int *)malloc((size_t)n * sizeof(int)); \
        scalar_t rcond = 0; \
        if (!ab || !afb || !ipiv || !r || !c || !b || !x || !ferr || !berr || !work || !iwork) { \
            free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(x); free(ferr); free(berr); free(work); free(iwork); return FB_BAND_CALL_ALLOC_FAILURE; \
        } \
        char fact = 'N', trans = 'N', equed = 'N'; \
        int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs; \
        int ldab_ = fact_rows, ldafb_ = fact_rows, ldb_ = n, ldx_ = n, info = 0; \
        pack_col_fn(A_dense, n, n, kl, ku, ab, ldab_); \
        row_to_col_fn(b_in, n, nrhs, b); \
        ((fortran_t)fortran_fn)(&fact, &trans, &n_, &kl_, &ku_, &nrhs_, ab, &ldab_, afb, &ldafb_, ipiv, &equed, r, c, b, &ldb_, x, &ldx_, &rcond, ferr, berr, work, iwork, &info); \
        if (info == 0) col_to_row_fn(x, n, nrhs, x_out); \
        free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(x); free(ferr); free(berr); free(work); free(iwork); \
        return info; \
    } \
    return FB_BAND_CALL_NOT_IMPL; \
}

#define FB_DEFINE_GBSVX_COMPLEX_CALLER(name, cblas_t, fortran_t, scalar_t, real_t, pack_row_fn, pack_col_fn, row_to_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *A_dense, int n, int kl, int ku, int nrhs, \
                const scalar_t *b_in, scalar_t *x_out) \
{ \
    int compact_rows = kl + ku + 1; \
    int fact_rows = 2 * kl + ku + 1; \
    if (cblas_fn != NULL) { \
        scalar_t *ab = (scalar_t *)malloc((size_t)compact_rows * (size_t)n * sizeof(scalar_t)); \
        scalar_t *afb = (scalar_t *)malloc((size_t)fact_rows * (size_t)n * sizeof(scalar_t)); \
        int *ipiv = (int *)malloc((size_t)n * sizeof(int)); \
        real_t *r = (real_t *)malloc((size_t)n * sizeof(real_t)); \
        real_t *c = (real_t *)malloc((size_t)n * sizeof(real_t)); \
        scalar_t *b = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        real_t *ferr = (real_t *)malloc((size_t)nrhs * sizeof(real_t)); \
        real_t *berr = (real_t *)malloc((size_t)nrhs * sizeof(real_t)); \
        real_t rpvgrw = 0; \
        real_t rcond = 0; \
        char fact = 'N', trans = 'N', equed = 'N'; \
        if (!ab || !afb || !ipiv || !r || !c || !b || !ferr || !berr) { \
            free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(ferr); free(berr); return FB_BAND_CALL_ALLOC_FAILURE; \
        } \
        pack_row_fn(A_dense, n, n, kl, ku, ab, n); \
        memcpy(b, b_in, (size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        int info = ((cblas_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, fact, trans, n, kl, ku, nrhs, ab, n, afb, n, ipiv, &equed, r, c, b, nrhs, x_out, nrhs, &rcond, ferr, berr, &rpvgrw); \
        free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(ferr); free(berr); \
        return info; \
    } \
    if (fortran_fn != NULL) { \
        scalar_t *ab = (scalar_t *)malloc((size_t)compact_rows * (size_t)n * sizeof(scalar_t)); \
        scalar_t *afb = (scalar_t *)malloc((size_t)fact_rows * (size_t)n * sizeof(scalar_t)); \
        int *ipiv = (int *)malloc((size_t)n * sizeof(int)); \
        real_t *r = (real_t *)malloc((size_t)n * sizeof(real_t)); \
        real_t *c = (real_t *)malloc((size_t)n * sizeof(real_t)); \
        scalar_t *b = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        scalar_t *x = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        real_t *ferr = (real_t *)malloc((size_t)nrhs * sizeof(real_t)); \
        real_t *berr = (real_t *)malloc((size_t)nrhs * sizeof(real_t)); \
        scalar_t *work = (scalar_t *)malloc((size_t)2 * (size_t)n * sizeof(scalar_t)); \
        real_t *rwork = (real_t *)malloc((size_t)n * sizeof(real_t)); \
        real_t rcond = 0; \
        if (!ab || !afb || !ipiv || !r || !c || !b || !x || !ferr || !berr || !work || !rwork) { \
            free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(x); free(ferr); free(berr); free(work); free(rwork); return FB_BAND_CALL_ALLOC_FAILURE; \
        } \
        char fact = 'N', trans = 'N', equed = 'N'; \
        int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs; \
        int ldab_ = compact_rows, ldafb_ = fact_rows, ldb_ = n, ldx_ = n, info = 0; \
        pack_col_fn(A_dense, n, n, kl, ku, ab, ldab_); \
        row_to_col_fn(b_in, n, nrhs, b); \
        ((fortran_t)fortran_fn)(&fact, &trans, &n_, &kl_, &ku_, &nrhs_, ab, &ldab_, afb, &ldafb_, ipiv, &equed, r, c, b, &ldb_, x, &ldx_, &rcond, ferr, berr, work, rwork, &info); \
        if (info == 0) col_to_row_fn(x, n, nrhs, x_out); \
        free(ab); free(afb); free(ipiv); free(r); free(c); free(b); free(x); free(ferr); free(berr); free(work); free(rwork); \
        return info; \
    } \
    return FB_BAND_CALL_NOT_IMPL; \
}

FB_DEFINE_GBSV_CALLER(fb_call_sgbsv, fb_sgbsv_fn_t, fb_sgbsv_fortran_fn_t, float,
                      fb_pack_band_row_major_f32, fb_pack_band_col_major_f32,
                      fb_copy_row_to_col_f32, fb_copy_col_to_row_f32)
FB_DEFINE_GBSV_CALLER(fb_call_dgbsv, fb_dgbsv_fn_t, fb_dgbsv_fortran_fn_t, double,
                      fb_pack_band_row_major_f64, fb_pack_band_col_major_f64,
                      fb_copy_row_to_col_f64, fb_copy_col_to_row_f64)
FB_DEFINE_GBSV_CALLER(fb_call_cgbsv, fb_cgbsv_fn_t, fb_cgbsv_fortran_fn_t, fb_complex_float_t,
                      fb_pack_band_row_major_cf32, fb_pack_band_col_major_cf32,
                      fb_copy_row_to_col_cf32, fb_copy_col_to_row_cf32)
FB_DEFINE_GBSV_CALLER(fb_call_zgbsv, fb_zgbsv_fn_t, fb_zgbsv_fortran_fn_t, fb_complex_double_t,
                      fb_pack_band_row_major_cf64, fb_pack_band_col_major_cf64,
                      fb_copy_row_to_col_cf64, fb_copy_col_to_row_cf64)

#define FB_DEFINE_PGBSV_CALLER(name, fortran_t, scalar_t, pack_col_fn, row_to_col_fn, col_to_row_fn) \
static int name(fb_generic_fn cblas_fn, fb_generic_fn fortran_fn, \
                const scalar_t *A_dense, int n, int kl, int ku, int nrhs, \
                scalar_t *ab_work, scalar_t *b_work, int *ipiv) \
{ \
    (void)cblas_fn; \
    if (fortran_fn != NULL) { \
        int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs; \
        int iab_ = 1, jab_ = 1, ib_ = 1, jb_ = 1, info = 0; \
        int descab_[9] = {0}; \
        int descb_[9] = {0}; \
        int band_rows = 2 * kl + ku + 1; \
        scalar_t *b_col = (scalar_t *)malloc((size_t)n * (size_t)nrhs * sizeof(scalar_t)); \
        if (!b_col) return FB_BAND_CALL_ALLOC_FAILURE; \
        pack_col_fn(A_dense, n, n, kl, ku, ab_work, band_rows); \
        row_to_col_fn(b_work, n, nrhs, b_col); \
        ((fortran_t)fortran_fn)(&n_, &kl_, &ku_, &nrhs_, ab_work, &iab_, &jab_, descab_, \
                                ipiv, b_col, &ib_, &jb_, descb_, &info); \
        if (info == 0) col_to_row_fn(b_col, n, nrhs, b_work); \
        free(b_col); \
        return info; \
    } \
    return FB_BAND_CALL_NOT_IMPL; \
}

FB_DEFINE_PGBSV_CALLER(fb_call_psgbsv, fb_psgbsv_fortran_fn_t, float,
                       fb_pack_band_col_major_f32,
                       fb_copy_row_to_col_f32,
                       fb_copy_col_to_row_f32)
FB_DEFINE_PGBSV_CALLER(fb_call_pdgbsv, fb_pdgbsv_fortran_fn_t, double,
                       fb_pack_band_col_major_f64,
                       fb_copy_row_to_col_f64,
                       fb_copy_col_to_row_f64)
FB_DEFINE_PGBSV_CALLER(fb_call_pcgbsv, fb_pcgbsv_fortran_fn_t, fb_complex_float_t,
                       fb_pack_band_col_major_cf32,
                       fb_copy_row_to_col_cf32,
                       fb_copy_col_to_row_cf32)
FB_DEFINE_PGBSV_CALLER(fb_call_pzgbsv, fb_pzgbsv_fortran_fn_t, fb_complex_double_t,
                       fb_pack_band_col_major_cf64,
                       fb_copy_row_to_col_cf64,
                       fb_copy_col_to_row_cf64)

FB_DEFINE_GBTRF_CALLER(fb_call_sgbtrf, fb_sgbtrf_fn_t, fb_sgbtrf_fortran_fn_t, float,
                       fb_pack_band_row_major_f32, fb_pack_band_col_major_f32)
FB_DEFINE_GBTRF_CALLER(fb_call_dgbtrf, fb_dgbtrf_fn_t, fb_dgbtrf_fortran_fn_t, double,
                       fb_pack_band_row_major_f64, fb_pack_band_col_major_f64)
FB_DEFINE_GBTRF_CALLER(fb_call_cgbtrf, fb_cgbtrf_fn_t, fb_cgbtrf_fortran_fn_t, fb_complex_float_t,
                       fb_pack_band_row_major_cf32, fb_pack_band_col_major_cf32)
FB_DEFINE_GBTRF_CALLER(fb_call_zgbtrf, fb_zgbtrf_fn_t, fb_zgbtrf_fortran_fn_t, fb_complex_double_t,
                       fb_pack_band_row_major_cf64, fb_pack_band_col_major_cf64)

FB_DEFINE_GBTRS_CALLER(fb_call_sgbtrs, fb_sgbtrs_fn_t, fb_sgbtrs_fortran_fn_t, float,
                       fb_copy_row_to_col_f32, fb_copy_col_to_row_f32)
FB_DEFINE_GBTRS_CALLER(fb_call_dgbtrs, fb_dgbtrs_fn_t, fb_dgbtrs_fortran_fn_t, double,
                       fb_copy_row_to_col_f64, fb_copy_col_to_row_f64)
FB_DEFINE_GBTRS_CALLER(fb_call_cgbtrs, fb_cgbtrs_fn_t, fb_cgbtrs_fortran_fn_t, fb_complex_float_t,
                       fb_copy_row_to_col_cf32, fb_copy_col_to_row_cf32)
FB_DEFINE_GBTRS_CALLER(fb_call_zgbtrs, fb_zgbtrs_fn_t, fb_zgbtrs_fortran_fn_t, fb_complex_double_t,
                       fb_copy_row_to_col_cf64, fb_copy_col_to_row_cf64)

FB_DEFINE_GBSVX_REAL_CALLER(fb_call_sgbsvx, fb_sgbsvx_fn_t, fb_sgbsvx_fortran_fn_t, float,
                            fb_pack_compact_band_row_major_f32, fb_pack_band_col_major_f32,
                            fb_copy_row_to_col_f32, fb_copy_col_to_row_f32)
FB_DEFINE_GBSVX_REAL_CALLER(fb_call_dgbsvx, fb_dgbsvx_fn_t, fb_dgbsvx_fortran_fn_t, double,
                            fb_pack_compact_band_row_major_f64, fb_pack_band_col_major_f64,
                            fb_copy_row_to_col_f64, fb_copy_col_to_row_f64)
FB_DEFINE_GBSVX_COMPLEX_CALLER(fb_call_cgbsvx, fb_cgbsvx_fn_t, fb_cgbsvx_fortran_fn_t, fb_complex_float_t, float,
                               fb_pack_compact_band_row_major_cf32, fb_pack_compact_band_col_major_cf32,
                               fb_copy_row_to_col_cf32, fb_copy_col_to_row_cf32)
FB_DEFINE_GBSVX_COMPLEX_CALLER(fb_call_zgbsvx, fb_zgbsvx_fn_t, fb_zgbsvx_fortran_fn_t, fb_complex_double_t, double,
                               fb_pack_compact_band_row_major_cf64, fb_pack_compact_band_col_major_cf64,
                               fb_copy_row_to_col_cf64, fb_copy_col_to_row_cf64)

#define FB_DEFINE_GBSV_RUNNER(name, op_id, call_fn, scalar_t, build_fn, bwerr_fn) \
static fb_judge_status_t name(\
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,\
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)\
{\
    fb_generic_fn oracle_cblas = oracle->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_fortran = oracle->ext_ops[op_id][FB_CONV_FORTRAN];\
    fb_generic_fn cand_cblas = cand->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn cand_fortran = cand->ext_ops[op_id][FB_CONV_FORTRAN];\
    if ((oracle_cblas == NULL && oracle_fortran == NULL) || (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;\
    int n = (int)tc->n, nrhs = (int)tc->k;\
    int lda = (int)tc->lda;\
    int ldb = nrhs;\
    const scalar_t *A0 = (const scalar_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);\
    const scalar_t *b0 = (const scalar_t *)tc->B;\
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }\
    int kl = fb_band_test_lower_width(n);\
    int ku = fb_band_test_upper_width(n);\
    int band_rows = 2 * kl + ku + 1;\
    size_t dense_Asz = (size_t)n * (size_t)n * sizeof(scalar_t);\
    size_t Absz = (size_t)band_rows * (size_t)n * sizeof(scalar_t);\
    size_t Bsz = (size_t)n * (size_t)nrhs * sizeof(scalar_t);\
    scalar_t *A_band = (scalar_t *)malloc(dense_Asz);\
    scalar_t *Abo = (scalar_t *)malloc(Absz);\
    scalar_t *Bo = (scalar_t *)malloc(Bsz);\
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));\
    if (!A_band || !Abo || !Bo || !ipiv) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); return FB_JUDGE_ERR_ALLOC;\
    }\
    build_fn(A0, A_band, n, lda, kl, ku);\
    memcpy(Bo, b0, Bsz);\
    int info_o = call_fn(oracle_cblas, oracle_fortran, A_band, n, kl, ku, nrhs, Abo, Bo, ipiv);\
    if (info_o == FB_BAND_CALL_ALLOC_FAILURE) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); return FB_JUDGE_ERR_ALLOC;\
    }\
    if (info_o == FB_BAND_CALL_NOT_IMPL) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); return FB_JUDGE_ERR_NOT_IMPL;\
    }\
    if (info_o != 0) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK;\
    }\
    scalar_t *Abc = (scalar_t *)malloc(Absz);\
    scalar_t *Bc = (scalar_t *)malloc(Bsz);\
    int *ipivc = (int *)malloc((size_t)n * sizeof(int));\
    if (!Abc || !Bc || !ipivc) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); return FB_JUDGE_ERR_ALLOC;\
    }\
    if (ns_out) {\
        uint64_t best = UINT64_MAX;\
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {\
            memcpy(Bc, b0, Bsz);\
            int info_w = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Abc, Bc, ipivc);\
            if (info_w == FB_BAND_CALL_ALLOC_FAILURE) {\
                free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); return FB_JUDGE_ERR_ALLOC;\
            }\
        }\
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {\
            memcpy(Bc, b0, Bsz);\
            uint64_t t0 = fb_judge_time_ns();\
            int info_t = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Abc, Bc, ipivc);\
            if (info_t == FB_BAND_CALL_ALLOC_FAILURE) {\
                free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); return FB_JUDGE_ERR_ALLOC;\
            }\
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;\
        }\
        *ns_out = best;\
    }\
    memcpy(Bc, b0, Bsz);\
    int info_c = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Abc, Bc, ipivc);\
    if (info_c == FB_BAND_CALL_ALLOC_FAILURE) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); return FB_JUDGE_ERR_ALLOC;\
    }\
    if (info_c == FB_BAND_CALL_NOT_IMPL) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); return FB_JUDGE_ERR_NOT_IMPL;\
    }\
    if (info_c != 0) {\
        free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc); mark_ca_fatal(res); return FB_JUDGE_OK;\
    }\
    res->residual = make_result(bwerr_fn(A_band, n, n, n, b0, ldb, Bc, ldb, nrhs));\
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};\
    res->kappa_estimate = (double)(n * (kl + ku + 1));\
    free(A_band); free(Abo); free(Bo); free(ipiv); free(Abc); free(Bc); free(ipivc);\
    return FB_JUDGE_OK;\
}

#define FB_DEFINE_GBTRS_RUNNER(name, solve_op_id, solve_call_fn, fact_op_id, fact_call_fn, scalar_t, build_fn, bwerr_fn) \
static fb_judge_status_t name(\
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,\
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)\
{\
    fb_generic_fn oracle_fact_cblas = oracle->ext_ops[fact_op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_fact_fortran = oracle->ext_ops[fact_op_id][FB_CONV_FORTRAN];\
    fb_generic_fn oracle_solve_cblas = oracle->ext_ops[solve_op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_solve_fortran = oracle->ext_ops[solve_op_id][FB_CONV_FORTRAN];\
    fb_generic_fn cand_solve_cblas = cand->ext_ops[solve_op_id][FB_CONV_CBLAS];\
    fb_generic_fn cand_solve_fortran = cand->ext_ops[solve_op_id][FB_CONV_FORTRAN];\
    if ((oracle_fact_cblas == NULL && oracle_fact_fortran == NULL) ||\
        (oracle_solve_cblas == NULL && oracle_solve_fortran == NULL) ||\
        (cand_solve_cblas == NULL && cand_solve_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;\
    int n = (int)tc->n, nrhs = (int)tc->k;\
    int lda = (int)tc->lda;\
    int ldb = nrhs;\
    const scalar_t *A0 = (const scalar_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);\
    const scalar_t *b0 = (const scalar_t *)tc->B;\
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }\
    int kl = fb_band_test_lower_width(n);\
    int ku = fb_band_test_upper_width(n);\
    int band_rows = 2 * kl + ku + 1;\
    size_t dense_Asz = (size_t)n * (size_t)n * sizeof(scalar_t);\
    size_t Absz = (size_t)band_rows * (size_t)n * sizeof(scalar_t);\
    size_t Bsz = (size_t)n * (size_t)nrhs * sizeof(scalar_t);\
    scalar_t *A_band = (scalar_t *)malloc(dense_Asz);\
    scalar_t *LU = (scalar_t *)malloc(Absz);\
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));\
    if (!A_band || !LU || !ipiv) { free(A_band); free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }\
    build_fn(A0, A_band, n, lda, kl, ku);\
    fb_conv_t lu_conv = FB_CONV_COUNT;\
    int fact_info = fact_call_fn(oracle_fact_cblas, oracle_fact_fortran, A_band, n, kl, ku, LU, ipiv, &lu_conv);\
    if (fact_info == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(LU); free(ipiv); return FB_JUDGE_ERR_ALLOC; }\
    if (fact_info == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(LU); free(ipiv); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (fact_info != 0) { free(A_band); free(LU); free(ipiv); mark_oc_fatal(res); return FB_JUDGE_OK; }\
    scalar_t *LUo = (scalar_t *)malloc(Absz);\
    scalar_t *Bo = (scalar_t *)malloc(Bsz);\
    if (!LUo || !Bo) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }\
    memcpy(LUo, LU, Absz); memcpy(Bo, b0, Bsz);\
    int info_o = solve_call_fn(oracle_solve_cblas, oracle_solve_fortran, LUo, lu_conv, n, kl, ku, nrhs, ipiv, Bo);\
    if (info_o == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_ALLOC; }\
    if (info_o == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_o != 0) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }\
    scalar_t *LUc = (scalar_t *)malloc(Absz);\
    scalar_t *Bc = (scalar_t *)malloc(Bsz);\
    if (!LUc || !Bc) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); return FB_JUDGE_ERR_ALLOC; }\
    if (ns_out) {\
        uint64_t best = UINT64_MAX;\
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {\
            memcpy(LUc, LU, Absz); memcpy(Bc, b0, Bsz);\
            int info_w = solve_call_fn(cand_solve_cblas, cand_solve_fortran, LUc, lu_conv, n, kl, ku, nrhs, ipiv, Bc);\
            if (info_w == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); return FB_JUDGE_ERR_ALLOC; }\
        }\
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {\
            memcpy(LUc, LU, Absz); memcpy(Bc, b0, Bsz);\
            uint64_t t0 = fb_judge_time_ns();\
            int info_t = solve_call_fn(cand_solve_cblas, cand_solve_fortran, LUc, lu_conv, n, kl, ku, nrhs, ipiv, Bc);\
            if (info_t == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); return FB_JUDGE_ERR_ALLOC; }\
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;\
        }\
        *ns_out = best;\
    }\
    memcpy(LUc, LU, Absz); memcpy(Bc, b0, Bsz);\
    int info_c = solve_call_fn(cand_solve_cblas, cand_solve_fortran, LUc, lu_conv, n, kl, ku, nrhs, ipiv, Bc);\
    if (info_c == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); return FB_JUDGE_ERR_ALLOC; }\
    if (info_c == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_c != 0) { free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc); mark_ca_fatal(res); return FB_JUDGE_OK; }\
    res->residual = make_result(bwerr_fn(A_band, n, n, n, b0, ldb, Bc, ldb, nrhs));\
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};\
    res->kappa_estimate = (double)(n * (kl + ku + 1));\
    free(A_band); free(LU); free(ipiv); free(LUo); free(Bo); free(LUc); free(Bc);\
    return FB_JUDGE_OK;\
}

#define FB_DEFINE_GBSVX_RUNNER(name, op_id, call_fn, scalar_t, build_fn, rhs_fn, bwerr_fn) \
static fb_judge_status_t name(\
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,\
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)\
{\
    fb_generic_fn oracle_cblas = oracle->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_fortran = oracle->ext_ops[op_id][FB_CONV_FORTRAN];\
    fb_generic_fn cand_cblas = cand->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn cand_fortran = cand->ext_ops[op_id][FB_CONV_FORTRAN];\
    if ((oracle_cblas == NULL && oracle_fortran == NULL) || (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;\
    int n = (int)tc->n, nrhs = (int)tc->k;\
    int lda = (int)tc->lda;\
    int ldb = nrhs;\
    const scalar_t *A0 = (const scalar_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);\
    if (n <= 0 || nrhs <= 0 || !A0) { mark_oc_fatal(res); return FB_JUDGE_OK; }\
    int kl = fb_band_test_lower_width(n);\
    int ku = fb_band_test_upper_width(n);\
    size_t Asz = (size_t)n * (size_t)n * sizeof(scalar_t);\
    size_t Bsz = (size_t)n * (size_t)nrhs * sizeof(scalar_t);\
    size_t Xsz = (size_t)n * (size_t)nrhs * sizeof(scalar_t);\
    scalar_t *A_band = (scalar_t *)malloc(Asz);\
    scalar_t *Btest = (scalar_t *)malloc(Bsz);\
    scalar_t *Xo = (scalar_t *)malloc(Xsz);\
    scalar_t *Xc = (scalar_t *)malloc(Xsz);\
    if (!A_band || !Btest || !Xo || !Xc) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    build_fn(A0, A_band, n, lda, kl, ku);\
    rhs_fn(A_band, n, n, nrhs, Btest);\
    int info_o = call_fn(oracle_cblas, oracle_fortran, A_band, n, kl, ku, nrhs, Btest, Xo);\
    if (info_o == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    if (info_o == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_o != 0) { free(A_band); free(Btest); free(Xo); free(Xc); mark_oc_fatal(res); return FB_JUDGE_OK; }\
    if (ns_out) {\
        uint64_t best = UINT64_MAX;\
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {\
            int info_w = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
            if (info_w == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
        }\
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {\
            uint64_t t0 = fb_judge_time_ns();\
            int info_t = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
            if (info_t == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;\
        }\
        *ns_out = best;\
    }\
    int info_c = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
    if (info_c == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    if (info_c == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_c != 0) { free(A_band); free(Btest); free(Xo); free(Xc); mark_ca_fatal(res); return FB_JUDGE_OK; }\
    res->residual = make_result(bwerr_fn(A_band, n, n, n, Btest, ldb, Xc, ldb, nrhs));\
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};\
    res->kappa_estimate = (double)(n * (kl + ku + 1));\
    free(A_band); free(Btest); free(Xo); free(Xc);\
    return FB_JUDGE_OK;\
}

static fb_complex_float_t fb_make_cf32(float real_part, float imag_part)
{
    fb_complex_float_t value = 0.0f;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

static fb_complex_double_t fb_make_cf64(double real_part, double imag_part)
{
    fb_complex_double_t value = 0.0;
    __real__ value = real_part;
    __imag__ value = imag_part;
    return value;
}

static void fb_build_gbsvx_reference_case_cf32(fb_complex_float_t *A,
                                               fb_complex_float_t *b)
{
    memset(A, 0, (size_t)9 * sizeof(*A));
    A[0] = fb_make_cf32(4.0f, 0.0f);
    A[1] = fb_make_cf32(1.0f, 0.0f);
    A[3] = fb_make_cf32(2.0f, 0.0f);
    A[4] = fb_make_cf32(5.0f, 0.0f);
    A[5] = fb_make_cf32(1.0f, 0.0f);
    A[7] = fb_make_cf32(3.0f, 0.0f);
    A[8] = fb_make_cf32(6.0f, 0.0f);

    b[0] = fb_make_cf32(6.0f, 3.0f);
    b[1] = fb_make_cf32(11.0f, -1.0f);
    b[2] = fb_make_cf32(0.0f, 9.0f);
}

static void fb_build_gbsvx_reference_case_cf64(fb_complex_double_t *A,
                                               fb_complex_double_t *b)
{
    memset(A, 0, (size_t)9 * sizeof(*A));
    A[0] = fb_make_cf64(4.0, 0.0);
    A[1] = fb_make_cf64(1.0, 0.0);
    A[3] = fb_make_cf64(2.0, 0.0);
    A[4] = fb_make_cf64(5.0, 0.0);
    A[5] = fb_make_cf64(1.0, 0.0);
    A[7] = fb_make_cf64(3.0, 0.0);
    A[8] = fb_make_cf64(6.0, 0.0);

    b[0] = fb_make_cf64(6.0, 3.0);
    b[1] = fb_make_cf64(11.0, -1.0);
    b[2] = fb_make_cf64(0.0, 9.0);
}

static int fb_call_cgbsvx_reference_case(fb_generic_fn cblas_fn,
                                         fb_generic_fn fortran_fn,
                                         const fb_complex_float_t *A_dense,
                                         int n,
                                         int kl,
                                         int ku,
                                         int nrhs,
                                         const fb_complex_float_t *b_in,
                                         fb_complex_float_t *x_out)
{
    if (fortran_fn != NULL) {
        int ldab = kl + ku + 1;
        int ldafb = 2 * kl + ku + 1;
        int ldb = n;
        int ldx = n;
        int ipiv[3] = {0, 0, 0};
        float r_vec[3] = {0.0f, 0.0f, 0.0f};
        float c_vec[3] = {0.0f, 0.0f, 0.0f};
        float rcond = 0.0f;
        float ferr[1] = {0.0f};
        float berr[1] = {0.0f};
        float rwork[3] = {0.0f, 0.0f, 0.0f};
        fb_complex_float_t ab[9];
        fb_complex_float_t afb[12];
        fb_complex_float_t b[3];
        fb_complex_float_t x[3];
        fb_complex_float_t work[6];
        char fact = 'N';
        char trans = 'N';
        char equed = '?';
        int info = 0;

        memset(afb, 0, sizeof(afb));
        memset(x, 0, sizeof(x));
        memset(work, 0, sizeof(work));
        fb_pack_compact_band_col_major_cf32(A_dense, n, n, kl, ku, ab, ldab);
        memcpy(b, b_in, sizeof(b));

        ((fb_cgbsvx_fortran_fn_t)fortran_fn)(&fact, &trans, &n, &kl, &ku,
                                             &nrhs, ab, &ldab, afb, &ldafb,
                                             ipiv, &equed, r_vec, c_vec, b,
                                             &ldb, x, &ldx, &rcond, ferr,
                                             berr, work, rwork, &info);
        if (info == 0) {
            memcpy(x_out, x, sizeof(x));
        }
        return info;
    }

    return fb_call_cgbsvx(cblas_fn, NULL, A_dense, n, kl, ku, nrhs, b_in,
                          x_out);
}

static int fb_call_zgbsvx_reference_case(fb_generic_fn cblas_fn,
                                         fb_generic_fn fortran_fn,
                                         const fb_complex_double_t *A_dense,
                                         int n,
                                         int kl,
                                         int ku,
                                         int nrhs,
                                         const fb_complex_double_t *b_in,
                                         fb_complex_double_t *x_out)
{
    if (fortran_fn != NULL) {
        int ldab = kl + ku + 1;
        int ldafb = 2 * kl + ku + 1;
        int ldb = n;
        int ldx = n;
        int ipiv[3] = {0, 0, 0};
        double r_vec[3] = {0.0, 0.0, 0.0};
        double c_vec[3] = {0.0, 0.0, 0.0};
        double rcond = 0.0;
        double ferr[1] = {0.0};
        double berr[1] = {0.0};
        double rwork[3] = {0.0, 0.0, 0.0};
        fb_complex_double_t ab[9];
        fb_complex_double_t afb[12];
        fb_complex_double_t b[3];
        fb_complex_double_t x[3];
        fb_complex_double_t work[6];
        char fact = 'N';
        char trans = 'N';
        char equed = '?';
        int info = 0;

        memset(afb, 0, sizeof(afb));
        memset(x, 0, sizeof(x));
        memset(work, 0, sizeof(work));
        fb_pack_compact_band_col_major_cf64(A_dense, n, n, kl, ku, ab, ldab);
        memcpy(b, b_in, sizeof(b));

        ((fb_zgbsvx_fortran_fn_t)fortran_fn)(&fact, &trans, &n, &kl, &ku,
                                             &nrhs, ab, &ldab, afb, &ldafb,
                                             ipiv, &equed, r_vec, c_vec, b,
                                             &ldb, x, &ldx, &rcond, ferr,
                                             berr, work, rwork, &info);
        if (info == 0) {
            memcpy(x_out, x, sizeof(x));
        }
        return info;
    }

    return fb_call_zgbsvx(cblas_fn, NULL, A_dense, n, kl, ku, nrhs, b_in,
                          x_out);
}

#define FB_DEFINE_GBSVX_REFERENCE_COMPLEX_RUNNER(name, op_id, call_fn, scalar_t, build_case_fn, bwerr_fn) \
static fb_judge_status_t name(\
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,\
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)\
{\
    (void)tc;\
    fb_generic_fn oracle_cblas = oracle->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn oracle_fortran = oracle->ext_ops[op_id][FB_CONV_FORTRAN];\
    fb_generic_fn cand_cblas = cand->ext_ops[op_id][FB_CONV_CBLAS];\
    fb_generic_fn cand_fortran = cand->ext_ops[op_id][FB_CONV_FORTRAN];\
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||\
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;\
    const int n = 3, nrhs = 1, kl = 1, ku = 1, ldb = 1;\
    size_t Asz = (size_t)9 * sizeof(scalar_t);\
    size_t Bsz = (size_t)3 * sizeof(scalar_t);\
    scalar_t *A_band = (scalar_t *)malloc(Asz);\
    scalar_t *Btest = (scalar_t *)malloc(Bsz);\
    scalar_t *Xo = (scalar_t *)malloc(Bsz);\
    scalar_t *Xc = (scalar_t *)malloc(Bsz);\
    if (!A_band || !Btest || !Xo || !Xc) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    build_case_fn(A_band, Btest);\
    int info_o = call_fn(oracle_cblas, oracle_fortran, A_band, n, kl, ku, nrhs, Btest, Xo);\
    if (info_o == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    if (info_o == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_o != 0) { free(A_band); free(Btest); free(Xo); free(Xc); mark_oc_fatal(res); return FB_JUDGE_OK; }\
    if (ns_out) {\
        uint64_t best = UINT64_MAX;\
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {\
            int info_w = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
            if (info_w == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
        }\
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {\
            uint64_t t0 = fb_judge_time_ns();\
            int info_t = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
            if (info_t == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;\
        }\
        *ns_out = best;\
    }\
    int info_c = call_fn(cand_cblas, cand_fortran, A_band, n, kl, ku, nrhs, Btest, Xc);\
    if (info_c == FB_BAND_CALL_ALLOC_FAILURE) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_ALLOC; }\
    if (info_c == FB_BAND_CALL_NOT_IMPL) { free(A_band); free(Btest); free(Xo); free(Xc); return FB_JUDGE_ERR_NOT_IMPL; }\
    if (info_c != 0) { free(A_band); free(Btest); free(Xo); free(Xc); mark_ca_fatal(res); return FB_JUDGE_OK; }\
    res->residual = make_result(bwerr_fn(A_band, n, n, n, Btest, ldb, Xc, ldb, nrhs));\
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};\
    res->kappa_estimate = 9.0;\
    free(A_band); free(Btest); free(Xo); free(Xc);\
    return FB_JUDGE_OK;\
}

static void mark_oc_fatal(fb_judge_solve_result_t *r)
    { memset(r, 0, sizeof(*r)); r->residual.is_oracle_fatal = true; }
static void mark_ca_fatal(fb_judge_solve_result_t *r)
    { memset(r, 0, sizeof(*r)); r->residual.is_fatal = true; }

FB_DEFINE_GBSV_RUNNER(run_sgbsv, FB_OP_SGBSV, fb_call_sgbsv, float, fb_build_band_dense_f32, bwerr_f32)
FB_DEFINE_GBSV_RUNNER(run_dgbsv, FB_OP_DGBSV, fb_call_dgbsv, double, fb_build_band_dense_f64, bwerr_f64)
FB_DEFINE_GBSV_RUNNER(run_cgbsv, FB_OP_CGBSV, fb_call_cgbsv, fb_complex_float_t, fb_build_band_dense_cf32, bwerr_cf32)
FB_DEFINE_GBSV_RUNNER(run_zgbsv, FB_OP_ZGBSV, fb_call_zgbsv, fb_complex_double_t, fb_build_band_dense_cf64, bwerr_cf64)
FB_DEFINE_GBSV_RUNNER(run_psgbsv, FB_OP_PSGBSV, fb_call_psgbsv, float, fb_build_band_dense_f32, bwerr_f32)
FB_DEFINE_GBSV_RUNNER(run_pdgbsv, FB_OP_PDGBSV, fb_call_pdgbsv, double, fb_build_band_dense_f64, bwerr_f64)
FB_DEFINE_GBSV_RUNNER(run_pcgbsv, FB_OP_PCGBSV, fb_call_pcgbsv, fb_complex_float_t, fb_build_band_dense_cf32, bwerr_cf32)
FB_DEFINE_GBSV_RUNNER(run_pzgbsv, FB_OP_PZGBSV, fb_call_pzgbsv, fb_complex_double_t, fb_build_band_dense_cf64, bwerr_cf64)

FB_DEFINE_GBSVX_RUNNER(run_sgbsvx, FB_OP_SGBSVX, fb_call_sgbsvx, float, fb_build_gbsvx_dense_f32, fb_build_rhs_f32, bwerr_f32)
FB_DEFINE_GBSVX_RUNNER(run_dgbsvx, FB_OP_DGBSVX, fb_call_dgbsvx, double, fb_build_gbsvx_dense_f64, fb_build_rhs_f64, bwerr_f64)
FB_DEFINE_GBSVX_REFERENCE_COMPLEX_RUNNER(run_cgbsvx, FB_OP_CGBSVX, fb_call_cgbsvx_reference_case, fb_complex_float_t, fb_build_gbsvx_reference_case_cf32, bwerr_cf32)
FB_DEFINE_GBSVX_REFERENCE_COMPLEX_RUNNER(run_zgbsvx, FB_OP_ZGBSVX, fb_call_zgbsvx_reference_case, fb_complex_double_t, fb_build_gbsvx_reference_case_cf64, bwerr_cf64)

FB_DEFINE_GBTRS_RUNNER(run_sgbtrs, FB_OP_SGBTRS, fb_call_sgbtrs, FB_OP_SGBTRF, fb_call_sgbtrf, float, fb_build_band_dense_f32, bwerr_f32)
FB_DEFINE_GBTRS_RUNNER(run_dgbtrs, FB_OP_DGBTRS, fb_call_dgbtrs, FB_OP_DGBTRF, fb_call_dgbtrf, double, fb_build_band_dense_f64, bwerr_f64)
FB_DEFINE_GBTRS_RUNNER(run_cgbtrs, FB_OP_CGBTRS, fb_call_cgbtrs, FB_OP_CGBTRF, fb_call_cgbtrf, fb_complex_float_t, fb_build_band_dense_cf32, bwerr_cf32)
FB_DEFINE_GBTRS_RUNNER(run_zgbtrs, FB_OP_ZGBTRS, fb_call_zgbtrs, FB_OP_ZGBTRF, fb_call_zgbtrf, fb_complex_double_t, fb_build_band_dense_cf64, bwerr_cf64)

/* Base GESV runners are generated later from the Fortran ext_ops helpers. */

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
static int fb_call_sgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *b_out);

static int fb_call_dgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *b_out);

static int fb_call_cgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    fb_complex_float_t *b_out);

static int fb_call_zgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    fb_complex_double_t *b_out);

static fb_judge_status_t run_sgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->sgels ? (fb_generic_fn)oracle->sgels : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SGELS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->sgels ? (fb_generic_fn)cand->sgels : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SGELS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *Bo = (float*)malloc(Bsz);
    if (!Bo) return FB_JUDGE_ERR_ALLOC;
    info = fb_call_sgels_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    float *Bc = (float*)malloc(Bsz);
    if (!Bc) { free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_sgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_sgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_sgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    /* For overdetermined (m>=n): solution in Bc[0:n,:]; compare ‖b−Ax‖ over m rows. */
    res->residual = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)((m > n ? m : n) * 10);
    free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->dgels ? (fb_generic_fn)oracle->dgels : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DGELS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->dgels ? (fb_generic_fn)cand->dgels : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DGELS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *Bo = (double*)malloc(Bsz);
    if (!Bo) return FB_JUDGE_ERR_ALLOC;
    info = fb_call_dgels_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    double *Bc = (double*)malloc(Bsz);
    if (!Bc) { free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_dgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_dgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)((m > n ? m : n) * 10);
    free(Bc); free(Bo);
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
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij_re = (double)__real__(b[i*ldb + j]);
            double rij_im = (double)__imag__(b[i*ldb + j]);
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            for (int k = 0; k < n; k++) {
                double ar = (double)__real__(A[i*lda + k]), ai = (double)__imag__(A[i*lda + k]);
                double xr = (double)__real__(x[k*ldx + j]), xi = (double)__imag__(x[k*ldx + j]);
                if (!isfinite(ar) || !isfinite(ai) || !isfinite(xr) || !isfinite(xi)) return NAN;
                rij_re -= ar*xr - ai*xi;
                rij_im -= ar*xi + ai*xr;
            }
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            fb_scaled_sumsq_add(&r_acc, hypot(rij_re, rij_im));
            fb_scaled_sumsq_add(&b_acc, hypot((double)__real__(b[i*ldb+j]), (double)__imag__(b[i*ldb+j])));
        }
        for (int i = 0; i < n; i++) {
            double xr = (double)__real__(x[i*ldx+j]), xi = (double)__imag__(x[i*ldx+j]);
            if (!isfinite(xr) || !isfinite(xi)) return NAN;
            fb_scaled_sumsq_add(&x_acc, hypot(xr, xi));
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            double ar = (double)__real__(A[i*lda+k]), ai = (double)__imag__(A[i*lda+k]);
            if (!isfinite(ar) || !isfinite(ai)) return NAN;
            fb_scaled_sumsq_add(&a_acc, hypot(ar, ai));
        }
    double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                            fb_scaled_sumsq_norm(&x_acc),
                                            fb_scaled_sumsq_norm(&b_acc));
    if (denom < (double)FLT_EPSILON)
        return (fb_scaled_sumsq_norm(&r_acc) < (double)FLT_EPSILON*(double)FLT_EPSILON) ? 0.0 : 1.0;
    if (isinf(denom)) return 0.0;
    return fb_scaled_sumsq_norm(&r_acc) / denom;
}

static double bwerr_cf64(const fb_complex_double_t *A, int m, int n, int lda,
                          const fb_complex_double_t *b, int ldb,
                          const fb_complex_double_t *x, int ldx, int nrhs)
{
    fb_scaled_sumsq_t r_acc, a_acc, x_acc, b_acc;
    fb_scaled_sumsq_init(&r_acc);
    fb_scaled_sumsq_init(&a_acc);
    fb_scaled_sumsq_init(&x_acc);
    fb_scaled_sumsq_init(&b_acc);

    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double rij_re = __real__(b[i*ldb + j]), rij_im = __imag__(b[i*ldb + j]);
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            for (int k = 0; k < n; k++) {
                double ar = __real__(A[i*lda+k]), ai = __imag__(A[i*lda+k]);
                double xr = __real__(x[k*ldx+j]), xi = __imag__(x[k*ldx+j]);
                if (!isfinite(ar) || !isfinite(ai) || !isfinite(xr) || !isfinite(xi)) return NAN;
                rij_re -= ar*xr - ai*xi;
                rij_im -= ar*xi + ai*xr;
            }
            if (!isfinite(rij_re) || !isfinite(rij_im)) return NAN;
            fb_scaled_sumsq_add(&r_acc, hypot(rij_re, rij_im));
            fb_scaled_sumsq_add(&b_acc, hypot(__real__(b[i*ldb+j]), __imag__(b[i*ldb+j])));
        }
        for (int i = 0; i < n; i++) {
            double xr = __real__(x[i*ldx+j]);
            double xi = __imag__(x[i*ldx+j]);
            if (!isfinite(xr) || !isfinite(xi)) return NAN;
            fb_scaled_sumsq_add(&x_acc, hypot(xr, xi));
        }
    }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            double ar = __real__(A[i*lda+k]);
            double ai = __imag__(A[i*lda+k]);
            if (!isfinite(ar) || !isfinite(ai)) return NAN;
            fb_scaled_sumsq_add(&a_acc, hypot(ar, ai));
        }
    double denom = fb_safe_positive_mul_add(fb_scaled_sumsq_norm(&a_acc),
                                            fb_scaled_sumsq_norm(&x_acc),
                                            fb_scaled_sumsq_norm(&b_acc));
    if (denom < DBL_EPSILON) return (fb_scaled_sumsq_norm(&r_acc) < DBL_EPSILON*DBL_EPSILON) ? 0.0 : 1.0;
    if (isinf(denom)) return 0.0;
    return fb_scaled_sumsq_norm(&r_acc) / denom;
}

/* Base complex GESV runners are generated later from the Fortran ext_ops helpers. */

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
    fb_generic_fn oracle_cblas = oracle->cgels ? (fb_generic_fn)oracle->cgels : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CGELS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->cgels ? (fb_generic_fn)cand->cgels : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CGELS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Bo || !Bc) { free(Bo); free(Bc); return FB_JUDGE_ERR_ALLOC; }
    info = fb_call_cgels_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Bo);
        free(Bc);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_cgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_cgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_cgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgels(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->zgels ? (fb_generic_fn)oracle->zgels : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZGELS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->zgels ? (fb_generic_fn)cand->zgels : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZGELS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m*(size_t)lda*sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Bo || !Bc) { free(Bo); free(Bc); return FB_JUDGE_ERR_ALLOC; }
    info = fb_call_zgels_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Bo);
        free(Bc);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_zgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                         A0, lda, b0, ldb, Asz, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0; if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_zgels_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                 A0, lda, b0, ldb, Asz, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(Bc);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bc); free(Bo);
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
typedef void (*fb_sgelsd_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       float *a, int *lda,
                                       float *b, int *ldb,
                                       float *s, float *rcond, int *rank,
                                       float *work, int *lwork,
                                       int *iwork, int *info);
typedef void (*fb_dgelsd_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       double *a, int *lda,
                                       double *b, int *ldb,
                                       double *s, double *rcond, int *rank,
                                       double *work, int *lwork,
                                       int *iwork, int *info);
typedef void (*fb_cgelsd_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *lda,
                                       fb_complex_float_t *b, int *ldb,
                                       float *s, float *rcond, int *rank,
                                       fb_complex_float_t *work,
                                       int *lwork, float *rwork,
                                       int *iwork, int *info);
typedef void (*fb_zgelsd_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *lda,
                                       fb_complex_double_t *b, int *ldb,
                                       double *s, double *rcond, int *rank,
                                       fb_complex_double_t *work,
                                       int *lwork, double *rwork,
                                       int *iwork, int *info);

static int fb_call_sgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    float *b_out);
static int fb_call_dgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    double *b_out);
static int fb_call_cgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    fb_complex_float_t *b_out);
static int fb_call_zgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    fb_complex_double_t *b_out);

/* =========================================================================
 * SGELSD — minimum-norm least squares via SVD (divide and conquer)
 * ========================================================================= */
static fb_judge_status_t run_sgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SGELSD][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SGELSD][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);

    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;

    float *Bo = (float *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_sgelsd_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);

    float *Bc = (float *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_sgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_sgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_sgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SGELSY — minimum-norm least squares via complete orthogonal factorization
 * ========================================================================= */
typedef void (*fb_sgelsy_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       float *a, int *lda,
                                       float *b, int *ldb,
                                       int *jpvt, float *rcond, int *rank,
                                       float *work, int *lwork, int *info);

typedef void (*fb_dgelsy_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       double *a, int *lda,
                                       double *b, int *ldb,
                                       int *jpvt, double *rcond, int *rank,
                                       double *work, int *lwork, int *info);

typedef void (*fb_sgels_fortran_fn_t)(const char *trans,
                                      int *m, int *n, int *nrhs,
                                      float *a, int *lda,
                                      float *b, int *ldb,
                                      float *work, int *lwork, int *info);

typedef void (*fb_dgels_fortran_fn_t)(const char *trans,
                                      int *m, int *n, int *nrhs,
                                      double *a, int *lda,
                                      double *b, int *ldb,
                                      double *work, int *lwork, int *info);

typedef void (*fb_cgels_fortran_fn_t)(const char *trans,
                                      int *m, int *n, int *nrhs,
                                      fb_complex_float_t *a, int *lda,
                                      fb_complex_float_t *b, int *ldb,
                                      fb_complex_float_t *work,
                                      int *lwork, int *info);

typedef void (*fb_zgels_fortran_fn_t)(const char *trans,
                                      int *m, int *n, int *nrhs,
                                      fb_complex_double_t *a, int *lda,
                                      fb_complex_double_t *b, int *ldb,
                                      fb_complex_double_t *work,
                                      int *lwork, int *info);

static int fb_solve_query_size_from_float(float query_size)
{
    return (query_size > 1.0f) ? (int)ceilf(query_size) : 1;
}

static int fb_solve_query_size_from_double(double query_size)
{
    return (query_size > 1.0) ? (int)ceil(query_size) : 1;
}

static int fb_solve_query_size_from_cfloat(fb_complex_float_t query_size)
{
    float real_part = (float)__real__ query_size;
    return (real_part > 1.0f) ? (int)ceilf(real_part) : 1;
}

static int fb_solve_query_size_from_cdouble(fb_complex_double_t query_size)
{
    double real_part = (double)__real__ query_size;
    return (real_part > 1.0) ? (int)ceil(real_part) : 1;
}

/* Some complex least-squares backends only partially populate workspace
 * queries. Keep judge-side fallbacks conservative so successful queries are
 * not required for correct residual measurement. */
static int fb_gelsd_nlvl(int minmn, int smlsiz)
{
    int nlvl = 0;
    int span = minmn;

    if (span < 1) {
        return 0;
    }
    if (smlsiz < 1) {
        smlsiz = 1;
    }

    while (span > smlsiz + 1) {
        span = (span + 1) / 2;
        ++nlvl;
    }

    return nlvl;
}

static int fb_gelss_complex_rwork_size(int m, int n)
{
    int minmn = (m < n) ? m : n;

    if (minmn < 1) {
        return 1;
    }

    return 5 * minmn;
}

[[maybe_unused]] static int fb_gelsd_complex_rwork_size(int m, int n, int nrhs)
{
    const int smlsiz = 25;
    int minmn = (m < n) ? m : n;
    int nlvl;
    int square_term;
    int rhs_term;
    int max_term;

    if (minmn < 1) {
        return 1;
    }

    nlvl = fb_gelsd_nlvl(minmn, smlsiz);
    square_term = (smlsiz + 1) * (smlsiz + 1);
    rhs_term = n * (nrhs + 1) + nrhs * 2;
    max_term = (square_term > rhs_term) ? square_term : rhs_term;

    return minmn * 10 + (minmn << 1) * smlsiz + (minmn << 3) * nlvl +
           smlsiz * 3 * nrhs + max_term;
}

static int fb_gelsd_iwork_size(int m, int n)
{
    int minmn = (m < n) ? m : n;
    int nlvl = 0;
    int span;

    if (minmn < 1) {
        return 1;
    }

    span = minmn;
    while (span > 1) {
        span = (span + 1) / 2;
        ++nlvl;
    }

    return minmn * (3 * nlvl + 11);
}

static int fb_call_sgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    float *b_out)
{
    if (cblas_fn != NULL) {
        float *a_tmp = (float *)malloc(a_bytes);
        float *b_tmp = (float *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_sgelsd_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        s_out, -1.0f, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(float)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        int iwork_query = 0;
        float rcond = -1.0f;
        float work_query = 0.0f;
        float *a_col = (float *)calloc((size_t)lda_col * (size_t)n, sizeof(float));
        float *b_col = (float *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(float));
        float *work = NULL;
        int *iwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             &iwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_float(work_query);
        int iwork_len = fb_gelsd_iwork_size(m, n);
        work = (float *)calloc((size_t)lwork, sizeof(float));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
        if (!work || !iwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(iwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, iwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(iwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_dgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    double *b_out)
{
    if (cblas_fn != NULL) {
        double *a_tmp = (double *)malloc(a_bytes);
        double *b_tmp = (double *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_dgelsd_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        s_out, -1.0, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(double)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        int iwork_query = 0;
        double rcond = -1.0;
        double work_query = 0.0;
        double *a_col = (double *)calloc((size_t)lda_col * (size_t)n, sizeof(double));
        double *b_col = (double *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(double));
        double *work = NULL;
        int *iwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             &iwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_double(work_query);
        int iwork_len = fb_gelsd_iwork_size(m, n);
        work = (double *)calloc((size_t)lwork, sizeof(double));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
        if (!work || !iwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(iwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, iwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(iwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_cgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    fb_complex_float_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_float_t *a_tmp = (fb_complex_float_t *)malloc(a_bytes);
        fb_complex_float_t *b_tmp = (fb_complex_float_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_cgelsd_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        s_out, -1.0f, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_float_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float rcond = -1.0f;
        float rwork_query[1] = {0.0f};
        fb_complex_float_t work_query = 0.0f;
        int iwork_query[1] = {0};
        fb_complex_float_t *a_query =
            (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *b_query =
            (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_float_t));
        fb_complex_float_t *a_work =
            (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *b_work =
            (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_float_t));
        fb_complex_float_t *work = NULL;
        float *rwork = NULL;
        int *iwork = NULL;

        if (!a_query || !b_query || !a_work || !b_work) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                fb_complex_float_t value = a_in[(size_t)i * (size_t)lda_in + (size_t)j];
                a_query[(size_t)i + (size_t)j * (size_t)lda_col] = value;
                a_work[(size_t)i + (size_t)j * (size_t)lda_col] = value;
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                fb_complex_float_t value = b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
                b_query[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
                b_work[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
            }
        }

        ((fb_cgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_query,
                                             &lda_col, b_query, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             rwork_query, iwork_query,
                                             &info);
        if (info != 0) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            return info;
        }

        lwork = fb_solve_query_size_from_cfloat(work_query);
        int rwork_len = fb_solve_query_size_from_float(rwork_query[0]);
        int iwork_len = (iwork_query[0] > 0) ? iwork_query[0] : 1;
        work = (fb_complex_float_t *)calloc((size_t)lwork, sizeof(fb_complex_float_t));
        rwork = (float *)calloc((size_t)rwork_len, sizeof(float));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
        if (!work || !rwork || !iwork) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            free(work);
            free(rwork);
            free(iwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_cgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_work,
                                             &lda_col, b_work, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, rwork, iwork,
                                             &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_work[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_query);
        free(b_query);
        free(a_work);
        free(b_work);
        free(work);
        free(rwork);
        free(iwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_zgelsd_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    fb_complex_double_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_double_t *a_tmp = (fb_complex_double_t *)malloc(a_bytes);
        fb_complex_double_t *b_tmp = (fb_complex_double_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_zgelsd_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        s_out, -1.0, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_double_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double rcond = -1.0;
        double rwork_query[1] = {0.0};
        fb_complex_double_t work_query = 0.0;
        int iwork_query[1] = {0};
        fb_complex_double_t *a_query =
            (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *b_query =
            (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_double_t));
        fb_complex_double_t *a_work =
            (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *b_work =
            (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_double_t));
        fb_complex_double_t *work = NULL;
        double *rwork = NULL;
        int *iwork = NULL;

        if (!a_query || !b_query || !a_work || !b_work) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                fb_complex_double_t value = a_in[(size_t)i * (size_t)lda_in + (size_t)j];
                a_query[(size_t)i + (size_t)j * (size_t)lda_col] = value;
                a_work[(size_t)i + (size_t)j * (size_t)lda_col] = value;
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                fb_complex_double_t value = b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
                b_query[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
                b_work[(size_t)i + (size_t)j * (size_t)ldb_col] = value;
            }
        }

        ((fb_zgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_query,
                                             &lda_col, b_query, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             rwork_query, iwork_query,
                                             &info);
        if (info != 0) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            return info;
        }

        lwork = fb_solve_query_size_from_cdouble(work_query);
        int rwork_len = fb_solve_query_size_from_double(rwork_query[0]);
        int iwork_len = (iwork_query[0] > 0) ? iwork_query[0] : 1;
        work = (fb_complex_double_t *)calloc((size_t)lwork, sizeof(fb_complex_double_t));
        rwork = (double *)calloc((size_t)rwork_len, sizeof(double));
        iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
        if (!work || !rwork || !iwork) {
            free(a_query);
            free(b_query);
            free(a_work);
            free(b_work);
            free(work);
            free(rwork);
            free(iwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_zgelsd_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_work,
                                             &lda_col, b_work, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, rwork, iwork,
                                             &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_work[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_query);
        free(b_query);
        free(a_work);
        free(b_work);
        free(work);
        free(rwork);
        free(iwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_sgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *b_out)
{
    if (cblas_fn != NULL) {
        float *a_tmp = (float *)malloc(a_bytes);
        float *b_tmp = (float *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_sgels_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS,
                                       m, n, nrhs, a_tmp, lda_in, b_tmp,
                                       ldb_in);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        char trans = 'N';
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(float)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float work_query = 0.0f;
        float *a_col = (float *)calloc((size_t)lda_col * (size_t)n, sizeof(float));
        float *b_col = (float *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(float));
        float *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_float(work_query);
        work = (float *)malloc((size_t)lwork * sizeof(float));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_sgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_dgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *b_out)
{
    if (cblas_fn != NULL) {
        double *a_tmp = (double *)malloc(a_bytes);
        double *b_tmp = (double *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_dgels_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS,
                                       m, n, nrhs, a_tmp, lda_in, b_tmp,
                                       ldb_in);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        char trans = 'N';
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(double)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double work_query = 0.0;
        double *a_col = (double *)calloc((size_t)lda_col * (size_t)n, sizeof(double));
        double *b_col = (double *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(double));
        double *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_double(work_query);
        work = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_dgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_cgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    fb_complex_float_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_float_t *a_tmp = (fb_complex_float_t *)malloc(a_bytes);
        fb_complex_float_t *b_tmp = (fb_complex_float_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_cgels_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS,
                                       m, n, nrhs, a_tmp, lda_in, b_tmp,
                                       ldb_in);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        char trans = 'N';
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_float_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *a_col =
            (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *b_col =
            (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_float_t));
        fb_complex_float_t *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_cgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cfloat(work_query);
        work = (fb_complex_float_t *)malloc((size_t)lwork * sizeof(fb_complex_float_t));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_cgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_zgels_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    fb_complex_double_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_double_t *a_tmp = (fb_complex_double_t *)malloc(a_bytes);
        fb_complex_double_t *b_tmp = (fb_complex_double_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_zgels_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS,
                                       m, n, nrhs, a_tmp, lda_in, b_tmp,
                                       ldb_in);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        char trans = 'N';
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_double_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *a_col =
            (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *b_col =
            (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_double_t));
        fb_complex_double_t *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_zgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cdouble(work_query);
        work = (fb_complex_double_t *)malloc((size_t)lwork * sizeof(fb_complex_double_t));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_zgels_fortran_fn_t)fortran_fn)(&trans, &m_, &n_, &nrhs_, a_col,
                                            &lda_col, b_col, &ldb_col,
                                            work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_sgelsy_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    int *jpvt, int *rank_out,
    float *b_out)
{
    if (cblas_fn != NULL) {
        float *a_tmp = (float *)malloc(a_bytes);
        float *b_tmp = (float *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_sgelsy_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        jpvt, -1.0f, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(float)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float rcond = -1.0f;
        float work_query = 0.0f;
        float *a_col = (float *)calloc((size_t)lda_col * (size_t)n, sizeof(float));
        float *b_col = (float *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(float));
        float *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_float(work_query);
        work = (float *)malloc((size_t)lwork * sizeof(float));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_sgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_dgelsy_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    int *jpvt, int *rank_out,
    double *b_out)
{
    if (cblas_fn != NULL) {
        double *a_tmp = (double *)malloc(a_bytes);
        double *b_tmp = (double *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_dgelsy_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        jpvt, -1.0, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(double)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double rcond = -1.0;
        double work_query = 0.0;
        double *a_col = (double *)calloc((size_t)lda_col * (size_t)n, sizeof(double));
        double *b_col = (double *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(double));
        double *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_double(work_query);
        work = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_dgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static fb_judge_status_t run_sgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->sgelsy ? (fb_generic_fn)oracle->sgelsy : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SGELSY][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->sgelsy ? (fb_generic_fn)cand->sgelsy : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SGELSY][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    float *Bo = (float *)malloc(Bsz);
    if (!Bo || !jpvt) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_sgelsy_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    float *Bc = (float *)malloc(Bsz);
    if (!Bc) { free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            info = fb_call_sgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_sgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_sgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bo); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}


/* =========================================================================
 * DGELSD — double precision min-norm least squares via divide-and-conquer SVD
 * ========================================================================= */
static fb_judge_status_t run_dgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DGELSD][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DGELSD][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    double *Bo = (double *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_dgelsd_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);
    double *Bc = (double *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_dgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_dgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGELSD — complex single precision min-norm least squares via SVD
 * ========================================================================= */
static fb_judge_status_t run_cgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    bool same_backend = (oracle == cand);
    fb_generic_fn oracle_cblas = NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CGELSD][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CGELSD][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;
    fb_complex_float_t *Bo = same_backend ? NULL : (fb_complex_float_t *)malloc(Bsz);
    if ((!same_backend && !Bo) || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (!same_backend) {
        memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
        rank_out = 0;
        info = fb_call_cgelsd_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                      A0, lda, b0, ldb, Asz, Bsz, s_work,
                                      &rank_out, Bo);
        if (info == FB_BAND_CALL_ALLOC_FAILURE) {
            free(Bo);
            free(s_work);
            return FB_JUDGE_ERR_ALLOC;
        }
        if (info == FB_BAND_CALL_NOT_IMPL) {
            free(Bo);
            free(s_work);
            return FB_JUDGE_ERR_NOT_IMPL;
        }
        if (info != 0) {
            free(Bo);
            free(s_work);
            mark_oc_fatal(res);
            return FB_JUDGE_OK;
        }
        free(Bo);
    }
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
            rank_out = 0;
            info = fb_call_cgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
            rank_out = 0;
            info = fb_call_cgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
    rank_out = 0;
    info = fb_call_cgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGELSD — complex double precision min-norm least squares via SVD
 * ========================================================================= */
static fb_judge_status_t run_zgelsd(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    bool same_backend = (oracle == cand);
    fb_generic_fn oracle_cblas = NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZGELSD][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZGELSD][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    fb_complex_double_t *Bo = same_backend ? NULL : (fb_complex_double_t *)malloc(Bsz);
    if ((!same_backend && !Bo) || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (!same_backend) {
        memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
        rank_out = 0;
        info = fb_call_zgelsd_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                      A0, lda, b0, ldb, Asz, Bsz, s_work,
                                      &rank_out, Bo);
        if (info == FB_BAND_CALL_ALLOC_FAILURE) {
            free(Bo);
            free(s_work);
            return FB_JUDGE_ERR_ALLOC;
        }
        if (info == FB_BAND_CALL_NOT_IMPL) {
            free(Bo);
            free(s_work);
            return FB_JUDGE_ERR_NOT_IMPL;
        }
        if (info != 0) {
            free(Bo);
            free(s_work);
            mark_oc_fatal(res);
            return FB_JUDGE_OK;
        }
        free(Bo);
    }
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
            rank_out = 0;
            info = fb_call_zgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
            rank_out = 0;
            info = fb_call_zgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memset(s_work, 0, (size_t)minmn * sizeof(*s_work));
    rank_out = 0;
    info = fb_call_zgelsd_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * GELSS — minimum-norm least squares via SVD
 * ========================================================================= */
typedef void (*fb_sgelss_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       float *a, int *lda,
                                       float *b, int *ldb,
                                       float *s, float *rcond, int *rank,
                                       float *work, int *lwork, int *info);
typedef void (*fb_dgelss_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       double *a, int *lda,
                                       double *b, int *ldb,
                                       double *s, double *rcond, int *rank,
                                       double *work, int *lwork, int *info);
typedef void (*fb_cgelss_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *lda,
                                       fb_complex_float_t *b, int *ldb,
                                       float *s, float *rcond, int *rank,
                                       fb_complex_float_t *work,
                                       int *lwork, float *rwork, int *info);
typedef void (*fb_zgelss_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *lda,
                                       fb_complex_double_t *b, int *ldb,
                                       double *s, double *rcond, int *rank,
                                       fb_complex_double_t *work,
                                       int *lwork, double *rwork, int *info);

static int fb_call_sgelss_backend(
    fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const float *a_in, int lda_in,
    const float *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    float *b_out)
{
    (void)a_bytes;
    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(float)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float rcond = -1.0f;
        float work_query = 0.0f;
        float *a_col = (float *)calloc((size_t)lda_col * (size_t)n, sizeof(float));
        float *b_col = (float *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(float));
        float *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_float(work_query);
        work = (float *)calloc((size_t)lwork, sizeof(float));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        /* Some GELSS backends clobber A/B during the workspace query path. */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_sgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_dgelss_backend(
    fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const double *a_in, int lda_in,
    const double *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    double *b_out)
{
    (void)a_bytes;
    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(double)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double rcond = -1.0;
        double work_query = 0.0;
        double *a_col = (double *)calloc((size_t)lda_col * (size_t)n, sizeof(double));
        double *b_col = (double *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(double));
        double *work = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_double(work_query);
        work = (double *)calloc((size_t)lwork, sizeof(double));
        if (!work) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        /* Some GELSS backends clobber A/B during the workspace query path. */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_dgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_cgelss_backend(
    fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    float *s_out, int *rank_out,
    fb_complex_float_t *b_out)
{
    (void)a_bytes;
    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_float_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float rcond = -1.0f;
        float rwork_query[1] = {0.0f};
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *a_col =
            (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *b_col =
            (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_float_t));
        fb_complex_float_t *work = NULL;
        float *rwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_cgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             rwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cfloat(work_query);
        int rwork_len = fb_gelss_complex_rwork_size(m, n);
        int queried_rwork_len = fb_solve_query_size_from_float(rwork_query[0]);
        if (queried_rwork_len > rwork_len) {
            rwork_len = queried_rwork_len;
        }
        work = (fb_complex_float_t *)calloc((size_t)lwork, sizeof(fb_complex_float_t));
        rwork = (float *)calloc((size_t)rwork_len, sizeof(float));
        if (!work || !rwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(rwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        /* Some GELSS backends clobber A/B during the workspace query path. */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_cgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, rwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(rwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_zgelss_backend(
    fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    double *s_out, int *rank_out,
    fb_complex_double_t *b_out)
{
    (void)a_bytes;
    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_double_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double rcond = -1.0;
        double rwork_query[1] = {0.0};
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *a_col =
            (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *b_col =
            (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_double_t));
        fb_complex_double_t *work = NULL;
        double *rwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_zgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             &work_query, &lwork,
                                             rwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cdouble(work_query);
        int rwork_len = fb_gelss_complex_rwork_size(m, n);
        int queried_rwork_len = fb_solve_query_size_from_double(rwork_query[0]);
        if (queried_rwork_len > rwork_len) {
            rwork_len = queried_rwork_len;
        }
        work = (fb_complex_double_t *)calloc((size_t)lwork, sizeof(fb_complex_double_t));
        rwork = (double *)calloc((size_t)rwork_len, sizeof(double));
        if (!work || !rwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(rwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        /* Some GELSS backends clobber A/B during the workspace query path. */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_in[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_zgelss_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             s_out, &rcond, rank_out,
                                             work, &lwork, rwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(rwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static fb_judge_status_t run_sgelss(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SGELSS][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SGELSS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const float *A0 = (const float *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const float *b0 = (const float *)tc->B;
    int info = 0;

    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(float);
    size_t Bsz = tc->B_elems * sizeof(float);
    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;
    float *Bo = (float *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_sgelss_backend(oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);
    float *Bc = (float *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_sgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_sgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_sgelss_backend(cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_dgelss(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DGELSS][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DGELSS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double *)tc->B;
    int info = 0;

    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    double *Bo = (double *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_dgelss_backend(oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);
    double *Bc = (double *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_dgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_dgelss_backend(cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_cgelss(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CGELSS][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CGELSS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 = (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    int info = 0;

    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    float *s_work = (float *)malloc((size_t)minmn * sizeof(float));
    int rank_out = 0;
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_cgelss_backend(oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_cgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_cgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_cgelss_backend(cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zgelss(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZGELSS][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZGELSS][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    int info = 0;

    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    int minmn = m < n ? m : n;
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    double *s_work = (double *)malloc((size_t)minmn * sizeof(double));
    int rank_out = 0;
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Bo || !s_work) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    info = fb_call_zgelss_backend(oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(s_work);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Bo);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Bc) {
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_zgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zgelss_backend(cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, s_work,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bc);
                free(s_work);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    info = fb_call_zgelss_backend(cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, s_work,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bc);
        free(s_work);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bc);
        free(s_work);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(minmn * 10);
    free(Bc);
    free(s_work);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * DGELSY — double precision min-norm via complete orthogonal factorization
 * ========================================================================= */
static fb_judge_status_t run_dgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->dgelsy ? (fb_generic_fn)oracle->dgelsy : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DGELSY][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->dgelsy ? (fb_generic_fn)cand->dgelsy : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DGELSY][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const double *A0 = (const double*)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const double *b0 = (const double*)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(double);
    size_t Bsz = tc->B_elems * sizeof(double);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    double *Bo = (double *)malloc(Bsz);
    if (!Bo || !jpvt) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_dgelsy_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    double *Bc = (double *)malloc(Bsz);
    if (!Bc) { free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            info = fb_call_dgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_dgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_f64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bo); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * CGELSY — complex single precision min-norm via complete orthogonal factorization
 * ========================================================================= */
static int fb_solve_query_size_from_cfloat(fb_complex_float_t query_size);
static int fb_call_cgelsy_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    int *jpvt, int *rank_out,
    fb_complex_float_t *b_out);

static fb_judge_status_t run_cgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->cgelsy ? (fb_generic_fn)oracle->cgelsy : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CGELSY][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->cgelsy ? (fb_generic_fn)cand->cgelsy : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CGELSY][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *A0 =
        (const fb_complex_float_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Bo || !jpvt) {
        free(Bo);
        free(jpvt);
        return FB_JUDGE_ERR_ALLOC;
    }

    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_cgelsy_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(jpvt);
        return FB_JUDGE_ERR_ALLOC;
    }

    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Bc) {
        free(Bo);
        free(jpvt);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Bo);
        free(Bc);
        free(jpvt);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            info = fb_call_cgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                free(jpvt);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_cgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo);
                free(Bc);
                free(jpvt);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }

    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_cgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo);
        free(Bc);
        free(jpvt);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo);
        free(Bc);
        free(jpvt);
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo);
        free(Bc);
        free(jpvt);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }

    res->residual = make_result(bwerr_cf32(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bo);
    free(Bc);
    free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * ZGELSY — complex double precision min-norm via complete orthogonal factorization
 * ========================================================================= */
typedef void (*fb_cgelsy_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *lda,
                                       fb_complex_float_t *b, int *ldb,
                                       int *jpvt, float *rcond, int *rank,
                                       fb_complex_float_t *work,
                                       int *lwork, float *rwork, int *info);
typedef void (*fb_zgelsy_fortran_fn_t)(int *m, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *lda,
                                       fb_complex_double_t *b, int *ldb,
                                       int *jpvt, double *rcond, int *rank,
                                       fb_complex_double_t *work,
                                       int *lwork, double *rwork, int *info);

static int fb_call_cgelsy_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_float_t *a_in, int lda_in,
    const fb_complex_float_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    int *jpvt, int *rank_out,
    fb_complex_float_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_float_t *a_tmp = (fb_complex_float_t *)malloc(a_bytes);
        fb_complex_float_t *b_tmp = (fb_complex_float_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_cgelsy_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        jpvt, -1.0f, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_float_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        float rcond = -1.0f;
        float rwork_query = 0.0f;
        fb_complex_float_t work_query = 0.0f;
        fb_complex_float_t *a_col =
            (fb_complex_float_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_float_t));
        fb_complex_float_t *b_col =
            (fb_complex_float_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_float_t));
        fb_complex_float_t *work = NULL;
        float *rwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_cgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             &work_query, &lwork,
                                             &rwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cfloat(work_query);
        int rwork_len = (rwork_query > 1.0f) ? (int)rwork_query : ((n > 0) ? (2 * n) : 1);
        work = (fb_complex_float_t *)malloc((size_t)lwork * sizeof(fb_complex_float_t));
        rwork = (float *)malloc((size_t)rwork_len * sizeof(float));
        if (!work || !rwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(rwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_cgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             work, &lwork, rwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(rwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static int fb_call_zgelsy_backend(
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,
    int m, int n, int nrhs,
    const fb_complex_double_t *a_in, int lda_in,
    const fb_complex_double_t *b_in, int ldb_in,
    size_t a_bytes, size_t b_bytes,
    int *jpvt, int *rank_out,
    fb_complex_double_t *b_out)
{
    if (cblas_fn != NULL) {
        fb_complex_double_t *a_tmp = (fb_complex_double_t *)malloc(a_bytes);
        fb_complex_double_t *b_tmp = (fb_complex_double_t *)malloc(b_bytes);
        int info = 0;
        if (!a_tmp || !b_tmp) {
            free(a_tmp);
            free(b_tmp);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }
        memcpy(a_tmp, a_in, a_bytes);
        memcpy(b_tmp, b_in, b_bytes);
        info = ((fb_zgelsy_fn)cblas_fn)(FB_LAYOUT_ROW_MAJOR, m, n, nrhs,
                                        a_tmp, lda_in, b_tmp, ldb_in,
                                        jpvt, -1.0, rank_out);
        if (info == 0) memcpy(b_out, b_tmp, b_bytes);
        free(a_tmp);
        free(b_tmp);
        return info;
    }

    if (fortran_fn != NULL) {
        int m_ = m;
        int n_ = n;
        int nrhs_ = nrhs;
        int lda_col = (m > 0) ? m : 1;
        int solve_rows = (m > n) ? m : n;
        int stored_rows = (ldb_in > 0)
            ? (int)(b_bytes / ((size_t)ldb_in * sizeof(fb_complex_double_t)))
            : 0;
        int row_copy_rows = (stored_rows < solve_rows) ? stored_rows : solve_rows;
        int ldb_col = (stored_rows > solve_rows) ? stored_rows : solve_rows;
        int lwork = -1;
        int info = 0;
        double rcond = -1.0;
        double rwork_query = 0.0;
        fb_complex_double_t work_query = 0.0;
        fb_complex_double_t *a_col =
            (fb_complex_double_t *)calloc((size_t)lda_col * (size_t)n, sizeof(fb_complex_double_t));
        fb_complex_double_t *b_col =
            (fb_complex_double_t *)calloc((size_t)ldb_col * (size_t)nrhs, sizeof(fb_complex_double_t));
        fb_complex_double_t *work = NULL;
        double *rwork = NULL;

        if (!a_col || !b_col) {
            free(a_col);
            free(b_col);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        memcpy(b_out, b_in, b_bytes);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =
                    a_in[(size_t)i * (size_t)lda_in + (size_t)j];
            }
        }
        for (int i = 0; i < row_copy_rows; i++) {
            for (int j = 0; j < nrhs; j++) {
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j];
            }
        }

        ((fb_zgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             &work_query, &lwork,
                                             &rwork_query, &info);
        if (info != 0) {
            free(a_col);
            free(b_col);
            return info;
        }

        lwork = fb_solve_query_size_from_cdouble(work_query);
        int rwork_len = (rwork_query > 1.0) ? (int)rwork_query : ((n > 0) ? (2 * n) : 1);
        work = (fb_complex_double_t *)malloc((size_t)lwork * sizeof(fb_complex_double_t));
        rwork = (double *)malloc((size_t)rwork_len * sizeof(double));
        if (!work || !rwork) {
            free(a_col);
            free(b_col);
            free(work);
            free(rwork);
            return FB_BAND_CALL_ALLOC_FAILURE;
        }

        ((fb_zgelsy_fortran_fn_t)fortran_fn)(&m_, &n_, &nrhs_, a_col,
                                             &lda_col, b_col, &ldb_col,
                                             jpvt, &rcond, rank_out,
                                             work, &lwork, rwork, &info);
        if (info == 0) {
            for (int i = 0; i < row_copy_rows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    b_out[(size_t)i * (size_t)ldb_in + (size_t)j] =
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];
                }
            }
        }

        free(a_col);
        free(b_col);
        free(work);
        free(rwork);
        return info;
    }

    return FB_BAND_CALL_NOT_IMPL;
}

static fb_judge_status_t run_zgelsy(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->zgelsy ? (fb_generic_fn)oracle->zgelsy : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZGELSY][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->zgelsy ? (fb_generic_fn)cand->zgelsy : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZGELSY][FB_CONV_FORTRAN];
    int m = (int)tc->m, n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *A0 = (const fb_complex_double_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    int info = 0;

    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (m <= 0 || n <= 0 || nrhs <= 0 || !A0 || !b0) { mark_oc_fatal(res); return FB_JUDGE_OK; }
    size_t Asz = (size_t)m * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);
    int *jpvt = (int *)calloc((size_t)n, sizeof(int));
    int rank_out = 0;
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Bo || !jpvt) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_zgelsy_backend(oracle_cblas, oracle_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Bc) { free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            info = fb_call_zgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memset(jpvt, 0, (size_t)n * sizeof(int));
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                          A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                          &rank_out, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memset(jpvt, 0, (size_t)n * sizeof(int));
    info = fb_call_zgelsy_backend(cand_cblas, cand_fortran, m, n, nrhs,
                                  A0, lda, b0, ldb, Asz, Bsz, jpvt,
                                  &rank_out, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(Bo); free(Bc); free(jpvt); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(Bo); free(Bc); free(jpvt); mark_ca_fatal(res); return FB_JUDGE_OK;
    }
    res->residual       = make_result(bwerr_cf64(A0, m, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Bo); free(Bc); free(jpvt);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * STRTRS / DTRTRS / CTRTRS / ZTRTRS
 * Triangular system solve: T·x = B  (T already triangular — no pre-factor)
 * Synthetic upper triangular T: forced positive diagonal + scaled off-diag
 * ========================================================================= */

typedef int (*fb_strtrs_c_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                fb_diag_t, int, int, const float *, int,
                                float *, int);
typedef int (*fb_dtrtrs_c_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                fb_diag_t, int, int, const double *, int,
                                double *, int);
typedef int (*fb_ctrtrs_c_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                fb_diag_t, int, int,
                                const fb_complex_float_t *, int,
                                fb_complex_float_t *, int);
typedef int (*fb_ztrtrs_c_fn_t)(fb_layout_t, fb_uplo_t, fb_transpose_t,
                                fb_diag_t, int, int,
                                const fb_complex_double_t *, int,
                                fb_complex_double_t *, int);

typedef void (*fb_strtrs_fortran_fn_t)(const char *, const char *, const char *,
                                       int *, int *, const float *, int *,
                                       float *, int *, int *);
typedef void (*fb_dtrtrs_fortran_fn_t)(const char *, const char *, const char *,
                                       int *, int *, const double *, int *,
                                       double *, int *, int *);
typedef void (*fb_ctrtrs_fortran_fn_t)(const char *, const char *, const char *,
                                       int *, int *, const fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, int *);
typedef void (*fb_ztrtrs_fortran_fn_t)(const char *, const char *, const char *,
                                       int *, int *, const fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, int *);

#define FB_DEFINE_TRTRS_BACKEND_CALL(name, scalar_t, c_fn_t, fortran_fn_t)       \
static int name(                                                                  \
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,                             \
    int n, int nrhs, const scalar_t *a_in, int lda,                               \
    const scalar_t *b_in, int ldb, size_t b_bytes, scalar_t *b_out)               \
{                                                                                  \
    if (cblas_fn != NULL) {                                                        \
        memcpy(b_out, b_in, b_bytes);                                              \
        return ((c_fn_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_NO_TRANS,      \
                                  FB_NON_UNIT, n, nrhs, a_in, lda, b_out, ldb);    \
    }                                                                              \
                                                                                   \
    if (fortran_fn != NULL) {                                                      \
        int n_ = n;                                                                \
        int nrhs_ = nrhs;                                                          \
        int lda_col = (n > 0) ? n : 1;                                             \
        int ldb_col = (n > 0) ? n : 1;                                             \
        int info = 0;                                                              \
        char uplo = 'U';                                                           \
        char trans = 'N';                                                          \
        char diag = 'N';                                                           \
        scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n,          \
                                             sizeof(scalar_t));                    \
        scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)nrhs,       \
                                             sizeof(scalar_t));                    \
                                                                                   \
        if (!a_col || !b_col) {                                                    \
            free(a_col);                                                           \
            free(b_col);                                                           \
            return FB_BAND_CALL_ALLOC_FAILURE;                                     \
        }                                                                          \
                                                                                   \
        memcpy(b_out, b_in, b_bytes);                                              \
        for (int i = 0; i < n; i++) {                                              \
            for (int j = 0; j < n; j++) {                                          \
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =                   \
                    a_in[(size_t)i * (size_t)lda + (size_t)j];                     \
            }                                                                      \
        }                                                                          \
        for (int i = 0; i < n; i++) {                                              \
            for (int j = 0; j < nrhs; j++) {                                       \
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =                   \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j];                    \
            }                                                                      \
        }                                                                          \
                                                                                   \
        ((fortran_fn_t)fortran_fn)(&uplo, &trans, &diag, &n_, &nrhs_,             \
                                   a_col, &lda_col, b_col, &ldb_col, &info);       \
        if (info == 0) {                                                           \
            for (int i = 0; i < n; i++) {                                          \
                for (int j = 0; j < nrhs; j++) {                                   \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j] =                   \
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];            \
                }                                                                  \
            }                                                                      \
        }                                                                          \
                                                                                   \
        free(a_col);                                                               \
        free(b_col);                                                               \
        return info;                                                               \
    }                                                                              \
                                                                                   \
    return FB_BAND_CALL_NOT_IMPL;                                                  \
}

FB_DEFINE_TRTRS_BACKEND_CALL(fb_call_strtrs_backend, float,
                             fb_strtrs_c_fn_t, fb_strtrs_fortran_fn_t)
FB_DEFINE_TRTRS_BACKEND_CALL(fb_call_dtrtrs_backend, double,
                             fb_dtrtrs_c_fn_t, fb_dtrtrs_fortran_fn_t)
FB_DEFINE_TRTRS_BACKEND_CALL(fb_call_ctrtrs_backend, fb_complex_float_t,
                             fb_ctrtrs_c_fn_t, fb_ctrtrs_fortran_fn_t)
FB_DEFINE_TRTRS_BACKEND_CALL(fb_call_ztrtrs_backend, fb_complex_double_t,
                             fb_ztrtrs_c_fn_t, fb_ztrtrs_fortran_fn_t)

#undef FB_DEFINE_TRTRS_BACKEND_CALL

static fb_judge_status_t run_strtrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->strtrs ? (fb_generic_fn)oracle->strtrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_STRTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->strtrs ? (fb_generic_fn)cand->strtrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_STRTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_strtrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    float *Tc = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_strtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_strtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_strtrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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
    fb_generic_fn oracle_cblas = oracle->dtrtrs ? (fb_generic_fn)oracle->dtrtrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DTRTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->dtrtrs ? (fb_generic_fn)cand->dtrtrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DTRTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_dtrtrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
        free(T); free(To); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK;
    }
    free(To);
    double *Tc = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!Tc || !Bc) { free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_dtrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dtrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_dtrtrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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
    fb_generic_fn oracle_cblas = oracle->ctrtrs ? (fb_generic_fn)oracle->ctrtrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CTRTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->ctrtrs ? (fb_generic_fn)cand->ctrtrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CTRTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_ctrtrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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
            info = fb_call_ctrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_ctrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_ctrtrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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
    fb_generic_fn oracle_cblas = oracle->ztrtrs ? (fb_generic_fn)oracle->ztrtrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZTRTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->ztrtrs ? (fb_generic_fn)cand->ztrtrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZTRTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_ztrtrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(To); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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
            info = fb_call_ztrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_ztrtrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, T, lda, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(Tc, T, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_ztrtrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, T, lda, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC;
    }
    if (info == FB_BAND_CALL_NOT_IMPL) {
        free(T); free(Tc); free(Bc); free(Bo); return FB_JUDGE_ERR_NOT_IMPL;
    }
    if (info != 0) {
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

typedef int (*fb_ssytrs_c_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                const float *, int, const int *, float *, int);
typedef int (*fb_dsytrs_c_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                const double *, int, const int *, double *, int);
typedef int (*fb_csytrs_c_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                const fb_complex_float_t *, int,
                                const int *, fb_complex_float_t *, int);
typedef int (*fb_zsytrs_c_fn_t)(fb_layout_t, fb_uplo_t, int, int,
                                const fb_complex_double_t *, int,
                                const int *, fb_complex_double_t *, int);

typedef void (*fb_ssytrs_fortran_fn_t)(const char *, const int *, const int *,
                                       const float *, const int *, const int *,
                                       float *, const int *, int *);
typedef void (*fb_dsytrs_fortran_fn_t)(const char *, const int *, const int *,
                                       const double *, const int *, const int *,
                                       double *, const int *, int *);
typedef void (*fb_csytrs_fortran_fn_t)(char *, int *, int *,
                                       fb_complex_float_t *, int *, int *,
                                       fb_complex_float_t *, int *, int *);
typedef void (*fb_zsytrs_fortran_fn_t)(char *, int *, int *,
                                       fb_complex_double_t *, int *, int *,
                                       fb_complex_double_t *, int *, int *);

#define FB_DEFINE_SYTRS_BACKEND_CALL(name, scalar_t, c_fn_t, fortran_fn_t)        \
static int name(                                                                   \
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,                              \
    int n, int nrhs, const scalar_t *fact_in, int lda, const int *ipiv_in,         \
    const scalar_t *b_in, int ldb, size_t b_bytes, scalar_t *b_out)                \
{                                                                                   \
    if (cblas_fn != NULL) {                                                         \
        memcpy(b_out, b_in, b_bytes);                                               \
        return ((c_fn_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, FB_LOWER,                    \
                                  n, nrhs, fact_in, lda, ipiv_in, b_out, ldb);      \
    }                                                                               \
                                                                                    \
    if (fortran_fn != NULL) {                                                       \
        int n_ = n;                                                                 \
        int nrhs_ = nrhs;                                                           \
        int lda_col = (n > 0) ? n : 1;                                              \
        int ldb_col = (n > 0) ? n : 1;                                              \
        int info = 0;                                                               \
        char uplo = 'L';                                                            \
        scalar_t *fact_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n,        \
                                                sizeof(scalar_t));                  \
        scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)nrhs,        \
                                             sizeof(scalar_t));                     \
        int *ipiv = (int *)malloc((size_t)n * sizeof(int));                         \
                                                                                    \
        if (!fact_col || !b_col || !ipiv) {                                         \
            free(fact_col);                                                         \
            free(b_col);                                                            \
            free(ipiv);                                                             \
            return FB_BAND_CALL_ALLOC_FAILURE;                                      \
        }                                                                           \
                                                                                    \
        memcpy(ipiv, ipiv_in, (size_t)n * sizeof(int));                             \
        memcpy(b_out, b_in, b_bytes);                                               \
        for (int i = 0; i < n; i++) {                                               \
            for (int j = 0; j < n; j++) {                                           \
                fact_col[(size_t)i + (size_t)j * (size_t)lda_col] =                 \
                    fact_in[(size_t)i * (size_t)lda + (size_t)j];                   \
            }                                                                       \
        }                                                                           \
        for (int i = 0; i < n; i++) {                                               \
            for (int j = 0; j < nrhs; j++) {                                        \
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =                    \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j];                     \
            }                                                                       \
        }                                                                           \
                                                                                    \
        ((fortran_fn_t)fortran_fn)(&uplo, &n_, &nrhs_, fact_col, &lda_col, ipiv,    \
                                   b_col, &ldb_col, &info);                         \
        if (info == 0) {                                                            \
            for (int i = 0; i < n; i++) {                                           \
                for (int j = 0; j < nrhs; j++) {                                    \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j] =                    \
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];             \
                }                                                                   \
            }                                                                       \
        }                                                                           \
                                                                                    \
        free(fact_col);                                                             \
        free(b_col);                                                                \
        free(ipiv);                                                                 \
        return info;                                                                \
    }                                                                               \
                                                                                    \
    return FB_BAND_CALL_NOT_IMPL;                                                   \
}

FB_DEFINE_SYTRS_BACKEND_CALL(fb_call_ssytrs_backend,
                             float,
                             fb_ssytrs_c_fn_t,
                             fb_ssytrs_fortran_fn_t)
FB_DEFINE_SYTRS_BACKEND_CALL(fb_call_dsytrs_backend,
                             double,
                             fb_dsytrs_c_fn_t,
                             fb_dsytrs_fortran_fn_t)
FB_DEFINE_SYTRS_BACKEND_CALL(fb_call_csytrs_backend,
                             fb_complex_float_t,
                             fb_csytrs_c_fn_t,
                             fb_csytrs_fortran_fn_t)
FB_DEFINE_SYTRS_BACKEND_CALL(fb_call_zsytrs_backend,
                             fb_complex_double_t,
                             fb_zsytrs_c_fn_t,
                             fb_zsytrs_fortran_fn_t)

#undef FB_DEFINE_SYTRS_BACKEND_CALL

static fb_judge_status_t run_ssytrs(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res, uint64_t *ns_out)
{
    fb_generic_fn oracle_cblas = oracle->ssytrs ? (fb_generic_fn)oracle->ssytrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SSYTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->ssytrs ? (fb_generic_fn)cand->ssytrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SSYTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_ssytrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, fo, lda, ipiv, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);

    /* Candidate solve */
    float *fc = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_ssytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_ssytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_ssytrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
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
    fb_generic_fn oracle_cblas = oracle->dsytrs ? (fb_generic_fn)oracle->dsytrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DSYTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->dsytrs ? (fb_generic_fn)cand->dsytrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DSYTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_dsytrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, fo, lda, ipiv, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);

    double *fc = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_dsytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dsytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_dsytrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
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
    fb_generic_fn oracle_cblas = oracle->csytrs ? (fb_generic_fn)oracle->csytrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CSYTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->csytrs ? (fb_generic_fn)cand->csytrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CSYTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_csytrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, fo, lda, ipiv, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);
    fb_complex_float_t *fc = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_csytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_csytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_csytrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
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
    fb_generic_fn oracle_cblas = oracle->zsytrs ? (fb_generic_fn)oracle->zsytrs : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZSYTRS][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->zsytrs ? (fb_generic_fn)cand->zsytrs : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZSYTRS][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL)) return FB_JUDGE_ERR_NOT_IMPL;
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
    info = fb_call_zsytrs_backend(oracle_cblas, oracle_fortran,
                                  n, nrhs, fo, lda, ipiv, b0, ldb, Bsz, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
        { free(fact); free(Sorig); free(ipiv); free(fo); free(Bo); mark_oc_fatal(res); return FB_JUDGE_OK; }
    free(fo);
    fb_complex_double_t *fc = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!fc || !Bc) { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            info = fb_call_zsytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zsytrs_backend(cand_cblas, cand_fortran,
                                          n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE)
                { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best) best = dt;
        }
        *ns_out = best;
    }
    memcpy(fc, fact, Asz); memcpy(Bc, b0, Bsz);
    info = fb_call_zsytrs_backend(cand_cblas, cand_fortran,
                                  n, nrhs, fc, lda, ipiv, b0, ldb, Bsz, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); return FB_JUDGE_ERR_ALLOC; }
    if (info != 0)
        { free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo); mark_ca_fatal(res); return FB_JUDGE_OK; }
    res->residual       = make_result(bwerr_cf64(Sorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality  = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(fact); free(Sorig); free(ipiv); free(fc); free(Bc); free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Symmetric Indefinite Driver (SSYSV, DSYSV)
 *
 * SYSV = factorize (SSYTRF) + solve (SSYTRS) in one LAPACKE call.
 * Build diagonally‑dominant symmetric A (same construction as SSYTRS's Sorig),
 * keep Aorig before factorization, then compute backward error on candidate.
 * ========================================================================= */

typedef int (*fb_ssysv_c_fn_t)(fb_layout_t, char, int, int,
                               float *, int, int *, float *, int);
typedef int (*fb_dsysv_c_fn_t)(fb_layout_t, char, int, int,
                               double *, int, int *, double *, int);
typedef int (*fb_csysv_c_fn_t)(fb_layout_t, char, int, int,
                               fb_complex_float_t *, int, int *,
                               fb_complex_float_t *, int);
typedef int (*fb_zsysv_c_fn_t)(fb_layout_t, char, int, int,
                               fb_complex_double_t *, int, int *,
                               fb_complex_double_t *, int);
typedef int (*fb_chesv_c_fn_t)(fb_layout_t, char, int, int,
                               fb_complex_float_t *, int, int *,
                               fb_complex_float_t *, int);
typedef int (*fb_zhesv_c_fn_t)(fb_layout_t, char, int, int,
                               fb_complex_double_t *, int, int *,
                               fb_complex_double_t *, int);

typedef void (*fb_ssysv_fortran_fn_t)(const char *, int *, int *,
                                      float *, int *, int *,
                                      float *, int *, float *, int *, int *);
typedef void (*fb_dsysv_fortran_fn_t)(const char *, int *, int *,
                                      double *, int *, int *,
                                      double *, int *, double *, int *, int *);
typedef void (*fb_csysv_fortran_fn_t)(const char *, int *, int *,
                                      fb_complex_float_t *, int *, int *,
                                      fb_complex_float_t *, int *,
                                      fb_complex_float_t *, int *, int *);
typedef void (*fb_zsysv_fortran_fn_t)(const char *, int *, int *,
                                      fb_complex_double_t *, int *, int *,
                                      fb_complex_double_t *, int *,
                                      fb_complex_double_t *, int *, int *);
typedef void (*fb_chesv_fortran_fn_t)(const char *, int *, int *,
                                      fb_complex_float_t *, int *, int *,
                                      fb_complex_float_t *, int *,
                                      fb_complex_float_t *, int *, int *);
typedef void (*fb_zhesv_fortran_fn_t)(const char *, int *, int *,
                                      fb_complex_double_t *, int *, int *,
                                      fb_complex_double_t *, int *,
                                      fb_complex_double_t *, int *, int *);

#define FB_DEFINE_SYSV_LIKE_BACKEND_CALL(name, scalar_t, c_fn_t, fortran_fn_t, \
                                         work_t, query_len_expr)               \
static int name(                                                                \
    fb_generic_fn cblas_fn, fb_generic_fn fortran_fn,                           \
    int n, int nrhs, const scalar_t *a_in, int lda, int *ipiv_out,             \
    const scalar_t *b_in, int ldb, size_t a_bytes, size_t b_bytes,             \
    scalar_t *a_work, scalar_t *b_out)                                          \
{                                                                                \
    if (cblas_fn != NULL) {                                                      \
        memcpy(a_work, a_in, a_bytes);                                           \
        memcpy(b_out, b_in, b_bytes);                                            \
        return ((c_fn_t)cblas_fn)(FB_LAYOUT_ROW_MAJOR, 'L',                      \
                                  n, nrhs, a_work, lda, ipiv_out, b_out, ldb);   \
    }                                                                            \
                                                                                 \
    if (fortran_fn != NULL) {                                                    \
        int n_ = n;                                                              \
        int nrhs_ = nrhs;                                                        \
        int lda_col = (n > 0) ? n : 1;                                           \
        int ldb_col = (n > 0) ? n : 1;                                           \
        int lwork = -1;                                                          \
        int info = 0;                                                            \
        char uplo = 'L';                                                         \
        scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n,        \
                                             sizeof(scalar_t));                  \
        scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)nrhs,     \
                                             sizeof(scalar_t));                  \
        work_t work_query = (work_t)0;                                           \
        work_t *work = NULL;                                                     \
                                                                                 \
        if (!a_col || !b_col) {                                                  \
            free(a_col);                                                         \
            free(b_col);                                                         \
            return FB_BAND_CALL_ALLOC_FAILURE;                                   \
        }                                                                        \
                                                                                 \
        memcpy(a_work, a_in, a_bytes);                                           \
        memcpy(b_out, b_in, b_bytes);                                            \
        for (int i = 0; i < n; i++) {                                            \
            for (int j = 0; j < n; j++) {                                        \
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =                 \
                    a_work[(size_t)i * (size_t)lda + (size_t)j];                 \
            }                                                                    \
        }                                                                        \
        for (int i = 0; i < n; i++) {                                            \
            for (int j = 0; j < nrhs; j++) {                                     \
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =                 \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j];                  \
            }                                                                    \
        }                                                                        \
                                                                                 \
        ((fortran_fn_t)fortran_fn)(&uplo, &n_, &nrhs_, a_col, &lda_col,          \
                                   ipiv_out, b_col, &ldb_col,                    \
                                   &work_query, &lwork, &info);                  \
        if (info != 0) {                                                         \
            free(a_col);                                                         \
            free(b_col);                                                         \
            return info;                                                         \
        }                                                                        \
                                                                                 \
        lwork = (query_len_expr);                                                \
        if (lwork < 1) {                                                         \
            lwork = (n > 0) ? n : 1;                                             \
        }                                                                        \
        work = (work_t *)malloc((size_t)lwork * sizeof(work_t));                 \
        if (!work) {                                                             \
            free(a_col);                                                         \
            free(b_col);                                                         \
            return FB_BAND_CALL_ALLOC_FAILURE;                                   \
        }                                                                        \
                                                                                 \
        /* Some reference exports also execute the solve during lwork=-1. */     \
        for (int i = 0; i < n; i++) {                                            \
            for (int j = 0; j < n; j++) {                                        \
                a_col[(size_t)i + (size_t)j * (size_t)lda_col] =                 \
                    a_work[(size_t)i * (size_t)lda + (size_t)j];                 \
            }                                                                    \
        }                                                                        \
        for (int i = 0; i < n; i++) {                                            \
            for (int j = 0; j < nrhs; j++) {                                     \
                b_col[(size_t)i + (size_t)j * (size_t)ldb_col] =                 \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j];                  \
            }                                                                    \
        }                                                                        \
                                                                                 \
        ((fortran_fn_t)fortran_fn)(&uplo, &n_, &nrhs_, a_col, &lda_col,          \
                                   ipiv_out, b_col, &ldb_col, work, &lwork,      \
                                   &info);                                       \
        if (info == 0) {                                                         \
            for (int i = 0; i < n; i++) {                                        \
                for (int j = 0; j < nrhs; j++) {                                 \
                    b_out[(size_t)i * (size_t)ldb + (size_t)j] =                 \
                        b_col[(size_t)i + (size_t)j * (size_t)ldb_col];          \
                }                                                                \
            }                                                                    \
        }                                                                        \
                                                                                 \
        free(a_col);                                                             \
        free(b_col);                                                             \
        free(work);                                                              \
        return info;                                                             \
    }                                                                            \
                                                                                 \
    return FB_BAND_CALL_NOT_IMPL;                                                \
}

FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_ssysv_backend,
                                 float,
                                 fb_ssysv_c_fn_t,
                                 fb_ssysv_fortran_fn_t,
                                 float,
                                 fb_solve_query_size_from_float(work_query))
FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_dsysv_backend,
                                 double,
                                 fb_dsysv_c_fn_t,
                                 fb_dsysv_fortran_fn_t,
                                 double,
                                 fb_solve_query_size_from_double(work_query))
FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_csysv_backend,
                                 fb_complex_float_t,
                                 fb_csysv_c_fn_t,
                                 fb_csysv_fortran_fn_t,
                                 fb_complex_float_t,
                                 fb_solve_query_size_from_float((float)__real__(work_query)))
FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_zsysv_backend,
                                 fb_complex_double_t,
                                 fb_zsysv_c_fn_t,
                                 fb_zsysv_fortran_fn_t,
                                 fb_complex_double_t,
                                 fb_solve_query_size_from_double((double)__real__(work_query)))
FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_chesv_backend,
                                 fb_complex_float_t,
                                 fb_chesv_c_fn_t,
                                 fb_chesv_fortran_fn_t,
                                 fb_complex_float_t,
                                 fb_solve_query_size_from_float((float)__real__(work_query)))
FB_DEFINE_SYSV_LIKE_BACKEND_CALL(fb_call_zhesv_backend,
                                 fb_complex_double_t,
                                 fb_zhesv_c_fn_t,
                                 fb_zhesv_fortran_fn_t,
                                 fb_complex_double_t,
                                 fb_solve_query_size_from_double((double)__real__(work_query)))

#undef FB_DEFINE_SYSV_LIKE_BACKEND_CALL

static fb_judge_status_t run_ssysv(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_solve_result_t *res,
                                   uint64_t *ns_out) {
  fb_generic_fn oracle_cblas = oracle->ssysv ? (fb_generic_fn)oracle->ssysv : NULL;
  fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_SSYSV][FB_CONV_FORTRAN];
  fb_generic_fn cand_cblas = cand->ssysv ? (fb_generic_fn)cand->ssysv : NULL;
  fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_SSYSV][FB_CONV_FORTRAN];
  int info = 0;
  if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
      (cand_cblas == NULL && cand_fortran == NULL))
    return FB_JUDGE_ERR_NOT_IMPL;
  int n = (int)tc->n, nrhs = (int)tc->k;
  int lda = (int)tc->lda, ldb = (int)tc->ldb;
  const float *b0 = (const float *)tc->B;
  if (n <= 0 || nrhs <= 0 || !b0) {
    mark_oc_fatal(res);
    return FB_JUDGE_OK;
  }

  size_t Asz = (size_t)n * (size_t)lda * sizeof(float);
  size_t Bsz = tc->B_elems * sizeof(float);

  /* Build diagonally‑dominant symmetric matrix from explicit LDL^T:
   * D[k]=(k+2), L_unit[i,k]=0.1/(i-k) for i>k.
   * Sorig[i,j] = Σ_{k=0}^{min(i,j)} L[i,k]·D[k]·L[j,k]              */
  float *fact = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
  float *Aorig = (float *)calloc((size_t)n * (size_t)lda, sizeof(float));
  int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
  int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
  if (!fact || !Aorig || !ipiv_o || !ipiv_c) {
    free(fact);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    return FB_JUDGE_ERR_ALLOC;
  }
  for (int i = 0; i < n; i++) {
    fact[i * lda + i] = (float)(i + 2);
    for (int k = 0; k < i; k++)
      fact[i * lda + k] = 0.1f / (float)(i - k);
  }
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k <= i; k++) {
        float lik = (k == i) ? 1.0f : fact[i * lda + k];
        float ljk = (k == j) ? 1.0f : fact[j * lda + k];
        sum += lik * fact[k * lda + k] * ljk;
      }
      Aorig[i * lda + j] = sum;
      Aorig[j * lda + i] = sum;
    }
  }
  free(fact);

  /* Oracle: pass FULL symmetric A (gets overwritten by factorization). */
  float *Ao = (float *)malloc(Asz), *Bo = (float *)malloc(Bsz);
  if (!Ao || !Bo) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ao);
    free(Bo);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(Ao, Aorig, Asz);
  memcpy(Bo, b0, Bsz);
    info = fb_call_ssysv_backend(oracle_cblas, oracle_fortran,
                                                             n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                                             Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ao);
    free(Bo);
    mark_oc_fatal(res);
    return FB_JUDGE_OK;
  }
  free(Ao);

  /* Candidate */
  float *Ac = (float *)malloc(Asz), *Bc = (float *)malloc(Bsz);
  if (!Ac || !Bc) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Bo);
    free(Ac);
    free(Bc);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_ssysv_backend(cand_cblas, cand_fortran,
                                                                     n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                                     Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
    }
    for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_ssysv_backend(cand_cblas, cand_fortran,
                                                                     n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                                     Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }
  memcpy(Ac, Aorig, Asz);
  memcpy(Bc, b0, Bsz);
    info = fb_call_ssysv_backend(cand_cblas, cand_fortran,
                                                             n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                             Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    mark_ca_fatal(res);
    return FB_JUDGE_OK;
  }
  res->residual =
      make_result(bwerr_f32(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
  res->orthogonality = (fb_judge_case_result_t){.digits = 16};
  res->kappa_estimate = (double)(n * 10);
  free(Aorig);
  free(ipiv_o);
  free(ipiv_c);
  free(Ac);
  free(Bc);
  free(Bo);
  return FB_JUDGE_OK;
}

static fb_judge_status_t run_dsysv(const fb_backend_vtable_t *oracle,
                                   const fb_backend_vtable_t *cand,
                                   const fb_corpus_case_t *tc,
                                   fb_judge_solve_result_t *res,
                                   uint64_t *ns_out) {
    fb_generic_fn oracle_cblas = oracle->dsysv ? (fb_generic_fn)oracle->dsysv : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_DSYSV][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->dsysv ? (fb_generic_fn)cand->dsysv : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_DSYSV][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
            (cand_cblas == NULL && cand_fortran == NULL))
    return FB_JUDGE_ERR_NOT_IMPL;
  int n = (int)tc->n, nrhs = (int)tc->k;
  int lda = (int)tc->lda, ldb = (int)tc->ldb;
  const double *b0 = (const double *)tc->B;
  if (n <= 0 || nrhs <= 0 || !b0) {
    mark_oc_fatal(res);
    return FB_JUDGE_OK;
  }

  size_t Asz = (size_t)n * (size_t)lda * sizeof(double);
  size_t Bsz = tc->B_elems * sizeof(double);

  double *fact = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
  double *Aorig = (double *)calloc((size_t)n * (size_t)lda, sizeof(double));
  int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
  int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
  if (!fact || !Aorig || !ipiv_o || !ipiv_c) {
    free(fact);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    return FB_JUDGE_ERR_ALLOC;
  }
  for (int i = 0; i < n; i++) {
    fact[i * lda + i] = (double)(i + 2);
    for (int k = 0; k < i; k++)
      fact[i * lda + k] = 0.1 / (double)(i - k);
  }
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k <= i; k++) {
        double lik = (k == i) ? 1.0 : fact[i * lda + k];
        double ljk = (k == j) ? 1.0 : fact[j * lda + k];
        sum += lik * fact[k * lda + k] * ljk;
      }
      Aorig[i * lda + j] = sum;
      Aorig[j * lda + i] = sum;
    }
  }
  free(fact);

  double *Ao = (double *)malloc(Asz), *Bo = (double *)malloc(Bsz);
  if (!Ao || !Bo) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ao);
    free(Bo);
    return FB_JUDGE_ERR_ALLOC;
  }
  memcpy(Ao, Aorig, Asz);
  memcpy(Bo, b0, Bsz);
    info = fb_call_dsysv_backend(oracle_cblas, oracle_fortran,
                                                             n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                                             Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ao);
    free(Bo);
    mark_oc_fatal(res);
    return FB_JUDGE_OK;
  }
  free(Ao);

  double *Ac = (double *)malloc(Asz), *Bc = (double *)malloc(Bsz);
  if (!Ac || !Bc) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Bo);
    free(Ac);
    free(Bc);
    return FB_JUDGE_ERR_ALLOC;
  }
  if (ns_out) {
    uint64_t best = UINT64_MAX;
    for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_dsysv_backend(cand_cblas, cand_fortran,
                                                                     n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                                     Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
    }
    for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_dsysv_backend(cand_cblas, cand_fortran,
                                                                     n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                                     Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
      uint64_t dt = fb_judge_time_ns() - t0;
      if (dt < best)
        best = dt;
    }
    *ns_out = best;
  }
  memcpy(Ac, Aorig, Asz);
  memcpy(Bc, b0, Bsz);
    info = fb_call_dsysv_backend(cand_cblas, cand_fortran,
                                                             n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                                             Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    mark_ca_fatal(res);
    return FB_JUDGE_OK;
  }
  res->residual =
      make_result(bwerr_f64(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
  res->orthogonality = (fb_judge_case_result_t){.digits = 16};
  res->kappa_estimate = (double)(n * 10);
  free(Aorig);
  free(ipiv_o);
  free(ipiv_c);
  free(Ac);
  free(Bc);
  free(Bo);
  return FB_JUDGE_OK;
}

static fb_judge_status_t run_csysv(const fb_backend_vtable_t *oracle,
                                                                     const fb_backend_vtable_t *cand,
                                                                     const fb_corpus_case_t *tc,
                                                                     fb_judge_solve_result_t *res,
                                                                     uint64_t *ns_out) {
    fb_generic_fn oracle_cblas = oracle->csysv ? (fb_generic_fn)oracle->csysv : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CSYSV][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->csysv ? (fb_generic_fn)cand->csysv : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CSYSV][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL))
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);

    fb_complex_float_t *Aorig =
            (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Aorig || !ipiv_o || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }
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

    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    memcpy(Bo, b0, Bsz);
    info = fb_call_csysv_backend(oracle_cblas, oracle_fortran,
                                 n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                 Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Ao);

    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Bo);
        free(Ac);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_csysv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_csysv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, Aorig, Asz);
    memcpy(Bc, b0, Bsz);
    info = fb_call_csysv_backend(cand_cblas, cand_fortran,
                                 n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                 Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual =
            make_result(bwerr_cf32(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zsysv(const fb_backend_vtable_t *oracle,
                                                                     const fb_backend_vtable_t *cand,
                                                                     const fb_corpus_case_t *tc,
                                                                     fb_judge_solve_result_t *res,
                                                                     uint64_t *ns_out) {
    fb_generic_fn oracle_cblas = oracle->zsysv ? (fb_generic_fn)oracle->zsysv : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZSYSV][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->zsysv ? (fb_generic_fn)cand->zsysv : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZSYSV][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL))
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);

    fb_complex_double_t *Aorig =
            (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Aorig || !ipiv_o || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }
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

    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    memcpy(Bo, b0, Bsz);
    info = fb_call_zsysv_backend(oracle_cblas, oracle_fortran,
                                 n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                 Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Ao);

    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Bo);
        free(Ac);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_zsysv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zsysv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, Aorig, Asz);
    memcpy(Bc, b0, Bsz);
    info = fb_call_zsysv_backend(cand_cblas, cand_fortran,
                                 n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                 Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual =
            make_result(bwerr_cf64(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_chesv(const fb_backend_vtable_t *oracle,
                                                                     const fb_backend_vtable_t *cand,
                                                                     const fb_corpus_case_t *tc,
                                                                     fb_judge_solve_result_t *res,
                                                                     uint64_t *ns_out) {
    fb_generic_fn oracle_cblas = oracle->chesv ? (fb_generic_fn)oracle->chesv : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CHESV][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->chesv ? (fb_generic_fn)cand->chesv : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CHESV][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL))
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_float_t *b0 = (const fb_complex_float_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_float_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_float_t);

    fb_complex_float_t *Aorig =
            (fb_complex_float_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Aorig || !ipiv_o || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }
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

    fb_complex_float_t *Ao = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bo = (fb_complex_float_t *)malloc(Bsz);
    if (!Ao || !Bo) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    memcpy(Bo, b0, Bsz);
    info = fb_call_chesv_backend(oracle_cblas, oracle_fortran,
                                 n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                 Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Ao);

    fb_complex_float_t *Ac = (fb_complex_float_t *)malloc(Asz);
    fb_complex_float_t *Bc = (fb_complex_float_t *)malloc(Bsz);
    if (!Ac || !Bc) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Bo);
        free(Ac);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_chesv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_chesv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, Aorig, Asz);
    memcpy(Bc, b0, Bsz);
    info = fb_call_chesv_backend(cand_cblas, cand_fortran,
                                 n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                 Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual =
            make_result(bwerr_cf32(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zhesv(const fb_backend_vtable_t *oracle,
                                                                     const fb_backend_vtable_t *cand,
                                                                     const fb_corpus_case_t *tc,
                                                                     fb_judge_solve_result_t *res,
                                                                     uint64_t *ns_out) {
    fb_generic_fn oracle_cblas = oracle->zhesv ? (fb_generic_fn)oracle->zhesv : NULL;
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZHESV][FB_CONV_FORTRAN];
    fb_generic_fn cand_cblas = cand->zhesv ? (fb_generic_fn)cand->zhesv : NULL;
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZHESV][FB_CONV_FORTRAN];
    int info = 0;
    if ((oracle_cblas == NULL && oracle_fortran == NULL) ||
        (cand_cblas == NULL && cand_fortran == NULL))
        return FB_JUDGE_ERR_NOT_IMPL;
    int n = (int)tc->n, nrhs = (int)tc->k;
    int lda = (int)tc->lda, ldb = (int)tc->ldb;
    const fb_complex_double_t *b0 = (const fb_complex_double_t *)tc->B;
    if (n <= 0 || nrhs <= 0 || !b0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    size_t Asz = (size_t)n * (size_t)lda * sizeof(fb_complex_double_t);
    size_t Bsz = tc->B_elems * sizeof(fb_complex_double_t);

    fb_complex_double_t *Aorig =
            (fb_complex_double_t *)calloc((size_t)n * (size_t)lda, sizeof(*Aorig));
    int *ipiv_o = (int *)malloc((size_t)n * sizeof(int));
    int *ipiv_c = (int *)malloc((size_t)n * sizeof(int));
    if (!Aorig || !ipiv_o || !ipiv_c) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        return FB_JUDGE_ERR_ALLOC;
    }
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

    fb_complex_double_t *Ao = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bo = (fb_complex_double_t *)malloc(Bsz);
    if (!Ao || !Bo) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    memcpy(Ao, Aorig, Asz);
    memcpy(Bo, b0, Bsz);
    info = fb_call_zhesv_backend(oracle_cblas, oracle_fortran,
                                 n, nrhs, Aorig, lda, ipiv_o, b0, ldb,
                                 Asz, Bsz, Ao, Bo);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ao);
        free(Bo);
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }
    free(Ao);

    fb_complex_double_t *Ac = (fb_complex_double_t *)malloc(Asz);
    fb_complex_double_t *Bc = (fb_complex_double_t *)malloc(Bsz);
    if (!Ac || !Bc) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Bo);
        free(Ac);
        free(Bc);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (ns_out) {
        uint64_t best = UINT64_MAX;
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {
            info = fb_call_zhesv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
        }
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {
            uint64_t t0 = fb_judge_time_ns();
            info = fb_call_zhesv_backend(cand_cblas, cand_fortran,
                                         n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                         Asz, Bsz, Ac, Bc);
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {
                free(Aorig);
                free(ipiv_o);
                free(ipiv_c);
                free(Bo);
                free(Ac);
                free(Bc);
                return FB_JUDGE_ERR_ALLOC;
            }
            uint64_t dt = fb_judge_time_ns() - t0;
            if (dt < best)
                best = dt;
        }
        *ns_out = best;
    }
    memcpy(Ac, Aorig, Asz);
    memcpy(Bc, b0, Bsz);
    info = fb_call_zhesv_backend(cand_cblas, cand_fortran,
                                 n, nrhs, Aorig, lda, ipiv_c, b0, ldb,
                                 Asz, Bsz, Ac, Bc);
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        return FB_JUDGE_ERR_ALLOC;
    }
    if (info != 0) {
        free(Aorig);
        free(ipiv_o);
        free(ipiv_c);
        free(Ac);
        free(Bc);
        free(Bo);
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }
    res->residual =
            make_result(bwerr_cf64(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = (double)(n * 10);
    free(Aorig);
    free(ipiv_o);
    free(ipiv_c);
    free(Ac);
    free(Bc);
    free(Bo);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * SSYSVX / DSYSVX / CSYSVX / ZSYSVX / CHESVX / ZHESVX
 * SSYSVXX / DSYSVXX / CSYSVXX / ZSYSVXX / CHESVXX / ZHESVXX
 * Dense symmetric / Hermitian expert drivers (Fortran exports only)
 * ========================================================================= */

#define FB_COPY_RM_TO_CM(dst, ld_dst, src, ld_src, rows, cols)                   \
    do {                                                                         \
        for (int fb_copy_i = 0; fb_copy_i < (rows); ++fb_copy_i) {              \
            for (int fb_copy_j = 0; fb_copy_j < (cols); ++fb_copy_j) {          \
                (dst)[(size_t)fb_copy_i +                                        \
                      (size_t)fb_copy_j * (size_t)(ld_dst)] =                    \
                    (src)[(size_t)fb_copy_i * (size_t)(ld_src) +                 \
                          (size_t)fb_copy_j];                                    \
            }                                                                    \
        }                                                                        \
    } while (0)

#define FB_COPY_CM_TO_RM(dst, ld_dst, src, ld_src, rows, cols)                   \
    do {                                                                         \
        for (int fb_copy_i = 0; fb_copy_i < (rows); ++fb_copy_i) {              \
            for (int fb_copy_j = 0; fb_copy_j < (cols); ++fb_copy_j) {          \
                (dst)[(size_t)fb_copy_i * (size_t)(ld_dst) +                     \
                      (size_t)fb_copy_j] =                                       \
                    (src)[(size_t)fb_copy_i +                                    \
                          (size_t)fb_copy_j * (size_t)(ld_src)];                \
            }                                                                    \
        }                                                                        \
    } while (0)

typedef void (*fb_sgerfs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       float *a, int *lda, float *af,
                                       int *ldaf, int *ipiv, float *b,
                                       int *ldb, float *x, int *ldx,
                                       float *ferr, float *berr,
                                       float *work, int *iwork, int *info);
typedef void (*fb_dgerfs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       double *a, int *lda, double *af,
                                       int *ldaf, int *ipiv, double *b,
                                       int *ldb, double *x, int *ldx,
                                       double *ferr, double *berr,
                                       double *work, int *iwork, int *info);
typedef void (*fb_cgerfs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *lda,
                                       fb_complex_float_t *af, int *ldaf,
                                       int *ipiv, fb_complex_float_t *b,
                                       int *ldb, fb_complex_float_t *x,
                                       int *ldx, float *ferr, float *berr,
                                       fb_complex_float_t *work, float *rwork,
                                       int *info);
typedef void (*fb_zgerfs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *lda,
                                       fb_complex_double_t *af, int *ldaf,
                                       int *ipiv, fb_complex_double_t *b,
                                       int *ldb, fb_complex_double_t *x,
                                       int *ldx, double *ferr, double *berr,
                                       fb_complex_double_t *work, double *rwork,
                                       int *info);
typedef void (*fb_sgetrf_fortran_fn_t)(int *m, int *n, float *a, int *lda,
                                       int *ipiv, int *info);
typedef void (*fb_dgetrf_fortran_fn_t)(int *m, int *n, double *a, int *lda,
                                       int *ipiv, int *info);
typedef void (*fb_cgetrf_fortran_fn_t)(int *m, int *n,
                                       fb_complex_float_t *a, int *lda,
                                       int *ipiv, int *info);
typedef void (*fb_zgetrf_fortran_fn_t)(int *m, int *n,
                                       fb_complex_double_t *a, int *lda,
                                       int *ipiv, int *info);
typedef void (*fb_sgetrs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       float *a, int *lda, int *ipiv,
                                       float *b, int *ldb, int *info);
typedef void (*fb_dgetrs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       double *a, int *lda, int *ipiv,
                                       double *b, int *ldb, int *info);
typedef void (*fb_cgetrs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       fb_complex_float_t *a, int *lda,
                                       int *ipiv, fb_complex_float_t *b,
                                       int *ldb, int *info);
typedef void (*fb_zgetrs_fortran_fn_t)(char *trans, int *n, int *nrhs,
                                       fb_complex_double_t *a, int *lda,
                                       int *ipiv, fb_complex_double_t *b,
                                       int *ldb, int *info);
typedef void (*fb_sgesv_fortran_fn_t)(int *n, int *nrhs, float *a, int *lda,
                                      int *ipiv, float *b, int *ldb,
                                      int *info);
typedef void (*fb_dgesv_fortran_fn_t)(int *n, int *nrhs, double *a, int *lda,
                                      int *ipiv, double *b, int *ldb,
                                      int *info);
typedef void (*fb_cgesv_fortran_fn_t)(int *n, int *nrhs,
                                      fb_complex_float_t *a, int *lda,
                                      int *ipiv, fb_complex_float_t *b,
                                      int *ldb, int *info);
typedef void (*fb_zgesv_fortran_fn_t)(int *n, int *nrhs,
                                      fb_complex_double_t *a, int *lda,
                                      int *ipiv, fb_complex_double_t *b,
                                      int *ldb, int *info);

#define FB_DEFINE_GESV_CALLER(name, fortran_fn_t, scalar_t)                      \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int info = 0;                                                                 \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !b_col || !ipiv) {                                              \
        free(a_col);                                                              \
        free(b_col);                                                              \
        free(ipiv);                                                               \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&n_, &nrhs_, a_col, &lda_col, ipiv,               \
                               b_col, &ldb_col, &info);                          \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, b_col, ldb_col, n, nrhs);                   \
    }                                                                             \
    free(a_col);                                                                  \
    free(b_col);                                                                  \
    free(ipiv);                                                                   \
    return info;                                                                  \
}

#define FB_DEFINE_GERFS_REAL_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, const scalar_t *af_in,                     \
                const int *ipiv_in, const scalar_t *b_in,                        \
                const scalar_t *x_in, int lda, int ldb,                          \
                size_t x_bytes, scalar_t *x_out)                                 \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf_col = lda_col;                                                       \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx_col = ldb_col;                                                        \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? n : 1;                                               \
    char trans = 'N';                                                             \
    int info = 0;                                                                 \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,      \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf_col * (size_t)n_order,    \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,        \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx_col * (size_t)n_rhs,        \
                                         sizeof(scalar_t));                       \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));     \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                     \
    int *ipiv = (int *)calloc((size_t)n_order, sizeof(int));                      \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    memset(x_out, 0, x_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !work || !iwork || !ipiv ||     \
        !ferr || !berr) {                                                         \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(iwork); free(ipiv); free(ferr); free(berr);                          \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(af_col, ldaf_col, af_in, lda, n, n);                        \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    FB_COPY_RM_TO_CM(x_col, ldx_col, x_in, ldb, n, nrhs);                        \
    memcpy(ipiv, ipiv_in, (size_t)n_order * sizeof(int));                        \
    ((fortran_fn_t)fortran_fn)(&trans, &n_, &nrhs_, a_col, &lda_col, af_col,     \
                               &ldaf_col, ipiv, b_col, &ldb_col, x_col,          \
                               &ldx_col, ferr, berr, work, iwork, &info);        \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx_col, n, nrhs);                   \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);             \
    free(iwork); free(ipiv); free(ferr); free(berr);                             \
    return info;                                                                  \
}

#define FB_DEFINE_GERFS_COMPLEX_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                          \
                const scalar_t *a_in, const scalar_t *af_in,                        \
                const int *ipiv_in, const scalar_t *b_in,                           \
                const scalar_t *x_in, int lda, int ldb,                             \
                size_t x_bytes, scalar_t *x_out)                                    \
{                                                                                    \
    if (fortran_fn == NULL) {                                                        \
        return FB_BAND_CALL_NOT_IMPL;                                                \
    }                                                                                \
    int n_ = n;                                                                      \
    int nrhs_ = nrhs;                                                                \
    int lda_col = (n > 0) ? n : 1;                                                   \
    int ldaf_col = lda_col;                                                          \
    int ldb_col = (n > 0) ? n : 1;                                                   \
    int ldx_col = ldb_col;                                                           \
    int n_order = (n > 0) ? n : 1;                                                   \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                               \
    int work_len = (n > 0) ? n : 1;                                                  \
    int rwork_len = (n > 0) ? n : 1;                                                 \
    char trans = 'N';                                                                \
    int info = 0;                                                                    \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,         \
                                         sizeof(scalar_t));                          \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf_col * (size_t)n_order,       \
                                          sizeof(scalar_t));                         \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,           \
                                         sizeof(scalar_t));                          \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx_col * (size_t)n_rhs,           \
                                         sizeof(scalar_t));                          \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));        \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));             \
    int *ipiv = (int *)calloc((size_t)n_order, sizeof(int));                         \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                  \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                  \
    memset(x_out, 0, x_bytes);                                                       \
    if (!a_col || !af_col || !b_col || !x_col || !work || !rwork || !ipiv ||        \
        !ferr || !berr) {                                                            \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);             \
        free(rwork); free(ipiv); free(ferr); free(berr);                             \
        return FB_BAND_CALL_ALLOC_FAILURE;                                           \
    }                                                                                \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                              \
    FB_COPY_RM_TO_CM(af_col, ldaf_col, af_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                           \
    FB_COPY_RM_TO_CM(x_col, ldx_col, x_in, ldb, n, nrhs);                           \
    memcpy(ipiv, ipiv_in, (size_t)n_order * sizeof(int));                           \
    ((fortran_fn_t)fortran_fn)(&trans, &n_, &nrhs_, a_col, &lda_col, af_col,        \
                               &ldaf_col, ipiv, b_col, &ldb_col, x_col,             \
                               &ldx_col, ferr, berr, work, rwork, &info);           \
    if (info == 0) {                                                                 \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx_col, n, nrhs);                      \
    }                                                                                \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);                \
    free(rwork); free(ipiv); free(ferr); free(berr);                                \
    return info;                                                                     \
}

FB_DEFINE_GERFS_REAL_FORTRAN_CALL(fb_call_sgerfs_backend, float, float,
                                  fb_sgerfs_fortran_fn_t)
FB_DEFINE_GERFS_REAL_FORTRAN_CALL(fb_call_dgerfs_backend, double, double,
                                  fb_dgerfs_fortran_fn_t)
FB_DEFINE_GERFS_COMPLEX_FORTRAN_CALL(fb_call_cgerfs_backend,
                                     fb_complex_float_t, float,
                                     fb_cgerfs_fortran_fn_t)
FB_DEFINE_GERFS_COMPLEX_FORTRAN_CALL(fb_call_zgerfs_backend,
                                     fb_complex_double_t, double,
                                     fb_zgerfs_fortran_fn_t)

#define FB_DEFINE_DENSE_GETRF_FORTRAN_CALL(name, scalar_t, fortran_fn_t)      \
static int name(fb_generic_fn fortran_fn, int n, scalar_t *a_inout, int lda,  \
                int *ipiv_inout)                                               \
{                                                                              \
    if (fortran_fn == NULL) {                                                  \
        return FB_BAND_CALL_NOT_IMPL;                                          \
    }                                                                          \
    int m_ = n;                                                                \
    int n_ = n;                                                                \
    int lda_col = (n > 0) ? n : 1;                                             \
    int n_order = (n > 0) ? n : 1;                                             \
    int info = 0;                                                              \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,    \
                                         sizeof(scalar_t));                    \
    if (!a_col) {                                                              \
        return FB_BAND_CALL_ALLOC_FAILURE;                                     \
    }                                                                          \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_inout, lda, n, n);                     \
    ((fortran_fn_t)fortran_fn)(&m_, &n_, a_col, &lda_col, ipiv_inout, &info); \
    if (info == 0) {                                                           \
        FB_COPY_CM_TO_RM(a_inout, lda, a_col, lda_col, n, n);                 \
    }                                                                          \
    free(a_col);                                                               \
    return info;                                                               \
}

#define FB_DEFINE_DENSE_GETRS_FORTRAN_CALL(name, scalar_t, fortran_fn_t)       \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                     \
                const scalar_t *a_in, int lda, const int *ipiv_in,             \
                scalar_t *b_inout, int ldb)                                     \
{                                                                              \
    if (fortran_fn == NULL) {                                                  \
        return FB_BAND_CALL_NOT_IMPL;                                          \
    }                                                                          \
    int n_ = n;                                                                \
    int nrhs_ = nrhs;                                                          \
    int lda_col = (n > 0) ? n : 1;                                             \
    int ldb_col = (n > 0) ? n : 1;                                             \
    int n_order = (n > 0) ? n : 1;                                             \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                         \
    int info = 0;                                                              \
    char trans = 'N';                                                          \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,    \
                                         sizeof(scalar_t));                    \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,      \
                                         sizeof(scalar_t));                    \
    int *ipiv = (int *)calloc((size_t)n_order, sizeof(int));                   \
    if (!a_col || !b_col || !ipiv) {                                           \
        free(a_col);                                                           \
        free(b_col);                                                           \
        free(ipiv);                                                            \
        return FB_BAND_CALL_ALLOC_FAILURE;                                     \
    }                                                                          \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                        \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_inout, ldb, n, nrhs);                  \
    memcpy(ipiv, ipiv_in, (size_t)n_order * sizeof(int));                      \
    ((fortran_fn_t)fortran_fn)(&trans, &n_, &nrhs_, a_col, &lda_col, ipiv,     \
                               b_col, &ldb_col, &info);                        \
    if (info == 0) {                                                           \
        FB_COPY_CM_TO_RM(b_inout, ldb, b_col, ldb_col, n, nrhs);              \
    }                                                                          \
    free(a_col);                                                               \
    free(b_col);                                                               \
    free(ipiv);                                                                \
    return info;                                                               \
}

FB_DEFINE_DENSE_GETRF_FORTRAN_CALL(fb_call_sgetrf_fortran_backend, float,
                                   fb_sgetrf_fortran_fn_t)
FB_DEFINE_DENSE_GETRF_FORTRAN_CALL(fb_call_dgetrf_fortran_backend, double,
                                   fb_dgetrf_fortran_fn_t)
FB_DEFINE_DENSE_GETRF_FORTRAN_CALL(fb_call_cgetrf_fortran_backend,
                                   fb_complex_float_t,
                                   fb_cgetrf_fortran_fn_t)
FB_DEFINE_DENSE_GETRF_FORTRAN_CALL(fb_call_zgetrf_fortran_backend,
                                   fb_complex_double_t,
                                   fb_zgetrf_fortran_fn_t)
FB_DEFINE_DENSE_GETRS_FORTRAN_CALL(fb_call_sgetrs_fortran_backend, float,
                                   fb_sgetrs_fortran_fn_t)
FB_DEFINE_DENSE_GETRS_FORTRAN_CALL(fb_call_dgetrs_fortran_backend, double,
                                   fb_dgetrs_fortran_fn_t)
FB_DEFINE_DENSE_GETRS_FORTRAN_CALL(fb_call_cgetrs_fortran_backend,
                                   fb_complex_float_t,
                                   fb_cgetrs_fortran_fn_t)
FB_DEFINE_DENSE_GETRS_FORTRAN_CALL(fb_call_zgetrs_fortran_backend,
                                   fb_complex_double_t,
                                   fb_zgetrs_fortran_fn_t)

#define FB_DEFINE_GERFS_RUNNER(name, op_id, scalar_t, getrf_op_id, getrs_op_id, \
                               getrf_call_fn, getrs_call_fn, call_fn, bwerr_fn) \
static fb_judge_status_t name(const fb_backend_vtable_t *oracle,                  \
                              const fb_backend_vtable_t *cand,                    \
                              const fb_corpus_case_t *tc,                         \
                              fb_judge_solve_result_t *res,                       \
                              uint64_t *ns_out)                                   \
{                                                                                  \
    fb_generic_fn oracle_fortran = oracle->ext_ops[(op_id)][FB_CONV_FORTRAN];     \
    fb_generic_fn cand_fortran = cand->ext_ops[(op_id)][FB_CONV_FORTRAN];         \
    fb_generic_fn oracle_getrf_fortran =                                          \
        oracle->ext_ops[(getrf_op_id)][FB_CONV_FORTRAN];                          \
    fb_generic_fn oracle_getrs_fortran =                                          \
        oracle->ext_ops[(getrs_op_id)][FB_CONV_FORTRAN];                          \
    if (oracle_fortran == NULL || cand_fortran == NULL ||                          \
        oracle_getrf_fortran == NULL || oracle_getrs_fortran == NULL) {           \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    int n = (int)tc->n;                                                            \
    int nrhs = (int)tc->k;                                                         \
    int lda = (int)tc->lda;                                                        \
    int ldb = (int)tc->ldb;                                                        \
    const scalar_t *A0 =                                                            \
        (const scalar_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);              \
    const scalar_t *b0 = (const scalar_t *)tc->B;                                  \
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) {                                       \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    size_t Asz = (size_t)n * (size_t)lda * sizeof(scalar_t);                       \
    size_t Bsz = tc->B_elems * sizeof(scalar_t);                                   \
    scalar_t *LU = (scalar_t *)malloc(Asz);                                        \
    scalar_t *Xseed = (scalar_t *)malloc(Bsz);                                     \
    scalar_t *Bo = (scalar_t *)malloc(Bsz);                                        \
    scalar_t *Bc = (scalar_t *)malloc(Bsz);                                        \
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));                            \
    int info = 0;                                                                  \
    if (!LU || !Xseed || !Bo || !Bc || !ipiv) {                                    \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    memcpy(LU, A0, Asz);                                                           \
    if (getrf_call_fn(oracle_getrf_fortran, n, LU, lda, ipiv) != 0) {             \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    memcpy(Xseed, b0, Bsz);                                                        \
    if (getrs_call_fn(oracle_getrs_fortran, n, nrhs, LU, lda, ipiv, Xseed, ldb)   \
        != 0) {                                                                    \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    info = call_fn(oracle_fortran, n, nrhs, A0, LU, ipiv, b0, Xseed, lda, ldb,    \
                   Bsz, Bo);                                                        \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    if (ns_out) {                                                                  \
        uint64_t best = UINT64_MAX;                                                \
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; ++w) {                           \
            info = call_fn(cand_fortran, n, nrhs, A0, LU, ipiv, b0, Xseed, lda,   \
                           ldb, Bsz, Bc);                                          \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);             \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
        }                                                                          \
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; ++t) {                           \
            uint64_t t0 = fb_judge_time_ns();                                      \
            info = call_fn(cand_fortran, n, nrhs, A0, LU, ipiv, b0, Xseed, lda,   \
                           ldb, Bsz, Bc);                                          \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);             \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
            uint64_t dt = fb_judge_time_ns() - t0;                                 \
            if (dt < best) {                                                       \
                best = dt;                                                         \
            }                                                                      \
        }                                                                          \
        *ns_out = best;                                                            \
    }                                                                              \
    info = call_fn(cand_fortran, n, nrhs, A0, LU, ipiv, b0, Xseed, lda, ldb,      \
                   Bsz, Bc);                                                        \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                     \
        mark_ca_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    res->residual = make_result(bwerr_fn(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs)); \
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};                  \
    res->kappa_estimate = (double)(n * 10);                                        \
    free(LU); free(Xseed); free(Bo); free(Bc); free(ipiv);                         \
    return FB_JUDGE_OK;                                                            \
}

FB_DEFINE_GERFS_RUNNER(run_sgerfs, FB_OP_SGERFS, float,
                       FB_OP_SGETRF, FB_OP_SGETRS,
                       fb_call_sgetrf_fortran_backend,
                       fb_call_sgetrs_fortran_backend,
                       fb_call_sgerfs_backend, bwerr_f32)
FB_DEFINE_GERFS_RUNNER(run_dgerfs, FB_OP_DGERFS, double,
                       FB_OP_DGETRF, FB_OP_DGETRS,
                       fb_call_dgetrf_fortran_backend,
                       fb_call_dgetrs_fortran_backend,
                       fb_call_dgerfs_backend, bwerr_f64)
FB_DEFINE_GERFS_RUNNER(run_cgerfs, FB_OP_CGERFS, fb_complex_float_t,
                       FB_OP_CGETRF, FB_OP_CGETRS,
                       fb_call_cgetrf_fortran_backend,
                       fb_call_cgetrs_fortran_backend,
                       fb_call_cgerfs_backend, bwerr_cf32)
FB_DEFINE_GERFS_RUNNER(run_zgerfs, FB_OP_ZGERFS, fb_complex_double_t,
                       FB_OP_ZGETRF, FB_OP_ZGETRS,
                       fb_call_zgetrf_fortran_backend,
                       fb_call_zgetrs_fortran_backend,
                       fb_call_zgerfs_backend, bwerr_cf64)

#undef FB_DEFINE_GERFS_RUNNER
#undef FB_DEFINE_GERFS_REAL_FORTRAN_CALL
#undef FB_DEFINE_GERFS_COMPLEX_FORTRAN_CALL

static void fb_build_dense_symm_f32(float *a, int n, int lda) {
    float *fact = (float *)calloc((size_t)n * (size_t)lda, sizeof(*fact));
    if (!fact) {
        return;
    }
    for (int i = 0; i < n; i++) {
        fact[i * lda + i] = (float)(i + 2);
        for (int k = 0; k < i; k++) {
            fact[i * lda + k] = 0.1f / (float)(i - k);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                float lik = (k == i) ? 1.0f : fact[i * lda + k];
                float ljk = (k == j) ? 1.0f : fact[j * lda + k];
                sum += lik * fact[k * lda + k] * ljk;
            }
            a[i * lda + j] = sum;
            a[j * lda + i] = sum;
        }
    }
    free(fact);
}

static void fb_build_dense_symm_f64(double *a, int n, int lda) {
    double *fact = (double *)calloc((size_t)n * (size_t)lda, sizeof(*fact));
    if (!fact) {
        return;
    }
    for (int i = 0; i < n; i++) {
        fact[i * lda + i] = (double)(i + 2);
        for (int k = 0; k < i; k++) {
            fact[i * lda + k] = 0.1 / (double)(i - k);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                double lik = (k == i) ? 1.0 : fact[i * lda + k];
                double ljk = (k == j) ? 1.0 : fact[j * lda + k];
                sum += lik * fact[k * lda + k] * ljk;
            }
            a[i * lda + j] = sum;
            a[j * lda + i] = sum;
        }
    }
    free(fact);
}

static void fb_build_dense_symm_cf32(fb_complex_float_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(a[i * lda + i]) = (float)(n + 2 * (i + 1));
        __imag__(a[i * lda + i]) = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float re = 0.5f / (float)(j - i + 1);
            float im = 0.05f * (float)(i + 1) / (float)(j + 1);
            __real__(a[i * lda + j]) = re;
            __imag__(a[i * lda + j]) = im;
            __real__(a[j * lda + i]) = re;
            __imag__(a[j * lda + i]) = im;
        }
    }
}

static void fb_build_dense_symm_cf64(fb_complex_double_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(a[i * lda + i]) = (double)(n + 2 * (i + 1));
        __imag__(a[i * lda + i]) = 0.0;
        for (int j = i + 1; j < n; j++) {
            double re = 0.5 / (double)(j - i + 1);
            double im = 0.05 * (double)(i + 1) / (double)(j + 1);
            __real__(a[i * lda + j]) = re;
            __imag__(a[i * lda + j]) = im;
            __real__(a[j * lda + i]) = re;
            __imag__(a[j * lda + i]) = im;
        }
    }
}

static void fb_build_dense_herm_cf32(fb_complex_float_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(a[i * lda + i]) = (float)(n + 2 * (i + 1));
        __imag__(a[i * lda + i]) = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float re = 0.5f / (float)(j - i + 1);
            float im = 0.05f * (float)(i + 1) / (float)(j + 1);
            __real__(a[i * lda + j]) = re;
            __imag__(a[i * lda + j]) = im;
            __real__(a[j * lda + i]) = re;
            __imag__(a[j * lda + i]) = -im;
        }
    }
}

static void fb_build_dense_herm_cf64(fb_complex_double_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        __real__(a[i * lda + i]) = (double)(n + 2 * (i + 1));
        __imag__(a[i * lda + i]) = 0.0;
        for (int j = i + 1; j < n; j++) {
            double re = 0.5 / (double)(j - i + 1);
            double im = 0.05 * (double)(i + 1) / (double)(j + 1);
            __real__(a[i * lda + j]) = re;
            __imag__(a[i * lda + j]) = im;
            __real__(a[j * lda + i]) = re;
            __imag__(a[j * lda + i]) = -im;
        }
    }
}

[[maybe_unused]] static void fb_build_diag_herm_cf32(fb_complex_float_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __real__(a[i * lda + j]) = 0.0f;
            __imag__(a[i * lda + j]) = 0.0f;
        }
        __real__(a[i * lda + i]) = (i % 2 == 0) ? (float)(i + 2) : (float)(-(i + 2));
    }
}

[[maybe_unused]] static void fb_build_diag_herm_cf64(fb_complex_double_t *a, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __real__(a[i * lda + j]) = 0.0;
            __imag__(a[i * lda + j]) = 0.0;
        }
        __real__(a[i * lda + i]) = (i % 2 == 0) ? (double)(i + 2) : (double)(-(i + 2));
    }
}

typedef void (*fb_ssysvx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                       float *, int *, int *, float *, int *,
                                       float *, int *, float *, float *, float *,
                                       float *, int *, int *, int *);
typedef void (*fb_dsysvx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                       double *, int *, int *, double *, int *,
                                       double *, int *, double *, double *, double *,
                                       double *, int *, int *, int *);
typedef void (*fb_csysvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, float *,
                                       float *, float *, fb_complex_float_t *,
                                       int *, int *);
typedef void (*fb_zsysvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, double *,
                                       double *, double *, fb_complex_double_t *,
                                       int *, double *, int *);
typedef void (*fb_chesvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, float *,
                                       float *, float *, fb_complex_float_t *,
                                       int *, float *, int *);
typedef void (*fb_zhesvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, double *,
                                       double *, double *, fb_complex_double_t *,
                                       int *, double *, int *);

typedef void (*fb_sgesvx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                       float *, int *, int *, char *, float *,
                                       float *, float *, int *, float *, int *,
                                       float *, float *, float *, float *, int *,
                                       int *);
typedef void (*fb_dgesvx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                       double *, int *, int *, char *, double *,
                                       double *, double *, int *, double *, int *,
                                       double *, double *, double *, double *, int *,
                                       int *);
typedef void (*fb_cgesvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, int *, char *,
                                       float *, float *, fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, float *, float *,
                                       float *, fb_complex_float_t *, float *, int *);
typedef void (*fb_zgesvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, int *, char *,
                                       double *, double *, fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, double *,
                                       double *, double *, fb_complex_double_t *,
                                       double *, int *);

typedef void (*fb_sposvx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                       float *, int *, char *, float *, float *,
                                       int *, float *, int *, float *, float *,
                                       float *, float *, int *, int *);
typedef void (*fb_dposvx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                       double *, int *, char *, double *, double *,
                                       int *, double *, int *, double *, double *,
                                       double *, double *, int *, int *);
typedef void (*fb_cposvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, char *, float *,
                                       fb_complex_float_t *, int *,
                                       fb_complex_float_t *, int *, float *, float *,
                                       float *, fb_complex_float_t *, float *, int *);
typedef void (*fb_zposvx_fortran_fn_t)(char *, char *, int *, int *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, char *, double *,
                                       fb_complex_double_t *, int *,
                                       fb_complex_double_t *, int *, double *,
                                       double *, double *, fb_complex_double_t *,
                                       double *, int *);

typedef void (*fb_ssysvxx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                        float *, int *, int *, char *, float *,
                                        float *, int *, float *, int *, float *,
                                        float *, float *, int *, float *, float *,
                                        int *, float *, float *, int *, int *);
typedef void (*fb_dsysvxx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                        double *, int *, int *, char *, double *,
                                        double *, int *, double *, int *, double *,
                                        double *, double *, int *, double *, double *,
                                        int *, double *, double *, int *, int *);
typedef void (*fb_csysvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, int *, char *,
                                        float *, fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, float *,
                                        float *, float *, int *, float *, float *,
                                        int *, float *, fb_complex_float_t *,
                                        float *, int *);
typedef void (*fb_zsysvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, int *, char *,
                                        double *, fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, double *,
                                        double *, double *, int *, double *, double *,
                                        int *, double *, fb_complex_double_t *,
                                        double *, int *);
typedef void (*fb_chesvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, int *, char *,
                                        float *, fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, float *,
                                        float *, float *, int *, float *, float *,
                                        int *, float *, fb_complex_float_t *,
                                        float *, int *);
typedef void (*fb_zhesvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, int *, char *,
                                        double *, fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, double *,
                                        double *, double *, int *, double *, double *,
                                        int *, double *, fb_complex_double_t *,
                                        double *, int *);

typedef void (*fb_sgesvxx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                        float *, int *, int *, char *, float *,
                                        float *, float *, int *, float *, int *, float *,
                                        float *, float *, int *, float *, float *,
                                        int *, float *, float *, int *, int *);
typedef void (*fb_dgesvxx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                        double *, int *, int *, char *, double *,
                                        double *, double *, int *, double *, int *,
                                        double *, double *, double *, int *, double *,
                                        double *, int *, double *, double *, int *,
                                        int *);
typedef void (*fb_cgesvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, int *, char *,
                                        float *, float *, fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, float *, float *,
                                        float *, int *, float *, float *, int *, float *,
                                        fb_complex_float_t *, float *, int *);
typedef void (*fb_zgesvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, int *, char *,
                                        double *, double *, fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, double *,
                                        double *, double *, int *, double *, double *,
                                        int *, double *, fb_complex_double_t *,
                                        double *, int *);

typedef void (*fb_sposvxx_fortran_fn_t)(char *, char *, int *, int *, float *, int *,
                                        float *, int *, char *, float *, float *,
                                        int *, float *, int *, float *, float *, float *,
                                        int *, float *, float *, int *, float *, float *,
                                        int *, int *);
typedef void (*fb_dposvxx_fortran_fn_t)(char *, char *, int *, int *, double *, int *,
                                        double *, int *, char *, double *, double *,
                                        int *, double *, int *, double *, double *,
                                        double *, int *, double *, double *, int *,
                                        double *, double *, int *, int *);
typedef void (*fb_cposvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, char *, float *,
                                        fb_complex_float_t *, int *,
                                        fb_complex_float_t *, int *, float *, float *,
                                        float *, int *, float *, float *, int *, float *,
                                        fb_complex_float_t *, float *, int *);
typedef void (*fb_zposvxx_fortran_fn_t)(char *, char *, int *, int *,
                                        fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, char *, double *,
                                        fb_complex_double_t *, int *,
                                        fb_complex_double_t *, int *, double *,
                                        double *, double *, int *, double *, double *,
                                        int *, double *, fb_complex_double_t *,
                                        double *, int *);

#define FB_DEFINE_SVX_REAL_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t,    \
                                        query_len_expr)                           \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int lwork = -1;                                                                \
    int info = 0;                                                                  \
    int n_ipiv = (n > 0) ? n : 1;                                                  \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    real_t rcond = (real_t)0;                                                      \
    scalar_t work_query = (scalar_t)0;                                             \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_ipiv,        \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_ipiv,          \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = NULL;                                                         \
    int *ipiv = (int *)malloc((size_t)n_ipiv * sizeof(int));                      \
    int *iwork = (int *)malloc((size_t)n_ipiv * sizeof(int));                     \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !iwork || !ferr ||      \
        !berr) {                                                                   \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(iwork); free(ferr); free(berr);                                       \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, &work_query, &lwork, iwork,    \
                               &info);                                             \
    if (info != 0) {                                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(iwork); free(ferr); free(berr);                                       \
        return info;                                                               \
    }                                                                              \
    lwork = (query_len_expr);                                                      \
    if (lwork < 1) {                                                               \
        lwork = (n > 0) ? n : 1;                                                   \
    }                                                                              \
    work = (scalar_t *)malloc((size_t)lwork * sizeof(scalar_t));                  \
    if (!work) {                                                                   \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(iwork); free(ferr); free(berr);                                       \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    memset(af_col, 0, (size_t)ldaf * (size_t)n_ipiv * sizeof(scalar_t));          \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    memset(x_col, 0, (size_t)ldx * (size_t)n_rhs * sizeof(scalar_t));             \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, work, &lwork, iwork, &info);   \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(iwork); free(ferr); free(berr);                              \
    return info;                                                                   \
}

#define FB_DEFINE_SVX_COMPLEX_FORTRAN_CALL(name, scalar_t, real_t,                \
                                           fortran_fn_t, query_len_expr)          \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int lwork = -1;                                                                \
    int info = 0;                                                                  \
    int n_ipiv = (n > 0) ? n : 1;                                                  \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    real_t rcond = (real_t)0;                                                      \
    scalar_t work_query = (scalar_t)0;                                             \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_ipiv,        \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_ipiv,          \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = NULL;                                                         \
    int *ipiv = (int *)malloc((size_t)n_ipiv * sizeof(int));                      \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !ferr || !berr) {      \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr);                                                    \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, &work_query, &lwork, &info);   \
    if (info != 0) {                                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr);                                                    \
        return info;                                                               \
    }                                                                              \
    lwork = (query_len_expr);                                                      \
    if (lwork < 1) {                                                               \
        lwork = (n > 0) ? n : 1;                                                   \
    }                                                                              \
    work = (scalar_t *)malloc((size_t)lwork * sizeof(scalar_t));                  \
    if (!work) {                                                                   \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr);                                                    \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    memset(af_col, 0, (size_t)ldaf * (size_t)n_ipiv * sizeof(scalar_t));          \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    memset(x_col, 0, (size_t)ldx * (size_t)n_rhs * sizeof(scalar_t));             \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, work, &lwork, &info);          \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(ferr); free(berr);                                            \
    return info;                                                                   \
}

#define FB_DEFINE_SVX_COMPLEX_RWORK_FORTRAN_CALL(name, scalar_t, real_t,          \
                                                 fortran_fn_t, query_len_expr,    \
                                                 rwork_len_expr)                  \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int lwork = -1;                                                                \
    int info = 0;                                                                  \
    int n_ipiv = (n > 0) ? n : 1;                                                  \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int rwork_len = (rwork_len_expr);                                              \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    real_t rcond = (real_t)0;                                                      \
    scalar_t work_query = (scalar_t)0;                                             \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_ipiv,        \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_ipiv,          \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = NULL;                                                         \
    int *ipiv = (int *)malloc((size_t)n_ipiv * sizeof(int));                      \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *rwork = (real_t *)calloc((size_t)((rwork_len > 0) ? rwork_len : 1),   \
                                     sizeof(real_t));                              \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !ferr || !berr ||       \
        !rwork) {                                                                  \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr); free(rwork);                                       \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, &work_query, &lwork, rwork,    \
                               &info);                                             \
    if (info != 0) {                                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr); free(rwork);                                       \
        return info;                                                               \
    }                                                                              \
    lwork = (query_len_expr);                                                      \
    if (lwork < 1) {                                                               \
        lwork = (n > 0) ? n : 1;                                                   \
    }                                                                              \
    work = (scalar_t *)malloc((size_t)lwork * sizeof(scalar_t));                  \
    if (!work) {                                                                   \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(ferr); free(berr); free(rwork);                                       \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    memset(af_col, 0, (size_t)ldaf * (size_t)n_ipiv * sizeof(scalar_t));          \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    memset(x_col, 0, (size_t)ldx * (size_t)n_rhs * sizeof(scalar_t));             \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, b_col, &ldb_col, x_col, &ldx, \
                               &rcond, ferr, berr, work, &lwork, rwork, &info);   \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(ferr); free(berr); free(rwork);                              \
    return info;                                                                   \
}

#define FB_DEFINE_SVXX_REAL_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t)    \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int info = 0;                                                                  \
    int n_ipiv = (n > 0) ? n : 1;                                                  \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (4 * n) : 1;                                          \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    char equed = 'N';                                                              \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_ipiv,        \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_ipiv,          \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *ipiv = (int *)malloc((size_t)n_ipiv * sizeof(int));                      \
    int *iwork = (int *)calloc((size_t)n_ipiv, sizeof(int));                      \
    real_t *s = (real_t *)calloc((size_t)n_ipiv, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));           \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));           \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !ipiv || !iwork ||      \
        !s || !berr || !err_norm || !err_comp) {                                   \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(ipiv); free(iwork); free(s); free(berr); free(err_norm);             \
        free(err_comp);                                                            \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, &equed, s, b_col, &ldb_col,   \
                               x_col, &ldx, &rcond, &rpvgrw, berr, &n_err_bnds,   \
                               err_norm, err_comp, &nparams, params, work, iwork, \
                               &info);                                             \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(iwork); free(s); free(berr); free(err_norm);                 \
    free(err_comp);                                                                \
    return info;                                                                   \
}

#define FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
[[maybe_unused]] static int name(fb_generic_fn fortran_fn, int n, int nrhs,       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int info = 0;                                                                  \
    int n_ipiv = (n > 0) ? n : 1;                                                  \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (8 * n) : 1;                                          \
    int rwork_len = (n > 0) ? (4 * n) : 1;                                         \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    char equed = 'N';                                                              \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_ipiv,        \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_ipiv,          \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *ipiv = (int *)malloc((size_t)n_ipiv * sizeof(int));                      \
    real_t *s = (real_t *)calloc((size_t)n_ipiv, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));           \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));           \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));          \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !ipiv || !s || !berr || \
        !err_norm || !err_comp || !rwork) {                                        \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(ipiv); free(s); free(berr); free(err_norm); free(err_comp);          \
        free(rwork);                                                               \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, ipiv, &equed, s, b_col, &ldb_col,   \
                               x_col, &ldx, &rcond, &rpvgrw, berr, &n_err_bnds,   \
                               err_norm, err_comp, &nparams, params, work, rwork, \
                               &info);                                             \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(s); free(berr); free(err_norm); free(err_comp);              \
    free(rwork);                                                                   \
    return info;                                                                   \
}

#define FB_DEFINE_GESVXX_REAL_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int n_order = (n > 0) ? n : 1;                                                 \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (8 * n) : 1;                                          \
    char fact = 'N';                                                               \
    char trans = 'N';                                                              \
    char equed = 'N';                                                              \
    int info = 0;                                                                  \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                      \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                      \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !ipiv || !iwork ||      \
        !r || !c || !berr || !err_norm || !err_comp) {                             \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(ipiv); free(iwork); free(r); free(c); free(berr); free(err_norm);    \
        free(err_comp);                                                            \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &lda_col,       \
                               af_col, &ldaf, ipiv, &equed, r, c, b_col,          \
                               &ldb_col, x_col, &ldx, &rcond, &rpvgrw, berr,      \
                               &n_err_bnds, err_norm, err_comp, &nparams, params, \
                               work, iwork, &info);                                \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(iwork); free(r); free(c); free(berr); free(err_norm);        \
    free(err_comp);                                                                \
    return info;                                                                   \
}

#define FB_DEFINE_GESVXX_COMPLEX_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int n_order = (n > 0) ? n : 1;                                                 \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (8 * n) : 1;                                          \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                         \
    char fact = 'N';                                                               \
    char trans = 'N';                                                              \
    char equed = 'N';                                                              \
    int info = 0;                                                                  \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                      \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));           \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !ipiv || !r || !c ||    \
        !berr || !err_norm || !err_comp || !rwork) {                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(ipiv); free(r); free(c); free(berr); free(err_norm); free(err_comp); \
        free(rwork);                                                               \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &lda_col,       \
                               af_col, &ldaf, ipiv, &equed, r, c, b_col,          \
                               &ldb_col, x_col, &ldx, &rcond, &rpvgrw, berr,      \
                               &n_err_bnds, err_norm, err_comp, &nparams, params, \
                               work, rwork, &info);                                \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(ipiv); free(r); free(c); free(berr); free(err_norm); free(err_comp);     \
    free(rwork);                                                                   \
    return info;                                                                   \
}

#define FB_DEFINE_POSVXX_REAL_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int n_order = (n > 0) ? n : 1;                                                 \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (8 * n) : 1;                                          \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    char equed = 'N';                                                              \
    int info = 0;                                                                  \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                      \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !iwork || !s || !berr ||\
        !err_norm || !err_comp) {                                                  \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(iwork); free(s); free(berr); free(err_norm); free(err_comp);         \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, &equed, s, b_col, &ldb_col,         \
                               x_col, &ldx, &rcond, &rpvgrw, berr, &n_err_bnds,   \
                               err_norm, err_comp, &nparams, params, work, iwork, \
                               &info);                                             \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(iwork); free(s); free(berr); free(err_norm); free(err_comp);             \
    return info;                                                                   \
}

#define FB_DEFINE_POSVXX_COMPLEX_FORTRAN_CALL(name, scalar_t, real_t, fortran_fn_t) \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                        \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,     \
                size_t b_bytes, scalar_t *x_out)                                   \
{                                                                                  \
    if (fortran_fn == NULL) {                                                      \
        return FB_BAND_CALL_NOT_IMPL;                                              \
    }                                                                              \
    int n_ = n;                                                                    \
    int nrhs_ = nrhs;                                                              \
    int lda_col = (n > 0) ? n : 1;                                                 \
    int ldaf = lda_col;                                                            \
    int ldb_col = (n > 0) ? n : 1;                                                 \
    int ldx = ldb_col;                                                             \
    int n_order = (n > 0) ? n : 1;                                                 \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                             \
    int n_err_bnds = 1;                                                            \
    int nparams = 0;                                                               \
    int work_len = (n > 0) ? (8 * n) : 1;                                          \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                         \
    char fact = 'N';                                                               \
    char uplo = 'L';                                                               \
    char equed = 'N';                                                              \
    int info = 0;                                                                  \
    real_t rcond = (real_t)0;                                                      \
    real_t rpvgrw = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                        \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                        \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                        \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));           \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                 \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));                \
    real_t *err_norm = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t *err_comp = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));            \
    real_t params[1] = {(real_t)0};                                                \
    memset(x_out, 0, b_bytes);                                                     \
    if (!a_col || !af_col || !b_col || !x_col || !work || !rwork || !s || !berr ||\
        !err_norm || !err_comp) {                                                  \
        free(a_col); free(af_col); free(b_col); free(x_col); free(work);          \
        free(rwork); free(s); free(berr); free(err_norm); free(err_comp);         \
        return FB_BAND_CALL_ALLOC_FAILURE;                                         \
    }                                                                              \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                            \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                         \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,        \
                               af_col, &ldaf, &equed, s, b_col, &ldb_col,         \
                               x_col, &ldx, &rcond, &rpvgrw, berr, &n_err_bnds,   \
                               err_norm, err_comp, &nparams, params, work, rwork, \
                               &info);                                             \
    if (info == 0) {                                                               \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                        \
    }                                                                              \
    free(a_col); free(af_col); free(b_col); free(x_col); free(work);              \
    free(rwork); free(s); free(berr); free(err_norm); free(err_comp);             \
    return info;                                                                   \
}

FB_DEFINE_SVX_REAL_FORTRAN_CALL(fb_call_ssysvx_backend, float, float,
                                fb_ssysvx_fortran_fn_t,
                                fb_solve_query_size_from_float(work_query))
FB_DEFINE_SVX_REAL_FORTRAN_CALL(fb_call_dsysvx_backend, double, double,
                                fb_dsysvx_fortran_fn_t,
                                fb_solve_query_size_from_double(work_query))
FB_DEFINE_SVX_COMPLEX_FORTRAN_CALL(fb_call_csysvx_backend, fb_complex_float_t,
                                   float, fb_csysvx_fortran_fn_t,
                                   fb_solve_query_size_from_float((float)__real__(work_query)))
FB_DEFINE_SVX_COMPLEX_RWORK_FORTRAN_CALL(
    fb_call_zsysvx_backend, fb_complex_double_t, double,
    fb_zsysvx_fortran_fn_t,
    fb_solve_query_size_from_double((double)__real__(work_query)),
    (n > 0) ? n : 1)
FB_DEFINE_SVX_COMPLEX_RWORK_FORTRAN_CALL(
    fb_call_chesvx_backend, fb_complex_float_t, float,
    fb_chesvx_fortran_fn_t,
    fb_solve_query_size_from_float((float)__real__(work_query)),
    (n > 0) ? n : 1)
FB_DEFINE_SVX_COMPLEX_RWORK_FORTRAN_CALL(
    fb_call_zhesvx_backend, fb_complex_double_t, double,
    fb_zhesvx_fortran_fn_t,
    fb_solve_query_size_from_double((double)__real__(work_query)),
    (n > 0) ? n : 1)

FB_DEFINE_SVXX_REAL_FORTRAN_CALL(fb_call_ssysvxx_backend, float, float,
                                 fb_ssysvxx_fortran_fn_t)
FB_DEFINE_SVXX_REAL_FORTRAN_CALL(fb_call_dsysvxx_backend, double, double,
                                 fb_dsysvxx_fortran_fn_t)
FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL(fb_call_csysvxx_backend, fb_complex_float_t,
                                    float, fb_csysvxx_fortran_fn_t)
FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL(fb_call_zsysvxx_backend, fb_complex_double_t,
                                    double, fb_zsysvxx_fortran_fn_t)
FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL(fb_call_chesvxx_backend, fb_complex_float_t,
                                    float, fb_chesvxx_fortran_fn_t)
FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL(fb_call_zhesvxx_backend, fb_complex_double_t,
                                    double, fb_zhesvxx_fortran_fn_t)
FB_DEFINE_GESVXX_REAL_FORTRAN_CALL(fb_call_sgesvxx_backend, float, float,
                                   fb_sgesvxx_fortran_fn_t)
FB_DEFINE_GESVXX_REAL_FORTRAN_CALL(fb_call_dgesvxx_backend, double, double,
                                   fb_dgesvxx_fortran_fn_t)
FB_DEFINE_GESVXX_COMPLEX_FORTRAN_CALL(fb_call_cgesvxx_backend,
                                      fb_complex_float_t, float,
                                      fb_cgesvxx_fortran_fn_t)
FB_DEFINE_GESVXX_COMPLEX_FORTRAN_CALL(fb_call_zgesvxx_backend,
                                      fb_complex_double_t, double,
                                      fb_zgesvxx_fortran_fn_t)
FB_DEFINE_POSVXX_REAL_FORTRAN_CALL(fb_call_sposvxx_backend, float, float,
                                   fb_sposvxx_fortran_fn_t)
FB_DEFINE_POSVXX_REAL_FORTRAN_CALL(fb_call_dposvxx_backend, double, double,
                                   fb_dposvxx_fortran_fn_t)
FB_DEFINE_POSVXX_COMPLEX_FORTRAN_CALL(fb_call_cposvxx_backend,
                                      fb_complex_float_t, float,
                                      fb_cposvxx_fortran_fn_t)
FB_DEFINE_POSVXX_COMPLEX_FORTRAN_CALL(fb_call_zposvxx_backend,
                                      fb_complex_double_t, double,
                                      fb_zposvxx_fortran_fn_t)

#undef FB_DEFINE_SVX_REAL_FORTRAN_CALL
#undef FB_DEFINE_SVX_COMPLEX_FORTRAN_CALL
#undef FB_DEFINE_SVX_COMPLEX_RWORK_FORTRAN_CALL
#undef FB_DEFINE_SVXX_REAL_FORTRAN_CALL
#undef FB_DEFINE_SVXX_COMPLEX_FORTRAN_CALL
#undef FB_DEFINE_GESVXX_REAL_FORTRAN_CALL
#undef FB_DEFINE_GESVXX_COMPLEX_FORTRAN_CALL
#undef FB_DEFINE_POSVXX_REAL_FORTRAN_CALL
#undef FB_DEFINE_POSVXX_COMPLEX_FORTRAN_CALL

#define FB_DEFINE_DENSE_EXPERT_RUNNER(name, op_id, scalar_t, build_fn,            \
                                      call_fn, bwerr_fn)                          \
static fb_judge_status_t name(const fb_backend_vtable_t *oracle,                  \
                              const fb_backend_vtable_t *cand,                    \
                              const fb_corpus_case_t *tc,                         \
                              fb_judge_solve_result_t *res,                       \
                              uint64_t *ns_out)                                   \
{                                                                                  \
    fb_generic_fn oracle_fortran = oracle->ext_ops[(op_id)][FB_CONV_FORTRAN];     \
    fb_generic_fn cand_fortran = cand->ext_ops[(op_id)][FB_CONV_FORTRAN];         \
    int info = 0;                                                                  \
    if (oracle_fortran == NULL || cand_fortran == NULL) {                         \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    int n = (int)tc->n;                                                            \
    int nrhs = (int)tc->k;                                                         \
    int lda = (int)tc->lda;                                                        \
    int ldb = (int)tc->ldb;                                                        \
    const scalar_t *b0 = (const scalar_t *)tc->B;                                  \
    if (n <= 0 || nrhs <= 0 || !b0) {                                              \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    size_t Bsz = tc->B_elems * sizeof(scalar_t);                                   \
    scalar_t *Aorig = (scalar_t *)calloc((size_t)n * (size_t)lda,                 \
                                         sizeof(scalar_t));                        \
    scalar_t *Bo = (scalar_t *)malloc(Bsz);                                        \
    if (!Aorig || !Bo) {                                                           \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    build_fn(Aorig, n, lda);                                                       \
    info = call_fn(oracle_fortran, n, nrhs, Aorig, lda, b0, ldb, Bsz, Bo);       \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    scalar_t *Bc = (scalar_t *)malloc(Bsz);                                        \
    if (!Bc) {                                                                     \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (ns_out) {                                                                  \
        uint64_t best = UINT64_MAX;                                                \
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; ++w) {                           \
            info = call_fn(cand_fortran, n, nrhs, Aorig, lda, b0, ldb, Bsz, Bc); \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(Aorig);                                                       \
                free(Bo);                                                          \
                free(Bc);                                                          \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
        }                                                                          \
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; ++t) {                           \
            uint64_t t0 = fb_judge_time_ns();                                      \
            info = call_fn(cand_fortran, n, nrhs, Aorig, lda, b0, ldb, Bsz, Bc); \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(Aorig);                                                       \
                free(Bo);                                                          \
                free(Bc);                                                          \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
            uint64_t dt = fb_judge_time_ns() - t0;                                 \
            if (dt < best) {                                                       \
                best = dt;                                                         \
            }                                                                      \
        }                                                                          \
        *ns_out = best;                                                            \
    }                                                                              \
    info = call_fn(cand_fortran, n, nrhs, Aorig, lda, b0, ldb, Bsz, Bc);         \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        free(Bc);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        free(Bc);                                                                  \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(Aorig);                                                               \
        free(Bo);                                                                  \
        free(Bc);                                                                  \
        mark_ca_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    res->residual =                                                                \
        make_result(bwerr_fn(Aorig, n, n, lda, b0, ldb, Bc, ldb, nrhs));          \
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};                  \
    res->kappa_estimate = (double)(n * 10);                                        \
    free(Aorig);                                                                   \
    free(Bo);                                                                      \
    free(Bc);                                                                      \
    return FB_JUDGE_OK;                                                            \
}

FB_DEFINE_DENSE_EXPERT_RUNNER(run_ssysvx, FB_OP_SSYSVX, float,
                              fb_build_dense_symm_f32,
                              fb_call_ssysvx_backend, bwerr_f32)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_dsysvx, FB_OP_DSYSVX, double,
                              fb_build_dense_symm_f64,
                              fb_call_dsysvx_backend, bwerr_f64)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_csysvx, FB_OP_CSYSVX, fb_complex_float_t,
                              fb_build_dense_symm_cf32,
                              fb_call_csysvx_backend, bwerr_cf32)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_zsysvx, FB_OP_ZSYSVX, fb_complex_double_t,
                              fb_build_dense_symm_cf64,
                              fb_call_zsysvx_backend, bwerr_cf64)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_chesvx, FB_OP_CHESVX, fb_complex_float_t,
                              fb_build_dense_herm_cf32,
                              fb_call_chesvx_backend, bwerr_cf32)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_zhesvx, FB_OP_ZHESVX, fb_complex_double_t,
                              fb_build_dense_herm_cf64,
                              fb_call_zhesvx_backend, bwerr_cf64)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_ssysvxx, FB_OP_SSYSVXX, float,
                              fb_build_dense_symm_f32,
                              fb_call_ssysvxx_backend, bwerr_f32)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_dsysvxx, FB_OP_DSYSVXX, double,
                              fb_build_dense_symm_f64,
                              fb_call_dsysvxx_backend, bwerr_f64)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_csysvxx, FB_OP_CSYSVXX, fb_complex_float_t,
                              fb_build_dense_symm_cf32,
                              fb_call_csysvxx_backend, bwerr_cf32)
FB_DEFINE_DENSE_EXPERT_RUNNER(run_zsysvxx, FB_OP_ZSYSVXX, fb_complex_double_t,
                              fb_build_dense_symm_cf64,
                              fb_call_zsysvxx_backend, bwerr_cf64)

static fb_judge_status_t run_chesvxx(const fb_backend_vtable_t *oracle,
                                     const fb_backend_vtable_t *cand,
                                     const fb_corpus_case_t *tc,
                                     fb_judge_solve_result_t *res,
                                     uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_CHESVXX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_CHESVXX][FB_CONV_FORTRAN];
    fb_chesvxx_fortran_fn_t oracle_fn = (fb_chesvxx_fortran_fn_t)oracle_fortran;
    fb_chesvxx_fortran_fn_t cand_fn = (fb_chesvxx_fortran_fn_t)cand_fortran;

    (void)tc;
    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }

    char fact = 'N';
    char uplo = 'L';
    char equed = '?';
    int n = 1;
    int nrhs = 1;
    int lda = 1;
    int ldaf = 1;
    int ldb = 1;
    int ldx = 1;
    int n_err_bnds = 1;
    int nparams = 0;
    int info = -1;
    fb_complex_float_t a[1] = {0};
    fb_complex_float_t af[1] = {0};
    int ipiv[1] = {0};
    float s[1] = {1.0f};
    fb_complex_float_t b[1] = {0};
    fb_complex_float_t x[1] = {0};
    float rcond = 0.0f;
    float rpvgrw = 0.0f;
    float berr[1] = {0.0f};
    float err_bnds_norm[1] = {0.0f};
    float err_bnds_comp[1] = {0.0f};
    float params[1] = {0.0f};
    fb_complex_float_t work[8] = {0};
    float rwork[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __real__(a[0]) = 2.0f;
    __imag__(a[0]) = 0.0f;
    __real__(b[0]) = 10.0f;
    __imag__(b[0]) = 2.0f;

    oracle_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
              (fb_complex_float_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
              &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
              work, rwork, &info);
    if (info != 0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fact = 'N';
        uplo = 'L';
        equed = '?';
        info = -1;
        __real__(af[0]) = 0.0f;
        __imag__(af[0]) = 0.0f;
        ipiv[0] = 0;
        s[0] = 1.0f;
        __real__(x[0]) = 0.0f;
        __imag__(x[0]) = 0.0f;
        rcond = 0.0f;
        rpvgrw = 0.0f;
        berr[0] = 0.0f;
        err_bnds_norm[0] = 0.0f;
        err_bnds_comp[0] = 0.0f;
        __real__(work[0]) = 0.0f;
        __imag__(work[0]) = 0.0f;
        rwork[0] = rwork[1] = rwork[2] = rwork[3] = 0.0f;
        uint64_t t0 = fb_judge_time_ns();
        cand_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
                (fb_complex_float_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
                &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
                work, rwork, &info);
        *ns_out = fb_judge_time_ns() - t0;
    }

    fact = 'N';
    uplo = 'L';
    equed = '?';
    info = -1;
    __real__(af[0]) = 0.0f;
    __imag__(af[0]) = 0.0f;
    ipiv[0] = 0;
    s[0] = 1.0f;
    __real__(x[0]) = 0.0f;
    __imag__(x[0]) = 0.0f;
    rcond = 0.0f;
    rpvgrw = 0.0f;
    berr[0] = 0.0f;
    err_bnds_norm[0] = 0.0f;
    err_bnds_comp[0] = 0.0f;
    __real__(work[0]) = 0.0f;
    __imag__(work[0]) = 0.0f;
    rwork[0] = rwork[1] = rwork[2] = rwork[3] = 0.0f;
    cand_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
            (fb_complex_float_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
            &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
            work, rwork, &info);
    if (info != 0) {
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }

    res->residual = make_result(bwerr_cf32(a, 1, 1, 1, b, 1, x, 1, 1));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = 10.0;
    return FB_JUDGE_OK;
}

static fb_judge_status_t run_zhesvxx(const fb_backend_vtable_t *oracle,
                                     const fb_backend_vtable_t *cand,
                                     const fb_corpus_case_t *tc,
                                     fb_judge_solve_result_t *res,
                                     uint64_t *ns_out)
{
    fb_generic_fn oracle_fortran = oracle->ext_ops[FB_OP_ZHESVXX][FB_CONV_FORTRAN];
    fb_generic_fn cand_fortran = cand->ext_ops[FB_OP_ZHESVXX][FB_CONV_FORTRAN];
    fb_zhesvxx_fortran_fn_t oracle_fn = (fb_zhesvxx_fortran_fn_t)oracle_fortran;
    fb_zhesvxx_fortran_fn_t cand_fn = (fb_zhesvxx_fortran_fn_t)cand_fortran;

    (void)tc;
    if (oracle_fortran == NULL || cand_fortran == NULL) {
        return FB_JUDGE_ERR_NOT_IMPL;
    }

    char fact = 'N';
    char uplo = 'L';
    char equed = '?';
    int n = 1;
    int nrhs = 1;
    int lda = 1;
    int ldaf = 1;
    int ldb = 1;
    int ldx = 1;
    int n_err_bnds = 1;
    int nparams = 0;
    int info = -1;
    fb_complex_double_t a[1] = {0};
    fb_complex_double_t af[1] = {0};
    int ipiv[1] = {0};
    double s[1] = {1.0};
    fb_complex_double_t b[1] = {0};
    fb_complex_double_t x[1] = {0};
    double rcond = 0.0;
    double rpvgrw = 0.0;
    double berr[1] = {0.0};
    double err_bnds_norm[1] = {0.0};
    double err_bnds_comp[1] = {0.0};
    double params[1] = {0.0};
    fb_complex_double_t work[8] = {0};
    double rwork[4] = {0.0, 0.0, 0.0, 0.0};

    __real__(a[0]) = 2.0;
    __imag__(a[0]) = 0.0;
    __real__(b[0]) = 10.0;
    __imag__(b[0]) = 2.0;

    oracle_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
              (fb_complex_double_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
              &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
              work, rwork, &info);
    if (info != 0) {
        mark_oc_fatal(res);
        return FB_JUDGE_OK;
    }

    if (ns_out) {
        fact = 'N';
        uplo = 'L';
        equed = '?';
        info = -1;
        __real__(af[0]) = 0.0;
        __imag__(af[0]) = 0.0;
        ipiv[0] = 0;
        s[0] = 1.0;
        __real__(x[0]) = 0.0;
        __imag__(x[0]) = 0.0;
        rcond = 0.0;
        rpvgrw = 0.0;
        berr[0] = 0.0;
        err_bnds_norm[0] = 0.0;
        err_bnds_comp[0] = 0.0;
        __real__(work[0]) = 0.0;
        __imag__(work[0]) = 0.0;
        rwork[0] = rwork[1] = rwork[2] = rwork[3] = 0.0;
        uint64_t t0 = fb_judge_time_ns();
        cand_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
                (fb_complex_double_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
                &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
                work, rwork, &info);
        *ns_out = fb_judge_time_ns() - t0;
    }

    fact = 'N';
    uplo = 'L';
    equed = '?';
    info = -1;
    __real__(af[0]) = 0.0;
    __imag__(af[0]) = 0.0;
    ipiv[0] = 0;
    s[0] = 1.0;
    __real__(x[0]) = 0.0;
    __imag__(x[0]) = 0.0;
    rcond = 0.0;
    rpvgrw = 0.0;
    berr[0] = 0.0;
    err_bnds_norm[0] = 0.0;
    err_bnds_comp[0] = 0.0;
    __real__(work[0]) = 0.0;
    __imag__(work[0]) = 0.0;
    rwork[0] = rwork[1] = rwork[2] = rwork[3] = 0.0;
    cand_fn(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, s,
            (fb_complex_double_t *)b, &ldb, x, &ldx, &rcond, &rpvgrw, berr,
            &n_err_bnds, err_bnds_norm, err_bnds_comp, &nparams, params,
            work, rwork, &info);
    if (info != 0) {
        mark_ca_fatal(res);
        return FB_JUDGE_OK;
    }

    res->residual = make_result(bwerr_cf64(a, 1, 1, 1, b, 1, x, 1, 1));
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};
    res->kappa_estimate = 10.0;
    return FB_JUDGE_OK;
}

#define FB_DEFINE_PGESV_CALLER(name, fortran_fn_t, scalar_t)                     \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int ia = 1, ja = 1, ib = 1, jb = 1, info = 0;                                \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int desca[9] = {0};                                                           \
    int descb[9] = {0};                                                           \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !b_col || !ipiv) {                                              \
        free(a_col);                                                              \
        free(b_col);                                                              \
        free(ipiv);                                                               \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&n_, &nrhs_, a_col, &ia, &ja, desca, ipiv,        \
                               b_col, &ib, &jb, descb, &info);                   \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, b_col, ldb_col, n, nrhs);                   \
    }                                                                             \
    free(a_col);                                                                  \
    free(b_col);                                                                  \
    free(ipiv);                                                                   \
    return info;                                                                  \
}

#define FB_DEFINE_PPOSV_CALLER(name, fortran_fn_t, scalar_t)                     \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int ia = 1, ja = 1, ib = 1, jb = 1, info = 0;                                \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int desca[9] = {0};                                                           \
    int descb[9] = {0};                                                           \
    char uplo = 'L';                                                              \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !b_col) {                                                       \
        free(a_col);                                                              \
        free(b_col);                                                              \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&uplo, &n_, &nrhs_, a_col, &ia, &ja, desca,       \
                               b_col, &ib, &jb, descb, &info);                   \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, b_col, ldb_col, n, nrhs);                   \
    }                                                                             \
    free(a_col);                                                                  \
    free(b_col);                                                                  \
    return info;                                                                  \
}

#define FB_DEFINE_GESVX_REAL_CALLER(name, fortran_fn_t, scalar_t, real_t)        \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int info = 0;                                                                 \
    char fact = 'N';                                                              \
    char trans = 'N';                                                             \
    char equed = 'N';                                                             \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                     \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !r || !c || !ferr ||   \
        !berr || !work || !iwork) {                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(r); free(c); free(ferr); free(berr); free(work); free(iwork);        \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &lda_col,      \
                               af_col, &ldaf, ipiv, &equed, r, c, b_col,         \
                               &ldb_col, x_col, &ldx, &rcond, ferr, berr, work,  \
                               iwork, &info);                                     \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);             \
    free(r); free(c); free(ferr); free(berr); free(work); free(iwork);           \
    return info;                                                                  \
}

#define FB_DEFINE_GESVX_COMPLEX_CALLER(name, fortran_fn_t, scalar_t, real_t)     \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                        \
    int info = 0;                                                                 \
    char fact = 'N';                                                              \
    char trans = 'N';                                                             \
    char equed = 'N';                                                             \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));          \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !r || !c || !ferr ||   \
        !berr || !work || !rwork) {                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(r); free(c); free(ferr); free(berr); free(work); free(rwork);        \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &lda_col,      \
                               af_col, &ldaf, ipiv, &equed, r, c, b_col,         \
                               &ldb_col, x_col, &ldx, &rcond, ferr, berr, work,  \
                               rwork, &info);                                     \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);             \
    free(r); free(c); free(ferr); free(berr); free(work); free(rwork);           \
    return info;                                                                  \
}

#define FB_DEFINE_POSVX_REAL_CALLER(name, fortran_fn_t, scalar_t, real_t)        \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int info = 0;                                                                 \
    char fact = 'N';                                                              \
    char uplo = 'L';                                                              \
    char equed = 'N';                                                             \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                     \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !s || !ferr || !berr || !work ||\
        !iwork) {                                                                 \
        free(a_col); free(af_col); free(b_col); free(x_col); free(s);             \
        free(ferr); free(berr); free(work); free(iwork);                          \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,       \
                               af_col, &ldaf, &equed, s, b_col, &ldb_col,        \
                               x_col, &ldx, &rcond, ferr, berr, work, iwork,     \
                               &info);                                            \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(s);                \
    free(ferr); free(berr); free(work); free(iwork);                              \
    return info;                                                                  \
}

#define FB_DEFINE_POSVX_COMPLEX_CALLER(name, fortran_fn_t, scalar_t, real_t)     \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n;                                                                   \
    int nrhs_ = nrhs;                                                             \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                        \
    int info = 0;                                                                 \
    char fact = 'N';                                                              \
    char uplo = 'L';                                                              \
    char equed = 'N';                                                             \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));          \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !s || !ferr || !berr || !work ||\
        !rwork) {                                                                 \
        free(a_col); free(af_col); free(b_col); free(x_col); free(s);             \
        free(ferr); free(berr); free(work); free(rwork);                          \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &lda_col,       \
                               af_col, &ldaf, &equed, s, b_col, &ldb_col,        \
                               x_col, &ldx, &rcond, ferr, berr, work, rwork,     \
                               &info);                                            \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(s);                \
    free(ferr); free(berr); free(work); free(rwork);                              \
    return info;                                                                  \
}

#define FB_DEFINE_PGESVX_REAL_CALLER(name, fortran_fn_t, scalar_t, real_t)       \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n, nrhs_ = nrhs;                                                     \
    int ia = 1, ja = 1, iaf = 1, jaf = 1;                                         \
    int ib = 1, jb = 1, ix = 1, jx = 1, info = 0;                                 \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int desca[9] = {0}, descaf[9] = {0}, descb[9] = {0}, descx[9] = {0};         \
    char fact = 'N', trans = 'N', equed = 'N';                                    \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *iwork = (int *)calloc((size_t)n_order, sizeof(int));                     \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !r || !c || !ferr ||   \
        !berr || !work || !iwork) {                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(r); free(c); free(ferr); free(berr); free(work); free(iwork);        \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &ia, &ja,      \
                               desca, af_col, &iaf, &jaf, descaf, ipiv, &equed,  \
                               r, c, b_col, &ib, &jb, descb, x_col, &ix, &jx,    \
                               descx, &rcond, ferr, berr, work, iwork, &info);   \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);             \
    free(r); free(c); free(ferr); free(berr); free(work); free(iwork);           \
    return info;                                                                  \
}

#define FB_DEFINE_PGESVX_COMPLEX_CALLER(name, fortran_fn_t, scalar_t, real_t)    \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n, nrhs_ = nrhs;                                                     \
    int ia = 1, ja = 1, iaf = 1, jaf = 1;                                         \
    int ib = 1, jb = 1, ix = 1, jx = 1, info = 0;                                 \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                        \
    int desca[9] = {0}, descaf[9] = {0}, descb[9] = {0}, descx[9] = {0};         \
    char fact = 'N', trans = 'N', equed = 'N';                                    \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    int *ipiv = (int *)malloc((size_t)n_order * sizeof(int));                     \
    real_t *r = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *c = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));          \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !ipiv || !r || !c || !ferr ||   \
        !berr || !work || !rwork) {                                               \
        free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);          \
        free(r); free(c); free(ferr); free(berr); free(work); free(rwork);        \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &trans, &n_, &nrhs_, a_col, &ia, &ja,      \
                               desca, af_col, &iaf, &jaf, descaf, ipiv, &equed,  \
                               r, c, b_col, &ib, &jb, descb, x_col, &ix, &jx,    \
                               descx, &rcond, ferr, berr, work, rwork, &info);   \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(ipiv);             \
    free(r); free(c); free(ferr); free(berr); free(work); free(rwork);           \
    return info;                                                                  \
}

#define FB_DEFINE_PPOSVX_REAL_CALLER(name, fortran_fn_t, scalar_t, real_t)       \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n, nrhs_ = nrhs;                                                     \
    int ia = 1, ja = 1, iaf = 1, jaf = 1;                                         \
    int ib = 1, jb = 1, ix = 1, jx = 1, info = 0;                                 \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int iwork_len = (n > 0) ? n : 1;                                              \
    int lwork = work_len, liwork = iwork_len;                                     \
    int desca[9] = {0}, descaf[9] = {0}, descb[9] = {0}, descx[9] = {0};         \
    char fact = 'N', uplo = 'L', equed = 'N';                                     \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    int *iwork = (int *)calloc((size_t)iwork_len, sizeof(int));                   \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !s || !ferr || !berr || !work ||\
        !iwork) {                                                                 \
        free(a_col); free(af_col); free(b_col); free(x_col); free(s);             \
        free(ferr); free(berr); free(work); free(iwork);                          \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &ia, &ja,       \
                               desca, af_col, &iaf, &jaf, descaf, &equed, s,     \
                               b_col, &ib, &jb, descb, x_col, &ix, &jx, descx,   \
                               &rcond, ferr, berr, work, &lwork, iwork, &liwork, \
                               &info);                                            \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(s);                \
    free(ferr); free(berr); free(work); free(iwork);                              \
    return info;                                                                  \
}

#define FB_DEFINE_PPOSVX_COMPLEX_CALLER(name, fortran_fn_t, scalar_t, real_t)    \
static int name(fb_generic_fn fortran_fn, int n, int nrhs,                       \
                const scalar_t *a_in, int lda, const scalar_t *b_in, int ldb,    \
                size_t b_bytes, scalar_t *x_out)                                  \
{                                                                                 \
    if (fortran_fn == NULL) {                                                     \
        return FB_BAND_CALL_NOT_IMPL;                                             \
    }                                                                             \
    int n_ = n, nrhs_ = nrhs;                                                     \
    int ia = 1, ja = 1, iaf = 1, jaf = 1;                                         \
    int ib = 1, jb = 1, ix = 1, jx = 1, info = 0;                                 \
    int lda_col = (n > 0) ? n : 1;                                                \
    int ldaf = lda_col;                                                           \
    int ldb_col = (n > 0) ? n : 1;                                                \
    int ldx = ldb_col;                                                            \
    int n_order = (n > 0) ? n : 1;                                                \
    int n_rhs = (nrhs > 0) ? nrhs : 1;                                            \
    int work_len = (n > 0) ? (8 * n) : 1;                                         \
    int rwork_len = (n > 0) ? (8 * n) : 1;                                        \
    int lwork = work_len, lrwork = rwork_len;                                     \
    int desca[9] = {0}, descaf[9] = {0}, descb[9] = {0}, descx[9] = {0};         \
    char fact = 'N', uplo = 'L', equed = 'N';                                     \
    real_t rcond = (real_t)0;                                                     \
    scalar_t *a_col = (scalar_t *)calloc((size_t)lda_col * (size_t)n_order,       \
                                         sizeof(scalar_t));                       \
    scalar_t *af_col = (scalar_t *)calloc((size_t)ldaf * (size_t)n_order,         \
                                          sizeof(scalar_t));                      \
    scalar_t *b_col = (scalar_t *)calloc((size_t)ldb_col * (size_t)n_rhs,         \
                                         sizeof(scalar_t));                       \
    scalar_t *x_col = (scalar_t *)calloc((size_t)ldx * (size_t)n_rhs,             \
                                         sizeof(scalar_t));                       \
    real_t *s = (real_t *)calloc((size_t)n_order, sizeof(real_t));                \
    real_t *ferr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    real_t *berr = (real_t *)calloc((size_t)n_rhs, sizeof(real_t));               \
    scalar_t *work = (scalar_t *)calloc((size_t)work_len, sizeof(scalar_t));      \
    real_t *rwork = (real_t *)calloc((size_t)rwork_len, sizeof(real_t));          \
    memset(x_out, 0, b_bytes);                                                    \
    if (!a_col || !af_col || !b_col || !x_col || !s || !ferr || !berr || !work ||\
        !rwork) {                                                                 \
        free(a_col); free(af_col); free(b_col); free(x_col); free(s);             \
        free(ferr); free(berr); free(work); free(rwork);                          \
        return FB_BAND_CALL_ALLOC_FAILURE;                                        \
    }                                                                             \
    FB_COPY_RM_TO_CM(a_col, lda_col, a_in, lda, n, n);                           \
    FB_COPY_RM_TO_CM(b_col, ldb_col, b_in, ldb, n, nrhs);                        \
    ((fortran_fn_t)fortran_fn)(&fact, &uplo, &n_, &nrhs_, a_col, &ia, &ja,       \
                               desca, af_col, &iaf, &jaf, descaf, &equed, s,     \
                               b_col, &ib, &jb, descb, x_col, &ix, &jx, descx,   \
                               &rcond, ferr, berr, work, &lwork, rwork, &lrwork, \
                               &info);                                            \
    if (info == 0) {                                                              \
        FB_COPY_CM_TO_RM(x_out, ldb, x_col, ldx, n, nrhs);                       \
    }                                                                             \
    free(a_col); free(af_col); free(b_col); free(x_col); free(s);                \
    free(ferr); free(berr); free(work); free(rwork);                              \
    return info;                                                                  \
}

#define FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(name, op_id, call_fn, scalar_t,   \
                                               bwerr_fn)                          \
static fb_judge_status_t name(                                                    \
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,           \
    const fb_corpus_case_t *tc, fb_judge_solve_result_t *res,                     \
    uint64_t *ns_out)                                                              \
{                                                                                 \
    fb_generic_fn oracle_fortran = oracle->ext_ops[op_id][FB_CONV_FORTRAN];      \
    fb_generic_fn cand_fortran = cand->ext_ops[op_id][FB_CONV_FORTRAN];          \
    int n = (int)tc->n, nrhs = (int)tc->k;                                        \
    int lda = (int)tc->lda, ldb = (int)tc->ldb;                                   \
    const scalar_t *A0 =                                                           \
        (const scalar_t *)(tc->A_snapshot ? tc->A_snapshot : tc->A);              \
    const scalar_t *b0 = (const scalar_t *)tc->B;                                 \
    int info = 0;                                                                  \
    if (oracle_fortran == NULL || cand_fortran == NULL) {                         \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (n <= 0 || nrhs <= 0 || !A0 || !b0) {                                      \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    size_t Bsz = tc->B_elems * sizeof(scalar_t);                                   \
    scalar_t *Bo = (scalar_t *)malloc(Bsz);                                        \
    if (!Bo) {                                                                     \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    info = call_fn(oracle_fortran, n, nrhs, A0, lda, b0, ldb, Bsz, Bo);          \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(Bo);                                                                  \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(Bo);                                                                  \
        mark_oc_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    free(Bo);                                                                      \
    scalar_t *Bc = (scalar_t *)malloc(Bsz);                                        \
    if (!Bc) {                                                                     \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (ns_out) {                                                                  \
        uint64_t best = UINT64_MAX;                                                \
        for (int w = 0; w < FB_SOLVE_WARMUP_RUNS; w++) {                          \
            info = call_fn(cand_fortran, n, nrhs, A0, lda, b0, ldb, Bsz, Bc);    \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(Bc);                                                          \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
        }                                                                          \
        for (int t = 0; t < FB_SOLVE_TIMING_RUNS; t++) {                          \
            uint64_t t0 = fb_judge_time_ns();                                      \
            info = call_fn(cand_fortran, n, nrhs, A0, lda, b0, ldb, Bsz, Bc);    \
            if (info == FB_BAND_CALL_ALLOC_FAILURE) {                              \
                free(Bc);                                                          \
                return FB_JUDGE_ERR_ALLOC;                                         \
            }                                                                      \
            uint64_t dt = fb_judge_time_ns() - t0;                                 \
            if (dt < best) {                                                       \
                best = dt;                                                         \
            }                                                                      \
        }                                                                          \
        *ns_out = best;                                                            \
    }                                                                              \
    info = call_fn(cand_fortran, n, nrhs, A0, lda, b0, ldb, Bsz, Bc);            \
    if (info == FB_BAND_CALL_ALLOC_FAILURE) {                                      \
        free(Bc);                                                                  \
        return FB_JUDGE_ERR_ALLOC;                                                 \
    }                                                                              \
    if (info == FB_BAND_CALL_NOT_IMPL) {                                           \
        free(Bc);                                                                  \
        return FB_JUDGE_ERR_NOT_IMPL;                                              \
    }                                                                              \
    if (info != 0) {                                                               \
        free(Bc);                                                                  \
        mark_ca_fatal(res);                                                        \
        return FB_JUDGE_OK;                                                        \
    }                                                                              \
    res->residual = make_result(bwerr_fn(A0, n, n, lda, b0, ldb, Bc, ldb, nrhs)); \
    res->orthogonality = (fb_judge_case_result_t){.digits = 16};                  \
    res->kappa_estimate = (double)(n * 10);                                        \
    free(Bc);                                                                      \
    return FB_JUDGE_OK;                                                            \
}

FB_DEFINE_PGESV_CALLER(fb_call_psgesv, fb_psgesv_fortran_fn_t, float)
FB_DEFINE_PGESV_CALLER(fb_call_pdgesv, fb_pdgesv_fortran_fn_t, double)
FB_DEFINE_PGESV_CALLER(fb_call_pcgesv, fb_pcgesv_fortran_fn_t, fb_complex_float_t)
FB_DEFINE_PGESV_CALLER(fb_call_pzgesv, fb_pzgesv_fortran_fn_t, fb_complex_double_t)
FB_DEFINE_GESV_CALLER(fb_call_sgesv_backend, fb_sgesv_fortran_fn_t, float)
FB_DEFINE_GESV_CALLER(fb_call_dgesv_backend, fb_dgesv_fortran_fn_t, double)
FB_DEFINE_GESV_CALLER(fb_call_cgesv_backend, fb_cgesv_fortran_fn_t,
                      fb_complex_float_t)
FB_DEFINE_GESV_CALLER(fb_call_zgesv_backend, fb_zgesv_fortran_fn_t,
                      fb_complex_double_t)

FB_DEFINE_PPOSV_CALLER(fb_call_psposv, fb_psposv_fortran_fn_t, float)
FB_DEFINE_PPOSV_CALLER(fb_call_pdposv, fb_pdposv_fortran_fn_t, double)
FB_DEFINE_PPOSV_CALLER(fb_call_pcposv, fb_pcposv_fortran_fn_t, fb_complex_float_t)
FB_DEFINE_PPOSV_CALLER(fb_call_pzposv, fb_pzposv_fortran_fn_t, fb_complex_double_t)

FB_DEFINE_GESVX_REAL_CALLER(fb_call_sgesvx, fb_sgesvx_fortran_fn_t, float, float)
FB_DEFINE_GESVX_REAL_CALLER(fb_call_dgesvx, fb_dgesvx_fortran_fn_t, double, double)
FB_DEFINE_GESVX_COMPLEX_CALLER(fb_call_cgesvx, fb_cgesvx_fortran_fn_t,
                               fb_complex_float_t, float)
FB_DEFINE_GESVX_COMPLEX_CALLER(fb_call_zgesvx, fb_zgesvx_fortran_fn_t,
                               fb_complex_double_t, double)

FB_DEFINE_POSVX_REAL_CALLER(fb_call_sposvx, fb_sposvx_fortran_fn_t, float, float)
FB_DEFINE_POSVX_REAL_CALLER(fb_call_dposvx, fb_dposvx_fortran_fn_t, double, double)
FB_DEFINE_POSVX_COMPLEX_CALLER(fb_call_cposvx, fb_cposvx_fortran_fn_t,
                               fb_complex_float_t, float)
FB_DEFINE_POSVX_COMPLEX_CALLER(fb_call_zposvx, fb_zposvx_fortran_fn_t,
                               fb_complex_double_t, double)

FB_DEFINE_PGESVX_REAL_CALLER(fb_call_psgesvx, fb_psgesvx_fortran_fn_t, float, float)
FB_DEFINE_PGESVX_REAL_CALLER(fb_call_pdgesvx, fb_pdgesvx_fortran_fn_t, double, double)
FB_DEFINE_PGESVX_COMPLEX_CALLER(fb_call_pcgesvx, fb_pcgesvx_fortran_fn_t,
                                fb_complex_float_t, float)
FB_DEFINE_PGESVX_COMPLEX_CALLER(fb_call_pzgesvx, fb_pzgesvx_fortran_fn_t,
                                fb_complex_double_t, double)

FB_DEFINE_PPOSVX_REAL_CALLER(fb_call_psposvx, fb_psposvx_fortran_fn_t, float, float)
FB_DEFINE_PPOSVX_REAL_CALLER(fb_call_pdposvx, fb_pdposvx_fortran_fn_t, double, double)
FB_DEFINE_PPOSVX_COMPLEX_CALLER(fb_call_pcposvx, fb_pcposvx_fortran_fn_t,
                                fb_complex_float_t, float)
FB_DEFINE_PPOSVX_COMPLEX_CALLER(fb_call_pzposvx, fb_pzposvx_fortran_fn_t,
                                fb_complex_double_t, double)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_sgesv, FB_OP_SGESV,
                                       fb_call_sgesv_backend, float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_dgesv, FB_OP_DGESV,
                                       fb_call_dgesv_backend, double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_cgesv, FB_OP_CGESV,
                                       fb_call_cgesv_backend,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_zgesv, FB_OP_ZGESV,
                                       fb_call_zgesv_backend,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_psgesv, FB_OP_PSGESV, fb_call_psgesv,
                                       float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pdgesv, FB_OP_PDGESV, fb_call_pdgesv,
                                       double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pcgesv, FB_OP_PCGESV, fb_call_pcgesv,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pzgesv, FB_OP_PZGESV, fb_call_pzgesv,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_sgesvxx, FB_OP_SGESVXX,
                                       fb_call_sgesvxx_backend, float,
                                       bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_dgesvxx, FB_OP_DGESVXX,
                                       fb_call_dgesvxx_backend, double,
                                       bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_cgesvxx, FB_OP_CGESVXX,
                                       fb_call_cgesvxx_backend,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_zgesvxx, FB_OP_ZGESVXX,
                                       fb_call_zgesvxx_backend,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_sgesvx, FB_OP_SGESVX, fb_call_sgesvx,
                                       float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_dgesvx, FB_OP_DGESVX, fb_call_dgesvx,
                                       double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_cgesvx, FB_OP_CGESVX, fb_call_cgesvx,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_zgesvx, FB_OP_ZGESVX, fb_call_zgesvx,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_psgesvx, FB_OP_PSGESVX,
                                       fb_call_psgesvx, float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pdgesvx, FB_OP_PDGESVX,
                                       fb_call_pdgesvx, double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pcgesvx, FB_OP_PCGESVX,
                                       fb_call_pcgesvx, fb_complex_float_t,
                                       bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pzgesvx, FB_OP_PZGESVX,
                                       fb_call_pzgesvx, fb_complex_double_t,
                                       bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_psposv, FB_OP_PSPOSV, fb_call_psposv,
                                       float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pdposv, FB_OP_PDPOSV, fb_call_pdposv,
                                       double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pcposv, FB_OP_PCPOSV, fb_call_pcposv,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pzposv, FB_OP_PZPOSV, fb_call_pzposv,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_sposvxx, FB_OP_SPOSVXX,
                                       fb_call_sposvxx_backend, float,
                                       bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_dposvxx, FB_OP_DPOSVXX,
                                       fb_call_dposvxx_backend, double,
                                       bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_cposvxx, FB_OP_CPOSVXX,
                                       fb_call_cposvxx_backend,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_zposvxx, FB_OP_ZPOSVXX,
                                       fb_call_zposvxx_backend,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_sposvx, FB_OP_SPOSVX, fb_call_sposvx,
                                       float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_dposvx, FB_OP_DPOSVX, fb_call_dposvx,
                                       double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_cposvx, FB_OP_CPOSVX, fb_call_cposvx,
                                       fb_complex_float_t, bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_zposvx, FB_OP_ZPOSVX, fb_call_zposvx,
                                       fb_complex_double_t, bwerr_cf64)

FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_psposvx, FB_OP_PSPOSVX,
                                       fb_call_psposvx, float, bwerr_f32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pdposvx, FB_OP_PDPOSVX,
                                       fb_call_pdposvx, double, bwerr_f64)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pcposvx, FB_OP_PCPOSVX,
                                       fb_call_pcposvx, fb_complex_float_t,
                                       bwerr_cf32)
FB_DEFINE_SCALAPACK_DENSE_SOLVE_RUNNER(run_pzposvx, FB_OP_PZPOSVX,
                                       fb_call_pzposvx, fb_complex_double_t,
                                       bwerr_cf64)

#undef FB_DEFINE_DENSE_EXPERT_RUNNER
#undef FB_COPY_RM_TO_CM
#undef FB_COPY_CM_TO_RM

/* =========================================================================
 * Dispatch table
 * ========================================================================= */

typedef fb_judge_status_t (*fb_solve_runner_fn)(
    const fb_backend_vtable_t *, const fb_backend_vtable_t *,
    const fb_corpus_case_t *, fb_judge_solve_result_t *, uint64_t *);

static const fb_solve_runner_fn fb_solve_dispatch[] = {
    [FB_OP_SGESV] = run_sgesv,   [FB_OP_DGESV] = run_dgesv,
    [FB_OP_CGESV] = run_cgesv,   [FB_OP_ZGESV] = run_zgesv,
    [FB_OP_SGESVX] = run_sgesvx, [FB_OP_DGESVX] = run_dgesvx,
    [FB_OP_CGESVX] = run_cgesvx, [FB_OP_ZGESVX] = run_zgesvx,
    [FB_OP_SGESVXX] = run_sgesvxx, [FB_OP_DGESVXX] = run_dgesvxx,
    [FB_OP_CGESVXX] = run_cgesvxx, [FB_OP_ZGESVXX] = run_zgesvxx,
    [FB_OP_PSGESV] = run_psgesv, [FB_OP_PDGESV] = run_pdgesv,
    [FB_OP_PCGESV] = run_pcgesv, [FB_OP_PZGESV] = run_pzgesv,
    [FB_OP_PSGESVX] = run_psgesvx, [FB_OP_PDGESVX] = run_pdgesvx,
    [FB_OP_PCGESVX] = run_pcgesvx, [FB_OP_PZGESVX] = run_pzgesvx,
    [FB_OP_SGBSV] = run_sgbsv,   [FB_OP_DGBSV] = run_dgbsv,
    [FB_OP_CGBSV] = run_cgbsv,   [FB_OP_ZGBSV] = run_zgbsv,
    [FB_OP_PSGBSV] = run_psgbsv, [FB_OP_PDGBSV] = run_pdgbsv,
    [FB_OP_PCGBSV] = run_pcgbsv, [FB_OP_PZGBSV] = run_pzgbsv,
    [FB_OP_SGBSVX] = run_sgbsvx, [FB_OP_DGBSVX] = run_dgbsvx,
    [FB_OP_CGBSVX] = run_cgbsvx, [FB_OP_ZGBSVX] = run_zgbsvx,
    [FB_OP_SPOSV] = run_sposv,   [FB_OP_DPOSV] = run_dposv,
    [FB_OP_CPOSV] = run_cposv,   [FB_OP_ZPOSV] = run_zposv,
    [FB_OP_SPOSVX] = run_sposvx, [FB_OP_DPOSVX] = run_dposvx,
    [FB_OP_CPOSVX] = run_cposvx, [FB_OP_ZPOSVX] = run_zposvx,
    [FB_OP_SPOSVXX] = run_sposvxx, [FB_OP_DPOSVXX] = run_dposvxx,
    [FB_OP_CPOSVXX] = run_cposvxx, [FB_OP_ZPOSVXX] = run_zposvxx,
    [FB_OP_PSPOSV] = run_psposv, [FB_OP_PDPOSV] = run_pdposv,
    [FB_OP_PCPOSV] = run_pcposv, [FB_OP_PZPOSV] = run_pzposv,
    [FB_OP_PSPOSVX] = run_psposvx, [FB_OP_PDPOSVX] = run_pdposvx,
    [FB_OP_PCPOSVX] = run_pcposvx, [FB_OP_PZPOSVX] = run_pzposvx,
    [FB_OP_SGELS] = run_sgels,   [FB_OP_DGELS] = run_dgels,
    [FB_OP_CGELS] = run_cgels,   [FB_OP_ZGELS] = run_zgels,
    [FB_OP_SGETRS] = run_sgetrs, [FB_OP_DGETRS] = run_dgetrs,
    [FB_OP_CGETRS] = run_cgetrs, [FB_OP_ZGETRS] = run_zgetrs,
    [FB_OP_SGERFS] = run_sgerfs, [FB_OP_DGERFS] = run_dgerfs,
    [FB_OP_CGERFS] = run_cgerfs, [FB_OP_ZGERFS] = run_zgerfs,
    [FB_OP_SGBTRS] = run_sgbtrs, [FB_OP_DGBTRS] = run_dgbtrs,
    [FB_OP_CGBTRS] = run_cgbtrs, [FB_OP_ZGBTRS] = run_zgbtrs,
    [FB_OP_SPOTRS] = run_spotrs, [FB_OP_DPOTRS] = run_dpotrs,
    [FB_OP_CPOTRS] = run_cpotrs, [FB_OP_ZPOTRS] = run_zpotrs,
    [FB_OP_STRTRS] = run_strtrs, [FB_OP_DTRTRS] = run_dtrtrs,
    [FB_OP_CTRTRS] = run_ctrtrs, [FB_OP_ZTRTRS] = run_ztrtrs,
    [FB_OP_SSYTRS] = run_ssytrs, [FB_OP_DSYTRS] = run_dsytrs,
    [FB_OP_CSYTRS] = run_csytrs, [FB_OP_ZSYTRS] = run_zsytrs,
    [FB_OP_SSYSV] = run_ssysv,   [FB_OP_DSYSV] = run_dsysv,
    [FB_OP_CSYSV] = run_csysv,   [FB_OP_ZSYSV] = run_zsysv,
    [FB_OP_CHESV] = run_chesv,   [FB_OP_ZHESV] = run_zhesv,
    [FB_OP_SSYSVX] = run_ssysvx, [FB_OP_DSYSVX] = run_dsysvx,
    [FB_OP_CSYSVX] = run_csysvx, [FB_OP_ZSYSVX] = run_zsysvx,
    [FB_OP_CHESVX] = run_chesvx, [FB_OP_ZHESVX] = run_zhesvx,
    [FB_OP_SSYSVXX] = run_ssysvxx, [FB_OP_DSYSVXX] = run_dsysvxx,
    [FB_OP_CSYSVXX] = run_csysvxx, [FB_OP_ZSYSVXX] = run_zsysvxx,
    [FB_OP_CHESVXX] = run_chesvxx, [FB_OP_ZHESVXX] = run_zhesvxx,
    [FB_OP_SGELSD] = run_sgelsd, [FB_OP_DGELSD] = run_dgelsd,
    [FB_OP_CGELSD] = run_cgelsd, [FB_OP_ZGELSD] = run_zgelsd,
    [FB_OP_SGELSS] = run_sgelss, [FB_OP_DGELSS] = run_dgelss,
    [FB_OP_CGELSS] = run_cgelss, [FB_OP_ZGELSS] = run_zgelss,
    [FB_OP_SGELSY] = run_sgelsy, [FB_OP_DGELSY] = run_dgelsy,
    [FB_OP_CGELSY] = run_cgelsy, [FB_OP_ZGELSY] = run_zgelsy,
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