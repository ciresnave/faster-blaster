/**
 * @file judge_norm.c
 * @brief Norm and distance utilities for the judge module.
 *
 * Uses scaled accumulation (same strategy as faster-blaster-reference dnrm2)
 * to avoid overflow/underflow for extreme-scale inputs.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_norm.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* =========================================================================
 * Internal: scaled L2 norm of a real double array
 * ========================================================================= */

static double frob_f64_raw(const double* x, size_t n) {
    if (n == 0 || x == NULL) return 0.0;

    double scale = 0.0;
    double ssq   = 1.0;

    for (size_t i = 0; i < n; i++) {
        double absxi = fabs(x[i]);
        if (absxi > 0.0) {
            if (scale < absxi) {
                double r = scale / absxi;
                ssq  = 1.0 + ssq * r * r;
                scale = absxi;
            } else {
                double r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    return scale * sqrt(ssq);
}

/* =========================================================================
 * Public: typed Frobenius norms
 * ========================================================================= */

double fb_norm_frob_f64(const double* x, size_t n) {
    return frob_f64_raw(x, n);
}

double fb_norm_frob_f32(const float* x, size_t n) {
    if (n == 0 || x == NULL) return 0.0;
    /* Upcasting to double for accumulation gives ~14-digit accuracy. */
    double scale = 0.0;
    double ssq   = 1.0;
    for (size_t i = 0; i < n; i++) {
        double absxi = fabs((double)x[i]);
        if (absxi > 0.0) {
            if (scale < absxi) {
                double r = scale / absxi;
                ssq  = 1.0 + ssq * r * r;
                scale = absxi;
            } else {
                double r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    return scale * sqrt(ssq);
}

double fb_norm_frob_cf32(const float* x, size_t n) {
    /* Complex float: each element is (re, im) so underlying length is 2n. */
    return fb_norm_frob_f32(x, 2 * n);
}

double fb_norm_frob_cf64(const double* x, size_t n) {
    return frob_f64_raw(x, 2 * n);
}

double fb_norm_frob(const void* data, size_t n, fb_dtype_t dtype) {
    switch (dtype) {
        case FB_DTYPE_F32:  return fb_norm_frob_f32 ((const float* )data, n);
        case FB_DTYPE_F64:  return fb_norm_frob_f64 ((const double*)data, n);
        case FB_DTYPE_CF32: return fb_norm_frob_cf32((const float* )data, n);
        case FB_DTYPE_CF64: return fb_norm_frob_cf64((const double*)data, n);
        default:            return 0.0;
    }
}

/* =========================================================================
 * Matrix Frobenius norms (column-major)
 * ========================================================================= */

double fb_matrix_norm_frob_f64(const double* A, int m, int n, int lda) {
    if (!A || m <= 0 || n <= 0) return 0.0;
    double scale = 0.0;
    double ssq   = 1.0;
    for (int j = 0; j < n; j++) {
        const double* col = A + (size_t)j * (size_t)lda;
        for (int i = 0; i < m; i++) {
            double absv = fabs(col[i]);
            if (absv > 0.0) {
                if (scale < absv) {
                    double r = scale / absv;
                    ssq  = 1.0 + ssq * r * r;
                    scale = absv;
                } else {
                    double r = absv / scale;
                    ssq += r * r;
                }
            }
        }
    }
    return scale * sqrt(ssq);
}

double fb_matrix_norm_frob_f32(const float* A, int m, int n, int lda) {
    if (!A || m <= 0 || n <= 0) return 0.0;
    double scale = 0.0;
    double ssq   = 1.0;
    for (int j = 0; j < n; j++) {
        const float* col = A + (size_t)j * (size_t)lda;
        for (int i = 0; i < m; i++) {
            double absv = fabs((double)col[i]);
            if (absv > 0.0) {
                if (scale < absv) {
                    double r = scale / absv;
                    ssq  = 1.0 + ssq * r * r;
                    scale = absv;
                } else {
                    double r = absv / scale;
                    ssq += r * r;
                }
            }
        }
    }
    return scale * sqrt(ssq);
}

/* =========================================================================
 * Relative error
 * ========================================================================= */

double fb_judge_relerr(
    const void* candidate,
    const void* reference,
    size_t      n,
    fb_dtype_t  dtype,
    double      scale_hint)
{
    if (!candidate || !reference || n == 0) return 1.0;

    /* Compute ||candidate - reference||_F using double accumulation. */
    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    switch (dtype) {
        case FB_DTYPE_F32: {
            const float* c = (const float*)candidate;
            const float* r = (const float*)reference;
            /* Use scaled accumulation for the diff to handle extreme scales. */
            double scale_d = 0.0, ssq_d = 1.0;
            double scale_r = 0.0, ssq_r = 1.0;
            for (size_t i = 0; i < n; i++) {
                double d  = fabs((double)c[i] - (double)r[i]);
                double rv = fabs((double)r[i]);
                if (d > 0.0) {
                    if (scale_d < d) { double t = scale_d/d; ssq_d = 1.0 + ssq_d*t*t; scale_d = d; }
                    else             { double t = d/scale_d; ssq_d += t*t; }
                }
                if (rv > 0.0) {
                    if (scale_r < rv) { double t = scale_r/rv; ssq_r = 1.0+ssq_r*t*t; scale_r = rv; }
                    else              { double t = rv/scale_r; ssq_r += t*t; }
                }
            }
            diff_norm = scale_d * sqrt(ssq_d);
            ref_norm  = scale_r * sqrt(ssq_r);
            break;
        }
        case FB_DTYPE_F64: {
            const double* c = (const double*)candidate;
            const double* r = (const double*)reference;
            double scale_d = 0.0, ssq_d = 1.0;
            double scale_r = 0.0, ssq_r = 1.0;
            for (size_t i = 0; i < n; i++) {
                double d  = fabs(c[i] - r[i]);
                double rv = fabs(r[i]);
                if (d > 0.0) {
                    if (scale_d < d) { double t = scale_d/d; ssq_d = 1.0+ssq_d*t*t; scale_d = d; }
                    else             { double t = d/scale_d; ssq_d += t*t; }
                }
                if (rv > 0.0) {
                    if (scale_r < rv) { double t = scale_r/rv; ssq_r = 1.0+ssq_r*t*t; scale_r = rv; }
                    else              { double t = rv/scale_r; ssq_r += t*t; }
                }
            }
            diff_norm = scale_d * sqrt(ssq_d);
            ref_norm  = scale_r * sqrt(ssq_r);
            break;
        }
        case FB_DTYPE_CF32: {
            /* Complex: treat as 2n real values. */
            return fb_judge_relerr(candidate, reference, 2*n, FB_DTYPE_F32, scale_hint);
        }
        case FB_DTYPE_CF64: {
            return fb_judge_relerr(candidate, reference, 2*n, FB_DTYPE_F64, scale_hint);
        }
        default:
            return 1.0;
    }

    /* tau = eps_type * scale_hint prevents log(0) for near-zero reference. */
    double eps = fb_dtype_eps[dtype];
    double tau = (scale_hint > 0.0) ? scale_hint * eps : eps;
    double denom = (ref_norm > tau) ? ref_norm : tau;

    return diff_norm / denom;
}

double fb_judge_relerr_matrix(
    const void* candidate,
    const void* reference,
    int         m,
    int         n,
    int         lda_cand,
    int         lda_ref,
    fb_dtype_t  dtype,
    double      scale_hint)
{
    if (!candidate || !reference || m <= 0 || n <= 0) return 1.0;

    /* For square leading dims, fall through to the flat version. */
    if (lda_cand == m && lda_ref == m) {
        return fb_judge_relerr(candidate, reference,
                               (size_t)m * (size_t)n, dtype, scale_hint);
    }

    /* Strided: compute per-column. */
    size_t elem_bytes = fb_dtype_element_size[dtype];
    double diff_ssq = 0.0, diff_scale = 0.0;
    double ref_ssq  = 0.0, ref_scale  = 0.0;

    for (int j = 0; j < n; j++) {
        const uint8_t* c_col = (const uint8_t*)candidate + (size_t)j*(size_t)lda_cand*elem_bytes;
        const uint8_t* r_col = (const uint8_t*)reference  + (size_t)j*(size_t)lda_ref *elem_bytes;
        double col_relerr = fb_judge_relerr(c_col, r_col, (size_t)m, dtype, scale_hint);
        /* Accumulate squared relative errors as a rough aggregate. */
        diff_ssq += col_relerr * col_relerr;
        (void)ref_scale; (void)ref_ssq;
    }
    return sqrt(diff_ssq / (double)n);
}

/* =========================================================================
 * NaN / Inf detection
 * ========================================================================= */

bool fb_judge_has_nan_inf(const void* data, size_t n, fb_dtype_t dtype) {
    if (!data || n == 0) return false;
    switch (dtype) {
        case FB_DTYPE_F32:
        case FB_DTYPE_CF32: {
            const float* p = (const float*)data;
            size_t count = (dtype == FB_DTYPE_CF32) ? 2*n : n;
            for (size_t i = 0; i < count; i++) {
                if (!isfinite(p[i])) return true;
            }
            return false;
        }
        case FB_DTYPE_F64:
        case FB_DTYPE_CF64: {
            const double* p = (const double*)data;
            size_t count = (dtype == FB_DTYPE_CF64) ? 2*n : n;
            for (size_t i = 0; i < count; i++) {
                if (!isfinite(p[i])) return true;
            }
            return false;
        }
        default:
            return false;
    }
}

/* =========================================================================
 * Orthogonality residual  ||Q^T Q - I||_F / sqrt(n)
 * ========================================================================= */

double fb_judge_ortho_residual_f64(const double* Q, int m, int n, int ldq) {
    if (!Q || m <= 0 || n <= 0 || m < n) return 1.0;

    /* Compute G = Q^T Q (n×n) via straightforward inner products.
     * Cost is O(mn²) — acceptable for judge use (not on hot dispatch path). */
    double frob_sq = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            /* dot product of column i and column j */
            double dot = 0.0;
            const double* qi = Q + (size_t)i * (size_t)ldq;
            const double* qj = Q + (size_t)j * (size_t)ldq;
            for (int k = 0; k < m; k++) {
                dot += qi[k] * qj[k];
            }
            /* Expected: G[i,j] = delta_ij */
            double expected = (i == j) ? 1.0 : 0.0;
            double err = dot - expected;
            double contrib = err * err;
            /* Off-diagonal contributes twice (symmetric). */
            frob_sq += (i == j) ? contrib : 2.0 * contrib;
        }
    }
    /* Normalize by sqrt(n) to get a per-column measure. */
    return sqrt(frob_sq) / sqrt((double)n);
}

double fb_judge_ortho_residual_f32(const float* Q, int m, int n, int ldq) {
    if (!Q || m <= 0 || n <= 0 || m < n) return 1.0;

    double frob_sq = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double dot = 0.0;
            const float* qi = Q + (size_t)i * (size_t)ldq;
            const float* qj = Q + (size_t)j * (size_t)ldq;
            for (int k = 0; k < m; k++) {
                dot += (double)qi[k] * (double)qj[k];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double err = dot - expected;
            double contrib = err * err;
            frob_sq += (i == j) ? contrib : 2.0 * contrib;
        }
    }
    return sqrt(frob_sq) / sqrt((double)n);
}
