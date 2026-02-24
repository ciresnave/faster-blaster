/**
 * @file judge_norm.h
 * @brief Norm and distance utilities for the judge module.
 *
 * All functions operate on host memory. GPU backend results must be copied
 * to host before calling any function here.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_NORM_H
#define FB_JUDGE_NORM_H

#include <stddef.h>
#include <stdint.h>
#include "judge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Vector norms
 * ========================================================================= */

/** Frobenius (L2) norm of a real float array of length n. */
double fb_norm_frob_f32(const float*  x, size_t n);

/** Frobenius (L2) norm of a real double array of length n. */
double fb_norm_frob_f64(const double* x, size_t n);

/**
 * Frobenius norm of a complex float array (interleaved re/im) of length n.
 * n is the number of complex elements; the underlying array has 2n floats.
 */
double fb_norm_frob_cf32(const float*  x, size_t n);

/**
 * Frobenius norm of a complex double array of length n.
 */
double fb_norm_frob_cf64(const double* x, size_t n);

/**
 * Generic dispatcher — calls the appropriate typed version based on dtype.
 * 'data' points to the raw element array, 'n' is the element count.
 */
double fb_norm_frob(const void* data, size_t n, fb_dtype_t dtype);

/* =========================================================================
 * Matrix norms (column-major storage)
 * ========================================================================= */

/**
 * Frobenius norm of a column-major double matrix of size m×n with leading
 * dimension lda.
 */
double fb_matrix_norm_frob_f64(const double* A, int m, int n, int lda);

/**
 * Frobenius norm of a column-major float matrix.
 */
double fb_matrix_norm_frob_f32(const float* A, int m, int n, int lda);

/* =========================================================================
 * Relative error (norm-based)
 * ========================================================================= */

/**
 * Compute the relative error between candidate X and reference X_star using
 * scaled Frobenius norm:
 *
 *   relerr = ||X - X*||_F  /  max(||X*||_F, tau)
 *
 * where tau = eps_dtype * scale_hint prevents division by zero near zero
 * reference vectors. Pass scale_hint = 1.0 if the input/output scale is
 * not known; pass an input matrix norm for tighter bounds.
 *
 * Returns the raw relative error (convert to digits with fb_judge_digits()).
 */
double fb_judge_relerr(
    const void* candidate,
    const void* reference,
    size_t      n,
    fb_dtype_t  dtype,
    double      scale_hint
);

/**
 * Relative error for column-major matrices (accounts for leading dimension).
 */
double fb_judge_relerr_matrix(
    const void* candidate,
    const void* reference,
    int         m,
    int         n,
    int         lda_cand,
    int         lda_ref,
    fb_dtype_t  dtype,
    double      scale_hint
);

/* =========================================================================
 * NaN / Inf detection
 * ========================================================================= */

/**
 * Returns true if any element of the array contains NaN or Inf.
 * Works for real and complex dtypes (checks each component).
 */
bool fb_judge_has_nan_inf(const void* data, size_t n, fb_dtype_t dtype);

/* =========================================================================
 * Identity residual  ||Q^H Q - I||_F / sqrt(n)
 * Used to check orthogonality of factor matrices.
 * ========================================================================= */

/**
 * Compute ||Q^T Q - I||_F for a real column-major m×n matrix Q (m >= n).
 * Normalised by sqrt(n) to give a per-column measure.
 * Only the n×n Gram matrix is computed internally (O(mn²) cost).
 */
double fb_judge_ortho_residual_f64(const double* Q, int m, int n, int ldq);
double fb_judge_ortho_residual_f32(const float*  Q, int m, int n, int ldq);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_NORM_H */
