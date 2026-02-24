/**
 * @file reference_level1.c
 * @brief Reference implementation of BLAS Level 1 operations
 * 
 * This implementation prioritizes correctness and numerical accuracy over performance.
 * It serves as the "golden reference" for validating other backend implementations.
 * 
 * Design principles:
 * - Handle all edge cases (zero length, negative/zero increments, NaN, Inf)
 * - IEEE 754 compliance for floating point operations
 * - Proper handling of complex arithmetic
 * - No performance optimizations that sacrifice precision
 * - Clear, readable code that matches BLAS specification exactly
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../backends/backend_interface.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * COPY - Copy vector X to vector Y
 * Y[i] = X[i]
 * ========================================================================= */

void fb_ref_scopy(
    const int64_t n,
    const float *x,
    const int64_t incx,
    float *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;  /* Invalid stride */
    
    if (incx == 1 && incy == 1) {
        /* Contiguous - use memcpy for correctness */
        memcpy(y, x, (size_t)n * sizeof(float));
    } else {
        /* Strided access */
        for (int64_t i = 0; i < n; i++) {
            y[i * incy] = x[i * incx];
        }
    }
}

void fb_ref_dcopy(
    const int64_t n,
    const double *x,
    const int64_t incx,
    double *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(double));
    } else {
        for (int64_t i = 0; i < n; i++) {
            y[i * incy] = x[i * incx];
        }
    }
}

void fb_ref_ccopy(
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx,
    fb_complex_float_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(fb_complex_float_t));
    } else {
        for (int64_t i = 0; i < n; i++) {
            y[i * incy] = x[i * incx];
        }
    }
}

void fb_ref_zcopy(
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx,
    fb_complex_double_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(fb_complex_double_t));
    } else {
        for (int64_t i = 0; i < n; i++) {
            y[i * incy] = x[i * incx];
        }
    }
}

/* ============================================================================
 * SWAP - Swap vectors X and Y
 * X[i] <-> Y[i]
 * ========================================================================= */

void fb_ref_sswap(
    const int64_t n,
    float *x,
    const int64_t incx,
    float *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        float temp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = temp;
    }
}

void fb_ref_dswap(
    const int64_t n,
    double *x,
    const int64_t incx,
    double *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        double temp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = temp;
    }
}

void fb_ref_cswap(
    const int64_t n,
    fb_complex_float_t *x,
    const int64_t incx,
    fb_complex_float_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        fb_complex_float_t temp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = temp;
    }
}

void fb_ref_zswap(
    const int64_t n,
    fb_complex_double_t *x,
    const int64_t incx,
    fb_complex_double_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        fb_complex_double_t temp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = temp;
    }
}

/* ============================================================================
 * SCAL - Scale vector by constant
 * X[i] = alpha * X[i]
 * ========================================================================= */

void fb_ref_sscal(
    const int64_t n,
    const float alpha,
    float *x,
    const int64_t incx)
{
    if (n <= 0) return;
    if (incx == 0) return;
    
    /* Special case: alpha = 0 sets all to zero (even if x contains NaN) */
    if (alpha == 0.0f) {
        for (int64_t i = 0; i < n; i++) {
            x[i * incx] = 0.0f;
        }
        return;
    }
    
    /* Special case: alpha = 1 is no-op */
    if (alpha == 1.0f) {
        return;
    }
    
    /* General case */
    for (int64_t i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

void fb_ref_dscal(
    const int64_t n,
    const double alpha,
    double *x,
    const int64_t incx)
{
    if (n <= 0) return;
    if (incx == 0) return;
    
    if (alpha == 0.0) {
        for (int64_t i = 0; i < n; i++) {
            x[i * incx] = 0.0;
        }
        return;
    }
    
    if (alpha == 1.0) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

void fb_ref_cscal(
    const int64_t n,
    const fb_complex_float_t alpha,
    fb_complex_float_t *x,
    const int64_t incx)
{
    if (n <= 0) return;
    if (incx == 0) return;
    
    /* Check if alpha is zero */
    if (alpha.real == 0.0f && alpha.imag == 0.0f) {
        const fb_complex_float_t zero = {0.0f, 0.0f};
        for (int64_t i = 0; i < n; i++) {
            x[i * incx] = zero;
        }
        return;
    }
    
    /* Check if alpha is one */
    if (alpha.real == 1.0f && alpha.imag == 0.0f) {
        return;
    }
    
    /* General complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
    for (int64_t i = 0; i < n; i++) {
        fb_complex_float_t *xi = &x[i * incx];
        float real_part = alpha.real * xi->real - alpha.imag * xi->imag;
        float imag_part = alpha.real * xi->imag + alpha.imag * xi->real;
        xi->real = real_part;
        xi->imag = imag_part;
    }
}

void fb_ref_zscal(
    const int64_t n,
    const fb_complex_double_t alpha,
    fb_complex_double_t *x,
    const int64_t incx)
{
    if (n <= 0) return;
    if (incx == 0) return;
    
    if (alpha.real == 0.0 && alpha.imag == 0.0) {
        const fb_complex_double_t zero = {0.0, 0.0};
        for (int64_t i = 0; i < n; i++) {
            x[i * incx] = zero;
        }
        return;
    }
    
    if (alpha.real == 1.0 && alpha.imag == 0.0) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        fb_complex_double_t *xi = &x[i * incx];
        double real_part = alpha.real * xi->real - alpha.imag * xi->imag;
        double imag_part = alpha.real * xi->imag + alpha.imag * xi->real;
        xi->real = real_part;
        xi->imag = imag_part;
    }
}

/* ============================================================================
 * AXPY - Constant times vector plus vector
 * Y[i] = alpha * X[i] + Y[i]
 * ========================================================================= */

void fb_ref_saxpy(
    const int64_t n,
    const float alpha,
    const float *x,
    const int64_t incx,
    float *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    /* alpha = 0 is no-op (preserve y even if x contains NaN) */
    if (alpha == 0.0f) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}

void fb_ref_daxpy(
    const int64_t n,
    const double alpha,
    const double *x,
    const int64_t incx,
    double *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (alpha == 0.0) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}

void fb_ref_caxpy(
    const int64_t n,
    const fb_complex_float_t alpha,
    const fb_complex_float_t *x,
    const int64_t incx,
    fb_complex_float_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (alpha.real == 0.0f && alpha.imag == 0.0f) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        fb_complex_float_t *yi = &y[i * incy];
        
        /* Complex multiply-add: y += alpha * x */
        float real_part = alpha.real * xi->real - alpha.imag * xi->imag;
        float imag_part = alpha.real * xi->imag + alpha.imag * xi->real;
        
        yi->real += real_part;
        yi->imag += imag_part;
    }
}

void fb_ref_zaxpy(
    const int64_t n,
    const fb_complex_double_t alpha,
    const fb_complex_double_t *x,
    const int64_t incx,
    fb_complex_double_t *y,
    const int64_t incy)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    if (alpha.real == 0.0 && alpha.imag == 0.0) {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        fb_complex_double_t *yi = &y[i * incy];
        
        double real_part = alpha.real * xi->real - alpha.imag * xi->imag;
        double imag_part = alpha.real * xi->imag + alpha.imag * xi->real;
        
        yi->real += real_part;
        yi->imag += imag_part;
    }
}

/* ============================================================================
 * DOT - Dot product of two vectors
 * result = sum(X[i] * Y[i])
 * 
 * For complex: DOTU uses unconjugated, DOTC uses conjugated first argument
 * ========================================================================= */

float fb_ref_sdot(
    const int64_t n,
    const float *x,
    const int64_t incx,
    const float *y,
    const int64_t incy)
{
    if (n <= 0) return 0.0f;
    if (incx == 0 || incy == 0) return 0.0f;
    
    /* Use Kahan summation for improved numerical accuracy */
    float sum = 0.0f;
    float compensation = 0.0f;  /* Running compensation for lost low-order bits */
    
    for (int64_t i = 0; i < n; i++) {
        float product = x[i * incx] * y[i * incy];
        float corrected = product - compensation;
        float new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

double fb_ref_ddot(
    const int64_t n,
    const double *x,
    const int64_t incx,
    const double *y,
    const int64_t incy)
{
    if (n <= 0) return 0.0;
    if (incx == 0 || incy == 0) return 0.0;
    
    /* Kahan summation for double precision */
    double sum = 0.0;
    double compensation = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        double product = x[i * incx] * y[i * incy];
        double corrected = product - compensation;
        double new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

void fb_ref_cdotu(
    fb_complex_float_t *result,
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx,
    const fb_complex_float_t *y,
    const int64_t incy)
{
    result->real = 0.0f;
    result->imag = 0.0f;
    
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    /* Kahan summation for both real and imaginary parts */
    float sum_real = 0.0f, comp_real = 0.0f;
    float sum_imag = 0.0f, comp_imag = 0.0f;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        const fb_complex_float_t *yi = &y[i * incy];
        
        /* Unconjugated product: (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
        float real_prod = xi->real * yi->real - xi->imag * yi->imag;
        float imag_prod = xi->real * yi->imag + xi->imag * yi->real;
        
        /* Kahan summation for real part */
        float corrected_real = real_prod - comp_real;
        float new_sum_real = sum_real + corrected_real;
        comp_real = (new_sum_real - sum_real) - corrected_real;
        sum_real = new_sum_real;
        
        /* Kahan summation for imaginary part */
        float corrected_imag = imag_prod - comp_imag;
        float new_sum_imag = sum_imag + corrected_imag;
        comp_imag = (new_sum_imag - sum_imag) - corrected_imag;
        sum_imag = new_sum_imag;
    }
    
    result->real = sum_real;
    result->imag = sum_imag;
}

void fb_ref_zdotu(
    fb_complex_double_t *result,
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx,
    const fb_complex_double_t *y,
    const int64_t incy)
{
    result->real = 0.0;
    result->imag = 0.0;
    
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    double sum_real = 0.0, comp_real = 0.0;
    double sum_imag = 0.0, comp_imag = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        const fb_complex_double_t *yi = &y[i * incy];
        
        double real_prod = xi->real * yi->real - xi->imag * yi->imag;
        double imag_prod = xi->real * yi->imag + xi->imag * yi->real;
        
        double corrected_real = real_prod - comp_real;
        double new_sum_real = sum_real + corrected_real;
        comp_real = (new_sum_real - sum_real) - corrected_real;
        sum_real = new_sum_real;
        
        double corrected_imag = imag_prod - comp_imag;
        double new_sum_imag = sum_imag + corrected_imag;
        comp_imag = (new_sum_imag - sum_imag) - corrected_imag;
        sum_imag = new_sum_imag;
    }
    
    result->real = sum_real;
    result->imag = sum_imag;
}

void fb_ref_cdotc(
    fb_complex_float_t *result,
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx,
    const fb_complex_float_t *y,
    const int64_t incy)
{
    result->real = 0.0f;
    result->imag = 0.0f;
    
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    float sum_real = 0.0f, comp_real = 0.0f;
    float sum_imag = 0.0f, comp_imag = 0.0f;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        const fb_complex_float_t *yi = &y[i * incy];
        
        /* Conjugated product: conj(a+bi)(c+di) = (a-bi)(c+di) = (ac+bd) + (ad-bc)i */
        float real_prod = xi->real * yi->real + xi->imag * yi->imag;
        float imag_prod = xi->real * yi->imag - xi->imag * yi->real;
        
        float corrected_real = real_prod - comp_real;
        float new_sum_real = sum_real + corrected_real;
        comp_real = (new_sum_real - sum_real) - corrected_real;
        sum_real = new_sum_real;
        
        float corrected_imag = imag_prod - comp_imag;
        float new_sum_imag = sum_imag + corrected_imag;
        comp_imag = (new_sum_imag - sum_imag) - corrected_imag;
        sum_imag = new_sum_imag;
    }
    
    result->real = sum_real;
    result->imag = sum_imag;
}

void fb_ref_zdotc(
    fb_complex_double_t *result,
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx,
    const fb_complex_double_t *y,
    const int64_t incy)
{
    result->real = 0.0;
    result->imag = 0.0;
    
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    double sum_real = 0.0, comp_real = 0.0;
    double sum_imag = 0.0, comp_imag = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        const fb_complex_double_t *yi = &y[i * incy];
        
        double real_prod = xi->real * yi->real + xi->imag * yi->imag;
        double imag_prod = xi->real * yi->imag - xi->imag * yi->real;
        
        double corrected_real = real_prod - comp_real;
        double new_sum_real = sum_real + corrected_real;
        comp_real = (new_sum_real - sum_real) - corrected_real;
        sum_real = new_sum_real;
        
        double corrected_imag = imag_prod - comp_imag;
        double new_sum_imag = sum_imag + corrected_imag;
        comp_imag = (new_sum_imag - sum_imag) - corrected_imag;
        sum_imag = new_sum_imag;
    }
    
    result->real = sum_real;
    result->imag = sum_imag;
}

/* ============================================================================
 * NRM2 - Euclidean norm (2-norm) of vector
 * result = sqrt(sum(|X[i]|^2))
 * 
 * Uses scaled algorithm to avoid overflow/underflow
 * ========================================================================= */

float fb_ref_snrm2(
    const int64_t n,
    const float *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0f;
    
    /* Use scaled algorithm to avoid overflow/underflow */
    float scale = 0.0f;
    float ssq = 1.0f;  /* sum of squares */
    
    for (int64_t i = 0; i < n; i++) {
        float absxi = fabsf(x[i * incx]);
        
        if (absxi != 0.0f) {
            if (scale < absxi) {
                /* Rescale sum of squares */
                float ratio = scale / absxi;
                ssq = 1.0f + ssq * (ratio * ratio);
                scale = absxi;
            } else {
                float ratio = absxi / scale;
                ssq += ratio * ratio;
            }
        }
    }
    
    return scale * sqrtf(ssq);
}

double fb_ref_dnrm2(
    const int64_t n,
    const double *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0;
    
    double scale = 0.0;
    double ssq = 1.0;
    
    for (int64_t i = 0; i < n; i++) {
        double absxi = fabs(x[i * incx]);
        
        if (absxi != 0.0) {
            if (scale < absxi) {
                double ratio = scale / absxi;
                ssq = 1.0 + ssq * (ratio * ratio);
                scale = absxi;
            } else {
                double ratio = absxi / scale;
                ssq += ratio * ratio;
            }
        }
    }
    
    return scale * sqrt(ssq);
}

float fb_ref_scnrm2(
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0f;
    
    float scale = 0.0f;
    float ssq = 1.0f;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        
        /* Process real part */
        float abs_real = fabsf(xi->real);
        if (abs_real != 0.0f) {
            if (scale < abs_real) {
                float ratio = scale / abs_real;
                ssq = 1.0f + ssq * (ratio * ratio);
                scale = abs_real;
            } else {
                float ratio = abs_real / scale;
                ssq += ratio * ratio;
            }
        }
        
        /* Process imaginary part */
        float abs_imag = fabsf(xi->imag);
        if (abs_imag != 0.0f) {
            if (scale < abs_imag) {
                float ratio = scale / abs_imag;
                ssq = 1.0f + ssq * (ratio * ratio);
                scale = abs_imag;
            } else {
                float ratio = abs_imag / scale;
                ssq += ratio * ratio;
            }
        }
    }
    
    return scale * sqrtf(ssq);
}

double fb_ref_dznrm2(
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0;
    
    double scale = 0.0;
    double ssq = 1.0;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        
        double abs_real = fabs(xi->real);
        if (abs_real != 0.0) {
            if (scale < abs_real) {
                double ratio = scale / abs_real;
                ssq = 1.0 + ssq * (ratio * ratio);
                scale = abs_real;
            } else {
                double ratio = abs_real / scale;
                ssq += ratio * ratio;
            }
        }
        
        double abs_imag = fabs(xi->imag);
        if (abs_imag != 0.0) {
            if (scale < abs_imag) {
                double ratio = scale / abs_imag;
                ssq = 1.0 + ssq * (ratio * ratio);
                scale = abs_imag;
            } else {
                double ratio = abs_imag / scale;
                ssq += ratio * ratio;
            }
        }
    }
    
    return scale * sqrt(ssq);
}

/* ============================================================================
 * ASUM - Sum of absolute values
 * result = sum(|X[i]|)
 * For complex: |a+bi| = |a| + |b| (1-norm, not 2-norm)
 * ========================================================================= */

float fb_ref_sasum(
    const int64_t n,
    const float *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0f;
    
    /* Kahan summation for accuracy */
    float sum = 0.0f;
    float compensation = 0.0f;
    
    for (int64_t i = 0; i < n; i++) {
        float absxi = fabsf(x[i * incx]);
        float corrected = absxi - compensation;
        float new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

double fb_ref_dasum(
    const int64_t n,
    const double *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0;
    
    double sum = 0.0;
    double compensation = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        double absxi = fabs(x[i * incx]);
        double corrected = absxi - compensation;
        double new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

float fb_ref_scasum(
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0f;
    
    float sum = 0.0f;
    float compensation = 0.0f;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        /* BLAS ASUM uses 1-norm: |a+bi| = |a| + |b| */
        float abs_sum = fabsf(xi->real) + fabsf(xi->imag);
        
        float corrected = abs_sum - compensation;
        float new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

double fb_ref_dzasum(
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0.0;
    
    double sum = 0.0;
    double compensation = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        double abs_sum = fabs(xi->real) + fabs(xi->imag);
        
        double corrected = abs_sum - compensation;
        double new_sum = sum + corrected;
        compensation = (new_sum - sum) - corrected;
        sum = new_sum;
    }
    
    return sum;
}

/* ============================================================================
 * IAMAX - Index of maximum absolute value element
 * Returns 0-based index (CBLAS convention)
 * For complex: uses |real| + |imag| (same as ASUM)
 * ========================================================================= */

int64_t fb_ref_isamax(
    const int64_t n,
    const float *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0;
    
    int64_t max_index = 0;
    float max_value = fabsf(x[0]);
    
    for (int64_t i = 1; i < n; i++) {
        float absxi = fabsf(x[i * incx]);
        if (absxi > max_value) {
            max_value = absxi;
            max_index = i;
        }
    }
    
    return max_index;
}

int64_t fb_ref_idamax(
    const int64_t n,
    const double *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0;
    
    int64_t max_index = 0;
    double max_value = fabs(x[0]);
    
    for (int64_t i = 1; i < n; i++) {
        double absxi = fabs(x[i * incx]);
        if (absxi > max_value) {
            max_value = absxi;
            max_index = i;
        }
    }
    
    return max_index;
}

int64_t fb_ref_icamax(
    const int64_t n,
    const fb_complex_float_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0;
    
    int64_t max_index = 0;
    float max_value = fabsf(x[0].real) + fabsf(x[0].imag);
    
    for (int64_t i = 1; i < n; i++) {
        const fb_complex_float_t *xi = &x[i * incx];
        float abs_sum = fabsf(xi->real) + fabsf(xi->imag);
        if (abs_sum > max_value) {
            max_value = abs_sum;
            max_index = i;
        }
    }
    
    return max_index;
}

int64_t fb_ref_izamax(
    const int64_t n,
    const fb_complex_double_t *x,
    const int64_t incx)
{
    if (n <= 0 || incx <= 0) return 0;
    
    int64_t max_index = 0;
    double max_value = fabs(x[0].real) + fabs(x[0].imag);
    
    for (int64_t i = 1; i < n; i++) {
        const fb_complex_double_t *xi = &x[i * incx];
        double abs_sum = fabs(xi->real) + fabs(xi->imag);
        if (abs_sum > max_value) {
            max_value = abs_sum;
            max_index = i;
        }
    }
    
    return max_index;
}

/* ============================================================================
 * ROTG - Generate Givens rotation
 * Constructs Givens plane rotation that zeros out b
 * 
 * Given a and b, compute c, s, and r such that:
 * [ c  s ] [ a ]   [ r ]
 * [-s  c ] [ b ] = [ 0 ]
 * 
 * Also computes z for reconstruction
 * ========================================================================= */

void fb_ref_srotg(
    float *a,
    float *b,
    float *c,
    float *s)
{
    float abs_a = fabsf(*a);
    float abs_b = fabsf(*b);
    
    if (abs_b == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
        /* *a unchanged, *b already 0 */
        return;
    }
    
    if (abs_a == 0.0f) {
        *c = 0.0f;
        *s = (*b >= 0.0f) ? 1.0f : -1.0f;
        *a = abs_b;
        *b = 1.0f;
        return;
    }
    
    /* General case - use scaled algorithm to avoid overflow */
    float scale = abs_a + abs_b;
    float norm_a = *a / scale;
    float norm_b = *b / scale;
    float r = scale * sqrtf(norm_a * norm_a + norm_b * norm_b);
    
    /* Ensure r has same sign as element with larger magnitude */
    if (abs_a > abs_b) {
        r = (*a >= 0.0f) ? r : -r;
    } else {
        r = (*b >= 0.0f) ? r : -r;
    }
    
    *c = *a / r;
    *s = *b / r;
    
    /* Reconstruction parameter z */
    float z;
    if (abs_a > abs_b) {
        z = *s;
    } else if (*c != 0.0f) {
        z = 1.0f / *c;
    } else {
        z = 1.0f;
    }
    
    *a = r;
    *b = z;
}

void fb_ref_drotg(
    double *a,
    double *b,
    double *c,
    double *s)
{
    double abs_a = fabs(*a);
    double abs_b = fabs(*b);
    
    if (abs_b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        return;
    }
    
    if (abs_a == 0.0) {
        *c = 0.0;
        *s = (*b >= 0.0) ? 1.0 : -1.0;
        *a = abs_b;
        *b = 1.0;
        return;
    }
    
    double scale = abs_a + abs_b;
    double norm_a = *a / scale;
    double norm_b = *b / scale;
    double r = scale * sqrt(norm_a * norm_a + norm_b * norm_b);
    
    if (abs_a > abs_b) {
        r = (*a >= 0.0) ? r : -r;
    } else {
        r = (*b >= 0.0) ? r : -r;
    }
    
    *c = *a / r;
    *s = *b / r;
    
    double z;
    if (abs_a > abs_b) {
        z = *s;
    } else if (*c != 0.0) {
        z = 1.0 / *c;
    } else {
        z = 1.0;
    }
    
    *a = r;
    *b = z;
}

/* Complex rotations use real c and complex s */
void fb_ref_crotg(
    fb_complex_float_t *a,
    const fb_complex_float_t *b,
    float *c,
    fb_complex_float_t *s)
{
    /* Complex Givens rotation - more involved algorithm */
    float abs_a = sqrtf(a->real * a->real + a->imag * a->imag);
    float abs_b = sqrtf(b->real * b->real + b->imag * b->imag);
    
    if (abs_b == 0.0f) {
        *c = 1.0f;
        s->real = 0.0f;
        s->imag = 0.0f;
        return;
    }
    
    if (abs_a == 0.0f) {
        *c = 0.0f;
        s->real = 1.0f;
        s->imag = 0.0f;
        *a = *b;
        return;
    }
    
    float scale = abs_a + abs_b;
    float norm = scale * sqrtf((abs_a / scale) * (abs_a / scale) + 
                                (abs_b / scale) * (abs_b / scale));
    
    /* alpha = a / |a| */
    fb_complex_float_t alpha;
    alpha.real = a->real / abs_a;
    alpha.imag = a->imag / abs_a;
    
    *c = abs_a / norm;
    
    /* s = alpha * conj(b) / norm */
    s->real = alpha.real * b->real + alpha.imag * b->imag;
    s->imag = alpha.imag * b->real - alpha.real * b->imag;
    s->real /= norm;
    s->imag /= norm;
    
    /* a = alpha * norm */
    a->real = alpha.real * norm;
    a->imag = alpha.imag * norm;
}

void fb_ref_zrotg(
    fb_complex_double_t *a,
    const fb_complex_double_t *b,
    double *c,
    fb_complex_double_t *s)
{
    double abs_a = sqrt(a->real * a->real + a->imag * a->imag);
    double abs_b = sqrt(b->real * b->real + b->imag * b->imag);
    
    if (abs_b == 0.0) {
        *c = 1.0;
        s->real = 0.0;
        s->imag = 0.0;
        return;
    }
    
    if (abs_a == 0.0) {
        *c = 0.0;
        s->real = 1.0;
        s->imag = 0.0;
        *a = *b;
        return;
    }
    
    double scale = abs_a + abs_b;
    double norm = scale * sqrt((abs_a / scale) * (abs_a / scale) + 
                                (abs_b / scale) * (abs_b / scale));
    
    fb_complex_double_t alpha;
    alpha.real = a->real / abs_a;
    alpha.imag = a->imag / abs_a;
    
    *c = abs_a / norm;
    
    s->real = alpha.real * b->real + alpha.imag * b->imag;
    s->imag = alpha.imag * b->real - alpha.real * b->imag;
    s->real /= norm;
    s->imag /= norm;
    
    a->real = alpha.real * norm;
    a->imag = alpha.imag * norm;
}

/* ============================================================================
 * ROT - Apply Givens rotation
 * [ x[i] ]   [ c  s ] [ x[i] ]
 * [ y[i] ] = [-s  c ] [ y[i] ]
 * ========================================================================= */

void fb_ref_srot(
    const int64_t n,
    float *x,
    const int64_t incx,
    float *y,
    const int64_t incy,
    const float c,
    const float s)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        float temp_x = c * x[i * incx] + s * y[i * incy];
        float temp_y = c * y[i * incy] - s * x[i * incx];
        x[i * incx] = temp_x;
        y[i * incy] = temp_y;
    }
}

void fb_ref_drot(
    const int64_t n,
    double *x,
    const int64_t incx,
    double *y,
    const int64_t incy,
    const double c,
    const double s)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        double temp_x = c * x[i * incx] + s * y[i * incy];
        double temp_y = c * y[i * incy] - s * x[i * incx];
        x[i * incx] = temp_x;
        y[i * incy] = temp_y;
    }
}

void fb_ref_crot(
    const int64_t n,
    fb_complex_float_t *x,
    const int64_t incx,
    fb_complex_float_t *y,
    const int64_t incy,
    const float c,
    const fb_complex_float_t s)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        fb_complex_float_t *xi = &x[i * incx];
        fb_complex_float_t *yi = &y[i * incy];
        
        /* temp_x = c*x + s*y */
        fb_complex_float_t temp_x;
        temp_x.real = c * xi->real + (s.real * yi->real - s.imag * yi->imag);
        temp_x.imag = c * xi->imag + (s.real * yi->imag + s.imag * yi->real);
        
        /* temp_y = c*y - conj(s)*x */
        fb_complex_float_t temp_y;
        temp_y.real = c * yi->real - (s.real * xi->real + s.imag * xi->imag);
        temp_y.imag = c * yi->imag - (s.real * xi->imag - s.imag * xi->real);
        
        *xi = temp_x;
        *yi = temp_y;
    }
}

void fb_ref_zrot(
    const int64_t n,
    fb_complex_double_t *x,
    const int64_t incx,
    fb_complex_double_t *y,
    const int64_t incy,
    const double c,
    const fb_complex_double_t s)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    for (int64_t i = 0; i < n; i++) {
        fb_complex_double_t *xi = &x[i * incx];
        fb_complex_double_t *yi = &y[i * incy];
        
        fb_complex_double_t temp_x;
        temp_x.real = c * xi->real + (s.real * yi->real - s.imag * yi->imag);
        temp_x.imag = c * xi->imag + (s.real * yi->imag + s.imag * yi->real);
        
        fb_complex_double_t temp_y;
        temp_y.real = c * yi->real - (s.real * xi->real + s.imag * xi->imag);
        temp_y.imag = c * yi->imag - (s.real * xi->imag - s.imag * xi->real);
        
        *xi = temp_x;
        *yi = temp_y;
    }
}

/* ============================================================================
 * ROTMG - Generate modified Givens rotation
 * Constructs modified Givens transformation H such that:
 * [ sqrt(d1)*x1 ]   [ h11  h12 ] [ sqrt(d1)*x1 ]
 * [ sqrt(d2)*y1 ] = [ h21  h22 ] [ sqrt(d2)*y1 ]
 * 
 * with y1 becoming zero
 * 
 * Flag values:
 *  -2: H is identity
 *  -1: H has -h11, h21, -h12, h22 form
 *   0: H has h11, h21, -h12, h22 form
 *   1: H has -h11, h21, h12, h22 form
 * ========================================================================= */

void fb_ref_srotmg(
    float *d1,
    float *d2,
    float *x1,
    const float y1,
    float *param)
{
    /* This is one of the most complex BLAS routines.
     * Implementation follows the BLAS Technical Forum standard.
     */
    const float gam = 4096.0f;
    const float gamsq = 16777216.0f;    /* gam^2 */
    const float rgamsq = 5.9604645e-8f; /* 1/gam^2 */
    
    float flag, h11 = 0.0f, h12 = 0.0f, h21 = 0.0f, h22 = 0.0f;
    float p1, p2, q1, q2, temp, u;
    
    if (*d1 < 0.0f) {
        /* Error: d1 < 0 */
        flag = -2.0f;
        param[0] = flag;
        *d1 = 0.0f;
        *d2 = 0.0f;
        *x1 = 0.0f;
        return;
    }
    
    p2 = *d2 * y1;
    if (p2 == 0.0f) {
        flag = -2.0f;
        param[0] = flag;
        return;
    }
    
    p1 = *d1 * (*x1);
    q2 = p2 * y1;
    q1 = p1 * (*x1);
    
    if (fabsf(q1) > fabsf(q2)) {
        h21 = -y1 / (*x1);
        h12 = p2 / p1;
        u = 1.0f - h12 * h21;
        
        if (u > 0.0f) {
            flag = 0.0f;
            *d1 = *d1 / u;
            *d2 = *d2 / u;
            *x1 = (*x1) * u;
        } else {
            flag = -2.0f;
            *d1 = 0.0f;
            *d2 = 0.0f;
            *x1 = 0.0f;
        }
    } else {
        if (q2 < 0.0f) {
            flag = -2.0f;
            *d1 = 0.0f;
            *d2 = 0.0f;
            *x1 = 0.0f;
        } else {
            flag = 1.0f;
            h11 = p1 / p2;
            h22 = (*x1) / y1;
            u = 1.0f + h11 * h22;
            temp = *d2 / u;
            *d2 = *d1 / u;
            *d1 = temp;
            *x1 = y1 * u;
        }
    }
    
    /* Rescale if needed */
    if (*d1 != 0.0f) {
        while ((*d1 <= rgamsq) || (*d1 >= gamsq)) {
            if (flag == 0.0f) {
                h11 = 1.0f;
                h22 = 1.0f;
                flag = -1.0f;
            } else {
                h21 = -1.0f;
                h12 = 1.0f;
                flag = -1.0f;
            }
            
            if (*d1 <= rgamsq) {
                *d1 = *d1 * gamsq;
                *x1 = (*x1) / gam;
                h11 = h11 / gam;
                h12 = h12 / gam;
            } else {
                *d1 = *d1 / gamsq;
                *x1 = (*x1) * gam;
                h11 = h11 * gam;
                h12 = h12 * gam;
            }
        }
    }
    
    if (*d2 != 0.0f) {
        while ((fabsf(*d2) <= rgamsq) || (fabsf(*d2) >= gamsq)) {
            if (flag == 0.0f) {
                h11 = 1.0f;
                h22 = 1.0f;
                flag = -1.0f;
            } else {
                h21 = -1.0f;
                h12 = 1.0f;
                flag = -1.0f;
            }
            
            if (fabsf(*d2) <= rgamsq) {
                *d2 = *d2 * gamsq;
                h21 = h21 / gam;
                h22 = h22 / gam;
            } else {
                *d2 = *d2 / gamsq;
                h21 = h21 * gam;
                h22 = h22 * gam;
            }
        }
    }
    
    param[0] = flag;
    if (flag == -1.0f) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == 0.0f) {
        param[2] = h21;
        param[3] = h12;
    } else if (flag == 1.0f) {
        param[1] = h11;
        param[4] = h22;
    }
}

void fb_ref_drotmg(
    double *d1,
    double *d2,
    double *x1,
    const double y1,
    double *param)
{
    const double gam = 4096.0;
    const double gamsq = 16777216.0;
    const double rgamsq = 5.9604645e-8;
    
    double flag, h11 = 0.0, h12 = 0.0, h21 = 0.0, h22 = 0.0;
    double p1, p2, q1, q2, temp, u;
    
    if (*d1 < 0.0) {
        flag = -2.0;
        param[0] = flag;
        *d1 = 0.0;
        *d2 = 0.0;
        *x1 = 0.0;
        return;
    }
    
    p2 = *d2 * y1;
    if (p2 == 0.0) {
        flag = -2.0;
        param[0] = flag;
        return;
    }
    
    p1 = *d1 * (*x1);
    q2 = p2 * y1;
    q1 = p1 * (*x1);
    
    if (fabs(q1) > fabs(q2)) {
        h21 = -y1 / (*x1);
        h12 = p2 / p1;
        u = 1.0 - h12 * h21;
        
        if (u > 0.0) {
            flag = 0.0;
            *d1 = *d1 / u;
            *d2 = *d2 / u;
            *x1 = (*x1) * u;
        } else {
            flag = -2.0;
            *d1 = 0.0;
            *d2 = 0.0;
            *x1 = 0.0;
        }
    } else {
        if (q2 < 0.0) {
            flag = -2.0;
            *d1 = 0.0;
            *d2 = 0.0;
            *x1 = 0.0;
        } else {
            flag = 1.0;
            h11 = p1 / p2;
            h22 = (*x1) / y1;
            u = 1.0 + h11 * h22;
            temp = *d2 / u;
            *d2 = *d1 / u;
            *d1 = temp;
            *x1 = y1 * u;
        }
    }
    
    if (*d1 != 0.0) {
        while ((*d1 <= rgamsq) || (*d1 >= gamsq)) {
            if (flag == 0.0) {
                h11 = 1.0;
                h22 = 1.0;
                flag = -1.0;
            } else {
                h21 = -1.0;
                h12 = 1.0;
                flag = -1.0;
            }
            
            if (*d1 <= rgamsq) {
                *d1 = *d1 * gamsq;
                *x1 = (*x1) / gam;
                h11 = h11 / gam;
                h12 = h12 / gam;
            } else {
                *d1 = *d1 / gamsq;
                *x1 = (*x1) * gam;
                h11 = h11 * gam;
                h12 = h12 * gam;
            }
        }
    }
    
    if (*d2 != 0.0) {
        while ((fabs(*d2) <= rgamsq) || (fabs(*d2) >= gamsq)) {
            if (flag == 0.0) {
                h11 = 1.0;
                h22 = 1.0;
                flag = -1.0;
            } else {
                h21 = -1.0;
                h12 = 1.0;
                flag = -1.0;
            }
            
            if (fabs(*d2) <= rgamsq) {
                *d2 = *d2 * gamsq;
                h21 = h21 / gam;
                h22 = h22 / gam;
            } else {
                *d2 = *d2 / gamsq;
                h21 = h21 * gam;
                h22 = h22 * gam;
            }
        }
    }
    
    param[0] = flag;
    if (flag == -1.0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == 0.0) {
        param[2] = h21;
        param[3] = h12;
    } else if (flag == 1.0) {
        param[1] = h11;
        param[4] = h22;
    }
}

/* ============================================================================
 * ROTM - Apply modified Givens rotation
 * Uses parameter array from ROTMG
 * ========================================================================= */

void fb_ref_srotm(
    const int64_t n,
    float *x,
    const int64_t incx,
    float *y,
    const int64_t incy,
    const float *param)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    float flag = param[0];
    
    if (flag == -2.0f) {
        /* Identity transformation */
        return;
    }
    
    float h11, h12, h21, h22;
    
    if (flag == -1.0f) {
        h11 = param[1];
        h12 = param[3];
        h21 = param[2];
        h22 = param[4];
    } else if (flag == 0.0f) {
        h11 = 1.0f;
        h12 = param[3];
        h21 = param[2];
        h22 = 1.0f;
    } else if (flag == 1.0f) {
        h11 = param[1];
        h12 = 1.0f;
        h21 = -1.0f;
        h22 = param[4];
    } else {
        /* Invalid flag */
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        float w = x[i * incx];
        float z = y[i * incy];
        x[i * incx] = w * h11 + z * h12;
        y[i * incy] = w * h21 + z * h22;
    }
}

void fb_ref_drotm(
    const int64_t n,
    double *x,
    const int64_t incx,
    double *y,
    const int64_t incy,
    const double *param)
{
    if (n <= 0) return;
    if (incx == 0 || incy == 0) return;
    
    double flag = param[0];
    
    if (flag == -2.0) {
        return;
    }
    
    double h11, h12, h21, h22;
    
    if (flag == -1.0) {
        h11 = param[1];
        h12 = param[3];
        h21 = param[2];
        h22 = param[4];
    } else if (flag == 0.0) {
        h11 = 1.0;
        h12 = param[3];
        h21 = param[2];
        h22 = 1.0;
    } else if (flag == 1.0) {
        h11 = param[1];
        h12 = 1.0;
        h21 = -1.0;
        h22 = param[4];
    } else {
        return;
    }
    
    for (int64_t i = 0; i < n; i++) {
        double w = x[i * incx];
        double z = y[i * incy];
        x[i * incx] = w * h11 + z * h12;
        y[i * incy] = w * h21 + z * h22;
    }
}
