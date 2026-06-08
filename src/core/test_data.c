/**
 * @file test_data.c
 * @brief Test data generation implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "test_data.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Simple XorShift random number generator for reproducibility */
static uint g_rng_state = 12345;

void fb_test_init_rng(uint32_t seed) {
    g_rng_state = seed ? seed : 12345;
}

static uint xorshift64(void) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 7;
    g_rng_state ^= g_rng_state << 17;
    return g_rng_state;
}

static double random_double(void) {
    return (xorshift64() & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
}

static float random_float(void) {
    return (float)random_double();
}

size_t fb_dtype_size(fb_dtype_t dtype) {
    switch (dtype) {
        case FB_DTYPE_FLOAT32:   return sizeof(float);
        case FB_DTYPE_FLOAT64:   return sizeof(double);
        case FB_DTYPE_COMPLEX64: return sizeof(float) * 2;
        case FB_DTYPE_COMPLEX128: return sizeof(double) * 2;
        case FB_DTYPE_INT8:      return sizeof(int8_t);
        case FB_DTYPE_INT16:     return sizeof(int16_t);
        case FB_DTYPE_INT32:     return sizeof(int32_t);
        case FB_DTYPE_BFLOAT16:  return sizeof(uint16_t);
        case FB_DTYPE_FLOAT16:   return sizeof(uint16_t);
        default: return 0;
    }
}

int fb_test_generate_matrix(
    void *matrix,
    size_t rows,
    size_t cols,
    fb_dtype_t dtype,
    fb_test_matrix_type_t type,
    uint32_t seed)
{
    fb_test_init_rng(seed);
    
    switch (dtype) {
        case FB_DTYPE_FLOAT32: {
            float *m = (float*)matrix;
            switch (type) {
                case FB_TEST_RANDOM:
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = random_float();
                    }
                    break;
                    
                case FB_TEST_IDENTITY:
                    memset(m, 0, rows * cols * sizeof(float));
                    for (size_t i = 0; i < (rows < cols ? rows : cols); i++) {
                        m[i * cols + i] = 1.0f;
                    }
                    break;
                    
                case FB_TEST_ZEROS:
                    memset(m, 0, rows * cols * sizeof(float));
                    break;
                    
                case FB_TEST_ONES:
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = 1.0f;
                    }
                    break;
                    
                case FB_TEST_DIAGONAL:
                    memset(m, 0, rows * cols * sizeof(float));
                    for (size_t i = 0; i < (rows < cols ? rows : cols); i++) {
                        m[i * cols + i] = random_float() * 10.0f;
                    }
                    break;
                    
                case FB_TEST_SYMMETRIC:
                    if (rows != cols) return -1;
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j <= i; j++) {
                            float val = random_float();
                            m[i * cols + j] = val;
                            m[j * cols + i] = val;
                        }
                    }
                    break;
                    
                case FB_TEST_TRIANGULAR:
                    memset(m, 0, rows * cols * sizeof(float));
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j <= i && j < cols; j++) {
                            m[i * cols + j] = random_float();
                        }
                    }
                    break;
                    
                case FB_TEST_ILL_CONDITIONED:
                    /* Create matrix with very large condition number */
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = random_float();
                    }
                    /* Make one row very small */
                    if (rows > 0) {
                        for (size_t j = 0; j < cols; j++) {
                            m[j] *= 1e-8f;
                        }
                    }
                    break;
            }
            break;
        }
        
        case FB_DTYPE_FLOAT64: {
            double *m = (double*)matrix;
            switch (type) {
                case FB_TEST_RANDOM:
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = random_double();
                    }
                    break;
                    
                case FB_TEST_IDENTITY:
                    memset(m, 0, rows * cols * sizeof(double));
                    for (size_t i = 0; i < (rows < cols ? rows : cols); i++) {
                        m[i * cols + i] = 1.0;
                    }
                    break;
                    
                case FB_TEST_ZEROS:
                    memset(m, 0, rows * cols * sizeof(double));
                    break;
                    
                case FB_TEST_ONES:
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = 1.0;
                    }
                    break;
                    
                case FB_TEST_DIAGONAL:
                    memset(m, 0, rows * cols * sizeof(double));
                    for (size_t i = 0; i < (rows < cols ? rows : cols); i++) {
                        m[i * cols + i] = random_double() * 10.0;
                    }
                    break;
                    
                case FB_TEST_SYMMETRIC:
                    if (rows != cols) return -1;
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j <= i; j++) {
                            double val = random_double();
                            m[i * cols + j] = val;
                            m[j * cols + i] = val;
                        }
                    }
                    break;
                    
                case FB_TEST_TRIANGULAR:
                    memset(m, 0, rows * cols * sizeof(double));
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j <= i && j < cols; j++) {
                            m[i * cols + j] = random_double();
                        }
                    }
                    break;
                    
                case FB_TEST_ILL_CONDITIONED:
                    for (size_t i = 0; i < rows * cols; i++) {
                        m[i] = random_double();
                    }
                    if (rows > 0) {
                        for (size_t j = 0; j < cols; j++) {
                            m[j] *= 1e-15;
                        }
                    }
                    break;
            }
            break;
        }
        
        default:
            return -1;  /* Unsupported dtype for now */
    }
    
    return 0;
}

int fb_test_generate_vector(
    void *vector,
    size_t length,
    fb_dtype_t dtype,
    fb_test_matrix_type_t type,
    uint32_t seed)
{
    return fb_test_generate_matrix(vector, 1, length, dtype, type, seed);
}

bool fb_test_compare_arrays(
    const void *expected,
    const void *actual,
    size_t length,
    fb_dtype_t dtype,
    double tolerance,
    double *max_error,
    double *avg_error)
{
    double max_err = 0.0;
    double sum_err = 0.0;
    bool passed = true;
    
    switch (dtype) {
        case FB_DTYPE_FLOAT32: {
            const float *exp = (const float*)expected;
            const float *act = (const float*)actual;
            
            for (size_t i = 0; i < length; i++) {
                double err = fabs((double)exp[i] - (double)act[i]);
                if (err > max_err) max_err = err;
                sum_err += err;
                if (err > tolerance) passed = false;
            }
            break;
        }
        
        case FB_DTYPE_FLOAT64: {
            const double *exp = (const double*)expected;
            const double *act = (const double*)actual;
            
            for (size_t i = 0; i < length; i++) {
                double err = fabs(exp[i] - act[i]);
                if (err > max_err) max_err = err;
                sum_err += err;
                if (err > tolerance) passed = false;
            }
            break;
        }
        
        case FB_DTYPE_COMPLEX64: {
            const float *exp = (const float*)expected;
            const float *act = (const float*)actual;
            
            for (size_t i = 0; i < length * 2; i++) {
                double err = fabs((double)exp[i] - (double)act[i]);
                if (err > max_err) max_err = err;
                sum_err += err;
                if (err > tolerance) passed = false;
            }
            break;
        }
        
        case FB_DTYPE_COMPLEX128: {
            const double *exp = (const double*)expected;
            const double *act = (const double*)actual;
            
            for (size_t i = 0; i < length * 2; i++) {
                double err = fabs(exp[i] - act[i]);
                if (err > max_err) max_err = err;
                sum_err += err;
                if (err > tolerance) passed = false;
            }
            break;
        }
        
        default:
            return false;
    }
    
    if (max_error) *max_error = max_err;
    if (avg_error) *avg_error = sum_err / length;
    
    return passed;
}

double fb_test_condition_number(
    const void *matrix,
    size_t rows,
    size_t cols,
    fb_dtype_t dtype)
{
    /* Simple estimate using max/min singular value approximation */
    /* For now, return a placeholder - full SVD would be needed for accuracy */
    
    if (rows != cols) return -1.0;  /* Only for square matrices */
    
    double max_val = 0.0;
    double min_val = DBL_MAX;
    
    switch (dtype) {
        case FB_DTYPE_FLOAT32: {
            const float *m = (const float*)matrix;
            for (size_t i = 0; i < rows; i++) {
                double diag = fabs((double)m[i * cols + i]);
                if (diag > max_val) max_val = diag;
                if (diag < min_val && diag > 0.0) min_val = diag;
            }
            break;
        }
        
        case FB_DTYPE_FLOAT64: {
            const double *m = (const double*)matrix;
            for (size_t i = 0; i < rows; i++) {
                double diag = fabs(m[i * cols + i]);
                if (diag > max_val) max_val = diag;
                if (diag < min_val && diag > 0.0) min_val = diag;
            }
            break;
        }
        
        default:
            return -1.0;
    }
    
    if (min_val <= 0.0 || min_val == DBL_MAX) return -1.0;
    
    /* Rough estimate: condition number ≈ max_diag / min_diag */
    return max_val / min_val;
}
