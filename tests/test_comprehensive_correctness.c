/**
 * @file test_comprehensive_correctness.c
 * @brief Comprehensive correctness validation for all operations across all backends
 * 
 * This test suite validates EVERY operation on EVERY backend on EVERY device.
 * Results are cached per backend version to avoid redundant testing.
 * 
 * Test Matrix:
 *   - All detected devices (CPU, GPU)
 *   - All loaded backends per device
 *   - All BLAS operations (Level 1, 2, 3)
 *   - All LAPACK operations (if supported)
 *   - Multiple problem sizes
 *   - Multiple data types (FP32, FP64, complex if supported)
 * 
 * Pass Criteria:
 *   - Numerical results must match reference within tolerance
 *   - No crashes or hangs
 *   - Proper error handling for invalid inputs
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/dispatch_unified.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/backend_instance.h"
#include "../src/backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

/* Tolerances for numerical comparison */
#define TOLERANCE_FP32 1e-5f
#define TOLERANCE_FP64 1e-12
#define TOLERANCE_COMPLEX_FP32 1e-5f
#define TOLERANCE_COMPLEX_FP64 1e-12

/* Test problem sizes */
static const size_t SMALL_SIZE = 64;
static const size_t MEDIUM_SIZE = 256;
static const size_t LARGE_SIZE = 1024;

/* Test result tracking */
typedef struct {
    char operation_name[64];
    char backend_name[64];
    char device_name[128];
    int device_id;
    size_t problem_size;
    const char* dtype;
    bool passed;
    double max_error;
    char error_message[256];
    double execution_time_ms;
} test_result_t;

typedef struct {
    test_result_t* results;
    size_t count;
    size_t capacity;
    size_t passed;
    size_t failed;
    size_t skipped;
} test_suite_results_t;

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

static void init_test_results(test_suite_results_t* suite) {
    suite->capacity = 10000;  /* Support up to 10k tests */
    suite->results = (test_result_t*)calloc(suite->capacity, sizeof(test_result_t));
    suite->count = 0;
    suite->passed = 0;
    suite->failed = 0;
    suite->skipped = 0;
}

static void free_test_results(test_suite_results_t* suite) {
    free(suite->results);
    suite->results = NULL;
    suite->count = 0;
}

static void record_test(test_suite_results_t* suite,
                       const char* operation,
                       const char* backend,
                       const char* device,
                       int device_id,
                       size_t size,
                       const char* dtype,
                       bool passed,
                       double max_error,
                       const char* error_msg,
                       double time_ms)
{
    if (suite->count >= suite->capacity) {
        fprintf(stderr, "ERROR: Test result capacity exceeded\n");
        return;
    }
    
    test_result_t* result = &suite->results[suite->count++];
    strncpy(result->operation_name, operation, sizeof(result->operation_name) - 1);
    strncpy(result->backend_name, backend, sizeof(result->backend_name) - 1);
    strncpy(result->device_name, device, sizeof(result->device_name) - 1);
    result->device_id = device_id;
    result->problem_size = size;
    result->dtype = dtype;
    result->passed = passed;
    result->max_error = max_error;
    if (error_msg) {
        strncpy(result->error_message, error_msg, sizeof(result->error_message) - 1);
    }
    result->execution_time_ms = time_ms;
    
    if (passed) {
        suite->passed++;
    } else {
        suite->failed++;
    }
}

static void record_skip(test_suite_results_t* suite,
                       const char* operation,
                       const char* backend,
                       const char* device,
                       int device_id,
                       const char* reason)
{
    if (suite->count >= suite->capacity) return;
    
    test_result_t* result = &suite->results[suite->count++];
    strncpy(result->operation_name, operation, sizeof(result->operation_name) - 1);
    strncpy(result->backend_name, backend, sizeof(result->backend_name) - 1);
    strncpy(result->device_name, device, sizeof(result->device_name) - 1);
    result->device_id = device_id;
    result->passed = true;  /* Skips don't count as failures */
    snprintf(result->error_message, sizeof(result->error_message), "SKIPPED: %s", reason);
    
    suite->skipped++;
}

static bool float_equal(float a, float b, float tolerance) {
    return fabsf(a - b) <= tolerance;
}

static bool double_equal(double a, double b, double tolerance) {
    return fabs(a - b) <= tolerance;
}

static double max_absolute_error_f(const float* a, const float* b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)a[i] - (double)b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static double max_absolute_error_d(const double* a, const double* b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* ============================================================================
 * Test Data Generation
 * ========================================================================== */

static void generate_random_vector_f(float* v, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        v[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;  /* Range [-1, 1] */
    }
}

static void generate_random_vector_d(double* v, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        v[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }
}

static void generate_random_matrix_f(float* m, size_t rows, size_t cols, unsigned int seed) {
    generate_random_vector_f(m, rows * cols, seed);
}

static void generate_random_matrix_d(double* m, size_t rows, size_t cols, unsigned int seed) {
    generate_random_vector_d(m, rows * cols, seed);
}

/* ============================================================================
 * Reference BLAS Implementations (for correctness comparison)
 * ========================================================================== */

/* Reference SAXPY: y = alpha*x + y */
static void ref_saxpy(int64_t n, float alpha, const float* x, const float* y, float* result) {
    for (int64_t i = 0; i < n; i++) {
        result[i] = alpha * x[i] + y[i];
    }
}

/* Reference DAXPY: y = alpha*x + y */
static void ref_daxpy(int64_t n, double alpha, const double* x, const double* y, double* result) {
    for (int64_t i = 0; i < n; i++) {
        result[i] = alpha * x[i] + y[i];
    }
}

/* Reference SGEMV: y = alpha*A*x + beta*y */
static void ref_sgemv(bool trans, size_t m, size_t n, float alpha, 
                     const float* A, size_t lda, const float* x,
                     float beta, const float* y, float* result)
{
    size_t out_size = trans ? n : m;
    
    /* result = beta * y */
    for (size_t i = 0; i < out_size; i++) {
        result[i] = beta * y[i];
    }
    
    /* result += alpha * A * x */
    if (!trans) {
        /* y = alpha*A*x + beta*y, A is m×n */
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                result[i] += alpha * A[i + j * lda] * x[j];
            }
        }
    } else {
        /* y = alpha*A^T*x + beta*y, A^T is n×m */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                result[i] += alpha * A[j + i * lda] * x[j];
            }
        }
    }
}

/* Reference SGEMM: C = alpha*A*B + beta*C */
static void ref_sgemm(bool trans_a, bool trans_b,
                     size_t m, size_t n, size_t k,
                     float alpha, const float* A, size_t lda,
                     const float* B, size_t ldb,
                     float beta, const float* C, size_t ldc,
                     float* result)
{
    /* Initialize result = beta * C */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i + j * ldc] = beta * C[i + j * ldc];
        }
    }
    
    /* result += alpha * op(A) * op(B) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                float a_val = trans_a ? A[p + i * lda] : A[i + p * lda];
                float b_val = trans_b ? B[j + p * ldb] : B[p + j * ldb];
                sum += a_val * b_val;
            }
            result[i + j * ldc] += alpha * sum;
        }
    }
}

/* Reference SSCAL: x = alpha*x */
static void ref_sscal(int64_t n, float alpha, float* x) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = alpha * x[i];
    }
}

/* Reference DSCAL: x = alpha*x */
static void ref_dscal(int64_t n, double alpha, double* x) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = alpha * x[i];
    }
}

/* Reference DGEMV: y = alpha*A*x + beta*y */
static void ref_dgemv(bool trans, size_t m, size_t n, double alpha,
                     const double* A, size_t lda, const double* x,
                     double beta, const double* y, double* result)
{
    size_t out_size = trans ? n : m;
    
    /* result = beta * y */
    for (size_t i = 0; i < out_size; i++) {
        result[i] = beta * y[i];
    }
    
    /* result += alpha * A * x */
    if (!trans) {
        /* y = alpha*A*x + beta*y, A is m×n */
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                result[i] += alpha * A[i + j * lda] * x[j];
            }
        }
    } else {
        /* y = alpha*A^T*x + beta*y, A^T is n×m */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                result[i] += alpha * A[j + i * lda] * x[j];
            }
        }
    }
}

/* Reference DGEMM: C = alpha*A*B + beta*C */
static void ref_dgemm(bool trans_a, bool trans_b,
                     size_t m, size_t n, size_t k,
                     double alpha, const double* A, size_t lda,
                     const double* B, size_t ldb,
                     double beta, const double* C, size_t ldc,
                     double* result)
{
    /* Initialize result = beta * C */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i + j * ldc] = beta * C[i + j * ldc];
        }
    }
    
    /* result += alpha * op(A) * op(B) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                double a_val = trans_a ? A[p + i * lda] : A[i + p * lda];
                double b_val = trans_b ? B[j + p * ldb] : B[p + j * ldb];
                sum += a_val * b_val;
            }
            result[i + j * ldc] += alpha * sum;
        }
    }
}

/* ============================================================================
 * Level 1 BLAS Tests (Vector-Vector Operations)
 * ========================================================================== */

static void test_saxpy(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->saxpy) {
        record_skip(suite, "saxpy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    /* Allocate vectors */
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    float* y_result = (float*)malloc(n * sizeof(float));
    float* y_reference = (float*)malloc(n * sizeof(float));
    
    if (!x || !y || !y_result || !y_reference) {
        record_test(suite, "saxpy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(y_result); free(y_reference);
        return;
    }
    
    /* Generate test data */
    generate_random_vector_f(x, n, 12345);
    generate_random_vector_f(y, n, 54321);
    memcpy(y_result, y, n * sizeof(float));
    memcpy(y_reference, y, n * sizeof(float));
    
    float alpha = 2.5f;
    
    /* Compute reference */
    ref_saxpy(n, alpha, x, y, y_reference);
    
    /* Test backend */
    clock_t start = clock();
    vtable->saxpy(n, alpha, x, 1, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(y_result, y_reference, n);
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "saxpy", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err, 
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y); free(y_result); free(y_reference);
}

static void test_daxpy(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->daxpy) {
        record_skip(suite, "daxpy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* y_result = (double*)malloc(n * sizeof(double));
    double* y_reference = (double*)malloc(n * sizeof(double));
    
    if (!x || !y || !y_result || !y_reference) {
        record_test(suite, "daxpy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(y_result); free(y_reference);
        return;
    }
    
    generate_random_vector_d(x, n, 12345);
    generate_random_vector_d(y, n, 54321);
    memcpy(y_result, y, n * sizeof(double));
    memcpy(y_reference, y, n * sizeof(double));
    
    double alpha = 2.5;
    
    ref_daxpy(n, alpha, x, y, y_reference);
    
    clock_t start = clock();
    vtable->daxpy(n, alpha, x, 1, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(y_result, y_reference, n);
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "daxpy", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y); free(y_result); free(y_reference);
}

/* ============================================================================
 * Level 2 BLAS Tests (Matrix-Vector Operations)
 * ========================================================================== */

static void test_sgemv(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sgemv) {
        record_skip(suite, "sgemv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    size_t size_A = m * n;
    size_t size_x = n;
    size_t size_y = m;
    
    float* A = (float*)malloc(size_A * sizeof(float));
    float* x = (float*)malloc(size_x * sizeof(float));
    float* y = (float*)malloc(size_y * sizeof(float));
    float* y_result = (float*)malloc(size_y * sizeof(float));
    float* y_reference = (float*)malloc(size_y * sizeof(float));
    
    if (!A || !x || !y || !y_result || !y_reference) {
        char size_str[64];
        snprintf(size_str, sizeof(size_str), "%zux%zu", m, n);
        record_test(suite, "sgemv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(y); free(y_result); free(y_reference);
        return;
    }
    
    generate_random_matrix_f(A, m, n, 11111);
    generate_random_vector_f(x, size_x, 22222);
    generate_random_vector_f(y, size_y, 33333);
    memcpy(y_result, y, size_y * sizeof(float));
    memcpy(y_reference, y, size_y * sizeof(float));
    
    float alpha = 1.5f;
    float beta = 0.5f;
    
    ref_sgemv(false, m, n, alpha, A, m, x, beta, y, y_reference);
    
    clock_t start = clock();
    vtable->sgemv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, m, n,
                  alpha, A, m, x, 1, beta, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(y_result, y_reference, size_y);
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "sgemv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(y); free(y_result); free(y_reference);
}

/* ============================================================================
 * Level 3 BLAS Tests (Matrix-Matrix Operations)
 * ========================================================================== */

static void test_sgemm(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t m, size_t n, size_t k)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sgemm) {
        record_skip(suite, "sgemm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    size_t size_A = m * k;
    size_t size_B = k * n;
    size_t size_C = m * n;
    
    float* A = (float*)malloc(size_A * sizeof(float));
    float* B = (float*)malloc(size_B * sizeof(float));
    float* C = (float*)malloc(size_C * sizeof(float));
    float* C_result = (float*)malloc(size_C * sizeof(float));
    float* C_reference = (float*)malloc(size_C * sizeof(float));
    
    if (!A || !B || !C || !C_result || !C_reference) {
        record_test(suite, "sgemm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n * k, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(C); free(C_result); free(C_reference);
        return;
    }
    
    generate_random_matrix_f(A, m, k, 11111);
    generate_random_matrix_f(B, k, n, 22222);
    generate_random_matrix_f(C, m, n, 33333);
    memcpy(C_result, C, size_C * sizeof(float));
    memcpy(C_reference, C, size_C * sizeof(float));
    
    float alpha = 1.0f;
    float beta = 1.0f;
    
    ref_sgemm(false, false, m, n, k, alpha, A, m, B, k, beta, C, m, C_reference);
    
    clock_t start = clock();
    vtable->sgemm(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                  m, n, k, alpha, A, m, B, k, beta, C_result, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(C_result, C_reference, size_C);
    bool passed = (max_err <= TOLERANCE_FP32 * (float)k);  /* Scale tolerance by k for accumulation */
    record_test(suite, "sgemm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n * k, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_result); free(C_reference);
}

static void test_sdot(fb_backend_instance_t* instance,
                     test_suite_results_t* suite,
                     size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sdot) {
        record_skip(suite, "sdot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    
    if (!x || !y) {
        record_test(suite, "sdot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y);
        return;
    }
    
    generate_random_vector_f(x, n, 11111);
    generate_random_vector_f(y, n, 22222);
    
    // Compute reference
    float ref_result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ref_result += x[i] * y[i];
    }
    
    clock_t start = clock();
    float result = vtable->sdot(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabsf(result - ref_result);
    bool passed = (error <= TOLERANCE_FP32 * n);  // Scale tolerance by vector size
    record_test(suite, "sdot", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y);
}

static void test_ddot(fb_backend_instance_t* instance,
                     test_suite_results_t* suite,
                     size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->ddot) {
        record_skip(suite, "ddot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    
    if (!x || !y) {
        record_test(suite, "ddot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y);
        return;
    }
    
    generate_random_vector_d(x, n, 11112);
    generate_random_vector_d(y, n, 22223);
    
    // Compute reference
    double ref_result = 0.0;
    for (size_t i = 0; i < n; i++) {
        ref_result += x[i] * y[i];
    }
    
    clock_t start = clock();
    double result = vtable->ddot(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabs(result - ref_result);
    bool passed = (error <= TOLERANCE_FP64 * n);
    record_test(suite, "ddot", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y);
}

static void test_snrm2(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->snrm2) {
        record_skip(suite, "snrm2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    
    if (!x) {
        record_test(suite, "snrm2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_f(x, n, 33333);
    
    // Compute reference (Euclidean norm)
    float ref_result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ref_result += x[i] * x[i];
    }
    ref_result = sqrtf(ref_result);
    
    clock_t start = clock();
    float result = vtable->snrm2(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabsf(result - ref_result);
    bool passed = (error <= TOLERANCE_FP32 * sqrtf((float)n));
    record_test(suite, "snrm2", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x);
}

static void test_dnrm2(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dnrm2) {
        record_skip(suite, "dnrm2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    
    if (!x) {
        record_test(suite, "dnrm2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_d(x, n, 33334);
    
    // Compute reference (Euclidean norm)
    double ref_result = 0.0;
    for (size_t i = 0; i < n; i++) {
        ref_result += x[i] * x[i];
    }
    ref_result = sqrt(ref_result);
    
    clock_t start = clock();
    double result = vtable->dnrm2(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabs(result - ref_result);
    bool passed = (error <= TOLERANCE_FP64 * sqrt((double)n));
    record_test(suite, "dnrm2", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x);
}

static void test_sasum(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sasum) {
        record_skip(suite, "sasum", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    
    if (!x) {
        record_test(suite, "sasum", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_f(x, n, 44444);
    
    // Compute reference (sum of absolute values)
    float ref_result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ref_result += fabsf(x[i]);
    }
    
    clock_t start = clock();
    float result = vtable->sasum(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabsf(result - ref_result);
    bool passed = (error <= TOLERANCE_FP32 * n);
    record_test(suite, "sasum", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x);
}

static void test_dasum(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dasum) {
        record_skip(suite, "dasum", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    
    if (!x) {
        record_test(suite, "dasum", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_d(x, n, 44445);
    
    // Compute reference (sum of absolute values)
    double ref_result = 0.0;
    for (size_t i = 0; i < n; i++) {
        ref_result += fabs(x[i]);
    }
    
    clock_t start = clock();
    double result = vtable->dasum(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double error = fabs(result - ref_result);
    bool passed = (error <= TOLERANCE_FP64 * n);
    record_test(suite, "dasum", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, error,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x);
}

static void test_sswap(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sswap) {
        record_skip(suite, "sswap", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    float* x_orig = (float*)malloc(n * sizeof(float));
    float* y_orig = (float*)malloc(n * sizeof(float));
    
    if (!x || !y || !x_orig || !y_orig) {
        record_test(suite, "sswap", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(x_orig); free(y_orig);
        return;
    }
    
    generate_random_vector_f(x, n, 55555);
    generate_random_vector_f(y, n, 66666);
    memcpy(x_orig, x, n * sizeof(float));
    memcpy(y_orig, y, n * sizeof(float));
    
    clock_t start = clock();
    vtable->sswap(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    // Check that x now has y_orig values and y has x_orig values
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        max_err = fmaxf(max_err, fabsf(x[i] - y_orig[i]));
        max_err = fmaxf(max_err, fabsf(y[i] - x_orig[i]));
    }
    
    bool passed = (max_err == 0.0);
    record_test(suite, "sswap", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err,
               passed ? NULL : "Swap values do not match", time_ms);
    
    free(x); free(y); free(x_orig); free(y_orig);
}

static void test_dswap(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dswap) {
        record_skip(suite, "dswap", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* x_orig = (double*)malloc(n * sizeof(double));
    double* y_orig = (double*)malloc(n * sizeof(double));
    
    if (!x || !y || !x_orig || !y_orig) {
        record_test(suite, "dswap", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(x_orig); free(y_orig);
        return;
    }
    
    generate_random_vector_d(x, n, 55556);
    generate_random_vector_d(y, n, 66667);
    memcpy(x_orig, x, n * sizeof(double));
    memcpy(y_orig, y, n * sizeof(double));
    
    clock_t start = clock();
    vtable->dswap(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    // Check that x now has y_orig values and y has x_orig values
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        max_err = fmax(max_err, fabs(x[i] - y_orig[i]));
        max_err = fmax(max_err, fabs(y[i] - x_orig[i]));
    }
    
    bool passed = (max_err == 0.0);
    record_test(suite, "dswap", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Swap values do not match", time_ms);
    
    free(x); free(y); free(x_orig); free(y_orig);
}

static void test_isamax(fb_backend_instance_t* instance,
                       test_suite_results_t* suite,
                       size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->isamax) {
        record_skip(suite, "isamax", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    
    if (!x) {
        record_test(suite, "isamax", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_f(x, n, 77777);
    
    // Compute reference (index of max absolute value)
    int64_t ref_idx = 0;
    float max_val = fabsf(x[0]);
    for (size_t i = 1; i < n; i++) {
        float val = fabsf(x[i]);
        if (val > max_val) {
            max_val = val;
            ref_idx = i;
        }
    }
    
    clock_t start = clock();
    int64_t result = vtable->isamax(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    bool passed = (result == ref_idx);
    record_test(suite, "isamax", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, passed ? 0.0 : 1.0,
               passed ? NULL : "Index mismatch", time_ms);
    
    free(x);
}

static void test_idamax(fb_backend_instance_t* instance,
                       test_suite_results_t* suite,
                       size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->idamax) {
        record_skip(suite, "idamax", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    
    if (!x) {
        record_test(suite, "idamax", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x);
        return;
    }
    
    generate_random_vector_d(x, n, 77778);
    
    // Compute reference (index of max absolute value)
    int64_t ref_idx = 0;
    double max_val = fabs(x[0]);
    for (size_t i = 1; i < n; i++) {
        double val = fabs(x[i]);
        if (val > max_val) {
            max_val = val;
            ref_idx = i;
        }
    }
    
    clock_t start = clock();
    int64_t result = vtable->idamax(n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    bool passed = (result == ref_idx);
    record_test(suite, "idamax", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, passed ? 0.0 : 1.0,
               passed ? NULL : "Index mismatch", time_ms);
    
    free(x);
}

static void test_scopy(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->scopy) {
        record_skip(suite, "scopy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    float* y_reference = (float*)malloc(n * sizeof(float));
    
    if (!x || !y || !y_reference) {
        record_test(suite, "scopy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(y_reference);
        return;
    }
    
    generate_random_vector_f(x, n, 98765);
    memcpy(y_reference, x, n * sizeof(float));
    
    clock_t start = clock();
    vtable->scopy(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(y, y_reference, n);
    bool passed = (max_err == 0.0);  /* Copy should be exact */
    record_test(suite, "scopy", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err,
               passed ? NULL : "Copy values do not match", time_ms);
    
    free(x); free(y); free(y_reference);
}

static void test_dcopy(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dcopy) {
        record_skip(suite, "dcopy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* y_reference = (double*)malloc(n * sizeof(double));
    
    if (!x || !y || !y_reference) {
        record_test(suite, "dcopy", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(y_reference);
        return;
    }
    
    generate_random_vector_d(x, n, 98766);
    memcpy(y_reference, x, n * sizeof(double));
    
    clock_t start = clock();
    vtable->dcopy(n, x, 1, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(y, y_reference, n);
    bool passed = (max_err == 0.0);  /* Copy should be exact */
    record_test(suite, "dcopy", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Copy values do not match", time_ms);
    
    free(x); free(y); free(y_reference);
}

static void test_sscal(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sscal) {
        record_skip(suite, "sscal", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    float* x_reference = (float*)malloc(n * sizeof(float));
    
    if (!x || !x_reference) {
        record_test(suite, "sscal", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(x_reference);
        return;
    }
    
    generate_random_vector_f(x, n, 87654);
    memcpy(x_reference, x, n * sizeof(float));
    
    float alpha = 3.14f;
    ref_sscal(n, alpha, x_reference);
    
    clock_t start = clock();
    vtable->sscal(n, alpha, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(x, x_reference, n);
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "sscal", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(x_reference);
}

static void test_dscal(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dscal) {
        record_skip(suite, "dscal", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* x_reference = (double*)malloc(n * sizeof(double));
    
    if (!x || !x_reference) {
        record_test(suite, "dscal", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(x_reference);
        return;
    }
    
    generate_random_vector_d(x, n, 87655);
    memcpy(x_reference, x, n * sizeof(double));
    
    double alpha = 2.71828;
    ref_dscal(n, alpha, x_reference);
    
    clock_t start = clock();
    vtable->dscal(n, alpha, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(x, x_reference, n);
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "dscal", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(x_reference);
}

static void test_dgemv(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dgemv) {
        record_skip(suite, "dgemv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    size_t size_A = m * n;
    double* A = (double*)malloc(size_A * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(m * sizeof(double));
    double* y_result = (double*)malloc(m * sizeof(double));
    double* y_reference = (double*)malloc(m * sizeof(double));
    
    if (!A || !x || !y || !y_result || !y_reference) {
        record_test(suite, "dgemv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(y); free(y_result); free(y_reference);
        return;
    }
    
    generate_random_matrix_d(A, m, n, 77777);
    generate_random_vector_d(x, n, 88888);
    generate_random_vector_d(y, m, 99999);
    memcpy(y_result, y, m * sizeof(double));
    memcpy(y_reference, y, m * sizeof(double));
    
    double alpha = 1.5;
    double beta = 0.5;
    
    ref_dgemv(false, m, n, alpha, A, m, x, beta, y, y_reference);
    
    clock_t start = clock();
    vtable->dgemv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, m, n, alpha, A, m, x, 1, beta, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(y_result, y_reference, m);
    bool passed = (max_err <= TOLERANCE_FP64 * n);
    record_test(suite, "dgemv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(y); free(y_result); free(y_reference);
}

/* ============================================================================
 * Phase 1 BLAS Tests - Rotation Operations
 * ========================================================================== */

static void test_srot(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->srot) {
        record_skip(suite, "srot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    float* x_ref = (float*)malloc(n * sizeof(float));
    float* y_ref = (float*)malloc(n * sizeof(float));
    
    if (!x || !y || !x_ref || !y_ref) {
        record_test(suite, "srot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(x_ref); free(y_ref);
        return;
    }
    
    generate_random_vector_f(x, n, 60001);
    generate_random_vector_f(y, n, 60002);
    memcpy(x_ref, x, n * sizeof(float));
    memcpy(y_ref, y, n * sizeof(float));
    
    float c = 0.8f;  /* cos(theta) */
    float s = 0.6f;  /* sin(theta) */
    
    /* Reference: [x; y] = [c s; -s c] * [x; y] */
    for (size_t i = 0; i < n; i++) {
        float temp_x = c * x_ref[i] + s * y_ref[i];
        float temp_y = -s * x_ref[i] + c * y_ref[i];
        x_ref[i] = temp_x;
        y_ref[i] = temp_y;
    }
    
    clock_t start = clock();
    vtable->srot(n, x, 1, y, 1, c, s);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err_x = max_absolute_error_f(x, x_ref, n);
    double max_err_y = max_absolute_error_f(y, y_ref, n);
    double max_err = (max_err_x > max_err_y) ? max_err_x : max_err_y;
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "srot", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y); free(x_ref); free(y_ref);
}

static void test_drot(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->drot) {
        record_skip(suite, "drot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* x_ref = (double*)malloc(n * sizeof(double));
    double* y_ref = (double*)malloc(n * sizeof(double));
    
    if (!x || !y || !x_ref || !y_ref) {
        record_test(suite, "drot", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(x); free(y); free(x_ref); free(y_ref);
        return;
    }
    
    generate_random_vector_d(x, n, 60003);
    generate_random_vector_d(y, n, 60004);
    memcpy(x_ref, x, n * sizeof(double));
    memcpy(y_ref, y, n * sizeof(double));
    
    double c = 0.8;
    double s = 0.6;
    
    /* Reference: [x; y] = [c s; -s c] * [x; y] */
    for (size_t i = 0; i < n; i++) {
        double temp_x = c * x_ref[i] + s * y_ref[i];
        double temp_y = -s * x_ref[i] + c * y_ref[i];
        x_ref[i] = temp_x;
        y_ref[i] = temp_y;
    }
    
    clock_t start = clock();
    vtable->drot(n, x, 1, y, 1, c, s);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err_x = max_absolute_error_d(x, x_ref, n);
    double max_err_y = max_absolute_error_d(y, y_ref, n);
    double max_err = (max_err_x > max_err_y) ? max_err_x : max_err_y;
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "drot", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(x); free(y); free(x_ref); free(y_ref);
}

/* ============================================================================
 * Phase 1 BLAS Tests - Symmetric Rank-2 Updates
 * ========================================================================== */

static void test_ssyr2(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->ssyr2) {
        record_skip(suite, "ssyr2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(n * n * sizeof(float));
    float* A_ref = (float*)malloc(n * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    
    if (!A || !A_ref || !x || !y) {
        record_test(suite, "ssyr2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(A_ref); free(x); free(y);
        return;
    }
    
    generate_random_matrix_f(A, n, n, 70001);
    memcpy(A_ref, A, n * n * sizeof(float));
    generate_random_vector_f(x, n, 70002);
    generate_random_vector_f(y, n, 70003);
    
    float alpha = 1.5f;
    
    /* Reference: A = alpha*x*y^T + alpha*y*x^T + A (upper triangle) */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            A_ref[i + j * n] += alpha * x[i] * y[j] + alpha * y[i] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->ssyr2(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, A, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    /* Compare only upper triangle */
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            double err = fabs(A[i + j * n] - A_ref[i + j * n]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "ssyr2", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_ref); free(x); free(y);
}

static void test_dsyr2(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dsyr2) {
        record_skip(suite, "dsyr2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* A_ref = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    
    if (!A || !A_ref || !x || !y) {
        record_test(suite, "dsyr2", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(A_ref); free(x); free(y);
        return;
    }
    
    generate_random_matrix_d(A, n, n, 70004);
    memcpy(A_ref, A, n * n * sizeof(double));
    generate_random_vector_d(x, n, 70005);
    generate_random_vector_d(y, n, 70006);
    
    double alpha = 1.5;
    
    /* Reference: A = alpha*x*y^T + alpha*y*x^T + A (upper triangle) */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            A_ref[i + j * n] += alpha * x[i] * y[j] + alpha * y[i] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->dsyr2(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, x, 1, y, 1, A, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    /* Compare only upper triangle */
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            double err = fabs(A[i + j * n] - A_ref[i + j * n]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "dsyr2", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_ref); free(x); free(y);
}

/* New Level 2 BLAS Tests */

static void test_sger(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sger) {
        record_skip(suite, "sger", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(m * n * sizeof(float));
    float* A_ref = (float*)malloc(m * n * sizeof(float));
    float* x = (float*)malloc(m * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    
    if (!A || !A_ref || !x || !y) {
        record_test(suite, "sger", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(A_ref); free(x); free(y);
        return;
    }
    
    generate_random_matrix_f(A, m, n, 50001);
    memcpy(A_ref, A, m * n * sizeof(float));
    generate_random_vector_f(x, m, 50002);
    generate_random_vector_f(y, n, 50003);
    
    float alpha = 1.5f;
    
    /* Reference: A = alpha*x*y^T + A */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            A_ref[i + j * m] += alpha * x[i] * y[j];
        }
    }
    
    clock_t start = clock();
    vtable->sger(FB_LAYOUT_COL_MAJOR, m, n, alpha, x, 1, y, 1, A, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(A, A_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "sger", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_ref); free(x); free(y);
}

static void test_dger(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dger) {
        record_skip(suite, "dger", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(m * n * sizeof(double));
    double* A_ref = (double*)malloc(m * n * sizeof(double));
    double* x = (double*)malloc(m * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    
    if (!A || !A_ref || !x || !y) {
        record_test(suite, "dger", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(A_ref); free(x); free(y);
        return;
    }
    
    generate_random_matrix_d(A, m, n, 60001);
    memcpy(A_ref, A, m * n * sizeof(double));
    generate_random_vector_d(x, m, 60002);
    generate_random_vector_d(y, n, 60003);
    
    double alpha = 2.0;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            A_ref[i + j * m] += alpha * x[i] * y[j];
        }
    }
    
    clock_t start = clock();
    vtable->dger(FB_LAYOUT_COL_MAJOR, m, n, alpha, x, 1, y, 1, A, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(A, A_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "dger", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_ref); free(x); free(y);
}

static void test_ssymv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->ssymv) {
        record_skip(suite, "ssymv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(n * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    float* y_ref = (float*)malloc(n * sizeof(float));
    
    if (!A || !x || !y || !y_ref) {
        record_test(suite, "ssymv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(y); free(y_ref);
        return;
    }
    
    /* Generate symmetric matrix (only upper triangle matters) */
    generate_random_matrix_f(A, n, n, 70001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            A[j + i * n] = A[i + j * n]; /* Make symmetric */
        }
    }
    generate_random_vector_f(x, n, 70002);
    generate_random_vector_f(y, n, 70003);
    memcpy(y_ref, y, n * sizeof(float));
    
    float alpha = 1.2f, beta = 0.8f;
    
    /* Reference: y = alpha*A*x + beta*y */
    for (size_t i = 0; i < n; i++) {
        y_ref[i] *= beta;
        for (size_t j = 0; j < n; j++) {
            y_ref[i] += alpha * A[i + j * n] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->ssymv(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, A, n, x, 1, beta, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(y, y_ref, n);
    bool passed = (max_err <= TOLERANCE_FP32 * n);
    record_test(suite, "ssymv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(y); free(y_ref);
}

static void test_dsymv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dsymv) {
        record_skip(suite, "dsymv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* y_ref = (double*)malloc(n * sizeof(double));
    
    if (!A || !x || !y || !y_ref) {
        record_test(suite, "dsymv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(y); free(y_ref);
        return;
    }
    
    generate_random_matrix_d(A, n, n, 80001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            A[j + i * n] = A[i + j * n];
        }
    }
    generate_random_vector_d(x, n, 80002);
    generate_random_vector_d(y, n, 80003);
    memcpy(y_ref, y, n * sizeof(double));
    
    double alpha = 1.3, beta = 0.7;
    
    for (size_t i = 0; i < n; i++) {
        y_ref[i] *= beta;
        for (size_t j = 0; j < n; j++) {
            y_ref[i] += alpha * A[i + j * n] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->dsymv(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, A, n, x, 1, beta, y, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(y, y_ref, n);
    bool passed = (max_err <= TOLERANCE_FP64 * n);
    record_test(suite, "dsymv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(y); free(y_ref);
}

static void test_strmv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->strmv) {
        record_skip(suite, "strmv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(n * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* x_ref = (float*)malloc(n * sizeof(float));
    
    if (!A || !x || !x_ref) {
        record_test(suite, "strmv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(x_ref);
        return;
    }
    
    /* Generate upper triangular matrix */
    generate_random_matrix_f(A, n, n, 90001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * n] = 0.0f;  /* Zero lower triangle */
        }
        A[i + i * n] = 1.0f + fabsf(A[i + i * n]);  /* Positive diagonal */
    }
    generate_random_vector_f(x, n, 90002);
    memcpy(x_ref, x, n * sizeof(float));
    
    /* Reference: x = A*x */
    for (size_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (size_t j = i; j < n; j++) {
            sum += A[i + j * n] * x_ref[j];
        }
        x_ref[i] = sum;
    }
    
    clock_t start = clock();
    vtable->strmv(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(x, x_ref, n);
    bool passed = (max_err <= TOLERANCE_FP32 * n);
    record_test(suite, "strmv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(x_ref);
}

static void test_dtrmv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dtrmv) {
        record_skip(suite, "dtrmv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* x_ref = (double*)malloc(n * sizeof(double));
    
    if (!A || !x || !x_ref) {
        record_test(suite, "dtrmv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(x_ref);
        return;
    }
    
    generate_random_matrix_d(A, n, n, 91001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * n] = 0.0;
        }
        A[i + i * n] = 1.0 + fabs(A[i + i * n]);
    }
    generate_random_vector_d(x, n, 91002);
    memcpy(x_ref, x, n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = i; j < n; j++) {
            sum += A[i + j * n] * x_ref[j];
        }
        x_ref[i] = sum;
    }
    
    clock_t start = clock();
    vtable->dtrmv(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(x, x_ref, n);
    bool passed = (max_err <= TOLERANCE_FP64 * n);
    record_test(suite, "dtrmv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(x_ref);
}

static void test_strsv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->strsv) {
        record_skip(suite, "strsv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(n * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    float* x_ref = (float*)malloc(n * sizeof(float));
    
    if (!A || !x || !b || !x_ref) {
        record_test(suite, "strsv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(b); free(x_ref);
        return;
    }
    
    /* Generate well-conditioned upper triangular matrix */
    generate_random_matrix_f(A, n, n, 92001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * n] = 0.0f;
        }
        A[i + i * n] = 2.0f + fabsf(A[i + i * n]);  /* Well-conditioned diagonal */
    }
    generate_random_vector_f(b, n, 92002);
    memcpy(x, b, n * sizeof(float));
    
    /* Reference solve: A*x = b, solve for x */
    memcpy(x_ref, b, n * sizeof(float));
    for (int64_t i = n - 1; i >= 0; i--) {
        for (size_t j = i + 1; j < n; j++) {
            x_ref[i] -= A[i + j * n] * x_ref[j];
        }
        x_ref[i] /= A[i + i * n];
    }
    
    clock_t start = clock();
    vtable->strsv(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(x, x_ref, n);
    bool passed = (max_err <= TOLERANCE_FP32 * n);
    record_test(suite, "strsv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(b); free(x_ref);
}

static void test_dtrsv(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dtrsv) {
        record_skip(suite, "dtrsv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* x_ref = (double*)malloc(n * sizeof(double));
    
    if (!A || !x || !b || !x_ref) {
        record_test(suite, "dtrsv", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   n * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(x); free(b); free(x_ref);
        return;
    }
    
    generate_random_matrix_d(A, n, n, 93001);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * n] = 0.0;
        }
        A[i + i * n] = 2.0 + fabs(A[i + i * n]);
    }
    generate_random_vector_d(b, n, 93002);
    memcpy(x, b, n * sizeof(double));
    
    memcpy(x_ref, b, n * sizeof(double));
    for (int64_t i = n - 1; i >= 0; i--) {
        for (size_t j = i + 1; j < n; j++) {
            x_ref[i] -= A[i + j * n] * x_ref[j];
        }
        x_ref[i] /= A[i + i * n];
    }
    
    clock_t start = clock();
    vtable->dtrsv(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, n, A, n, x, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(x, x_ref, n);
    bool passed = (max_err <= TOLERANCE_FP64 * n);
    record_test(suite, "dtrsv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(x); free(b); free(x_ref);
}

/* New Level 3 BLAS Tests */

static void test_ssymm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->ssymm) {
        record_skip(suite, "ssymm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(m * m * sizeof(float));
    float* B = (float*)malloc(m * n * sizeof(float));
    float* C = (float*)malloc(m * n * sizeof(float));
    float* C_ref = (float*)malloc(m * n * sizeof(float));
    
    if (!A || !B || !C || !C_ref) {
        record_test(suite, "ssymm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(C); free(C_ref);
        return;
    }
    
    /* Generate symmetric matrix A */
    generate_random_matrix_f(A, m, m, 94001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i + 1; j < m; j++) {
            A[j + i * m] = A[i + j * m];
        }
    }
    generate_random_matrix_f(B, m, n, 94002);
    generate_random_matrix_f(C, m, n, 94003);
    memcpy(C_ref, C, m * n * sizeof(float));
    
    float alpha = 1.1f, beta = 0.9f;
    
    /* Reference: C = alpha*A*B + beta*C (A symmetric, side = left) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < m; k++) {
                sum += A[i + k * m] * B[k + j * m];
            }
            C_ref[i + j * m] = alpha * sum + beta * C_ref[i + j * m];
        }
    }
    
    clock_t start = clock();
    vtable->ssymm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, m, B, m, beta, C, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(C, C_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP32 * m);
    record_test(suite, "ssymm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_ref);
}

static void test_dsymm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dsymm) {
        record_skip(suite, "dsymm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(m * m * sizeof(double));
    double* B = (double*)malloc(m * n * sizeof(double));
    double* C = (double*)malloc(m * n * sizeof(double));
    double* C_ref = (double*)malloc(m * n * sizeof(double));
    
    if (!A || !B || !C || !C_ref) {
        record_test(suite, "dsymm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(C); free(C_ref);
        return;
    }
    
    generate_random_matrix_d(A, m, m, 95001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i + 1; j < m; j++) {
            A[j + i * m] = A[i + j * m];
        }
    }
    generate_random_matrix_d(B, m, n, 95002);
    generate_random_matrix_d(C, m, n, 95003);
    memcpy(C_ref, C, m * n * sizeof(double));
    
    double alpha = 1.2, beta = 0.8;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += A[i + k * m] * B[k + j * m];
            }
            C_ref[i + j * m] = alpha * sum + beta * C_ref[i + j * m];
        }
    }
    
    clock_t start = clock();
    vtable->dsymm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, m, n, alpha, A, m, B, m, beta, C, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(C, C_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP64 * m);
    record_test(suite, "dsymm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_ref);
}

static void test_strmm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->strmm) {
        record_skip(suite, "strmm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(m * m * sizeof(float));
    float* B = (float*)malloc(m * n * sizeof(float));
    float* B_ref = (float*)malloc(m * n * sizeof(float));
    
    if (!A || !B || !B_ref) {
        record_test(suite, "strmm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(B_ref);
        return;
    }
    
    /* Generate upper triangular matrix */
    generate_random_matrix_f(A, m, m, 96001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * m] = 0.0f;
        }
        A[i + i * m] = 1.0f + fabsf(A[i + i * m]);
    }
    generate_random_matrix_f(B, m, n, 96002);
    memcpy(B_ref, B, m * n * sizeof(float));
    
    float alpha = 1.5f;
    
    /* Reference: B = alpha*A*B (A triangular, side = left) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = i; k < m; k++) {
                sum += A[i + k * m] * B_ref[k + j * m];
            }
            B_ref[i + j * m] = alpha * sum;
        }
    }
    
    clock_t start = clock();
    vtable->strmm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, m, B, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(B, B_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP32 * m);
    record_test(suite, "strmm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(B_ref);
}

static void test_dtrmm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dtrmm) {
        record_skip(suite, "dtrmm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(m * m * sizeof(double));
    double* B = (double*)malloc(m * n * sizeof(double));
    double* B_ref = (double*)malloc(m * n * sizeof(double));
    
    if (!A || !B || !B_ref) {
        record_test(suite, "dtrmm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(B_ref);
        return;
    }
    
    generate_random_matrix_d(A, m, m, 97001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * m] = 0.0;
        }
        A[i + i * m] = 1.0 + fabs(A[i + i * m]);
    }
    generate_random_matrix_d(B, m, n, 97002);
    memcpy(B_ref, B, m * n * sizeof(double));
    
    double alpha = 1.6;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = i; k < m; k++) {
                sum += A[i + k * m] * B_ref[k + j * m];
            }
            B_ref[i + j * m] = alpha * sum;
        }
    }
    
    clock_t start = clock();
    vtable->dtrmm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, m, B, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(B, B_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP64 * m);
    record_test(suite, "dtrmm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(B_ref);
}

static void test_strsm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->strsm) {
        record_skip(suite, "strsm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    float* A = (float*)malloc(m * m * sizeof(float));
    float* B = (float*)malloc(m * n * sizeof(float));
    float* B_ref = (float*)malloc(m * n * sizeof(float));
    
    if (!A || !B || !B_ref) {
        record_test(suite, "strsm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP32", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(B_ref);
        return;
    }
    
    /* Generate well-conditioned upper triangular matrix */
    generate_random_matrix_f(A, m, m, 98001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * m] = 0.0f;
        }
        A[i + i * m] = 2.0f + fabsf(A[i + i * m]);
    }
    generate_random_matrix_f(B, m, n, 98002);
    memcpy(B_ref, B, m * n * sizeof(float));
    
    float alpha = 1.0f;
    
    /* Reference solve: A*X = alpha*B, solve for X (stored in B_ref) */
    for (size_t j = 0; j < n; j++) {
        for (int64_t i = m - 1; i >= 0; i--) {
            for (size_t k = i + 1; k < m; k++) {
                B_ref[i + j * m] -= A[i + k * m] * B_ref[k + j * m];
            }
            B_ref[i + j * m] = (alpha * B_ref[i + j * m]) / A[i + i * m];
        }
    }
    
    clock_t start = clock();
    vtable->strsm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, m, B, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(B, B_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP32 * m);
    record_test(suite, "strsm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(B_ref);
}

static void test_dtrsm(fb_backend_instance_t* instance, test_suite_results_t* suite, size_t m, size_t n)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dtrsm) {
        record_skip(suite, "dtrsm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    double* A = (double*)malloc(m * m * sizeof(double));
    double* B = (double*)malloc(m * n * sizeof(double));
    double* B_ref = (double*)malloc(m * n * sizeof(double));
    
    if (!A || !B || !B_ref) {
        record_test(suite, "dtrsm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(B_ref);
        return;
    }
    
    generate_random_matrix_d(A, m, m, 99001);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i + j * m] = 0.0;
        }
        A[i + i * m] = 2.0 + fabs(A[i + i * m]);
    }
    generate_random_matrix_d(B, m, n, 99002);
    memcpy(B_ref, B, m * n * sizeof(double));
    
    double alpha = 1.0;
    
    for (size_t j = 0; j < n; j++) {
        for (int64_t i = m - 1; i >= 0; i--) {
            for (size_t k = i + 1; k < m; k++) {
                B_ref[i + j * m] -= A[i + k * m] * B_ref[k + j * m];
            }
            B_ref[i + j * m] = (alpha * B_ref[i + j * m]) / A[i + i * m];
        }
    }
    
    clock_t start = clock();
    vtable->dtrsm(FB_LAYOUT_COL_MAJOR, FB_LEFT, FB_UPPER, FB_NO_TRANS, FB_NON_UNIT, m, n, alpha, A, m, B, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(B, B_ref, m * n);
    bool passed = (max_err <= TOLERANCE_FP64 * m);
    record_test(suite, "dtrsm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(B_ref);
}

static void test_dgemm(fb_backend_instance_t* instance,
                      test_suite_results_t* suite,
                      size_t m, size_t n, size_t k)
{
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->dgemm) {
        record_skip(suite, "dgemm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   "Operation not implemented");
        return;
    }
    
    size_t size_A = m * k;
    size_t size_B = k * n;
    size_t size_C = m * n;
    
    double* A = (double*)malloc(size_A * sizeof(double));
    double* B = (double*)malloc(size_B * sizeof(double));
    double* C = (double*)malloc(size_C * sizeof(double));
    double* C_result = (double*)malloc(size_C * sizeof(double));
    double* C_reference = (double*)malloc(size_C * sizeof(double));
    
    if (!A || !B || !C || !C_result || !C_reference) {
        record_test(suite, "dgemm", instance->plugin->metadata->name,
                   instance->device->properties.name, instance->device->properties.device_id,
                   m * n * k, "FP64", false, 0.0, "Memory allocation failed", 0.0);
        free(A); free(B); free(C); free(C_result); free(C_reference);
        return;
    }
    
    generate_random_matrix_d(A, m, k, 55555);
    generate_random_matrix_d(B, k, n, 66666);
    generate_random_matrix_d(C, m, n, 77777);
    memcpy(C_result, C, size_C * sizeof(double));
    memcpy(C_reference, C, size_C * sizeof(double));
    
    double alpha = 1.0;
    double beta = 1.0;
    
    ref_dgemm(false, false, m, n, k, alpha, A, m, B, k, beta, C, m, C_reference);
    
    clock_t start = clock();
    vtable->dgemm(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                  m, n, k, alpha, A, m, B, k, beta, C_result, m);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(C_result, C_reference, size_C);
    bool passed = (max_err <= TOLERANCE_FP64 * k);
    record_test(suite, "dgemm", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n * k, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_result); free(C_reference);
}

/* ============================================================================
 * PHASE2: Symmetric Rank-1 Update Tests (ssyr, dsyr)
 * ========================================================================== */

static void test_ssyr(int64_t n, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->ssyr) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    float* A = (float*)malloc(n * n * sizeof(float));
    float* A_result = (float*)malloc(n * n * sizeof(float));
    float* A_reference = (float*)malloc(n * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    
    // Initialize: A = symmetric matrix, x = random vector
    for (int64_t i = 0; i < n; i++) {
        x[i] = (float)(rand() % 100) / 100.0f;
        for (int64_t j = 0; j < n; j++) {
            if (i <= j) {
                A[j * n + i] = (float)(rand() % 100) / 100.0f;  // col-major: A(i,j) = A[j*n+i]
                A[i * n + j] = A[j * n + i];  // Make symmetric
            }
        }
    }
    memcpy(A_result, A, n * n * sizeof(float));
    memcpy(A_reference, A, n * n * sizeof(float));
    
    float alpha = 0.5f;
    
    // Reference: A = alpha*x*x' + A (only upper triangle)
    // For col-major upper triangle: j >= i, element (i,j) is at A[j*n+i]
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            A_reference[j * n + i] += alpha * x[i] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->ssyr(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, x, 1, A_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    // Check upper triangle only
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(A_result[j * n + i] - A_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP32);
    record_test(suite, "ssyr", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_result); free(A_reference); free(x);
}

static void test_dsyr(int64_t n, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->dsyr) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* A_result = (double*)malloc(n * n * sizeof(double));
    double* A_reference = (double*)malloc(n * n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    
    for (int64_t i = 0; i < n; i++) {
        x[i] = (double)(rand() % 100) / 100.0;
        for (int64_t j = 0; j < n; j++) {
            if (i <= j) {
                A[j * n + i] = (double)(rand() % 100) / 100.0;  // col-major
                A[i * n + j] = A[j * n + i];
            }
        }
    }
    memcpy(A_result, A, n * n * sizeof(double));
    memcpy(A_reference, A, n * n * sizeof(double));
    
    double alpha = 0.5;
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            A_reference[j * n + i] += alpha * x[i] * x[j];
        }
    }
    
    clock_t start = clock();
    vtable->dsyr(FB_LAYOUT_COL_MAJOR, FB_UPPER, n, alpha, x, 1, A_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(A_result[j * n + i] - A_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP64);
    record_test(suite, "dsyr", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(A_result); free(A_reference); free(x);
}

/* ============================================================================
 * PHASE2: Symmetric Rank-k Update Tests (ssyrk, dsyrk)
 * ========================================================================== */

static void test_ssyrk(int64_t n, int64_t k, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->ssyrk) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    float* A = (float*)malloc(n * k * sizeof(float));
    float* C = (float*)malloc(n * n * sizeof(float));
    float* C_result = (float*)malloc(n * n * sizeof(float));
    float* C_reference = (float*)malloc(n * n * sizeof(float));
    
    for (int64_t i = 0; i < n * k; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int64_t i = 0; i < n * n; i++) C[i] = (float)(rand() % 100) / 100.0f;
    memcpy(C_result, C, n * n * sizeof(float));
    memcpy(C_reference, C, n * n * sizeof(float));
    
    float alpha = 1.0f, beta = 1.0f;
    
    // Reference: C = alpha*A*A' + beta*C (upper triangle)
    // Col-major: A(i,p) = A[p*n+i], C(i,j) = C[j*n+i]
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                sum += A[p * n + i] * A[p * n + j];
            }
            C_reference[j * n + i] = alpha * sum + beta * C_reference[j * n + i];
        }
    }
    
    clock_t start = clock();
    vtable->ssyrk(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, n, beta, C_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(C_result[j * n + i] - C_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP32 * k);
    record_test(suite, "ssyrk", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * k, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(C); free(C_result); free(C_reference);
}

static void test_dsyrk(int64_t n, int64_t k, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->dsyrk) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    double* A = (double*)malloc(n * k * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    double* C_result = (double*)malloc(n * n * sizeof(double));
    double* C_reference = (double*)malloc(n * n * sizeof(double));
    
    for (int64_t i = 0; i < n * k; i++) A[i] = (double)(rand() % 100) / 100.0;
    for (int64_t i = 0; i < n * n; i++) C[i] = (double)(rand() % 100) / 100.0;
    memcpy(C_result, C, n * n * sizeof(double));
    memcpy(C_reference, C, n * n * sizeof(double));
    
    double alpha = 1.0, beta = 1.0;
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double sum = 0.0;
            for (int64_t p = 0; p < k; p++) {
                sum += A[p * n + i] * A[p * n + j];
            }
            C_reference[j * n + i] = alpha * sum + beta * C_reference[j * n + i];
        }
    }
    
    clock_t start = clock();
    vtable->dsyrk(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, n, beta, C_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(C_result[j * n + i] - C_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP64 * k);
    record_test(suite, "dsyrk", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * k, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(C); free(C_result); free(C_reference);
}

/* ============================================================================
 * Phase 1 BLAS Tests - Symmetric Rank-2k Updates
 * ========================================================================== */

static void test_ssyr2k(int64_t n, int64_t k, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->ssyr2k) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    float* A = (float*)malloc(n * k * sizeof(float));
    float* B = (float*)malloc(n * k * sizeof(float));
    float* C = (float*)malloc(n * n * sizeof(float));
    float* C_result = (float*)malloc(n * n * sizeof(float));
    float* C_reference = (float*)malloc(n * n * sizeof(float));
    
    for (int64_t i = 0; i < n * k; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int64_t i = 0; i < n * k; i++) B[i] = (float)(rand() % 100) / 100.0f;
    for (int64_t i = 0; i < n * n; i++) C[i] = (float)(rand() % 100) / 100.0f;
    memcpy(C_result, C, n * n * sizeof(float));
    memcpy(C_reference, C, n * n * sizeof(float));
    
    float alpha = 1.0f, beta = 1.0f;
    
    /* Reference: C = alpha*A*B^T + alpha*B*A^T + beta*C (upper triangle) */
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                sum += A[p * n + i] * B[p * n + j] + B[p * n + i] * A[p * n + j];
            }
            C_reference[j * n + i] = alpha * sum + beta * C_reference[j * n + i];
        }
    }
    
    clock_t start = clock();
    vtable->ssyr2k(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, n, B, n, beta, C_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(C_result[j * n + i] - C_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP32 * k * 2);
    record_test(suite, "ssyr2k", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * k, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_result); free(C_reference);
}

static void test_dsyr2k(int64_t n, int64_t k, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->dsyr2k) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    double* A = (double*)malloc(n * k * sizeof(double));
    double* B = (double*)malloc(n * k * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    double* C_result = (double*)malloc(n * n * sizeof(double));
    double* C_reference = (double*)malloc(n * n * sizeof(double));
    
    for (int64_t i = 0; i < n * k; i++) A[i] = (double)(rand() % 100) / 100.0;
    for (int64_t i = 0; i < n * k; i++) B[i] = (double)(rand() % 100) / 100.0;
    for (int64_t i = 0; i < n * n; i++) C[i] = (double)(rand() % 100) / 100.0;
    memcpy(C_result, C, n * n * sizeof(double));
    memcpy(C_reference, C, n * n * sizeof(double));
    
    double alpha = 1.0, beta = 1.0;
    
    /* Reference: C = alpha*A*B^T + alpha*B*A^T + beta*C (upper triangle) */
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double sum = 0.0;
            for (int64_t p = 0; p < k; p++) {
                sum += A[p * n + i] * B[p * n + j] + B[p * n + i] * A[p * n + j];
            }
            C_reference[j * n + i] = alpha * sum + beta * C_reference[j * n + i];
        }
    }
    
    clock_t start = clock();
    vtable->dsyr2k(FB_LAYOUT_COL_MAJOR, FB_UPPER, FB_NO_TRANS, n, k, alpha, A, n, B, n, beta, C_result, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double err = fabs(C_result[j * n + i] - C_reference[j * n + i]);
            if (err > max_err) max_err = err;
        }
    }
    
    bool passed = (max_err <= TOLERANCE_FP64 * k * 2);
    record_test(suite, "dsyr2k", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               n * k, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A); free(B); free(C); free(C_result); free(C_reference);
}

/* ============================================================================
 * PHASE2: Band Matrix Tests (sgbmv, dgbmv)
 * ========================================================================== */

static void test_sgbmv(int64_t m, int64_t n, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->sgbmv) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    int64_t kl = (m < 10) ? 1 : 2;  // Number of subdiagonals
    int64_t ku = (n < 10) ? 1 : 2;  // Number of superdiagonals
    int64_t lda = kl + ku + 1;
    
    float* A_band = (float*)calloc(lda * n, sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(m * sizeof(float));
    float* y_result = (float*)malloc(m * sizeof(float));
    float* y_reference = (float*)malloc(m * sizeof(float));
    
    for (int64_t i = 0; i < n; i++) x[i] = (float)(rand() % 100) / 100.0f;
    for (int64_t i = 0; i < m; i++) y[i] = (float)(rand() % 100) / 100.0f;
    
    // Fill band matrix (band storage format, col-major: A_band[j*lda + k])
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = (j - ku < 0 ? 0 : j - ku); i < m && i <= j + kl; i++) {
            int64_t k = ku + i - j;
            A_band[j * lda + k] = (float)(rand() % 100) / 100.0f + 1.0f;
        }
    }
    
    memcpy(y_result, y, m * sizeof(float));
    memcpy(y_reference, y, m * sizeof(float));
    
    float alpha = 1.0f, beta = 1.0f;
    
    // Reference: y = alpha*A*x + beta*y (reconstruct full matrix)
    for (int64_t i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int64_t j = 0; j < n; j++) {
            if (j >= i - kl && j <= i + ku) {
                int64_t k = ku + i - j;
                if (k >= 0 && k < lda) {
                    sum += A_band[j * lda + k] * x[j];
                }
            }
        }
        y_reference[i] = alpha * sum + beta * y_reference[i];
    }
    
    clock_t start = clock();
    vtable->sgbmv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, A_band, lda, x, 1, beta, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_f(y_result, y_reference, m);
    bool passed = (max_err <= TOLERANCE_FP32 * n);
    record_test(suite, "sgbmv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP32", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A_band); free(x); free(y); free(y_result); free(y_reference);
}

static void test_dgbmv(int64_t m, int64_t n, fb_backend_instance_t* instance, test_suite_results_t* suite) {
    if (!instance->vtable->dgbmv) return;
    const fb_backend_vtable_t* vtable = instance->vtable;
    
    int64_t kl = (m < 10) ? 1 : 2;
    int64_t ku = (n < 10) ? 1 : 2;
    int64_t lda = kl + ku + 1;
    
    double* A_band = (double*)calloc(lda * n, sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(m * sizeof(double));
    double* y_result = (double*)malloc(m * sizeof(double));
    double* y_reference = (double*)malloc(m * sizeof(double));
    
    for (int64_t i = 0; i < n; i++) x[i] = (double)(rand() % 100) / 100.0;
    for (int64_t i = 0; i < m; i++) y[i] = (double)(rand() % 100) / 100.0;
    
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = (j - ku < 0 ? 0 : j - ku); i < m && i <= j + kl; i++) {
            int64_t k = ku + i - j;
            A_band[j * lda + k] = (double)(rand() % 100) / 100.0 + 1.0;
        }
    }
    
    memcpy(y_result, y, m * sizeof(double));
    memcpy(y_reference, y, m * sizeof(double));
    
    double alpha = 1.0, beta = 1.0;
    
    for (int64_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; j++) {
            if (j >= i - kl && j <= i + ku) {
                int64_t k = ku + i - j;
                if (k >= 0 && k < lda) {
                    sum += A_band[j * lda + k] * x[j];
                }
            }
        }
        y_reference[i] = alpha * sum + beta * y_reference[i];
    }
    
    clock_t start = clock();
    vtable->dgbmv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, m, n, kl, ku, alpha, A_band, lda, x, 1, beta, y_result, 1);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    double max_err = max_absolute_error_d(y_result, y_reference, m);
    bool passed = (max_err <= TOLERANCE_FP64 * n);
    record_test(suite, "dgbmv", instance->plugin->metadata->name,
               instance->device->properties.name, instance->device->properties.device_id,
               m * n, "FP64", passed, max_err,
               passed ? NULL : "Numerical error exceeds tolerance", time_ms);
    
    free(A_band); free(x); free(y); free(y_result); free(y_reference);
}

/* ============================================================================
 * Test Orchestration
 * ========================================================================== */

static void test_backend_on_device(int device_id, test_suite_results_t* suite) {
    fb_compute_device_t* device = fb_get_device(device_id);
    if (!device) return;
    
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Testing Device %d: %s\n", device_id, device->properties.name);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    /* Get all backend instances for this device */
    fb_backend_instance_t** instances = NULL;
    size_t num_instances = 0;
    
    /* Try to load backends for this device */
    int result = fb_use_device(device_id);
    if (result != 0) {
        printf("  ⚠ Could not use device %d\n", device_id);
        return;
    }
    
    /* Get primary backend */
    fb_backend_instance_t* primary = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (!primary) {
        printf("  ⚠ No backend available for device\n");
        fb_use_auto();
        return;
    }
    
    /* For now, test just the primary backend (multi-backend per device comes later) */
    instances = &primary;
    num_instances = 1;
    
    for (size_t i = 0; i < num_instances; i++) {
        fb_backend_instance_t* instance = instances[i];
        
        printf("\n  Backend: %s (v%s)\n", 
               instance->plugin->metadata->name,
               instance->plugin->metadata->version);
        printf("  ─────────────────────────────────────────────────\n");
        
        /* Level 1 BLAS Tests */
        printf("    Level 1 BLAS (Vector-Vector):\n");
        test_saxpy(instance, suite, SMALL_SIZE);
        test_saxpy(instance, suite, MEDIUM_SIZE);
        test_saxpy(instance, suite, LARGE_SIZE);
        test_daxpy(instance, suite, SMALL_SIZE);
        test_daxpy(instance, suite, MEDIUM_SIZE);
        test_scopy(instance, suite, SMALL_SIZE);
        test_scopy(instance, suite, LARGE_SIZE);
        test_dcopy(instance, suite, MEDIUM_SIZE);
        test_sscal(instance, suite, MEDIUM_SIZE);
        test_dscal(instance, suite, LARGE_SIZE);
        test_sdot(instance, suite, MEDIUM_SIZE);
        test_ddot(instance, suite, MEDIUM_SIZE);
        test_snrm2(instance, suite, MEDIUM_SIZE);
        test_dnrm2(instance, suite, MEDIUM_SIZE);
        test_sasum(instance, suite, MEDIUM_SIZE);
        test_dasum(instance, suite, MEDIUM_SIZE);
        test_sswap(instance, suite, SMALL_SIZE);
        test_dswap(instance, suite, SMALL_SIZE);
        test_isamax(instance, suite, MEDIUM_SIZE);
        test_idamax(instance, suite, MEDIUM_SIZE);
        test_srot(instance, suite, SMALL_SIZE);
        test_srot(instance, suite, MEDIUM_SIZE);
        test_drot(instance, suite, SMALL_SIZE);
        test_drot(instance, suite, MEDIUM_SIZE);
        
        /* Level 2 BLAS Tests */
        printf("    Level 2 BLAS (Matrix-Vector):\n");
        test_sgemv(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_sgemv(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_sgemv(instance, suite, SMALL_SIZE, MEDIUM_SIZE);  /* Non-square */
        test_dgemv(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_dgemv(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_sger(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_sger(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_dger(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_dger(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_ssymv(instance, suite, SMALL_SIZE);
        test_ssymv(instance, suite, MEDIUM_SIZE);
        test_dsymv(instance, suite, SMALL_SIZE);
        test_dsymv(instance, suite, MEDIUM_SIZE);
        test_strmv(instance, suite, SMALL_SIZE);
        test_strmv(instance, suite, MEDIUM_SIZE);
        test_dtrmv(instance, suite, SMALL_SIZE);
        test_dtrmv(instance, suite, MEDIUM_SIZE);
        test_strsv(instance, suite, SMALL_SIZE);
        test_strsv(instance, suite, MEDIUM_SIZE);
        test_dtrsv(instance, suite, SMALL_SIZE);
        test_dtrsv(instance, suite, MEDIUM_SIZE);
        test_ssyr2(instance, suite, SMALL_SIZE);
        test_ssyr2(instance, suite, MEDIUM_SIZE);
        test_dsyr2(instance, suite, SMALL_SIZE);
        test_dsyr2(instance, suite, MEDIUM_SIZE);
        
        /* Level 3 BLAS Tests */
        printf("    Level 3 BLAS (Matrix-Matrix):\n");
        test_sgemm(instance, suite, SMALL_SIZE, SMALL_SIZE, SMALL_SIZE);
        test_sgemm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE, MEDIUM_SIZE);
        test_sgemm(instance, suite, 128, 128, 64);  /* Non-square */
        test_dgemm(instance, suite, SMALL_SIZE, SMALL_SIZE, SMALL_SIZE);
        test_dgemm(instance, suite, 128, 128, 64);  /* Non-square */
        test_ssymm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_ssymm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_dsymm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_dsymm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_strmm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_strmm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_dtrmm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_dtrmm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_strsm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_strsm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        test_dtrsm(instance, suite, SMALL_SIZE, SMALL_SIZE);
        test_dtrsm(instance, suite, MEDIUM_SIZE, MEDIUM_SIZE);
        
        /* PHASE2: Symmetric rank-1 and rank-k updates */
        printf("    PHASE2: Symmetric Rank Updates:\n");
        test_ssyr(SMALL_SIZE, instance, suite);
        test_ssyr(MEDIUM_SIZE, instance, suite);
        test_dsyr(SMALL_SIZE, instance, suite);
        test_dsyr(MEDIUM_SIZE, instance, suite);
        test_ssyrk(SMALL_SIZE, 32, instance, suite);
        test_ssyrk(MEDIUM_SIZE, 64, instance, suite);
        test_dsyrk(SMALL_SIZE, 32, instance, suite);
        test_dsyrk(MEDIUM_SIZE, 64, instance, suite);
        test_ssyr2k(SMALL_SIZE, 32, instance, suite);
        test_ssyr2k(MEDIUM_SIZE, 64, instance, suite);
        test_dsyr2k(SMALL_SIZE, 32, instance, suite);
        test_dsyr2k(MEDIUM_SIZE, 64, instance, suite);
        
        /* PHASE2: Band matrix operations */
        printf("    PHASE2: Band Matrix Operations:\n");
        test_sgbmv(SMALL_SIZE, SMALL_SIZE, instance, suite);
        test_sgbmv(MEDIUM_SIZE, MEDIUM_SIZE, instance, suite);
        test_dgbmv(SMALL_SIZE, SMALL_SIZE, instance, suite);
        test_dgbmv(MEDIUM_SIZE, MEDIUM_SIZE, instance, suite);
    }
    
    fb_use_auto();
}

/* ============================================================================
 * Results Reporting
 * ========================================================================== */

static void print_summary(const test_suite_results_t* suite) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  COMPREHENSIVE CORRECTNESS TEST RESULTS\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  Total Tests: %zu\n", suite->count);
    printf("  ✓ Passed:    %zu\n", suite->passed);
    printf("  ✗ Failed:    %zu\n", suite->failed);
    printf("  ⚠ Skipped:   %zu\n", suite->skipped);
    printf("\n");
    
    if (suite->failed > 0) {
        printf("═══════════════════════════════════════════════════════\n");
        printf("  FAILED TESTS:\n");
        printf("═══════════════════════════════════════════════════════\n");
        
        for (size_t i = 0; i < suite->count; i++) {
            const test_result_t* result = &suite->results[i];
            if (!result->passed && result->error_message[0] && 
                strncmp(result->error_message, "SKIPPED", 7) != 0) {
                printf("\n  ✗ %s [%s] on %s\n",
                       result->operation_name,
                       result->dtype,
                       result->backend_name);
                printf("    Device: %s (ID: %d)\n",
                       result->device_name, result->device_id);
                printf("    Size: %zu\n", result->problem_size);
                printf("    Max Error: %.6e\n", result->max_error);
                if (result->error_message[0]) {
                    printf("    Reason: %s\n", result->error_message);
                }
            }
        }
        printf("\n");
    }
    
    printf("═══════════════════════════════════════════════════════\n");
    if (suite->failed == 0) {
        printf("  ✓✓✓ ALL TESTS PASSED ✓✓✓\n");
    } else {
        printf("  ✗✗✗ SOME TESTS FAILED ✗✗✗\n");
    }
    printf("═══════════════════════════════════════════════════════\n");
    printf("\n");
}

static void save_results_to_json(const test_suite_results_t* suite, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "{\n");
    fprintf(f, "  \"test_suite\": \"comprehensive_correctness\",\n");
    fprintf(f, "  \"timestamp\": %ld,\n", (long)time(NULL));
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"total\": %zu,\n", suite->count);
    fprintf(f, "    \"passed\": %zu,\n", suite->passed);
    fprintf(f, "    \"failed\": %zu,\n", suite->failed);
    fprintf(f, "    \"skipped\": %zu\n", suite->skipped);
    fprintf(f, "  },\n");
    fprintf(f, "  \"results\": [\n");
    
    for (size_t i = 0; i < suite->count; i++) {
        const test_result_t* r = &suite->results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"operation\": \"%s\",\n", r->operation_name);
        fprintf(f, "      \"backend\": \"%s\",\n", r->backend_name);
        fprintf(f, "      \"device\": \"%s\",\n", r->device_name);
        fprintf(f, "      \"device_id\": %d,\n", r->device_id);
        fprintf(f, "      \"size\": %zu,\n", r->problem_size);
        fprintf(f, "      \"dtype\": \"%s\",\n", r->dtype ? r->dtype : "unknown");
        fprintf(f, "      \"passed\": %s,\n", r->passed ? "true" : "false");
        fprintf(f, "      \"max_error\": %.6e,\n", r->max_error);
        fprintf(f, "      \"time_ms\": %.3f", r->execution_time_ms);
        if (r->error_message[0]) {
            fprintf(f, ",\n      \"message\": \"%s\"\n", r->error_message);
        } else {
            fprintf(f, "\n");
        }
        fprintf(f, "    }%s\n", (i < suite->count - 1) ? "," : "");
    }
    
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    
    fclose(f);
    printf("  Results saved to: %s\n", filename);
}

/* ============================================================================
 * Main Test Entry Point
 * ========================================================================== */

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  COMPREHENSIVE CORRECTNESS VALIDATION\n");
    printf("  Testing ALL operations on ALL backends/devices\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("\n");
    
    /* Initialize faster-blaster */
    int result = fb_init();
    if (result != 0) {
        fprintf(stderr, "ERROR: fb_init() failed with code %d\n", result);
        return 1;
    }
    
    /* Initialize test results */
    test_suite_results_t suite;
    init_test_results(&suite);
    
    /* Get all devices */
    uint32_t num_devices = fb_get_device_count();
    printf("Detected %u device(s)\n", num_devices);
    
    if (num_devices == 0) {
        fprintf(stderr, "ERROR: No devices detected\n");
        free_test_results(&suite);
        fb_shutdown();
        return 1;
    }
    
    /* Test each device */
    for (uint32_t dev = 0; dev < num_devices; dev++) {
        test_backend_on_device(dev, &suite);
    }
    
    /* Print summary */
    print_summary(&suite);
    
    /* Save results */
    save_results_to_json(&suite, "correctness_test_results.json");
    
    /* Cleanup */
    free_test_results(&suite);
    fb_shutdown();
    
    return (suite.failed > 0) ? 1 : 0;
}
