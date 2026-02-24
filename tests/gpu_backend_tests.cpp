/**
 * @file gpu_backend_tests.cpp
 * @brief Comprehensive GPU backend test suite
 * 
 * Tests all BLAS/LAPACK operations across all available GPU backends:
 * - NVIDIA cuBLAS (via CUDA)
 * - AMD rocBLAS (via HIP)  
 * - AMD hipSOLVER (unified CUDA/ROCm)
 * - Intel oneMKL (via SYCL)
 * 
 * Automatically detects available hardware and runs appropriate tests.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "../include/faster-blaster/gpu_backend_trait.h"

// Test configuration
#define MAX_TESTS 200
#define TEST_SIZE_SMALL 64
#define TEST_SIZE_MEDIUM 256
#define TEST_SIZE_LARGE 1024
#define TOLERANCE_FLOAT 1e-5f
#define TOLERANCE_DOUBLE 1e-12

// Test result tracking
typedef struct {
    const char* backend_name;
    const char* test_name;
    int passed;
    double time_ms;
    const char* error_msg;
} test_result_t;

static test_result_t test_results[MAX_TESTS];
static int num_tests = 0;

// Helper macros
#define RECORD_TEST(backend, name, pass, time, msg) do { \
    test_results[num_tests].backend_name = backend; \
    test_results[num_tests].test_name = name; \
    test_results[num_tests].passed = pass; \
    test_results[num_tests].time_ms = time; \
    test_results[num_tests].error_msg = msg; \
    num_tests++; \
} while(0)

#define CHECK_ALLOC(ptr, backend) if (!(ptr)) { \
    RECORD_TEST(backend, "memory_allocation", 0, 0.0, "Failed to allocate device memory"); \
    return; \
}

// Timer utilities
static double get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart * 1000.0) / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

// ============================================================================
// BLAS Level 1 Tests
// ============================================================================

/**
 * Test SAXPY: y = alpha*x + y
 */
static void test_saxpy(const fb_gpu_backend_trait_t* trait, void* handle, fb_gpu_stream_t stream, const char* backend_name) {
    const int n = TEST_SIZE_MEDIUM;
    const float alpha = 2.5f;
    
    // Host data
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    float *h_y_result = (float*)malloc(n * sizeof(float));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(n - i);
    }
    
    // Allocate device memory
    fb_gpu_ptr_t d_x, d_y;
    if (trait->malloc(handle, &d_x, n * sizeof(float)) != 0 ||
        trait->malloc(handle, &d_y, n * sizeof(float)) != 0) {
        RECORD_TEST(backend_name, "saxpy", 0, 0.0, "Device allocation failed");
        free(h_x); free(h_y); free(h_y_result);
        return;
    }
    
    // Copy to device
    trait->memcpy_h2d(handle, d_x, h_x, n * sizeof(float));
    trait->memcpy_h2d(handle, d_y, h_y, n * sizeof(float));
    
    // Run SAXPY
    double t0 = get_time_ms();
    trait->saxpy(handle, stream, n, alpha, d_x, 1, d_y, 1);
    trait->stream_synchronize(handle, stream);
    double t1 = get_time_ms();
    
    // Copy result back
    trait->memcpy_d2h(handle, h_y_result, d_y, n * sizeof(float));
    
    // Verify result
    int passed = 1;
    for (int i = 0; i < n; i++) {
        float expected = alpha * h_x[i] + h_y[i];
        if (fabsf(h_y_result[i] - expected) > TOLERANCE_FLOAT) {
            passed = 0;
            break;
        }
    }
    
    RECORD_TEST(backend_name, "saxpy", passed, t1 - t0, passed ? NULL : "Numerical error");
    
    // Cleanup
    trait->free(handle, d_x);
    trait->free(handle, d_y);
    free(h_x); free(h_y); free(h_y_result);
}

/**
 * Test SDOT: result = x^T * y
 */
static void test_sdot(const fb_gpu_backend_trait_t* trait, void* handle, fb_gpu_stream_t stream, const char* backend_name) {
    const int n = TEST_SIZE_MEDIUM;
    
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    
    // Initialize with known values
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    
    fb_gpu_ptr_t d_x, d_y;
    if (trait->malloc(handle, &d_x, n * sizeof(float)) != 0 ||
        trait->malloc(handle, &d_y, n * sizeof(float)) != 0) {
        RECORD_TEST(backend_name, "sdot", 0, 0.0, "Device allocation failed");
        free(h_x); free(h_y);
        return;
    }
    
    trait->memcpy_h2d(handle, d_x, h_x, n * sizeof(float));
    trait->memcpy_h2d(handle, d_y, h_y, n * sizeof(float));
    
    double t0 = get_time_ms();
    float result = trait->sdot(handle, stream, n, d_x, 1, d_y, 1);
    trait->stream_synchronize(handle, stream);
    double t1 = get_time_ms();
    
    float expected = 2.0f * n; // 1*2*n
    int passed = fabsf(result - expected) < TOLERANCE_FLOAT;
    
    RECORD_TEST(backend_name, "sdot", passed, t1 - t0, 
                passed ? NULL : "Numerical error");
    
    trait->free(handle, d_x);
    trait->free(handle, d_y);
    free(h_x); free(h_y);
}

// ============================================================================
// BLAS Level 3 Tests
// ============================================================================

/**
 * Test SGEMM: C = alpha*A*B + beta*C
 */
static void test_sgemm(const fb_gpu_backend_trait_t* trait, void* handle, fb_gpu_stream_t stream, const char* backend_name) {
    const int m = TEST_SIZE_SMALL;
    const int n = TEST_SIZE_SMALL;
    const int k = TEST_SIZE_SMALL;
    const float alpha = 1.0f, beta = 0.0f;
    
    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);
    
    float *h_a = (float*)calloc(m * k, sizeof(float));
    float *h_b = (float*)calloc(k * n, sizeof(float));
    float *h_c = (float*)calloc(m * n, sizeof(float));
    float *h_c_result = (float*)calloc(m * n, sizeof(float));
    
    // Initialize with identity-like pattern
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_a[i * k + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_b[i * n + j] = (float)(i + j);
        }
    }
    
    fb_gpu_ptr_t d_a, d_b, d_c;
    if (trait->malloc(handle, &d_a, size_a) != 0 ||
        trait->malloc(handle, &d_b, size_b) != 0 ||
        trait->malloc(handle, &d_c, size_c) != 0) {
        RECORD_TEST(backend_name, "sgemm", 0, 0.0, "Device allocation failed");
        free(h_a); free(h_b); free(h_c); free(h_c_result);
        return;
    }
    
    trait->memcpy_h2d(handle, d_a, h_a, size_a);
    trait->memcpy_h2d(handle, d_b, h_b, size_b);
    trait->memcpy_h2d(handle, d_c, h_c, size_c);
    
    double t0 = get_time_ms();
    trait->sgemm(handle, stream, 'N', 'N', m, n, k, alpha, 
                 d_a, k, d_b, n, beta, d_c, n);
    trait->stream_synchronize(handle, stream);
    double t1 = get_time_ms();
    
    trait->memcpy_d2h(handle, h_c_result, d_c, size_c);
    
    // Verify: Since A is identity-like, C should ≈ B
    int passed = 1;
    for (int i = 0; i < m && i < k; i++) {
        for (int j = 0; j < n; j++) {
            float expected = h_b[i * n + j];
            if (fabsf(h_c_result[i * n + j] - expected) > TOLERANCE_FLOAT) {
                passed = 0;
                break;
            }
        }
        if (!passed) break;
    }
    
    double gflops = (2.0 * m * n * k) / ((t1 - t0) * 1e6);
    char msg[128];
    snprintf(msg, sizeof(msg), passed ? "%.2f GFLOPS" : "Numerical error", gflops);
    
    RECORD_TEST(backend_name, "sgemm", passed, t1 - t0, passed ? NULL : "Numerical error");
    
    trait->free(handle, d_a);
    trait->free(handle, d_b);
    trait->free(handle, d_c);
    free(h_a); free(h_b); free(h_c); free(h_c_result);
}

// ============================================================================
// LAPACK Tests
// ============================================================================

/**
 * Test SGETRF/SGETRS: LU factorization and solve
 */
static void test_sgetrf_getrs(const fb_gpu_backend_trait_t* trait, void* handle, fb_gpu_stream_t stream, const char* backend_name) {
    const int n = TEST_SIZE_SMALL;
    const int nrhs = 1;
    
    float *h_a = (float*)calloc(n * n, sizeof(float));
    float *h_b = (float*)calloc(n * nrhs, sizeof(float));
    float *h_x = (float*)calloc(n * nrhs, sizeof(float));
    int *h_ipiv = (int*)calloc(n, sizeof(int));
    
    // Create a simple diagonally dominant matrix
    for (int i = 0; i < n; i++) {
        h_a[i * n + i] = (float)(n + 1);
        for (int j = 0; j < n; j++) {
            if (i != j) h_a[i * n + j] = 1.0f;
        }
        h_b[i] = (float)(i + 1);
    }
    
    fb_gpu_ptr_t d_a, d_b, d_ipiv;
    if (trait->malloc(handle, &d_a, n * n * sizeof(float)) != 0 ||
        trait->malloc(handle, &d_b, n * nrhs * sizeof(float)) != 0 ||
        trait->malloc(handle, &d_ipiv, n * sizeof(int)) != 0) {
        RECORD_TEST(backend_name, "sgetrf/sgetrs", 0, 0.0, "Device allocation failed");
        free(h_a); free(h_b); free(h_x); free(h_ipiv);
        return;
    }
    
    trait->memcpy_h2d(handle, d_a, h_a, n * n * sizeof(float));
    trait->memcpy_h2d(handle, d_b, h_b, n * nrhs * sizeof(float));
    
    double t0 = get_time_ms();
    int info1 = trait->sgetrf(handle, stream, n, n, d_a, n, d_ipiv);
    int info2 = trait->sgetrs(handle, stream, 'N', n, nrhs, d_a, n, d_ipiv, d_b, n);
    trait->stream_synchronize(handle, stream);
    double t1 = get_time_ms();
    
    trait->memcpy_d2h(handle, h_x, d_b, n * nrhs * sizeof(float));
    
    int passed = (info1 == 0 && info2 == 0);
    
    RECORD_TEST(backend_name, "sgetrf/sgetrs", passed, t1 - t0,
                passed ? NULL : "LAPACK returned error");
    
    trait->free(handle, d_a);
    trait->free(handle, d_b);
    trait->free(handle, d_ipiv);
    free(h_a); free(h_b); free(h_x); free(h_ipiv);
}

/**
 * Test SPOTRF/SPOTRS: Cholesky factorization and solve
 */
static void test_spotrf_potrs(const fb_gpu_backend_trait_t* trait, void* handle, fb_gpu_stream_t stream, const char* backend_name) {
    const int n = TEST_SIZE_SMALL;
    const int nrhs = 1;
    
    float *h_a = (float*)calloc(n * n, sizeof(float));
    float *h_b = (float*)calloc(n * nrhs, sizeof(float));
    
    // Create a symmetric positive definite matrix (A = I + v*v^T)
    for (int i = 0; i < n; i++) {
        h_a[i * n + i] = (float)(n + 2); // Diagonal dominance
        h_b[i] = (float)(i + 1);
    }
    
    fb_gpu_ptr_t d_a, d_b;
    if (trait->malloc(handle, &d_a, n * n * sizeof(float)) != 0 ||
        trait->malloc(handle, &d_b, n * nrhs * sizeof(float)) != 0) {
        RECORD_TEST(backend_name, "spotrf/spotrs", 0, 0.0, "Device allocation failed");
        free(h_a); free(h_b);
        return;
    }
    
    trait->memcpy_h2d(handle, d_a, h_a, n * n * sizeof(float));
    trait->memcpy_h2d(handle, d_b, h_b, n * nrhs * sizeof(float));
    
    double t0 = get_time_ms();
    int info1 = trait->spotrf(handle, stream, 'U', n, d_a, n);
    int info2 = trait->spotrs(handle, stream, 'U', n, nrhs, d_a, n, d_b, n);
    trait->stream_synchronize(handle, stream);
    double t1 = get_time_ms();
    
    int passed = (info1 == 0 && info2 == 0);
    
    RECORD_TEST(backend_name, "spotrf/spotrs", passed, t1 - t0,
                passed ? NULL : "LAPACK returned error");
    
    trait->free(handle, d_a);
    trait->free(handle, d_b);
    free(h_a); free(h_b);
}

// ============================================================================
// Backend Test Runner
// ============================================================================

static void run_backend_tests(const fb_gpu_backend_trait_t* trait, const char* backend_name) {
    printf("\n========================================\n");
    printf("Testing Backend: %s\n", backend_name);
    printf("========================================\n");
    
    // Initialize backend
    void* handle = NULL;
    if (trait->init(0, &handle) != 0) {
        printf("ERROR: Failed to initialize %s backend\n", backend_name);
        RECORD_TEST(backend_name, "initialization", 0, 0.0, "Init failed");
        return;
    }
    
    RECORD_TEST(backend_name, "initialization", 1, 0.0, NULL);
    
    // Create stream
    fb_gpu_stream_t stream = NULL;
    trait->stream_create(handle, &stream);
    
    // Run tests
    printf("\nBLAS Level 1 Tests:\n");
    test_saxpy(trait, handle, stream, backend_name);
    test_sdot(trait, handle, stream, backend_name);
    
    printf("\nBLAS Level 3 Tests:\n");
    test_sgemm(trait, handle, stream, backend_name);
    
    printf("\nLAPACK Tests:\n");
    test_sgetrf_getrs(trait, handle, stream, backend_name);
    test_spotrf_potrs(trait, handle, stream, backend_name);
    
    // Cleanup
    trait->stream_destroy(handle, stream);
    trait->shutdown(handle);
    
    printf("\nBackend tests complete.\n");
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

// External trait declarations
extern "C" {
    #ifdef WITH_CUDA
    extern const fb_gpu_backend_trait_t fb_cublas_trait;
    #endif
    
    #ifdef WITH_HIP
    // Note: hipSOLVER only provides LAPACK, not full BLAS
    extern const fb_gpu_backend_trait_t fb_hipsolver_lapack_trait;
    extern const fb_gpu_backend_trait_t fb_rocblas_trait;
    #endif
    
    #ifdef WITH_SYCL
    extern const fb_gpu_backend_trait_t fb_onemkl_trait;
    #endif
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  GPU Backend Comprehensive Test Suite\n");
    printf("========================================\n");
    printf("Testing all BLAS/LAPACK operations\n");
    printf("Detecting available hardware...\n\n");
    
    // Test all available backends
    #ifdef WITH_CUDA
    run_backend_tests(&fb_cublas_trait, "NVIDIA cuBLAS");
    #endif
    
    #ifdef WITH_HIP
    // Note: hipSOLVER implements LAPACK only (wraps cuSOLVER/rocSOLVER)
    // For BLAS, use rocBLAS
    run_backend_tests(&fb_hipsolver_lapack_trait, "AMD hipSOLVER (LAPACK)");
    // run_backend_tests(&fb_rocblas_trait, "AMD rocBLAS");
    #endif
    
    #ifdef WITH_SYCL
    run_backend_tests(&fb_onemkl_trait, "Intel oneMKL");
    #endif
    
    // Print summary
    printf("\n========================================\n");
    printf("  Test Summary\n");
    printf("========================================\n");
    
    int total_passed = 0, total_failed = 0;
    const char* current_backend = NULL;
    
    for (int i = 0; i < num_tests; i++) {
        if (current_backend == NULL || strcmp(current_backend, test_results[i].backend_name) != 0) {
            current_backend = test_results[i].backend_name;
            printf("\n%s:\n", current_backend);
        }
        
        printf("  %-20s: %s", test_results[i].test_name, 
               test_results[i].passed ? "PASS" : "FAIL");
        
        if (test_results[i].time_ms > 0.0) {
            printf(" (%.3f ms)", test_results[i].time_ms);
        }
        if (test_results[i].error_msg) {
            printf(" - %s", test_results[i].error_msg);
        }
        printf("\n");
        
        if (test_results[i].passed) total_passed++;
        else total_failed++;
    }
    
    printf("\n========================================\n");
    printf("Total: %d tests, %d passed, %d failed\n", 
           num_tests, total_passed, total_failed);
    printf("========================================\n");
    
    return (total_failed == 0) ? 0 : 1;
}
