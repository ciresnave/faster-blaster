/**
 * @file test_onemkl_backend.cpp
 * @brief Test suite for Intel oneMKL GPU backend
 * 
 * Tests the oneMKL backend implementation for Intel GPUs.
 * Can also run on NVIDIA/AMD GPUs via oneMKL's CUDA/HIP backends.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "../include/faster-blaster/gpu_backend_trait.h"

// Forward declare the trait getter with C linkage
extern "C" {
    const fb_gpu_backend_trait* fb_onemkl_get_trait(void);
}

#define TOLERANCE_SINGLE 1e-4f
#define TOLERANCE_DOUBLE 1e-10
#define TEST_PASSED "✅ PASSED"
#define TEST_FAILED "❌ FAILED"

static int tests_passed = 0;
static int tests_failed = 0;
static void* backend_handle = nullptr;

/**
 * Compare floating point values with tolerance
 */
static bool float_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static bool double_equal(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

/**
 * Print test result
 */
static void report_test(const char* test_name, bool passed) {
    if (passed) {
        printf("  %s %s\n", TEST_PASSED, test_name);
        tests_passed++;
    } else {
        printf("  %s %s\n", TEST_FAILED, test_name);
        tests_failed++;
    }
}

/**
 * Test 1: Backend initialization
 */
static void test_init(void) {
    printf("\n📋 Test 1: oneMKL Backend Initialization\n");
    printf("═══════════════════════════════════════\n");
    
    // Get trait implementation

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    if (!trait) {
        printf("  ❌ Failed to get oneMKL trait\n");
        report_test("Get oneMKL trait", false);
        return;
    }
    
    printf("  oneMKL trait loaded successfully\n");
    
    // Initialize backend for device 0
    int ret = trait->init(0, &backend_handle);
    
    if (ret != 0) {
        printf("  ⚠️  oneMKL initialization failed (code %d)\n", ret);
        printf("  This is expected if:\n");
        printf("  - Intel GPU not present\n");
        printf("  - oneMKL not installed\n");
        printf("  - oneMKL CUDA backend not configured\n");
        report_test("Backend initialization", false);
        return;
    }
    
    printf("  Backend initialized successfully\n");
    printf("  Handle: %p\n", backend_handle);
    
    report_test("Backend initialization", ret == 0);
}

/**
 * Test 2: Device properties
 */
static void test_device_properties(void) {
    printf("\n📋 Test 2: Device Properties\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("Device properties", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    char name[256] = {0};
    size_t total_mem = 0;
    int ret = trait->get_device_properties(backend_handle, 0, name, sizeof(name), &total_mem);
    
    if (ret == 0) {
        printf("  Device name: %s\n", name);
        printf("  Total memory: %.2f GB\n", total_mem / (1024.0*1024.0*1024.0));
    }
    
    report_test("Device properties", ret == 0);
}

/**
 * Test 3: Memory operations
 */
static void test_memory_ops(void) {
    printf("\n📋 Test 3: Memory Operations\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("Memory operations", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    const size_t size = 1024 * sizeof(float);
    fb_gpu_ptr_t dev_ptr = nullptr;
    
    // Allocate GPU memory
    int ret = trait->malloc(backend_handle, &dev_ptr, size);
    if (ret != 0) {
        printf("  ❌ Memory allocation failed\n");
        report_test("Memory operations", false);
        return;
    }
    
    printf("  Allocated %zu bytes at %p\n", size, dev_ptr);
    
    // Test host-to-device copy
    float* host_data = (float*)malloc(size);
    for (int i = 0; i < 1024; i++) {
        host_data[i] = (float)i;
    }
    
    ret = trait->memcpy_h2d(backend_handle, dev_ptr, host_data, size);
    if (ret != 0) {
        printf("  ❌ Host-to-device copy failed\n");
        free(host_data);
        trait->free(backend_handle, dev_ptr);
        report_test("Memory operations", false);
        return;
    }
    
    printf("  Copied %zu bytes to device\n", size);
    
    // Test device-to-host copy
    float* verify_data = (float*)malloc(size);
    ret = trait->memcpy_d2h(backend_handle, verify_data, dev_ptr, size);
    if (ret != 0) {
        printf("  ❌ Device-to-host copy failed\n");
        free(host_data);
        free(verify_data);
        trait->free(backend_handle, dev_ptr);
        report_test("Memory operations", false);
        return;
    }
    
    printf("  Copied %zu bytes from device\n", size);
    
    // Verify data
    bool data_correct = true;
    for (int i = 0; i < 1024; i++) {
        if (!float_equal(verify_data[i], host_data[i], TOLERANCE_SINGLE)) {
            printf("  ❌ Data mismatch at index %d: %.1f != %.1f\n", 
                   i, verify_data[i], host_data[i]);
            data_correct = false;
            break;
        }
    }
    
    if (data_correct) {
        printf("  Data verification passed\n");
    }
    
    // Cleanup
    trait->free(backend_handle, dev_ptr);
    free(host_data);
    free(verify_data);
    
    report_test("Memory operations", ret == 0 && data_correct);
}

/**
 * Test 4: SAXPY (Level 1 BLAS)
 */
static void test_saxpy(void) {
    printf("\n📋 Test 4: BLAS Level 1 - SAXPY\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("SAXPY operation", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    const int n = 5;
    float alpha = 2.0f;
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_y[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float expected[] = {12.0f, 24.0f, 36.0f, 48.0f, 60.0f};
    
    // Allocate device memory
    fb_gpu_ptr_t d_x, d_y;
    trait->malloc(backend_handle, &d_x, n * sizeof(float));
    trait->malloc(backend_handle, &d_y, n * sizeof(float));
    
    // Copy to device
    trait->memcpy_h2d(backend_handle, d_x, h_x, n * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_y, h_y, n * sizeof(float));
    
    // Execute SAXPY: y = alpha*x + y
    trait->saxpy(backend_handle, nullptr, n, alpha, d_x, 1, d_y, 1);
    
    // Copy result back
    int ret = trait->memcpy_d2h(backend_handle, h_y, d_y, n * sizeof(float));
    
    printf("  Input: x = [1, 2, 3, 4, 5], y = [10, 20, 30, 40, 50]\n");
    printf("  Alpha: %.1f\n", alpha);
    printf("  Result: y = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           h_y[0], h_y[1], h_y[2], h_y[3], h_y[4]);
    
    bool passed = (ret == 0);
    for (int i = 0; i < n && passed; i++) {
        if (!float_equal(h_y[i], expected[i], TOLERANCE_SINGLE)) {
            printf("  ❌ Mismatch at %d: %.6f != %.6f\n", i, h_y[i], expected[i]);
            passed = false;
        }
    }
    
    trait->free(backend_handle, d_x);
    trait->free(backend_handle, d_y);
    
    report_test("SAXPY operation", passed);
}

/**
 * Test 5: SGEMV (Level 2 BLAS)
 */
static void test_sgemv(void) {
    printf("\n📋 Test 5: BLAS Level 2 - SGEMV\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("SGEMV operation", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    // y = A*x where A is 3x2
    const int m = 3, n = 2;
    float alpha = 1.0f, beta = 0.0f;
    float h_A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // Row-major
    float h_x[] = {1.0f, 2.0f};
    float h_y[] = {0.0f, 0.0f, 0.0f};
    float expected[] = {5.0f, 11.0f, 17.0f};
    
    fb_gpu_ptr_t d_A, d_x, d_y;
    trait->malloc(backend_handle, &d_A, m * n * sizeof(float));
    trait->malloc(backend_handle, &d_x, n * sizeof(float));
    trait->malloc(backend_handle, &d_y, m * sizeof(float));
    
    trait->memcpy_h2d(backend_handle, d_A, h_A, m * n * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_x, h_x, n * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_y, h_y, m * sizeof(float));
    
    // Execute SGEMV
    trait->sgemv(backend_handle, nullptr, 'N', m, n, alpha, d_A, n, d_x, 1, beta, d_y, 1);
    
    int ret = trait->memcpy_d2h(backend_handle, h_y, d_y, m * sizeof(float));
    
    printf("  Matrix A (3x2): [[1,2], [3,4], [5,6]]\n");
    printf("  Vector x: [1, 2]\n");
    printf("  Result y: [%.1f, %.1f, %.1f]\n", h_y[0], h_y[1], h_y[2]);
    
    bool passed = (ret == 0);
    for (int i = 0; i < m && passed; i++) {
        if (!float_equal(h_y[i], expected[i], TOLERANCE_SINGLE)) {
            passed = false;
        }
    }
    
    trait->free(backend_handle, d_A);
    trait->free(backend_handle, d_x);
    trait->free(backend_handle, d_y);
    
    report_test("SGEMV operation", passed);
}

/**
 * Test 6: SGEMM (Level 3 BLAS)
 */
static void test_sgemm(void) {
    printf("\n📋 Test 6: BLAS Level 3 - SGEMM\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("SGEMM operation", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    // C = A*B where A, B, C are 2x2
    const int m = 2, n = 2, k = 2;
    float alpha = 1.0f, beta = 0.0f;
    float h_A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    
    fb_gpu_ptr_t d_A, d_B, d_C;
    trait->malloc(backend_handle, &d_A, m * k * sizeof(float));
    trait->malloc(backend_handle, &d_B, k * n * sizeof(float));
    trait->malloc(backend_handle, &d_C, m * n * sizeof(float));
    
    trait->memcpy_h2d(backend_handle, d_A, h_A, m * k * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_B, h_B, k * n * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_C, h_C, m * n * sizeof(float));
    
    // Execute SGEMM
    trait->sgemm(backend_handle, nullptr, 'N', 'N', m, n, k, 
                 alpha, d_A, k, d_B, n, beta, d_C, n);
    
    int ret = trait->memcpy_d2h(backend_handle, h_C, d_C, m * n * sizeof(float));
    
    printf("  A = [[1,2], [3,4]], B = [[5,6], [7,8]]\n");
    printf("  Result C:\n");
    printf("    [%.0f, %.0f]\n", h_C[0], h_C[1]);
    printf("    [%.0f, %.0f]\n", h_C[2], h_C[3]);
    
    bool passed = (ret == 0);
    for (int i = 0; i < m*n && passed; i++) {
        if (!float_equal(h_C[i], expected[i], TOLERANCE_SINGLE)) {
            passed = false;
        }
    }
    
    trait->free(backend_handle, d_A);
    trait->free(backend_handle, d_B);
    trait->free(backend_handle, d_C);
    
    report_test("SGEMM operation", passed);
}

/**
 * Test 7: Large SGEMM for performance
 */
static void test_large_sgemm(void) {
    printf("\n📋 Test 7: Large SGEMM (512x512)\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("Large SGEMM", false);
        return;
    }
    

    const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
    
    const int size = 512;
    float* h_A = (float*)malloc(size * size * sizeof(float));
    float* h_B = (float*)malloc(size * size * sizeof(float));
    float* h_C = (float*)malloc(size * size * sizeof(float));
    
    // Initialize
    for (int i = 0; i < size * size; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i * 7) % 100) / 100.0f;
        h_C[i] = 0.0f;
    }
    
    fb_gpu_ptr_t d_A, d_B, d_C;
    trait->malloc(backend_handle, &d_A, size * size * sizeof(float));
    trait->malloc(backend_handle, &d_B, size * size * sizeof(float));
    trait->malloc(backend_handle, &d_C, size * size * sizeof(float));
    
    trait->memcpy_h2d(backend_handle, d_A, h_A, size * size * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_B, h_B, size * size * sizeof(float));
    trait->memcpy_h2d(backend_handle, d_C, h_C, size * size * sizeof(float));
    
    printf("  Computing C = A * B (%dx%d)...\n", size, size);
    
    trait->sgemm(backend_handle, nullptr, 'N', 'N', size, size, size,
                           1.0f, d_A, size, d_B, size, 0.0f, d_C, size);
    
    int ret = trait->memcpy_d2h(backend_handle, h_C, d_C, size * size * sizeof(float));
    
    
    // Validation
    int non_zero = 0, nan_count = 0;
    for (int i = 0; i < size * size; i++) {
        if (isnan(h_C[i]) || isinf(h_C[i])) nan_count++;
        if (fabsf(h_C[i]) > TOLERANCE_SINGLE) non_zero++;
    }
    
    printf("  Non-zero: %d/%d (%.1f%%)\n", non_zero, size*size, 
           100.0f * non_zero / (size*size));
    printf("  NaN/Inf: %d\n", nan_count);
    
    bool passed = (ret == 0) && (nan_count == 0) && (non_zero > 0);
    
    trait->free(backend_handle, d_A);
    trait->free(backend_handle, d_B);
    trait->free(backend_handle, d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    report_test("Large SGEMM", passed);
}

/**
 * Main test runner
 */
int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       Intel oneMKL GPU Backend Test Suite                ║\n");
    printf("║       Testing on: NVIDIA RTX 4090 (via CUDA backend)     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Run tests
    test_init();
    test_device_properties();
    test_memory_ops();
    test_saxpy();
    test_sgemv();
    test_sgemm();
    test_large_sgemm();
    
    // Cleanup
    if (backend_handle) {

        const fb_gpu_backend_trait* trait = fb_onemkl_get_trait();
        trait->shutdown(backend_handle);
    }
    
    // Summary
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Test Summary                            ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total Tests: %d                                           ║\n", tests_passed + tests_failed);
    printf("║  Passed: %d                                                ║\n", tests_passed);
    printf("║  Failed: %d                                                ║\n", tests_failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed == 0) ? 0 : 1;
}
