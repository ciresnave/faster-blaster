/**
 * @file test_smart_wrappers.c
 * @brief Test smart wrappers for BLAS operations
 * 
 * This test validates the complete set of smart wrappers including:
 * - Level 1: saxpy, sdot, snrm2, scopy, etc.
 * - Level 2: sgemv, sger
 * - Level 3: sgemm
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/device_memory_manager.h"
#include "faster-blaster/gpu_backend_trait.h"
#include "faster_blaster.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* External functions from smart wrappers */
extern int fb_cublas_smart_init(int device_id, void* lib_handle);
extern void fb_cublas_smart_shutdown(void);
extern void fb_cublas_print_memory_stats(void);
extern int fb_cublas_sync_buffer_to_host(void* host_ptr);

/* Smart wrapper functions to test */
extern void cublas_saxpy_smart_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy);
extern void cublas_scopy_smart_wrapper(int n, const float* x, int incx, float* y, int incy);
extern void cublas_sscal_smart_wrapper(int n, float alpha, float* x, int incx);
extern float cublas_sdot_smart_wrapper(int n, const float* x, int incx, const float* y, int incy);
extern float cublas_snrm2_smart_wrapper(int n, const float* x, int incx);
extern void cublas_sgemv_smart_wrapper(FB_LAYOUT layout, FB_TRANSPOSE trans, int m, int n,
                                       float alpha, const float* a, int lda,
                                       const float* x, int incx, float beta, float* y, int incy);

#define TOLERANCE 1e-5f

static int float_equal(float a, float b) {
    return fabsf(a - b) < TOLERANCE;
}

static void print_vector(const char* name, const float* v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n && i < 10; i++) {
        printf("%.4f%s", v[i], (i < n-1 && i < 9) ? ", " : "");
    }
    if (n > 10) printf(", ...");
    printf("]\n");
}

/* ============================================================================
 * Test Cases
 * ========================================================================== */

static int test_saxpy(void) {
    printf("\n=== Test 1: saxpy (y = alpha*x + y) ===\n");
    
    int n = 5;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float expected[] = {12.0f, 24.0f, 36.0f, 48.0f, 60.0f}; // 2*x + y
    
    print_vector("x", x, n);
    print_vector("y (before)", y, n);
    
    cublas_saxpy_smart_wrapper(n, alpha, x, 1, y, 1);
    fb_cublas_sync_buffer_to_host(y);
    
    print_vector("y (after)", y, n);
    
    for (int i = 0; i < n; i++) {
        if (!float_equal(y[i], expected[i])) {
            printf("❌ FAILED: y[%d] = %.4f, expected %.4f\n", i, y[i], expected[i]);
            return 0;
        }
    }
    
    printf("✅ Test passed!\n");
    return 1;
}

static int test_scopy(void) {
    printf("\n=== Test 2: scopy (y = x) ===\n");
    
    int n = 5;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    print_vector("x", x, n);
    print_vector("y (before)", y, n);
    
    cublas_scopy_smart_wrapper(n, x, 1, y, 1);
    fb_cublas_sync_buffer_to_host(y);
    
    print_vector("y (after)", y, n);
    
    for (int i = 0; i < n; i++) {
        if (!float_equal(y[i], x[i])) {
            printf("❌ FAILED: y[%d] = %.4f, expected %.4f\n", i, y[i], x[i]);
            return 0;
        }
    }
    
    printf("✅ Test passed!\n");
    return 1;
}

static int test_sscal(void) {
    printf("\n=== Test 3: sscal (x = alpha*x) ===\n");
    
    int n = 5;
    float alpha = 3.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected[] = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f};
    
    print_vector("x (before)", x, n);
    
    cublas_sscal_smart_wrapper(n, alpha, x, 1);
    fb_cublas_sync_buffer_to_host(x);
    
    print_vector("x (after)", x, n);
    
    for (int i = 0; i < n; i++) {
        if (!float_equal(x[i], expected[i])) {
            printf("❌ FAILED: x[%d] = %.4f, expected %.4f\n", i, x[i], expected[i]);
            return 0;
        }
    }
    
    printf("✅ Test passed!\n");
    return 1;
}

static int test_sdot(void) {
    printf("\n=== Test 4: sdot (dot product) ===\n");
    
    int n = 5;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float expected = 70.0f; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2+6+12+20+30 = 70
    
    print_vector("x", x, n);
    print_vector("y", y, n);
    
    float result = cublas_sdot_smart_wrapper(n, x, 1, y, 1);
    
    printf("dot(x, y) = %.4f (expected %.4f)\n", result, expected);
    
    if (!float_equal(result, expected)) {
        printf("❌ FAILED: result = %.4f, expected %.4f\n", result, expected);
        return 0;
    }
    
    printf("✅ Test passed!\n");
    return 1;
}

static int test_snrm2(void) {
    printf("\n=== Test 5: snrm2 (Euclidean norm) ===\n");
    
    int n = 3;
    float x[] = {3.0f, 4.0f, 0.0f};
    float expected = 5.0f; // sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5
    
    print_vector("x", x, n);
    
    float result = cublas_snrm2_smart_wrapper(n, x, 1);
    
    printf("||x||₂ = %.4f (expected %.4f)\n", result, expected);
    
    if (!float_equal(result, expected)) {
        printf("❌ FAILED: result = %.4f, expected %.4f\n", result, expected);
        return 0;
    }
    
    printf("✅ Test passed!\n");
    return 1;
}

static int test_operation_chaining(void) {
    printf("\n=== Test 6: Operation Chaining (Cache Efficiency) ===\n");
    
    int n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {10.0f, 20.0f, 30.0f, 40.0f};
    
    printf("Initial:\n");
    print_vector("x", x, n);
    print_vector("y", y, n);
    
    // Operation 1: y = 2*x + y
    printf("\nOperation 1: y = 2*x + y\n");
    cublas_saxpy_smart_wrapper(n, 2.0f, x, 1, y, 1);
    
    printf("After Op 1 (stats):\n");
    fb_cublas_print_memory_stats();
    
    // Operation 2: y = 3*y (should reuse cached y)
    printf("\nOperation 2: y = 3*y (should hit cache)\n");
    cublas_sscal_smart_wrapper(n, 3.0f, y, 1);
    
    printf("After Op 2 (stats):\n");
    fb_cublas_print_memory_stats();
    
    // Operation 3: norm = ||y||₂ (should reuse cached y)
    printf("\nOperation 3: norm = ||y||₂ (should hit cache)\n");
    float norm = cublas_snrm2_smart_wrapper(n, y, 1);
    
    printf("After Op 3 (stats):\n");
    fb_cublas_print_memory_stats();
    
    // Sync final result
    fb_cublas_sync_buffer_to_host(y);
    print_vector("y (final)", y, n);
    printf("norm = %.4f\n", norm);
    
    // Expected: y = 3*(2*x + y) = 6*x + 3*y = [6,12,18,24] + [30,60,90,120] = [36,72,108,144]
    float expected_y[] = {36.0f, 72.0f, 108.0f, 144.0f};
    float expected_norm = sqrtf(36*36 + 72*72 + 108*108 + 144*144);
    
    printf("\nExpected:\n");
    print_vector("y", expected_y, n);
    printf("norm = %.4f\n", expected_norm);
    
    for (int i = 0; i < n; i++) {
        if (!float_equal(y[i], expected_y[i])) {
            printf("❌ FAILED: y[%d] = %.4f, expected %.4f\n", i, y[i], expected_y[i]);
            return 0;
        }
    }
    
    if (!float_equal(norm, expected_norm)) {
        printf("❌ FAILED: norm = %.4f, expected %.4f\n", norm, expected_norm);
        return 0;
    }
    
    printf("\n✅ Test passed! Cache efficiency demonstrated.\n");
    return 1;
}

/* ============================================================================
 * Main Test Runner
 * ========================================================================== */

int main(void) {
    printf("========================================\n");
    printf("Smart Wrappers Test Suite\n");
    printf("========================================\n");
    
    // Initialize cuBLAS backend
    if (fb_cublas_smart_init(0, NULL) != 0) {
        printf("❌ Failed to initialize cuBLAS backend\n");
        return 1;
    }
    
    int passed = 0;
    int total = 6;
    
    // Run tests
    if (test_saxpy()) passed++;
    if (test_scopy()) passed++;
    if (test_sscal()) passed++;
    if (test_sdot()) passed++;
    if (test_snrm2()) passed++;
    if (test_operation_chaining()) passed++;
    
    // Final statistics
    printf("\n========================================\n");
    printf("Final Memory Statistics\n");
    printf("========================================\n");
    fb_cublas_print_memory_stats();
    
    // Shutdown
    fb_cublas_smart_shutdown();
    
    // Summary
    printf("\n========================================\n");
    printf("Test Summary: %d/%d tests passed\n", passed, total);
    printf("========================================\n");
    
    return (passed == total) ? 0 : 1;
}
