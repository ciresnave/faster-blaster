/**
 * @file test_aocl_backend.c
 * @brief Test suite for AMD AOCL CPU backend
 * 
 * Tests the AOCL backend implementation on AMD Ryzen processors.
 * Validates BLAS Level 1, 2, and 3 operations with numerical correctness checks.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "../src/backends/aocl_backend.h"

#define TOLERANCE_SINGLE 1e-5f
#define TOLERANCE_DOUBLE 1e-12
#define TEST_PASSED "✅ PASSED"
#define TEST_FAILED "❌ FAILED"

static int tests_passed = 0;
static int tests_failed = 0;

/**
 * Compare two floating point values with tolerance
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
 * Test 1: AOCL availability and version
 */
static void test_aocl_availability(void) {
    printf("\n📋 Test 1: AOCL Availability\n");
    printf("═══════════════════════════════════════\n");
    
    bool available = fb_aocl_is_available();
    printf("  AOCL library available: %s\n", available ? "Yes" : "No");
    
    if (!available) {
        printf("  ⚠️  AOCL not found on system. Install AMD AOCL from:\n");
        printf("      https://developer.amd.com/amd-aocl/\n");
        report_test("AOCL availability check", false);
        return;
    }
    
    int major, minor, update;
    int ret = fb_aocl_get_version(&major, &minor, &update);
    if (true) {
        printf("  AOCL version: %d.%d.%d\n", major, minor, update);
    }
    
    report_test("AOCL availability check", available);
}

/**
 * Test 2: AOCL initialization and threading
 */
static void test_aocl_init(void) {
    printf("\n📋 Test 2: AOCL Initialization\n");
    printf("═══════════════════════════════════════\n");
    
    int ret = fb_aocl_init();
    printf("  Initialization result: %d\n", ret);
    
    if (ret != 0) {
        printf("  ⚠️  AOCL initialization failed\n");
        report_test("AOCL initialization", false);
        return;
    }
    
    // Test threading configuration
    printf("  Default threads: %d\n", fb_aocl_get_num_threads());
    
    fb_aocl_set_num_threads(8);
    printf("  After setting to 8: %d\n", fb_aocl_get_num_threads());
    
    fb_aocl_set_num_threads(0); // Auto
    printf("  After auto: %d\n", fb_aocl_get_num_threads());
    
    report_test("AOCL initialization", true);
}

/**
 * Test 3: BLAS Level 1 - SAXPY (y = alpha*x + y)
 */
static void test_saxpy(void) {
    printf("\n📋 Test 3: BLAS Level 1 - SAXPY\n");
    printf("═══════════════════════════════════════\n");
    
    const int n = 5;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float expected[] = {12.0f, 24.0f, 36.0f, 48.0f, 60.0f}; // 2*x + y
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->saxpy) {
        printf("  ⚠️  SAXPY not available in vtable\n");
        report_test("SAXPY operation", false);
        return;
    }
    
    vtable->saxpy(n, alpha, x, 1, y, 1);
    
    printf("  Input: x = [1, 2, 3, 4, 5]\n");
    printf("  Input: y = [10, 20, 30, 40, 50]\n");
    printf("  Alpha: %.1f\n", alpha);
    printf("  Result: y = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           y[0], y[1], y[2], y[3], y[4]);
    printf("  Expected: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
           expected[0], expected[1], expected[2], expected[3], expected[4]);
    
    bool passed = true;
    for (int i = 0; i < n && passed; i++) {
        if (!float_equal(y[i], expected[i], TOLERANCE_SINGLE)) {
            printf("  ❌ Mismatch at index %d: %.6f != %.6f\n", i, y[i], expected[i]);
            passed = false;
        }
    }
    
    report_test("SAXPY operation", passed);
}

/**
 * Test 4: BLAS Level 1 - SDOT (dot product)
 */
static void test_sdot(void) {
    printf("\n📋 Test 4: BLAS Level 1 - SDOT\n");
    printf("═══════════════════════════════════════\n");
    
    const int n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected = 1*5 + 2*6 + 3*7 + 4*8; // = 70
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->sdot) {
        printf("  ⚠️  SDOT not available in vtable\n");
        report_test("SDOT operation", false);
        return;
    }
    
    float result = vtable->sdot(n, x, 1, y, 1);
    
    printf("  Input: x = [1, 2, 3, 4]\n");
    printf("  Input: y = [5, 6, 7, 8]\n");
    printf("  Result: %.1f\n", result);
    printf("  Expected: %.1f\n", expected);
    
    bool passed =  float_equal(result, expected, TOLERANCE_SINGLE);
    report_test("SDOT operation", passed);
}

/**
 * Test 5: BLAS Level 2 - SGEMV (matrix-vector product)
 */
static void test_sgemv(void) {
    printf("\n📋 Test 5: BLAS Level 2 - SGEMV\n");
    printf("═══════════════════════════════════════\n");
    
    // Test: y = alpha*A*x + beta*y
    // A = [[1, 2],     x = [1],    y = [0]
    //      [3, 4],          [2]         [0]
    //      [5, 6]]
    // Result: y = 1.0 * A*x + 0.0*y = [5, 11, 17]
    
    const int m = 3, n = 2;
    float alpha = 1.0f, beta = 0.0f;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // Row-major
    float x[] = {1.0f, 2.0f};
    float y[] = {0.0f, 0.0f, 0.0f};
    float expected[] = {5.0f, 11.0f, 17.0f};
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->sgemv) {
        printf("  ⚠️  SGEMV not available in vtable\n");
        report_test("SGEMV operation", false);
        return;
    }
    
    // FbRowMajor, FbNoTrans
    vtable->sgemv(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, m, n, 
                            alpha, A, n, x, 1, beta, y, 1);
    
    printf("  Matrix A (3x2):\n");
    printf("    [1, 2]\n");
    printf("    [3, 4]\n");
    printf("    [5, 6]\n");
    printf("  Vector x: [1, 2]\n");
    printf("  Result y: [%.1f, %.1f, %.1f]\n", y[0], y[1], y[2]);
    printf("  Expected: [%.1f, %.1f, %.1f]\n", expected[0], expected[1], expected[2]);
    
    bool passed = true;
    for (int i = 0; i < m && passed; i++) {
        if (!float_equal(y[i], expected[i], TOLERANCE_SINGLE)) {
            printf("  ❌ Mismatch at index %d: %.6f != %.6f\n", i, y[i], expected[i]);
            passed = false;
        }
    }
    
    report_test("SGEMV operation", passed);
}

/**
 * Test 6: BLAS Level 3 - SGEMM (matrix-matrix product)
 */
static void test_sgemm(void) {
    printf("\n📋 Test 6: BLAS Level 3 - SGEMM\n");
    printf("═══════════════════════════════════════\n");
    
    // Test: C = alpha*A*B + beta*C
    // A = [[1, 2],     B = [[5, 6],     C = [[0, 0]
    //      [3, 4]]          [7, 8]]          [0, 0]]
    // Result: C = A*B = [[19, 22], [43, 50]]
    
    const int m = 2, n = 2, k = 2;
    float alpha = 1.0f, beta = 0.0f;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Row-major 2x2
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f}; // Row-major 2x2
    float C[] = {0.0f, 0.0f, 0.0f, 0.0f}; // Row-major 2x2
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->sgemm) {
        printf("  ⚠️  SGEMM not available in vtable\n");
        report_test("SGEMM operation", false);
        return;
    }
    
    vtable->sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            m, n, k, alpha, A, k, B, n, beta, C, n);
    
    printf("  Matrix A (2x2):      Matrix B (2x2):\n");
    printf("    [1, 2]               [5, 6]\n");
    printf("    [3, 4]               [7, 8]\n");
    printf("\n");
    printf("  Result C (2x2):\n");
    printf("    [%.0f, %.0f]\n", C[0], C[1]);
    printf("    [%.0f, %.0f]\n", C[2], C[3]);
    printf("\n");
    printf("  Expected:\n");
    printf("    [%.0f, %.0f]\n", expected[0], expected[1]);
    printf("    [%.0f, %.0f]\n", expected[2], expected[3]);
    
    bool passed = true;
    for (int i = 0; i < m*n && passed; i++) {
        if (!float_equal(C[i], expected[i], TOLERANCE_SINGLE)) {
            printf("  ❌ Mismatch at index %d: %.6f != %.6f\n", i, C[i], expected[i]);
            passed = false;
        }
    }
    
    report_test("SGEMM operation", passed);
}

/**
 * Test 7: Large SGEMM for performance validation
 */
static void test_large_sgemm(void) {
    printf("\n📋 Test 7: Large SGEMM Performance\n");
    printf("═══════════════════════════════════════\n");
    
    const int size = 512;
    float* A = (float*)malloc(size * size * sizeof(float));
    float* B = (float*)malloc(size * size * sizeof(float));
    float* C = (float*)malloc(size * size * sizeof(float));
    
    if (!A || !B || !C) {
        printf("  ❌ Memory allocation failed\n");
        free(A); free(B); free(C);
        report_test("Large SGEMM allocation", false);
        return;
    }
    
    // Initialize matrices
    for (int i = 0; i < size * size; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i * 7) % 100) / 100.0f;
        C[i] = 0.0f;
    }
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->sgemm) {
        printf("  ⚠️  SGEMM not available\n");
        free(A); free(B); free(C);
        report_test("Large SGEMM operation", false);
        return;
    }
    
    printf("  Matrix size: %dx%d\n", size, size);
    printf("  Computing C = A * B...\n");
    
    vtable->sgemm(0, 0, 0, size, size, size,
                            1.0f, A, size, B, size, 0.0f, C, size);
    
    // Check for NaN/Inf and non-zero results
    int non_zero = 0;
    int nan_count = 0;
    for (int i = 0; i < size * size; i++) {
        if (isnan(C[i]) || isinf(C[i])) nan_count++;
        if (fabsf(C[i]) > TOLERANCE_SINGLE) non_zero++;
    }
    
    printf("  Non-zero elements: %d / %d (%.1f%%)\n", 
           non_zero, size*size, 100.0f * non_zero / (size*size));
    printf("  NaN/Inf elements: %d\n", nan_count);
    printf("  Sample C[0,0] = %.6f\n", C[0]);
    printf("  Sample C[%d,%d] = %.6f\n", size-1, size-1, C[size*size-1]);
    
    bool passed =  (nan_count == 0) && (non_zero > 0);
    
    free(A);
    free(B);
    free(C);
    
    report_test("Large SGEMM operation", passed);
}

/**
 * Test 8: Double precision DGEMM
 */
static void test_dgemm(void) {
    printf("\n📋 Test 8: Double Precision DGEMM\n");
    printf("═══════════════════════════════════════\n");
    
    const int m = 2, n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0, 7.0, 8.0};
    double C[] = {0.0, 0.0, 0.0, 0.0};
    double expected[] = {19.0, 22.0, 43.0, 50.0};
    
    const fb_backend_vtable_t* vtable = fb_aocl_get_vtable();
    if (!vtable || !vtable->dgemm) {
        printf("  ⚠️  DGEMM not available in vtable\n");
        report_test("DGEMM operation", false);
        return;
    }
    
    vtable->dgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS, m, n, k, alpha, A, k, B, n, beta, C, n);
    
    printf("  Result C: [%.0f, %.0f, %.0f, %.0f]\n", C[0], C[1], C[2], C[3]);
    printf("  Expected: [%.0f, %.0f, %.0f, %.0f]\n", 
           expected[0], expected[1], expected[2], expected[3]);
    
    bool passed = true;
    for (int i = 0; i < m*n && passed; i++) {
        if (!double_equal(C[i], expected[i], TOLERANCE_DOUBLE)) {
            printf("  ❌ Mismatch at index %d: %.12f != %.12f\n", i, C[i], expected[i]);
            passed = false;
        }
    }
    
    report_test("DGEMM operation", passed);
}

/**
 * Main test runner
 */
int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       AMD AOCL Backend Test Suite                         ║\n");
    printf("║       Testing on: AMD Ryzen 9 7950X (expected)           ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Run all tests
    test_aocl_availability();
    test_aocl_init();
    test_saxpy();
    test_sdot();
    test_sgemv();
    test_sgemm();
    test_large_sgemm();
    test_dgemm();
    
    // Cleanup
    fb_aocl_shutdown();
    
    // Print summary
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
