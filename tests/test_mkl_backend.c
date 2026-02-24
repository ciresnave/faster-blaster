/**
 * @file test_mkl_backend.c
 * @brief Test suite for Intel MKL CPU backend
 * 
 * Tests the Intel MKL backend implementation.
 * Works on both Intel and AMD x86-64 CPUs.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "../src/backends/mkl_backend.h"

#define TOLERANCE_SINGLE 1e-5f
#define TOLERANCE_DOUBLE 1e-12
#define TEST_PASSED "✅ PASSED"
#define TEST_FAILED "❌ FAILED"

static int tests_passed = 0;
static int tests_failed = 0;

static bool float_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static void report_test(const char* test_name, bool passed) {
    if (passed) {
        printf("  %s %s\n", TEST_PASSED, test_name);
        tests_passed++;
    } else {
        printf("  %s %s\n", TEST_FAILED, test_name);
        tests_failed++;
    }
}

static void test_mkl_availability(void) {
    printf("\n📋 Test 1: MKL Availability\n");
    printf("═══════════════════════════════════════\n");
    
    bool available = fb_mkl_is_available();
    printf("  MKL library available: %s\n", available ? "Yes" : "No");
    
    if (!available) {
        printf("  ⚠️  Intel MKL not found. Install from Intel oneAPI:\n");
        printf("      https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html\n");
        report_test("MKL availability check", false);
        return;
    }
    
    report_test("MKL availability check", available);
}

static void test_mkl_saxpy(void) {
    printf("\n📋 Test 2: SAXPY (y = alpha*x + y)\n");
    printf("═══════════════════════════════════════\n");
    
    if (!fb_mkl_is_available()) {
        printf("  ⚠️  MKL not available, skipping\n");
        report_test("SAXPY operation", false);
        return;
    }
    
    const int n = 5;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float expected[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    
    fb_mkl_saxpy(n, alpha, x, 1, y, 1);
    
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (!float_equal(y[i], expected[i], TOLERANCE_SINGLE)) {
            printf("  ❌ y[%d] = %.6f, expected %.6f\n", i, y[i], expected[i]);
            passed = false;
        }
    }
    
    if (passed) {
        printf("  All values correct\n");
    }
    
    report_test("SAXPY operation", passed);
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       Intel MKL CPU Backend Test Suite                   ║\n");
    printf("║       High-performance CPU linear algebra                ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_mkl_availability();
    test_mkl_saxpy();
    
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Test Summary                            ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total Tests: %d                                           ║\n", tests_passed + tests_failed);
    printf("║  Passed: %d                                                ║\n", tests_passed);
    printf("║  Failed: %d                                                ║\n", tests_failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed > 0) ? 1 : 0;
}
