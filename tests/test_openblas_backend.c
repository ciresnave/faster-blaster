/**
 * @file test_openblas_backend.c
 * @brief Test suite for OpenBLAS CPU backend
 * 
 * Tests the OpenBLAS backend implementation.
 * Portable BLAS library for all CPU architectures.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "../src/backends/openblas_backend.h"
#include "../src/backends/backend_auto_detect.h"

/*
 * Test-only stubs for optional auto-population paths used by openblas_backend.
 * The backend unit tests exercise core BLAS calls directly and do not require
 * LAPACKE table auto-fill.
 */
const fb_sym_entry_t k_lapacke_symbols[] = {0};
const size_t k_lapacke_symbols_count = 0;

void fb_auto_populate_ext_ops(fb_backend_vtable_t *vtable,
                              void *lib_handle,
                              const fb_sym_entry_t *syms,
                              size_t count) {
    (void)vtable;
    (void)lib_handle;
    (void)syms;
    (void)count;
}

uint32_t fb_stem_to_op_id(const char *stem) {
    (void)stem;
    return UINT32_MAX;
}

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

static void test_openblas_availability(void) {
    printf("\n📋 Test 1: OpenBLAS Availability\n");
    printf("═══════════════════════════════════════\n");
    
    bool available = fb_openblas_is_available();
    printf("  OpenBLAS library available: %s\n", available ? "Yes" : "No");
    
    if (!available) {
        printf("  ⚠️  OpenBLAS not found. Install from:\n");
        printf("      https://www.openblas.net/\n");
        report_test("OpenBLAS availability check", false);
        return;
    }
    
    const char* version = fb_openblas_get_version();
    if (version) {
        printf("  Version: %s\n", version);
    }
    
    report_test("OpenBLAS availability check", available);
}

static void test_openblas_saxpy(void) {
    printf("\n📋 Test 2: SAXPY (y = alpha*x + y)\n");
    printf("═══════════════════════════════════════\n");
    
    if (!fb_openblas_is_available()) {
        printf("  ⚠️  OpenBLAS not available, skipping\n");
        report_test("SAXPY operation", false);
        return;
    }
    
    const int n = 5;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float expected[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    
    const fb_backend_vtable_t* vtable = fb_openblas_get_vtable();
    if (!vtable || !vtable->saxpy) {
        printf("  ⚠️  SAXPY not available in vtable\n");
        report_test("SAXPY operation", false);
        return;
    }
    vtable->saxpy(n, alpha, x, 1, y, 1);
    
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
    printf("║       OpenBLAS CPU Backend Test Suite                    ║\n");
    printf("║       Portable optimized BLAS library                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_openblas_availability();
    test_openblas_saxpy();
    
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Test Summary                            ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total Tests: %d                                           ║\n", tests_passed + tests_failed);
    printf("║  Passed: %d                                                ║\n", tests_passed);
    printf("║  Failed: %d                                                ║\n", tests_failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed > 0) ? 1 : 0;
}
