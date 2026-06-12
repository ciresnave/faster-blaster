/**
 * @file test_hipblas_backend.cpp
 * @brief Test suite for hipBLAS GPU backend (portable AMD/NVIDIA)
 * 
 * Tests the hipBLAS backend implementation.
 * Works on both AMD and NVIDIA GPUs.
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

extern "C" {
    const fb_gpu_backend_trait* fb_hipblas_get_trait(void);
}

#define TOLERANCE_SINGLE 1e-4f
#define TEST_PASSED "✅ PASSED"
#define TEST_FAILED "❌ FAILED"

static int tests_passed = 0;
static int tests_failed = 0;
static void* backend_handle = nullptr;

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

static void test_init(void) {
    printf("\n📋 Test 1: hipBLAS Backend Initialization\n");
    printf("═══════════════════════════════════════\n");
    
    const fb_gpu_backend_trait* trait = fb_hipblas_get_trait();
    
    if (!trait) {
        printf("  ❌ Failed to get hipBLAS trait\n");
        report_test("Get hipBLAS trait", false);
        return;
    }
    
    printf("  hipBLAS trait loaded successfully\n");
    
    int ret = trait->init(0, nullptr, &backend_handle);
    
    if (ret != 0) {
        printf("  ⚠️  hipBLAS initialization failed (code %d)\n", ret);
        printf("  This is expected if no AMD/NVIDIA GPU is present\n");
        report_test("Backend initialization", false);
        return;
    }
    
    printf("  Backend initialized successfully\n");
    report_test("Backend initialization", ret == 0);
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       hipBLAS GPU Backend Test Suite                     ║\n");
    printf("║       Portable across AMD and NVIDIA GPUs                ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_init();
    
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Test Summary                            ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total Tests: %d                                           ║\n", tests_passed + tests_failed);
    printf("║  Passed: %d                                                ║\n", tests_passed);
    printf("║  Failed: %d                                                ║\n", tests_failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed > 0) ? 1 : 0;
}
