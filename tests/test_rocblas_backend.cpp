/**
 * @file test_rocblas_backend.cpp
 * @brief Test suite for AMD rocBLAS GPU backend
 * 
 * Tests the rocBLAS backend implementation for AMD GPUs.
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
    const fb_gpu_backend_trait* fb_rocblas_get_trait(void);
}

#define TOLERANCE_SINGLE 1e-4f
#define TOLERANCE_DOUBLE 1e-10
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
    printf("\n📋 Test 1: rocBLAS Backend Initialization\n");
    printf("═══════════════════════════════════════\n");
    
    const fb_gpu_backend_trait* trait = fb_rocblas_get_trait();
    
    if (!trait) {
        printf("  ❌ Failed to get rocBLAS trait\n");
        report_test("Get rocBLAS trait", false);
        return;
    }
    
    printf("  rocBLAS trait loaded successfully\n");
    
    int ret = trait->init(0, &backend_handle);
    
    if (ret != 0) {
        printf("  ⚠️  rocBLAS initialization failed (code %d)\n", ret);
        printf("  This is expected if no AMD GPU is present\n");
        report_test("Backend initialization", false);
        return;
    }
    
    printf("  Backend initialized successfully\n");
    report_test("Backend initialization", ret == 0);
}

static void test_device_properties(void) {
    printf("\n📋 Test 2: Device Properties\n");
    printf("═══════════════════════════════════════\n");
    
    if (!backend_handle) {
        printf("  ⚠️  Backend not initialized, skipping\n");
        report_test("Device properties", false);
        return;
    }
    
    const fb_gpu_backend_trait* trait = fb_rocblas_get_trait();
    char name[256];
    size_t total_mem;
    
    int ret = trait->get_device_properties(backend_handle, 0, name, sizeof(name), &total_mem);
    
    if (ret == 0) {
        printf("  Device: %s\n", name);
        printf("  Memory: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    }
    
    report_test("Device properties", ret == 0);
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       AMD rocBLAS GPU Backend Test Suite                 ║\n");
    printf("║       Testing on: AMD GPU (via ROCm backend)             ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_init();
    test_device_properties();
    
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Test Summary                            ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total Tests: %d                                           ║\n", tests_passed + tests_failed);
    printf("║  Passed: %d                                                ║\n", tests_passed);
    printf("║  Failed: %d                                                ║\n", tests_failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed > 0) ? 1 : 0;
}
