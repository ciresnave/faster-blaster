/**
 * @file test_clblast_execution.c
 * @brief CLBlast execution test - verify actual matrix operations work correctly
 * 
 * Tests CLBlast backend with real computations on both NVIDIA and AMD GPUs,
 * comparing results against reference implementation.
 */

#include "faster-blaster/dispatch_unified.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/backend_instance.h"
#include "backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* NOTE: GPU backends (CLBlast, cuBLAS, etc.) typically use column-major layout.
 * This test uses column-major to ensure compatibility across all backends. */
#define TEST_SIZE_SMALL 64
#define TEST_SIZE_MEDIUM 256
#define EPSILON 1e-4f

typedef struct {
    int passed;
    int failed;
} test_results_t;

/**
 * Compare two matrices for equality within epsilon
 */
static int compare_matrices(const float* a, const float* b, int m, int n, float epsilon) {
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > epsilon) {
            printf("      Mismatch at index %d: expected %.6f, got %.6f (diff: %.6e)\n", 
                   i, a[i], b[i], diff);
            return 0;
        }
    }
    return 1;
}

/**
 * Initialize matrix with test pattern
 */
static void init_matrix(float* mat, int rows, int cols, int pattern) {
    for (int i = 0; i < rows * cols; i++) {
        switch (pattern) {
            case 0: /* Identity-like */
                mat[i] = (i % (cols + 1) == 0) ? 1.0f : 0.0f;
                break;
            case 1: /* Sequential */
                mat[i] = (float)(i + 1) * 0.1f;
                break;
            case 2: /* Random-like pattern */
                mat[i] = ((i * 7919) % 1000) / 1000.0f;
                break;
            default:
                mat[i] = 1.0f;
        }
    }
}

/**
 * Test SGEMM (matrix multiplication) on a specific device
 */
static int test_sgemm(int device_id, const char* device_name, int m, int n, int k, test_results_t* results) {
    printf("\n  [Test] SGEMM (%dx%d) * (%dx%d) on %s\n", m, k, k, n, device_name);
    
    /* Pin to the device */
    int pin_result = fb_use_device(device_id);
    if (pin_result != 0) {
        printf("    ✗ FAILED: Could not pin to device %d\n", device_id);
        results->failed++;
        return 0;
    }
    
    /* Get backend instance */
    fb_backend_instance_t* instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (!instance) {
        printf("    ✗ FAILED: No backend instance available\n");
        fb_use_auto();
        results->failed++;
        return 0;
    }
    
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sgemm) {
        printf("    ⚠ SKIPPED: Backend doesn't implement SGEMM\n");
        fb_use_auto();
        return 1;  /* Not a failure */
    }
    
    /* Allocate matrices */
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C_device = (float*)calloc(m * n, sizeof(float));
    float* C_reference = (float*)calloc(m * n, sizeof(float));
    
    if (!A || !B || !C_device || !C_reference) {
        printf("    ✗ FAILED: Memory allocation failed\n");
        results->failed++;
        free(A); free(B); free(C_device); free(C_reference);
        fb_use_auto();
        return 0;
    }
    
    /* Initialize matrices */
    init_matrix(A, m, k, 1);
    init_matrix(B, k, n, 2);
    
    /* Compute reference result on CPU (device 0) */
    fb_use_device(0);
    fb_backend_instance_t* ref_instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (ref_instance) {
        const fb_backend_vtable_t* ref_vtable = fb_backend_get_vtable(ref_instance);
        if (ref_vtable && ref_vtable->sgemm) {
            ref_vtable->sgemm(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                            (int64_t)m, (int64_t)n, (int64_t)k, 1.0f,
                            A, (int64_t)m, B, (int64_t)k, 0.0f, C_reference, (int64_t)m);
        }
    }
    
    /* Compute result on specified device */
    fb_use_device(device_id);
    vtable->sgemm(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                 (int64_t)m, (int64_t)n, (int64_t)k, 1.0f,
                 A, (int64_t)m, B, (int64_t)k, 0.0f, C_device, (int64_t)m);
    
    fb_use_auto();
    
    /* Compare results */
    if (compare_matrices(C_reference, C_device, m, n, EPSILON)) {
        printf("    ✓ PASSED: Results match reference (epsilon: %.2e)\n", EPSILON);
        results->passed++;
        free(A); free(B); free(C_device); free(C_reference);
        return 1;
    } else {
        printf("    ✗ FAILED: Results differ from reference\n");
        results->failed++;
        free(A); free(B); free(C_device); free(C_reference);
        return 0;
    }
}

/**
 * Test SAXPY (vector operation) on a specific device
 */
static int test_saxpy(int device_id, const char* device_name, int n, test_results_t* results) {
    printf("\n  [Test] SAXPY (n=%d) on %s\n", n, device_name);
    
    /* Pin to the device */
    int pin_result = fb_use_device(device_id);
    if (pin_result != 0) {
        printf("    ✗ FAILED: Could not pin to device %d\n", device_id);
        results->failed++;
        return 0;
    }
    
    /* Get backend instance */
    fb_backend_instance_t* instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (!instance) {
        printf("    ✗ FAILED: No backend instance available\n");
        fb_use_auto();
        results->failed++;
        return 0;
    }
    
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->saxpy) {
        printf("    ⚠ SKIPPED: Backend doesn't implement SAXPY\n");
        fb_use_auto();
        return 1;
    }
    
    /* Allocate vectors */
    float* x = (float*)malloc(n * sizeof(float));
    float* y_device = (float*)malloc(n * sizeof(float));
    float* y_reference = (float*)malloc(n * sizeof(float));
    
    if (!x || !y_device || !y_reference) {
        printf("    ✗ FAILED: Memory allocation failed\n");
        results->failed++;
        free(x); free(y_device); free(y_reference);
        fb_use_auto();
        return 0;
    }
    
    /* Initialize vectors */
    for (int i = 0; i < n; i++) {
        x[i] = (float)(i + 1) * 0.1f;
        y_device[i] = (float)(n - i) * 0.2f;
        y_reference[i] = y_device[i];
    }
    
    float alpha = 2.5f;
    
    /* Compute reference result on CPU */
    fb_use_device(0);
    fb_backend_instance_t* ref_instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (ref_instance) {
        const fb_backend_vtable_t* ref_vtable = fb_backend_get_vtable(ref_instance);
        if (ref_vtable && ref_vtable->saxpy) {
            ref_vtable->saxpy((int64_t)n, alpha, x, 1, y_reference, 1);
        }
    }
    
    /* Compute result on specified device */
    fb_use_device(device_id);
    vtable->saxpy((int64_t)n, alpha, x, 1, y_device, 1);
    fb_use_auto();
    
    /* Compare results */
    int match = 1;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(y_reference[i] - y_device[i]);
        if (diff > EPSILON) {
            printf("      Mismatch at index %d: expected %.6f, got %.6f (diff: %.6e)\n",
                   i, y_reference[i], y_device[i], diff);
            match = 0;
            break;
        }
    }
    
    if (match) {
        printf("    ✓ PASSED: Results match reference\n");
        results->passed++;
    } else {
        printf("    ✗ FAILED: Results differ from reference\n");
        results->failed++;
    }
    
    free(x);
    free(y_device);
    free(y_reference);
    return match;
}

/**
 * Test SGEMV (matrix-vector multiply) on a specific device
 */
static int test_sgemv(int device_id, const char* device_name, int m, int n, test_results_t* results) {
    printf("\n  [Test] SGEMV (%dx%d) on %s\n", m, n, device_name);
    
    /* Pin to the device */
    int pin_result = fb_use_device(device_id);
    if (pin_result != 0) {
        printf("    ✗ FAILED: Could not pin to device %d\n", device_id);
        results->failed++;
        return 0;
    }
    
    /* Get backend instance */
    fb_backend_instance_t* instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (!instance) {
        printf("    ✗ FAILED: No backend instance available\n");
        fb_use_auto();
        results->failed++;
        return 0;
    }
    
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    if (!vtable || !vtable->sgemv) {
        printf("    ⚠ SKIPPED: Backend doesn't implement SGEMV\n");
        fb_use_auto();
        return 1;
    }
    
    /* Allocate matrices and vectors */
    float* A = (float*)malloc(m * n * sizeof(float));
    float* x = (float*)malloc(n * sizeof(float));
    float* y_device = (float*)calloc(m, sizeof(float));
    float* y_reference = (float*)calloc(m, sizeof(float));
    
    if (!A || !x || !y_device || !y_reference) {
        printf("    ✗ FAILED: Memory allocation failed\n");
        results->failed++;
        free(A); free(x); free(y_device); free(y_reference);
        fb_use_auto();
        return 0;
    }
    
    /* Initialize */
    init_matrix(A, m, n, 1);
    for (int i = 0; i < n; i++) {
        x[i] = (float)(i + 1) * 0.5f;
    }
    
    /* Compute reference */
    fb_use_device(0);
    fb_backend_instance_t* ref_instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    if (ref_instance) {
        const fb_backend_vtable_t* ref_vtable = fb_backend_get_vtable(ref_instance);
        if (ref_vtable && ref_vtable->sgemv) {
            ref_vtable->sgemv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, (int64_t)m, (int64_t)n,
                            1.0f, A, (int64_t)m, x, 1, 0.0f, y_reference, 1);
        }
    }
    
    /* Compute on device */
    fb_use_device(device_id);
    vtable->sgemv(FB_LAYOUT_COL_MAJOR, FB_NO_TRANS, (int64_t)m, (int64_t)n,
                 1.0f, A, (int64_t)m, x, 1, 0.0f, y_device, 1);
    fb_use_auto();
    
    /* Compare */
    int match = 1;
    for (int i = 0; i < m; i++) {
        float diff = fabsf(y_reference[i] - y_device[i]);
        if (diff > EPSILON) {
            printf("      Mismatch at index %d: expected %.6f, got %.6f (diff: %.6e)\n",
                   i, y_reference[i], y_device[i], diff);
            match = 0;
            break;
        }
    }
    
    if (match) {
        printf("    ✓ PASSED: Results match reference\n");
        results->passed++;
    } else {
        printf("    ✗ FAILED: Results differ from reference\n");
        results->failed++;
    }
    
    free(A);
    free(x);
    free(y_device);
    free(y_reference);
    return match;
}

int main(void) {
    printf("#############################################################\n");
    printf("# CLBlast Execution Test Suite\n");
    printf("#############################################################\n\n");
    
    test_results_t results = {0, 0};
    
    /* Initialize faster-blaster */
    printf("Initializing faster-blaster...\n");
    int init_result = fb_init();
    if (init_result != 0) {
        printf("✗ FAILED: fb_init() returned %d\n", init_result);
        return 1;
    }
    printf("✓ System initialized\n\n");
    
    /* Get device count */
    int num_devices = fb_get_device_count();
    printf("Detected %d device(s)\n\n", num_devices);
    
    if (num_devices < 2) {
        printf("✗ ERROR: Need at least 2 devices (CPU + GPU) for testing\n");
        fb_shutdown();
        return 1;
    }
    
    /* Test each GPU device with CLBlast backend */
    for (int dev = 1; dev < num_devices; dev++) {
        fb_compute_device_t* device = fb_get_device(dev);
        if (!device) continue;
        
        if (device->properties.type == FB_DEVICE_TYPE_GPU) {
            printf("=============================================================\n");
            printf("Testing Device %d: %s\n", dev, device->properties.name);
            printf("=============================================================\n");
            
            /* Test SAXPY (Level 1 BLAS) */
            test_saxpy(dev, device->properties.name, 1024, &results);
            test_saxpy(dev, device->properties.name, 10000, &results);
            
            /* Test SGEMV (Level 2 BLAS) */
            test_sgemv(dev, device->properties.name, TEST_SIZE_SMALL, TEST_SIZE_SMALL, &results);
            test_sgemv(dev, device->properties.name, TEST_SIZE_MEDIUM, TEST_SIZE_MEDIUM, &results);
            
            /* Test SGEMM (Level 3 BLAS) */
            test_sgemm(dev, device->properties.name, TEST_SIZE_SMALL, TEST_SIZE_SMALL, TEST_SIZE_SMALL, &results);
            test_sgemm(dev, device->properties.name, TEST_SIZE_MEDIUM, TEST_SIZE_MEDIUM, TEST_SIZE_MEDIUM, &results);
            
            /* Test non-square matrices */
            test_sgemm(dev, device->properties.name, 128, 256, 64, &results);
            test_sgemm(dev, device->properties.name, 64, 128, 256, &results);
        }
    }
    
    /* Shutdown */
    printf("\n=============================================================\n");
    printf("Shutting down...\n");
    fb_shutdown();
    
    /* Print summary */
    printf("\n#############################################################\n");
    printf("# Test Summary\n");
    printf("#############################################################\n");
    printf("  Tests passed: %d\n", results.passed);
    printf("  Tests failed: %d\n", results.failed);
    printf("\n");
    
    if (results.failed == 0) {
        printf("  ✓ ALL TESTS PASSED!\n\n");
        return 0;
    } else {
        printf("  ✗ SOME TESTS FAILED\n\n");
        return 1;
    }
}
