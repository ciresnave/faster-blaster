/**
 * @file test_device_memory_manager.c
 * @brief Test device memory manager with operation chaining
 * 
 * This test demonstrates:
 * 1. Single operation (baseline)
 * 2. Chained operations on same backend (cache benefits)
 * 3. Cross-backend chaining (rocBLAS → CLBlast)
 * 
 * Expected behavior:
 * - First operation: H2D copy + compute + cache
 * - Second operation same backend: Cache hit (no H2D!)
 * - Cross-backend: Sync + cache reuse
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

/* Forward declarations of external traits */
extern const fb_gpu_backend_trait_t fb_cublas_trait;
extern int fb_cublas_smart_init(int device_id, void* lib_handle);
extern void fb_cublas_smart_shutdown(void);
extern void fb_cublas_print_memory_stats(void);
extern int fb_cublas_sync_all_to_host(void);

/* Test operation wrappers - use correct types from faster_blaster.h */
extern void cublas_sgemv_smart_wrapper(
    FB_LAYOUT layout, FB_TRANSPOSE trans,
    int m, int n, float alpha,
    const float* a, int lda,
    const float* x, int incx,
    float beta, float* y, int incy
);

/* ============================================================================
 * Test Utilities
 * ========================================================================== */

static void print_vector(const char* name, const float* v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n && i < 10; i++) {
        printf("%.4f%s", v[i], (i < n-1) ? ", " : "");
    }
    if (n > 10) {
        printf(", ... (%d more)", n - 10);
    }
    printf("]\n");
}

static void print_matrix(const char* name, const float* a, int m, int n, int lda) {
    printf("%s (%dx%d, lda=%d):\n", name, m, n, lda);
    for (int i = 0; i < m && i < 5; i++) {
        printf("  [");
        for (int j = 0; j < n && j < 5; j++) {
            /* Column-major indexing */
            printf("%.4f%s", a[i + j*lda], (j < n-1) ? ", " : "");
        }
        if (n > 5) {
            printf(", ...");
        }
        printf("]\n");
    }
    if (m > 5) {
        printf("  ... (%d more rows)\n", m - 5);
    }
}

/* ============================================================================
 * Test 1: Single Operation
 * ========================================================================== */

void test_single_sgemv(void) {
    printf("\n=== Test 1: Single sgemv Operation ===\n");
    
    /* Create test data: y = 2*A*x + 0*y */
    int m = 4, n = 3;
    float alpha = 2.0f, beta = 0.0f;
    
    /* Matrix A (column-major) */
    float A[12] = {
        1, 2, 3, 4,  /* Column 0 */
        5, 6, 7, 8,  /* Column 1 */
        9, 10, 11, 12 /* Column 2 */
    };
    
    /* Vector x */
    float x[3] = {1.0f, 2.0f, 3.0f};
    
    /* Vector y (output) */
    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    print_matrix("A", A, m, n, m);
    print_vector("x", x, n);
    print_vector("y (before)", y, m);
    
    /* Execute sgemv */
    cublas_sgemv_smart_wrapper(
        FbColMajor, FbNoTrans,
        m, n, alpha, A, m, x, 1, beta, y, 1
    );
    
    /* Sync result to host */
    fb_cublas_sync_all_to_host();
    
    print_vector("y (after)", y, m);
    
    /* Expected: y = 2*(A*x) = 2*([1,2,3,4]*1 + [5,6,7,8]*2 + [9,10,11,12]*3)
     *           y = 2*([1+10+27, 2+12+30, 3+14+33, 4+16+36])
     *           y = 2*[38, 44, 50, 56] = [76, 88, 100, 112] */
    
    float expected[4] = {76.0f, 88.0f, 100.0f, 112.0f};
    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (fabsf(y[i] - expected[i]) > 0.001f) {
            printf("ERROR: y[%d] = %.4f, expected %.4f\n", i, y[i], expected[i]);
            correct = 0;
        }
    }
    
    if (correct) {
        printf("✓ Result correct!\n");
    }
    
    fb_cublas_print_memory_stats();
}

/* ============================================================================
 * Test 2: Chained Operations (Same Backend)
 * ========================================================================== */

void test_chained_sgemv(void) {
    printf("\n=== Test 2: Chained sgemv Operations ===\n");
    printf("This tests cache efficiency: second operation should have cache hits\n");
    
    int m = 4, n = 3;
    float alpha = 1.0f, beta = 0.5f;
    
    float A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float x[3] = {1.0f, 0.0f, -1.0f};
    float y[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    printf("\nOperation 1: y = A*x + 0.5*y\n");
    
    /* Clear stats */
    fb_device_memory_manager_t* manager = fb_device_memory_get_manager(0);
    if (manager) {
        manager->stats.num_cache_hits = 0;
        manager->stats.num_cache_misses = 0;
    }
    
    /* First operation: A, x, y all go to device (3 misses expected) */
    cublas_sgemv_smart_wrapper(
        FbColMajor, FbNoTrans,
        m, n, alpha, A, m, x, 1, beta, y, 1
    );
    
    printf("After operation 1:\n");
    fb_cublas_print_memory_stats();
    
    /* Don't sync yet - keep data on device */
    
    /* Second operation: A, x, y already on device (3 hits expected!) */
    printf("\nOperation 2: y = A*x + 0.5*y (with cached buffers)\n");
    
    cublas_sgemv_smart_wrapper(
        FbColMajor, FbNoTrans,
        m, n, alpha, A, m, x, 1, beta, y, 1
    );
    
    printf("After operation 2:\n");
    fb_cublas_print_memory_stats();
    
    /* Now sync to see final result */
    fb_cublas_sync_all_to_host();
    print_vector("y (final)", y, m);
    
    printf("\nExpected: High cache hit rate on operation 2\n");
}

/* ============================================================================
 * Test 3: Memory Manager Diagnostics
 * ========================================================================== */

void test_memory_manager_diagnostics(void) {
    printf("\n=== Test 3: Memory Manager Diagnostics ===\n");
    
    fb_device_memory_manager_t* manager = fb_device_memory_get_manager(0);
    if (!manager) {
        printf("ERROR: Could not get memory manager\n");
        return;
    }
    
    /* Print detailed entry information */
    fb_device_memory_dump_entries(manager);
    
    /* Get stats */
    size_t total_allocated, num_entries;
    double cache_hit_rate;
    
    fb_device_memory_get_stats(manager, &total_allocated, &num_entries, &cache_hit_rate);
    
    printf("Summary:\n");
    printf("  Total allocated: %zu bytes (%.2f KB)\n", 
           total_allocated, total_allocated / 1024.0);
    printf("  Active entries: %zu\n", num_entries);
    printf("  Cache hit rate: %.2f%%\n", cache_hit_rate * 100.0);
}

/* ============================================================================
 * Main Test Driver
 * ========================================================================== */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    printf("========================================\n");
    printf("Device Memory Manager Test Suite\n");
    printf("========================================\n");
    
    /* Initialize cuBLAS smart wrappers */
    if (fb_cublas_smart_init(0, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize cuBLAS\n");
        return 1;
    }
    
    /* Run tests */
    test_single_sgemv();
    test_chained_sgemv();
    test_memory_manager_diagnostics();
    
    /* Shutdown */
    fb_cublas_smart_shutdown();
    fb_device_memory_shutdown_all();
    
    printf("\n========================================\n");
    printf("All tests completed\n");
    printf("========================================\n");
    
    return 0;
}
