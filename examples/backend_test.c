/**
 * @file backend_test.c
 * @brief Test program for backend loading and switching
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TEST_SIZE 1000
#define BENCH_SIZE 10000
#define BENCH_ITERATIONS 1000

static void print_separator(void) {
    printf("========================================\n");
}

static void test_basic_operations(void) {
    printf("\n=== Testing Basic BLAS Operations ===\n");
    
    /* Allocate test vectors */
    float* x = (float*)malloc(TEST_SIZE * sizeof(float));
    float* y = (float*)malloc(TEST_SIZE * sizeof(float));
    
    /* Initialize vectors */
    for (int i = 0; i < TEST_SIZE; i++) {
        x[i] = (float)i;
        y[i] = (float)(TEST_SIZE - i);
    }
    
    /* Test SDOT */
    float dot_result = 0.0f;
    fb_backend_vtable_t vtable;
    if (fb_backend_load(fb_backend_get_current(), &vtable) == 0) {
        if (vtable.sdot) {
            vtable.sdot(TEST_SIZE, x, 1, y, 1, &dot_result);
            printf("SDOT result: %.2f\n", dot_result);
        }
        
        /* Test SAXPY */
        if (vtable.saxpy) {
            float alpha = 2.0f;
            vtable.saxpy(TEST_SIZE, alpha, x, 1, y, 1);
            printf("SAXPY completed (alpha=%.1f)\n", alpha);
        }
        
        /* Test SNRM2 */
        if (vtable.snrm2) {
            float norm = 0.0f;
            vtable.snrm2(TEST_SIZE, x, 1, &norm);
            printf("SNRM2 result: %.2f\n", norm);
        }
    }
    
    free(x);
    free(y);
}

static void benchmark_dgemm(const char* backend_name, int size, int iterations) {
    /* Allocate matrices */
    double* A = (double*)malloc(size * size * sizeof(double));
    double* B = (double*)malloc(size * size * sizeof(double));
    double* C = (double*)malloc(size * size * sizeof(double));
    
    /* Initialize matrices */
    for (int i = 0; i < size * size; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }
    
    /* Benchmark */
    fb_backend_vtable_t vtable;
    if (fb_backend_load(fb_backend_get_current(), &vtable) == 0 && vtable.dgemm) {
        clock_t start = clock();
        
        for (int iter = 0; iter < iterations; iter++) {
            vtable.dgemm('N', 'N', size, size, size,
                        1.0, A, size, B, size, 0.0, C, size);
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        
        /* Calculate GFLOPS: 2*m*n*k ops per iteration */
        double flops = 2.0 * size * size * size * iterations;
        double gflops = (flops / elapsed) / 1e9;
        
        printf("%-20s: %8.2f GFLOPS (%.3f sec)\n", backend_name, gflops, elapsed);
    } else {
        printf("%-20s: DGEMM not available\n", backend_name);
    }
    
    free(A);
    free(B);
    free(C);
}

static void run_benchmarks(void) {
    printf("\n=== DGEMM Benchmark (size=%d, iterations=%d) ===\n", 
           BENCH_SIZE / 10, 10);
    
    fb_backend_info_t backends[FB_BACKEND_COUNT];
    int count = fb_backend_list_available(backends, FB_BACKEND_COUNT);
    
    for (int i = 0; i < count; i++) {
        if (backends[i].capabilities & FB_CAP_LEVEL3) {
            /* Set backend */
            if (fb_backend_set_current(backends[i].type) == 0) {
                benchmark_dgemm(backends[i].name, BENCH_SIZE / 10, 10);
            }
        }
    }
}

int main(int argc, char** argv) {
    printf("Faster-Blaster Backend Test\n");
    print_separator();
    
    /* Initialize backend loader */
    printf("\nInitializing backend loader...\n");
    if (fb_backend_loader_init() != 0) {
        fprintf(stderr, "Failed to initialize backend loader\n");
        return 1;
    }
    
    /* List available backends */
    printf("\n=== Available Backends ===\n");
    fb_backend_info_t backends[FB_BACKEND_COUNT];
    int count = fb_backend_list_available(backends, FB_BACKEND_COUNT);
    
    printf("Found %d backend(s):\n\n", count);
    for (int i = 0; i < count; i++) {
        printf("  [%d] %s\n", i, backends[i].name);
        printf("      Version:      %s\n", backends[i].version ? backends[i].version : "Unknown");
        printf("      License:      %s\n", backends[i].license ? backends[i].license : "Unknown");
        printf("      Vendor:       %s\n", backends[i].vendor ? backends[i].vendor : "Unknown");
        printf("      Priority:     %d\n", backends[i].priority);
        printf("      Capabilities: ");
        
        if (backends[i].capabilities & FB_CAP_CPU) printf("CPU ");
        if (backends[i].capabilities & FB_CAP_GPU) printf("GPU ");
        if (backends[i].capabilities & FB_CAP_LEVEL1) printf("Level1 ");
        if (backends[i].capabilities & FB_CAP_LEVEL2) printf("Level2 ");
        if (backends[i].capabilities & FB_CAP_LEVEL3) printf("Level3 ");
        if (backends[i].capabilities & FB_CAP_LAPACK) printf("LAPACK ");
        if (backends[i].capabilities & FB_CAP_THREADSAFE) printf("ThreadSafe ");
        printf("\n\n");
    }
    
    /* Auto-select best backend */
    printf("=== Auto-selecting Backend ===\n");
    fb_backend_type_t best = fb_backend_auto_select(false);
    fb_backend_info_t best_info;
    if (fb_backend_get_info(best, &best_info) == 0) {
        printf("Selected: %s (priority %d)\n", best_info.name, best_info.priority);
        fb_backend_set_current(best);
    }
    
    /* Test basic operations */
    test_basic_operations();
    
    /* Run benchmarks if requested */
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        run_benchmarks();
    } else {
        printf("\nTip: Run with --benchmark flag to compare backend performance\n");
    }
    
    /* Cleanup */
    printf("\n");
    print_separator();
    printf("Shutting down...\n");
    fb_backend_loader_shutdown();
    
    printf("Test completed successfully!\n");
    return 0;
}
