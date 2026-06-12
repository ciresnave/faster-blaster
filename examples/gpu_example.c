/**
 * @file gpu_example.c
 * @brief Example demonstrating GPU backend usage
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_loader.h"
#include "gpu/gpu_backend.h"
#include "gpu/cublas_backend.h"
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 1000000
#define MATRIX_SIZE 1024

static void example_gpu_vector_operations(void) {
    printf("\n=== GPU Vector Operations Example ===\n");
    
    /* Check if CUDA is available */
    if (!fb_cublas_is_available()) {
        printf("CUDA not available on this system\n");
        return;
    }
    
    int device_count = fb_cublas_device_count();
    printf("Found %d CUDA device(s)\n", device_count);
    
    if (device_count == 0) {
        return;
    }
    
    /* Initialize GPU backend */
    fb_gpu_context_t ctx;
    if (fb_cublas_init(0, &ctx) != 0) {
        printf("Failed to initialize cuBLAS\n");
        return;
    }
    
    printf("GPU initialized successfully (device %d)\n", ctx.device_id);
    
    /* Allocate host memory */
    float* h_x = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_y = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_result = (float*)malloc(VECTOR_SIZE * sizeof(float));
    
    /* Initialize vectors */
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(VECTOR_SIZE - i);
    }
    
    /* Get GPU backend operations */
    const fb_gpu_backend_t* gpu_backend = fb_cublas_get_backend();
    if (!gpu_backend) {
        printf("Failed to get GPU backend\n");
        goto cleanup;
    }
    
    /* Allocate device memory */
    fb_gpu_ptr_t d_x, d_y, d_result;
    if (gpu_backend->malloc(&d_x, VECTOR_SIZE * sizeof(float)) != 0 ||
        gpu_backend->malloc(&d_y, VECTOR_SIZE * sizeof(float)) != 0 ||
        gpu_backend->malloc(&d_result, VECTOR_SIZE * sizeof(float)) != 0) {
        printf("Failed to allocate device memory\n");
        goto cleanup;
    }
    
    printf("Allocated %.2f MB on GPU\n", 
           (3 * VECTOR_SIZE * sizeof(float)) / (1024.0 * 1024.0));
    
    /* Copy data to GPU */
    gpu_backend->memcpy_h2d(d_x, h_x, VECTOR_SIZE * sizeof(float), NULL);
    gpu_backend->memcpy_h2d(d_y, h_y, VECTOR_SIZE * sizeof(float), NULL);
    
    printf("Data transferred to GPU\n");
    
    /* Set GPU backend as current */
    fb_backend_set_current(FB_BACKEND_CUBLAS);
    
    /* Perform GPU operations (if cuBLAS BLAS vtable is implemented) */
    printf("\nNote: GPU BLAS operations would execute here\n");
    printf("(cuBLAS vtable wrappers need to be implemented)\n");
    
    /* Copy result back */
    gpu_backend->memcpy_d2h(h_result, d_result, VECTOR_SIZE * sizeof(float), NULL);
    
    /* Synchronize */
    gpu_backend->device_synchronize();
    
    printf("Operations completed and synchronized\n");
    
    /* Free device memory */
    gpu_backend->free(d_x);
    gpu_backend->free(d_y);
    gpu_backend->free(d_result);
    
cleanup:
    free(h_x);
    free(h_y);
    free(h_result);
    
    /* Shutdown GPU */
    fb_cublas_shutdown(&ctx);
    printf("GPU backend shut down\n");
}

static void example_gpu_matrix_operations(void) {
    printf("\n=== GPU Matrix Operations Example ===\n");
    
    if (!fb_cublas_is_available()) {
        printf("CUDA not available\n");
        return;
    }
    
    /* Initialize */
    fb_gpu_context_t ctx;
    if (fb_cublas_init(0, &ctx) != 0) {
        printf("Failed to initialize cuBLAS\n");
        return;
    }
    
    const int N = MATRIX_SIZE;
    const size_t matrix_bytes = N * N * sizeof(float);
    
    printf("Matrix size: %dx%d (%.2f MB per matrix)\n", 
           N, N, matrix_bytes / (1024.0 * 1024.0));
    
    /* Allocate host matrices */
    float* h_A = (float*)malloc(matrix_bytes);
    float* h_B = (float*)malloc(matrix_bytes);
    float* h_C = (float*)malloc(matrix_bytes);
    
    /* Initialize matrices */
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_C[i] = 0.0f;
    }
    
    const fb_gpu_backend_t* gpu_backend = fb_cublas_get_backend();
    if (!gpu_backend) {
        printf("Failed to get GPU backend\n");
        goto cleanup_matrix;
    }
    
    /* Allocate device matrices */
    fb_gpu_ptr_t d_A, d_B, d_C;
    gpu_backend->malloc(&d_A, matrix_bytes);
    gpu_backend->malloc(&d_B, matrix_bytes);
    gpu_backend->malloc(&d_C, matrix_bytes);
    
    printf("Allocated %.2f MB on GPU\n", (3 * matrix_bytes) / (1024.0 * 1024.0));
    
    /* Transfer to GPU */
    gpu_backend->memcpy_h2d(d_A, h_A, matrix_bytes, NULL);
    gpu_backend->memcpy_h2d(d_B, h_B, matrix_bytes, NULL);
    
    printf("Matrices transferred to GPU\n");
    printf("\nNote: GPU matrix multiplication (SGEMM) would execute here\n");
    printf("Expected FLOPS: %.2f GFLOPS (theoretical)\n", 
           (2.0 * N * N * N) / 1e9);
    
    /* Copy result back */
    gpu_backend->memcpy_d2h(h_C, d_C, matrix_bytes, NULL);
    gpu_backend->device_synchronize();
    
    /* Free device memory */
    gpu_backend->free(d_A);
    gpu_backend->free(d_B);
    gpu_backend->free(d_C);
    
cleanup_matrix:
    free(h_A);
    free(h_B);
    free(h_C);
    
    fb_cublas_shutdown(&ctx);
}

int main(void) {
    printf("Faster-Blaster GPU Backend Example\n");
    printf("===================================\n");
    
    /* Initialize backend loader */
    fb_backend_loader_init();
    
    /* Check GPU availability */
    printf("\n=== GPU Backend Detection ===\n");
    
    if (fb_cublas_is_available()) {
        printf("✓ NVIDIA CUDA: Available\n");
        printf("  Devices: %d\n", fb_cublas_device_count());
    } else {
        printf("✗ NVIDIA CUDA: Not available\n");
    }
    
    if (fb_rocblas_is_available()) {
        printf("✓ AMD ROCm: Available\n");
        printf("  Devices: %d\n", fb_rocblas_device_count());
    } else {
        printf("✗ AMD ROCm: Not available\n");
    }
    
    /* Run examples */
    example_gpu_vector_operations();
    example_gpu_matrix_operations();
    
    /* Cleanup */
    fb_backend_loader_shutdown();
    
    printf("\n===================================\n");
    printf("Example completed\n");
    
    return 0;
}
