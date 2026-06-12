/**
 * @file multi_gpu_example.c
 * @brief Example: Using AMD and NVIDIA GPUs simultaneously
 * 
 * This example demonstrates transparent multi-GPU execution across vendors.
 * Your PC has both an AMD GPU and an NVIDIA GPU - this shows how to route
 * tasks to each GPU interchangeably using the unified GPU backend trait.
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VECTOR_SIZE 10000
#define MATRIX_SIZE 1024

/* ============================================================================
 * Example 1: Distribute vector operations across both GPUs
 * ========================================================================== */

void example_vector_operations(fb_gpu_manager_t* mgr) {
    printf("\n=== Example 1: Vector Operations on Multi-GPU System ===\n");
    
    int num_gpus = fb_gpu_get_device_count(mgr);
    printf("Found %d GPU(s)\n", num_gpus);
    
    if (num_gpus < 1) {
        printf("Need at least 1 GPU for this example\n");
        return;
    }
    
    // Prepare host data
    float* h_x = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_y = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_result = (float*)malloc(VECTOR_SIZE * sizeof(float));
    
    // Initialize vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(i * 2);
    }
    
    // Execute SAXPY on each GPU
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, gpu_id);
        const char* backend_name = 
            (ctx->backend_type == FB_GPU_BACKEND_CUBLAS) ? "NVIDIA cuBLAS" :
            (ctx->backend_type == FB_GPU_BACKEND_ROCBLAS) ? "AMD rocBLAS" : "Unknown";
        
        printf("\nUsing GPU %d: %s\n", gpu_id, backend_name);
        
        // Allocate device memory
        fb_gpu_ptr_t d_x = fb_gpu_malloc_on_device(mgr, gpu_id, VECTOR_SIZE * sizeof(float));
        fb_gpu_ptr_t d_y = fb_gpu_malloc_on_device(mgr, gpu_id, VECTOR_SIZE * sizeof(float));
        
        if (!d_x || !d_y) {
            printf("  Failed to allocate device memory\n");
            continue;
        }
        
        // Copy to device
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_x, h_x, VECTOR_SIZE * sizeof(float));
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_y, h_y, VECTOR_SIZE * sizeof(float));
        
        // Execute SAXPY: y = alpha*x + y
        float alpha = 3.0f;
        fb_gpu_saxpy_on_device(mgr, gpu_id, VECTOR_SIZE, alpha, d_x, 1, d_y, 1);
        
        // Copy result back
        fb_gpu_memcpy_d2h_on_device(mgr, gpu_id, h_result, d_y, VECTOR_SIZE * sizeof(float));
        
        // Verify result (first 5 elements)
        printf("  SAXPY result (first 5 elements):\n");
        for (int i = 0; i < 5; i++) {
            float expected = alpha * h_x[i] + (float)(i * 2);
            printf("    [%d] = %.2f (expected %.2f) %s\n",
                   i, h_result[i], expected,
                   (fabs(h_result[i] - expected) < 1e-4f) ? "✓" : "✗");
        }
        
        // Cleanup
        fb_gpu_free_on_device(mgr, gpu_id, d_x);
        fb_gpu_free_on_device(mgr, gpu_id, d_y);
    }
    
    free(h_x);
    free(h_y);
    free(h_result);
}

/* ============================================================================
 * Example 2: Route specific tasks to specific GPUs
 * ========================================================================== */

void example_gpu_routing(fb_gpu_manager_t* mgr) {
    printf("\n=== Example 2: Intelligent GPU Task Routing ===\n");
    
    int num_gpus = fb_gpu_get_device_count(mgr);
    if (num_gpus < 2) {
        printf("Need at least 2 GPUs for this example (you have %d)\n", num_gpus);
        return;
    }
    
    printf("Routing different tasks to different GPUs:\n");
    
    // Task 1: Small vector operation on GPU 0 (e.g., NVIDIA)
    {
        int gpu_id = 0;
        int n = 1000;
        
        fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, gpu_id);
        printf("\nTask 1 (small vector) -> GPU %d: %s\n", gpu_id,
               (ctx->backend_type == FB_GPU_BACKEND_CUBLAS) ? "NVIDIA" : "AMD");
        
        float* h_x = (float*)malloc(n * sizeof(float));
        float* h_y = (float*)malloc(n * sizeof(float));
        
        for (int i = 0; i < n; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }
        
        fb_gpu_ptr_t d_x = fb_gpu_malloc_on_device(mgr, gpu_id, n * sizeof(float));
        fb_gpu_ptr_t d_y = fb_gpu_malloc_on_device(mgr, gpu_id, n * sizeof(float));
        
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_x, h_x, n * sizeof(float));
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_y, h_y, n * sizeof(float));
        
        fb_gpu_saxpy_on_device(mgr, gpu_id, n, 5.0f, d_x, 1, d_y, 1);
        
        printf("  Executed SAXPY (n=%d)\n", n);
        
        fb_gpu_free_on_device(mgr, gpu_id, d_x);
        fb_gpu_free_on_device(mgr, gpu_id, d_y);
        free(h_x);
        free(h_y);
    }
    
    // Task 2: Large vector operation on GPU 1 (e.g., AMD)
    {
        int gpu_id = 1;
        int n = 100000;
        
        fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, gpu_id);
        printf("\nTask 2 (large vector) -> GPU %d: %s\n", gpu_id,
               (ctx->backend_type == FB_GPU_BACKEND_ROCBLAS) ? "AMD" : "NVIDIA");
        
        float* h_x = (float*)malloc(n * sizeof(float));
        float* h_y = (float*)malloc(n * sizeof(float));
        
        for (int i = 0; i < n; i++) {
            h_x[i] = 0.5f;
            h_y[i] = 1.5f;
        }
        
        fb_gpu_ptr_t d_x = fb_gpu_malloc_on_device(mgr, gpu_id, n * sizeof(float));
        fb_gpu_ptr_t d_y = fb_gpu_malloc_on_device(mgr, gpu_id, n * sizeof(float));
        
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_x, h_x, n * sizeof(float));
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_y, h_y, n * sizeof(float));
        
        fb_gpu_saxpy_on_device(mgr, gpu_id, n, 2.0f, d_x, 1, d_y, 1);
        
        printf("  Executed SAXPY (n=%d)\n", n);
        
        fb_gpu_free_on_device(mgr, gpu_id, d_x);
        fb_gpu_free_on_device(mgr, gpu_id, d_y);
        free(h_x);
        free(h_y);
    }
    
    printf("\nKey advantage: Same API, different GPUs - completely transparent!\n");
}

/* ============================================================================
 * Example 3: Parallel workload distribution
 * ========================================================================== */

void example_parallel_distribution(fb_gpu_manager_t* mgr) {
    printf("\n=== Example 3: Parallel Workload Distribution ===\n");
    
    int num_gpus = fb_gpu_get_device_count(mgr);
    if (num_gpus < 1) {
        printf("Need at least 1 GPU for this example\n");
        return;
    }
    
    int total_size = 100000;
    int chunk_size = total_size / num_gpus;
    
    printf("Distributing workload across %d GPU(s):\n", num_gpus);
    printf("  Total size: %d elements\n", total_size);
    printf("  Chunk size per GPU: %d elements\n", chunk_size);
    
    // Allocate host data
    float* h_x = (float*)malloc(total_size * sizeof(float));
    float* h_y = (float*)malloc(total_size * sizeof(float));
    
    for (int i = 0; i < total_size; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(i * 0.5);
    }
    
    // Distribute work to each GPU
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        int offset = gpu_id * chunk_size;
        int size = (gpu_id == num_gpus - 1) ? (total_size - offset) : chunk_size;
        
        fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, gpu_id);
        const char* backend_name = 
            (ctx->backend_type == FB_GPU_BACKEND_CUBLAS) ? "NVIDIA" : "AMD";
        
        printf("\nGPU %d (%s):\n", gpu_id, backend_name);
        printf("  Processing elements [%d - %d] (%d elements)\n",
               offset, offset + size - 1, size);
        
        // Allocate device memory for this chunk
        fb_gpu_ptr_t d_x = fb_gpu_malloc_on_device(mgr, gpu_id, size * sizeof(float));
        fb_gpu_ptr_t d_y = fb_gpu_malloc_on_device(mgr, gpu_id, size * sizeof(float));
        
        // Copy chunk to device
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_x, &h_x[offset], size * sizeof(float));
        fb_gpu_memcpy_h2d_on_device(mgr, gpu_id, d_y, &h_y[offset], size * sizeof(float));
        
        // Execute SAXPY on this chunk
        fb_gpu_saxpy_on_device(mgr, gpu_id, size, 2.0f, d_x, 1, d_y, 1);
        
        // Copy result back
        fb_gpu_memcpy_d2h_on_device(mgr, gpu_id, &h_y[offset], d_y, size * sizeof(float));
        
        printf("  ✓ Completed\n");
        
        // Cleanup
        fb_gpu_free_on_device(mgr, gpu_id, d_x);
        fb_gpu_free_on_device(mgr, gpu_id, d_y);
    }
    
    // Verify results
    printf("\nVerifying distributed computation:\n");
    int errors = 0;
    for (int i = 0; i < total_size; i++) {
        float expected = 2.0f * h_x[i] + (float)(i * 0.5);
        if (fabs(h_y[i] - expected) > 1e-4f) {
            if (errors < 5) {
                printf("  Error at [%d]: got %.2f, expected %.2f\n", i, h_y[i], expected);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("  ✓ All %d elements computed correctly!\n", total_size);
    } else {
        printf("  ✗ Found %d errors\n", errors);
    }
    
    free(h_x);
    free(h_y);
}

/* ============================================================================
 * Main Program
 * ========================================================================== */

int main(void) {
    printf("========================================\n");
    printf("Multi-GPU Example: AMD + NVIDIA\n");
    printf("========================================\n");
    
    // Initialize GPU manager - detects all GPUs automatically
    fb_gpu_manager_t* mgr = fb_gpu_manager_init();
    if (!mgr) {
        fprintf(stderr, "Failed to initialize GPU manager\n");
        return 1;
    }
    
    // Run examples
    example_vector_operations(mgr);
    example_gpu_routing(mgr);
    example_parallel_distribution(mgr);
    
    // Cleanup
    fb_gpu_manager_shutdown(mgr);
    
    printf("\n========================================\n");
    printf("All examples completed!\n");
    printf("========================================\n");
    
    return 0;
}
