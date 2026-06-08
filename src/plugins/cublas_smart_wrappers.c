/**
 * @file cublas_smart_wrappers.c
 * @brief Smart cuBLAS wrappers using device memory manager
 * 
 * This file provides intelligent wrappers that:
 * - Use device memory manager for efficient operation chaining
 * - Keep data on device between operations
 * - Support cross-backend operation sequences
 * - Automatically handle H2D/D2H copies only when needed
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/gpu_backend_trait.h"
#include "faster-blaster/device_memory_manager.h"
#include "backends/backend_interface.h"
#include "faster_blaster.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* External reference to cuBLAS GPU trait */
extern const fb_gpu_backend_trait_t fb_cublas_trait;

/* Global backend handle (TODO: Move to plugin context) */
void* g_cublas_handle = NULL;  /* Exported for use by complete wrappers */

/* Device ID for this backend (TODO: Support multi-GPU) */
static const int CUBLAS_DEVICE_ID = 0;

/* ============================================================================
 * Helper Functions
 * ========================================================================== */

/**
 * @brief Get device memory manager for cuBLAS operations
 */
static fb_device_memory_manager_t* get_device_manager(void) {
    return fb_device_memory_get_manager(CUBLAS_DEVICE_ID);
}

/**
 * @brief Calculate buffer size for different BLAS operations
 */
static size_t calculate_vector_size(int n, int inc) {
    if (n <= 0 || inc == 0) return 0;
    return (size_t)(1 + (n - 1) * abs(inc));
}

static size_t calculate_matrix_size_col_major(int rows, int cols, int ld) {
    /* Column-major: ld is leading dimension (>= rows) */
    (void)rows;
    return (size_t)ld * cols;
}

/* ============================================================================
 * Smart Memory Management Wrapper
 * ========================================================================== */

/**
 * @brief Get or allocate device memory for operation input
 * 
 * This handles the smart caching logic:
 * - If data is already on device, reuse it
 * - If not, allocate and H2D copy
 * - Track that we're using this buffer
 * 
 * @param host_ptr Host pointer
 * @param size Buffer size in bytes
 * @param device_ptr_out Output: device pointer
 * @return 0 on success, -1 on error
 */
static int get_device_buffer_input(
    const void* host_ptr,
    size_t size,
    fb_gpu_ptr_t* device_ptr_out)
{
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        fprintf(stderr, "Failed to get device memory manager\n");
        return -1;
    }
    
    return fb_device_memory_get_or_alloc(
        manager,
        &fb_cublas_trait,
        g_cublas_handle,
        host_ptr,
        size,
        FB_GPU_BACKEND_CUBLAS,
        device_ptr_out
    );
}

/**
 * @brief Get or allocate device memory for operation output
 * 
 * For outputs, we need to:
 * 1. Get or allocate device memory
 * 2. If it's an input-output parameter (beta != 0), copy current host value
 * 3. Mark as dirty after operation completes
 */
static int get_device_buffer_output(
    void* host_ptr,
    size_t size,
    int is_inout,  /* true if output also depends on input (e.g., beta*y) */
    fb_gpu_ptr_t* device_ptr_out)
{
    (void)is_inout;
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        return -1;
    }
    
    /* For input-output parameters, treat same as input (will have current value) */
    /* For pure outputs, still need to allocate but might not need H2D copy */
    /* Memory manager will handle this - if already cached, it has the value */
    
    return fb_device_memory_get_or_alloc(
        manager,
        &fb_cublas_trait,
        g_cublas_handle,
        host_ptr,
        size,
        FB_GPU_BACKEND_CUBLAS,
        device_ptr_out
    );
}

/**
 * @brief Sync output buffer to host immediately after GPU operation.
 * Mark dirty first (dirty=1) so sync_to_host proceeds, then sync.
 * Without marking dirty first, sync_to_host skips due to dirty==0 guard.
 */
static void mark_output_dirty(void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return;
    fb_device_memory_mark_dirty(manager, host_ptr, FB_GPU_BACKEND_CUBLAS);
    fb_device_memory_sync_to_host(manager, &fb_cublas_trait, g_cublas_handle, host_ptr);
}

/**
 * @brief Release reference to device buffer after operation
 */
static void release_device_buffer(const void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_release(manager, host_ptr);
    }
}

/**
 * @brief Sync output buffer to host (performs D2H if dirty)
 * 
 * Call this when:
 * - User explicitly requests sync
 * - Before returning from API call that user expects result in host memory
 * - End of operation chain
 */
static int sync_output_to_host(void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        return -1;
    }
    
    return fb_device_memory_sync_to_host(
        manager,
        &fb_cublas_trait,
        g_cublas_handle,
        host_ptr
    );
}

/* ============================================================================
 * BLAS Level 2: sgemv (Matrix-Vector Multiply)
 * ========================================================================== */

/**
 * @brief Smart sgemv wrapper with memory manager
 * 
 * y := alpha*A*x + beta*y  (or with A transposed)
 * 
 * This wrapper:
 * - Checks if A, x, y are already on device (cache hit = instant)
 * - If not, allocates and copies (cache miss)
 * - Executes operation on device
 * - Marks y as dirty (doesn't D2H yet - waits for explicit sync or chain end)
 * - Releases references (but keeps memory cached)
 * 
 * @param layout Row or column major (cuBLAS uses column-major)
 * @param trans Transpose flag for A
 * @param m Number of rows in A
 * @param n Number of columns in A  
 * @param alpha Scalar alpha
 * @param a Matrix A (host pointer)
 * @param lda Leading dimension of A
 * @param x Vector x (host pointer)
 * @param incx Stride of x
 * @param beta Scalar beta
 * @param y Vector y (host pointer, input-output)
 * @param incy Stride of y
 */
void cublas_sgemv_smart_wrapper(
    FB_LAYOUT layout,
    FB_TRANSPOSE trans,
    int m, int n,
    float alpha,
    const float* a, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy)
{
    (void)layout;
    /* Parameter validation */
    if (!a || !x || !y || m <= 0 || n <= 0) {
        fprintf(stderr, "Invalid parameters to sgemv\n");
        return;
    }
    
    if (!g_cublas_handle) {
        fprintf(stderr, "cuBLAS handle not initialized\n");
        return;
    }
    
    /* Calculate buffer sizes */
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(m, incy) * sizeof(float);
    
    /* Get device buffers (cache-aware) */
    fb_gpu_ptr_t d_a = NULL, d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(a, size_a, &d_a) != 0) {
        fprintf(stderr, "Failed to get device buffer for A\n");
        return;
    }
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        fprintf(stderr, "Failed to get device buffer for x\n");
        return;
    }
    
    /* y is input-output (beta*y term), so need current value */
    int is_inout = (beta != 0.0f);
    if (get_device_buffer_output(y, size_y, is_inout, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        fprintf(stderr, "Failed to get device buffer for y\n");
        return;
    }
    
    /* Convert transpose enum to char */
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    /* Call cuBLAS sgemv via trait
     * Note: cuBLAS expects column-major, which matches BLAS standard */
    if (fb_cublas_trait.sgemv) {
        fb_cublas_trait.sgemv(
            g_cublas_handle,
            NULL,  /* NULL stream = default/synchronous */
            trans_char,
            m, n,
            alpha,
            d_a, lda,
            d_x, incx,
            beta,
            d_y, incy
        );
        
        /* Mark output as dirty (device has newer data than host) */
        mark_output_dirty(y);
    }
    
    /* Release references (memory stays cached on device) */
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
    
    /* Note: We do NOT sync y back to host here!
     * Data stays on device for potential chaining.
     * User must call sync explicitly or it happens automatically
     * at chain end / when switching to different device. */
}

/* ============================================================================
 * BLAS Level 2: dgemv (Double Precision)
 * ========================================================================== */

void cublas_dgemv_smart_wrapper(
    FB_LAYOUT layout,
    FB_TRANSPOSE trans,
    int m, int n,
    double alpha,
    const double* a, int lda,
    const double* x, int incx,
    double beta,
    double* y, int incy)
{
    (void)layout;
    if (!a || !x || !y || m <= 0 || n <= 0 || !g_cublas_handle) {
        return;
    }
    
    /* Calculate buffer sizes */
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(m, incy) * sizeof(double);
    
    /* Get device buffers */
    fb_gpu_ptr_t d_a = NULL, d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(a, size_a, &d_a) != 0 ||
        get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, (beta != 0.0), &d_y) != 0) {
        /* Cleanup handled by get_device_buffer functions */
        return;
    }
    
    /* Convert transpose */
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    /* Call cuBLAS dgemv */
    if (fb_cublas_trait.dgemv) {
        fb_cublas_trait.dgemv(
            g_cublas_handle, NULL, trans_char, m, n,
            alpha, d_a, lda, d_x, incx, beta, d_y, incy
        );
        
        mark_output_dirty(y);
    }
    
    /* Release references */
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/* ============================================================================
 * API for Explicit Synchronization
 * ========================================================================== */

/**
 * @brief Synchronize a specific buffer from device to host
 * 
 * User can call this explicitly to get result back to host.
 * Also called automatically at operation chain boundaries.
 */
int fb_cublas_sync_buffer_to_host(void* host_ptr) {
    return sync_output_to_host(host_ptr);
}

/**
 * @brief Synchronize all dirty buffers to host
 * 
 * Call this at end of operation chain or when switching devices.
 */
int fb_cublas_sync_all_to_host(void) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        return -1;
    }
    
    return fb_device_memory_sync_all_to_host(
        manager,
        &fb_cublas_trait,
        g_cublas_handle
    );
}

/**
 * @brief Free device memory for a specific buffer
 * 
 * Removes from cache. Next operation will need to reallocate.
 */
int fb_cublas_free_device_buffer(void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        return -1;
    }
    
    return fb_device_memory_free(
        manager,
        &fb_cublas_trait,
        g_cublas_handle,
        host_ptr
    );
}

/**
 * @brief Print device memory manager statistics
 */
void fb_cublas_print_memory_stats(void) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_print_stats(manager);
    }
}

/* ============================================================================
 * Initialize backend handle
 * ========================================================================== */

/**
 * @brief Initialize cuBLAS backend and memory manager
 * 
 * Call this before using any smart wrappers.
 */
int fb_cublas_smart_init(int device_id, void* lib_handle) {
    /* Initialize cuBLAS backend */
    if (fb_cublas_trait.init) {
        int result = fb_cublas_trait.init(device_id, lib_handle, &g_cublas_handle);
        if (result != 0) {
            fprintf(stderr, "Failed to initialize cuBLAS backend\n");
            return -1;
        }
    }
    
    /* Initialize device memory manager (happens automatically via get_device_manager) */
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) {
        fprintf(stderr, "Failed to initialize device memory manager\n");
        return -1;
    }
    
    printf("cuBLAS smart wrappers initialized (device %d)\n", device_id);
    
    return 0;
}

/**
 * @brief Shutdown cuBLAS backend and memory manager
 */
void fb_cublas_smart_shutdown(void) {
    /* Sync all dirty buffers before shutdown */
    fb_cublas_sync_all_to_host();
    
    /* Shutdown cuBLAS backend */
    if (fb_cublas_trait.shutdown && g_cublas_handle) {
        fb_cublas_trait.shutdown(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    
    /* Memory manager shutdown happens globally via fb_device_memory_shutdown_all() */
    
    printf("cuBLAS smart wrappers shut down\n");
}

#ifdef __cplusplus
}
#endif
