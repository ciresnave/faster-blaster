/**
 * @file gpu_backend.h
 * @brief GPU backend interface for CUDA, ROCm, and other accelerators
 * 
 * This provides a unified abstraction layer over different GPU BLAS implementations
 * (cuBLAS, rocBLAS, oneMKL GPU) to enable seamless GPU acceleration.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_GPU_BACKEND_H
#define FASTER_BLASTER_GPU_BACKEND_H

#include "../backend_interface.h"
#include "../backend_loader.h"  /* For fb_backend_type_t */
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU memory management modes
 */
typedef enum {
    FB_GPU_MEM_DEVICE,      /**< Device memory (GPU-only) */
    FB_GPU_MEM_HOST,        /**< Pinned host memory */
    FB_GPU_MEM_MANAGED,     /**< Unified memory (CUDA/HIP managed) */
    FB_GPU_MEM_MAPPED       /**< Zero-copy mapped memory */
} fb_gpu_memory_type_t;

/**
 * GPU stream/queue handle (opaque)
 */
typedef void* fb_gpu_stream_t;

/**
 * GPU device memory pointer (opaque)
 */
typedef void* fb_gpu_ptr_t;

/**
 * GPU backend context
 */
typedef struct {
    int device_id;
    fb_gpu_stream_t default_stream;
    void* backend_context;  /**< Backend-specific context (cuBLAS handle, etc.) */
    bool synchronous;       /**< Auto-sync after operations */
} fb_gpu_context_t;

/**
 * GPU backend operations (extends base backend_vtable)
 */
typedef struct {
    /* Device management */
    int (*init)(int device_id, fb_gpu_context_t* ctx);
    void (*shutdown)(fb_gpu_context_t* ctx);
    int (*set_device)(int device_id);
    int (*get_device)(void);
    int (*device_count)(void);
    
    /* Stream management */
    int (*stream_create)(fb_gpu_stream_t* stream);
    void (*stream_destroy)(fb_gpu_stream_t stream);
    int (*stream_synchronize)(fb_gpu_stream_t stream);
    
    /* Memory management */
    int (*malloc)(fb_gpu_ptr_t* ptr, size_t size);
    void (*free)(fb_gpu_ptr_t ptr);
    int (*memcpy_h2d)(fb_gpu_ptr_t dst, const void* src, size_t size, fb_gpu_stream_t stream);
    int (*memcpy_d2h)(void* dst, fb_gpu_ptr_t src, size_t size, fb_gpu_stream_t stream);
    int (*memcpy_d2d)(fb_gpu_ptr_t dst, fb_gpu_ptr_t src, size_t size, fb_gpu_stream_t stream);
    int (*memset)(fb_gpu_ptr_t ptr, int value, size_t size, fb_gpu_stream_t stream);
    
    /* Synchronization */
    int (*device_synchronize)(void);
    
    /* BLAS vtable pointer (backend_vtable_t for GPU operations) */
    const fb_backend_vtable_t* blas_vtable;
    
} fb_gpu_backend_t;

/**
 * Initialize GPU backend and select device
 * 
 * @param device_id GPU device ID (0-based), or -1 for auto-select
 * @param ctx Output parameter for GPU context
 * @return 0 on success, negative on error
 */
int fb_gpu_init(int device_id, fb_gpu_context_t* ctx);

/**
 * Shutdown GPU backend and release resources
 * 
 * @param ctx GPU context to shutdown
 */
void fb_gpu_shutdown(fb_gpu_context_t* ctx);

/**
 * Allocate device memory
 * 
 * @param ptr Output parameter for device pointer
 * @param size Size in bytes
 * @return 0 on success, negative on error
 */
int fb_gpu_malloc(fb_gpu_ptr_t* ptr, size_t size);

/**
 * Free device memory
 * 
 * @param ptr Device pointer to free
 */
void fb_gpu_free(fb_gpu_ptr_t ptr);

/**
 * Copy data from host to device
 * 
 * @param dst Device destination pointer
 * @param src Host source pointer
 * @param size Size in bytes
 * @param stream Stream to use (NULL for default)
 * @return 0 on success, negative on error
 */
int fb_gpu_memcpy_h2d(fb_gpu_ptr_t dst, const void* src, size_t size, fb_gpu_stream_t stream);

/**
 * Copy data from device to host
 * 
 * @param dst Host destination pointer
 * @param src Device source pointer
 * @param size Size in bytes
 * @param stream Stream to use (NULL for default)
 * @return 0 on success, negative on error
 */
int fb_gpu_memcpy_d2h(void* dst, fb_gpu_ptr_t src, size_t size, fb_gpu_stream_t stream);

/**
 * Synchronize device
 * 
 * @return 0 on success, negative on error
 */
int fb_gpu_device_synchronize(void);

/**
 * Get GPU backend operations structure
 * 
 * @param type Backend type (cuBLAS, rocBLAS, etc.)
 * @return GPU backend operations, or NULL if not available
 */
const fb_gpu_backend_t* fb_gpu_get_backend(fb_backend_type_t type);

/**
 * Helper: Allocate and copy array to GPU
 * 
 * @param host_ptr Source host pointer
 * @param size Size in bytes
 * @param device_ptr Output parameter for device pointer
 * @return 0 on success, negative on error
 */
int fb_gpu_upload_array(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr);

/**
 * Helper: Copy array from GPU and free device memory
 * 
 * @param device_ptr Device pointer (will be freed)
 * @param size Size in bytes
 * @param host_ptr Output host pointer
 * @return 0 on success, negative on error
 */
int fb_gpu_download_array(fb_gpu_ptr_t device_ptr, size_t size, void* host_ptr);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_GPU_BACKEND_H */
