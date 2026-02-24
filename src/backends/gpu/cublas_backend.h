/**
 * @file cublas_backend.h
 * @brief NVIDIA cuBLAS backend implementation
 * 
 * Provides adapter layer between faster-blaster and NVIDIA cuBLAS library.
 * Requires CUDA Toolkit to be installed.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_CUBLAS_BACKEND_H
#define FASTER_BLASTER_CUBLAS_BACKEND_H

#include "gpu_backend.h"

/* Forward declaration for trait interface */
typedef struct fb_gpu_backend_trait fb_gpu_backend_trait_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get cuBLAS backend operations
 * 
 * @return cuBLAS GPU backend trait, or NULL if CUDA not available
 */
const fb_gpu_backend_trait_t* fb_cublas_get_backend(void);

/**
 * Initialize cuBLAS backend with specific device
 * 
 * @param device_id CUDA device ID
 * @param ctx Output parameter for GPU context
 * @return 0 on success, negative on error
 */
int fb_cublas_init(int device_id, fb_gpu_context_t* ctx);

/**
 * Shutdown cuBLAS backend
 * 
 * @param ctx GPU context to shutdown
 */
void fb_cublas_shutdown(fb_gpu_context_t* ctx);

/**
 * Get cuBLAS BLAS vtable
 * 
 * @return Backend vtable for cuBLAS operations
 */
const fb_backend_vtable_t* fb_cublas_get_vtable(void);

/**
 * Check if CUDA is available on this system
 * 
 * @return true if CUDA runtime is available
 */
bool fb_cublas_is_available(void);

/**
 * Get CUDA device count
 * 
 * @return Number of CUDA devices, or 0 if none
 */
int fb_cublas_device_count(void);

/**
 * Get CUDA device properties
 * 
 * @param device_id CUDA device ID
 * @param info Output parameter for device info
 * @return 0 on success, negative on error
 */
int fb_cublas_get_device_info(int device_id, fb_gpu_device_info_t* info);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_CUBLAS_BACKEND_H */
