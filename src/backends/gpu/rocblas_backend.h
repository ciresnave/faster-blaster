/**
 * @file rocblas_backend.h
 * @brief AMD rocBLAS backend implementation
 * 
 * Provides adapter layer between faster-blaster and AMD rocBLAS library.
 * Requires ROCm to be installed.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_ROCBLAS_BACKEND_H
#define FASTER_BLASTER_ROCBLAS_BACKEND_H

#include "gpu_backend.h"

/* Forward declaration for trait interface */
typedef struct fb_gpu_backend_trait fb_gpu_backend_trait_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get rocBLAS backend operations
 * 
 * @return rocBLAS GPU backend trait, or NULL if ROCm not available
 */
const fb_gpu_backend_trait_t* fb_rocblas_get_backend(void);

/**
 * Initialize rocBLAS backend with specific device
 * 
 * @param device_id HIP device ID
 * @param ctx Output parameter for GPU context
 * @return 0 on success, negative on error
 */
int fb_rocblas_init(int device_id, fb_gpu_context_t* ctx);

/**
 * Shutdown rocBLAS backend
 * 
 * @param ctx GPU context to shutdown
 */
void fb_rocblas_shutdown(fb_gpu_context_t* ctx);

/**
 * Get rocBLAS BLAS vtable
 * 
 * @return Backend vtable for rocBLAS operations
 */
const fb_backend_vtable_t* fb_rocblas_get_vtable(void);

/**
 * Check if ROCm is available on this system
 * 
 * @return true if HIP runtime is available
 */
bool fb_rocblas_is_available(void);

/**
 * Get HIP device count
 * 
 * @return Number of HIP devices, or 0 if none
 */
int fb_rocblas_device_count(void);

/**
 * Get HIP device properties
 * 
 * @param device_id HIP device ID
 * @param info Output parameter for device info
 * @return 0 on success, negative on error
 */
int fb_rocblas_get_device_info(int device_id, fb_gpu_device_info_t* info);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_ROCBLAS_BACKEND_H */
