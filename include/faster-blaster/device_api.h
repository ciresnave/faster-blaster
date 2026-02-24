/**
 * @file device_api.h
 * @brief Unified device API for CPU and GPU backends
 * 
 * This layer provides a uniform interface for memory management, stream
 * control, and backend properties regardless of whether the backend is
 * CPU or GPU based.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_DEVICE_API_H
#define FASTER_BLASTER_DEVICE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct fb_compute_device fb_compute_device_t;
typedef struct fb_backend_instance fb_backend_instance_t;

/**
 * @brief Allocate device memory
 * 
 * For CPU backends: Uses system malloc or returns error if not supported
 * For GPU backends: Calls cudaMalloc/hipMalloc/clCreateBuffer
 * 
 * @param device Device handle
 * @param ptr Output pointer to allocated memory
 * @param size Size in bytes to allocate
 * @return 0 on success, non-zero on error
 */
int fb_device_alloc(fb_compute_device_t* device, void** ptr, size_t size);

/**
 * @brief Free device memory
 * 
 * For CPU backends: Uses system free or no-op if not supported
 * For GPU backends: Calls cudaFree/hipFree/clReleaseMemObject
 * 
 * @param device Device handle
 * @param ptr Pointer to free
 */
void fb_device_free(fb_compute_device_t* device, void* ptr);

/**
 * @brief Upload data to device (host → device)
 * 
 * For CPU backends: No-op (data already accessible)
 * For GPU backends: Calls cudaMemcpy H2D/hipMemcpy H2D
 * 
 * @param device Device handle
 * @param dst Destination device pointer
 * @param src Source host pointer
 * @param size Number of bytes to transfer
 * @return 0 on success, non-zero on error
 */
int fb_device_upload(fb_compute_device_t* device, void* dst, const void* src, size_t size);

/**
 * @brief Download data from device (device → host)
 * 
 * For CPU backends: No-op (data already accessible)
 * For GPU backends: Calls cudaMemcpy D2H/hipMemcpy D2H
 * 
 * @param device Device handle
 * @param dst Destination host pointer
 * @param src Source device pointer
 * @param size Number of bytes to transfer
 * @return 0 on success, non-zero on error
 */
int fb_device_download(fb_compute_device_t* device, void* dst, const void* src, size_t size);

/**
 * @brief Copy data on device (device → device)
 * 
 * For CPU backends: Uses memcpy if supported
 * For GPU backends: Calls cudaMemcpy D2D/hipMemcpy D2D
 * 
 * @param device Device handle
 * @param dst Destination device pointer
 * @param src Source device pointer
 * @param size Number of bytes to copy
 * @return 0 on success, non-zero on error
 */
int fb_device_copy(fb_compute_device_t* device, void* dst, const void* src, size_t size);

/**
 * @brief Create device stream/queue
 * 
 * For CPU backends: No-op (operations are synchronous)
 * For GPU backends: Creates cudaStream/hipStream/cl_command_queue
 * 
 * @param device Device handle
 * @param stream Output stream handle
 * @return 0 on success, non-zero on error
 */
int fb_device_stream_create(fb_compute_device_t* device, void** stream);

/**
 * @brief Destroy device stream/queue
 * 
 * For CPU backends: No-op
 * For GPU backends: Destroys cudaStream/hipStream/cl_command_queue
 * 
 * @param device Device handle
 * @param stream Stream handle to destroy
 */
void fb_device_stream_destroy(fb_compute_device_t* device, void* stream);

/**
 * @brief Synchronize device stream/queue
 * 
 * For CPU backends: No-op (operations already complete)
 * For GPU backends: Calls cudaStreamSynchronize/hipStreamSynchronize
 * 
 * @param device Device handle
 * @param stream Stream handle (NULL = default stream/device sync)
 * @return 0 on success, non-zero on error
 */
int fb_device_sync(fb_compute_device_t* device, void* stream);

/**
 * @brief Set active device stream/queue
 * 
 * For CPU backends: No-op
 * For GPU backends: Sets active stream for subsequent operations
 * 
 * @param device Device handle
 * @param stream Stream handle to activate
 */
void fb_device_stream_set(fb_compute_device_t* device, void* stream);

/**
 * @brief Get device backend capabilities
 * 
 * Returns a bitfield of FB_CAP_* flags indicating what operations
 * and features the backend supports.
 * 
 * @param device Device handle
 * @return Capability flags (FB_CAP_CPU, FB_CAP_GPU, FB_CAP_ASYNC, etc.)
 */
uint32_t fb_device_get_capabilities(fb_compute_device_t* device);

/**
 * @brief Get number of compute threads
 * 
 * For CPU backends: Returns configured OpenMP/BLAS thread count
 * For GPU backends: Returns -1 (not applicable)
 * 
 * @param device Device handle
 * @return Thread count, or -1 if not applicable
 */
int fb_device_get_num_threads(fb_compute_device_t* device);

/**
 * @brief Set number of compute threads
 * 
 * For CPU backends: Sets OpenMP/BLAS thread count
 * For GPU backends: No-op (not applicable)
 * 
 * @param device Device handle
 * @param num_threads Desired thread count
 */
void fb_device_set_num_threads(fb_compute_device_t* device, int num_threads);

/* ========================================================================
 * BACKEND INSTANCE API (for internal use)
 * ======================================================================== */

/**
 * @brief Get backend instance for a device
 * 
 * Internal function to retrieve the backend instance from a device handle.
 * Used by the device API functions to access the backend vtable.
 * 
 * @param device Device handle
 * @return Backend instance, or NULL if not initialized
 */
fb_backend_instance_t* fb_device_get_backend_instance(fb_compute_device_t* device);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DEVICE_API_H */
