/**
 * @file rocblas_vtable_wrappers.c
 * @brief rocBLAS GPU trait to unified vtable wrapper functions
 * 
 * This file bridges the GPU trait interface (fb_rocblas_trait) to the unified
 * backend vtable interface (fb_backend_vtable_t). It wraps memory management,
 * stream management, and backend property operations.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../backends/backend_interface.h"
#include "faster-blaster/gpu_backend_trait.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* External reference to rocBLAS GPU trait (defined in gpu_backend_trait.c) */
extern const fb_gpu_backend_trait_t fb_rocblas_trait;

/* ========================================================================
 * UNIFIED CPU/GPU INTERFACE WRAPPERS FOR rocBLAS
 * ========================================================================
 * 
 * These functions adapt the rocBLAS GPU trait to the unified backend vtable.
 * They handle type conversions (void* ↔ fb_gpu_ptr_t, void* ↔ fb_gpu_stream_t)
 * and delegate to the rocBLAS trait implementation.
 * 
 * For GPU backends:
 * - Memory operations delegate to hipMalloc/hipMemcpy
 * - Stream operations delegate to hipStream* functions
 * - Backend properties return GPU-specific capabilities
 * ======================================================================== */

/* ------------------------------------------------------------------------
 * MEMORY MANAGEMENT OPERATIONS
 * ------------------------------------------------------------------------ */

/**
 * @brief Allocate GPU memory (wraps fb_rocblas_trait.malloc)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_mem_alloc(void* handle, void** ptr, size_t size) {
    if (!fb_rocblas_trait.malloc || !handle) {
        return -1;
    }
    fb_gpu_ptr_t gpu_ptr;
    int result = fb_rocblas_trait.malloc(handle, &gpu_ptr, size);
    if (result == 0) {
        *ptr = (void*)gpu_ptr;  /* Cast GPU pointer to void* for unified interface */
    }
    return result;
}

/**
 * @brief Free GPU memory (wraps fb_rocblas_trait.free)
 */
static void rocblas_mem_free(void* handle, void* ptr) {
    if (fb_rocblas_trait.free && handle) {
        fb_rocblas_trait.free(handle, (fb_gpu_ptr_t)ptr);
    }
}

/**
 * @brief Transfer data to GPU (host → device, wraps fb_rocblas_trait.memcpy_h2d)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_mem_upload(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_rocblas_trait.memcpy_h2d || !handle) {
        return -1;
    }
    return fb_rocblas_trait.memcpy_h2d(handle, (fb_gpu_ptr_t)dst, src, size);
}

/**
 * @brief Transfer data from GPU (device → host, wraps fb_rocblas_trait.memcpy_d2h)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_mem_download(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_rocblas_trait.memcpy_d2h || !handle) {
        return -1;
    }
    return fb_rocblas_trait.memcpy_d2h(handle, dst, (fb_gpu_ptr_t)src, size);
}

/**
 * @brief Copy data on GPU (device → device, wraps fb_rocblas_trait.memcpy_d2d)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_mem_copy(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_rocblas_trait.memcpy_d2d || !handle) {
        return -1;
    }
    return fb_rocblas_trait.memcpy_d2d(handle, (fb_gpu_ptr_t)dst, (fb_gpu_ptr_t)src, size);
}

/* ------------------------------------------------------------------------
 * STREAM MANAGEMENT OPERATIONS
 * ------------------------------------------------------------------------ */

/**
 * @brief Create GPU stream (wraps fb_rocblas_trait.stream_create)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_stream_create(void* handle, void** stream) {
    if (!fb_rocblas_trait.stream_create || !handle) {
        return -1;
    }
    fb_gpu_stream_t gpu_stream;
    int result = fb_rocblas_trait.stream_create(handle, &gpu_stream);
    if (result == 0) {
        *stream = (void*)gpu_stream;  /* Cast GPU stream to void* for unified interface */
    }
    return result;
}

/**
 * @brief Destroy GPU stream (wraps fb_rocblas_trait.stream_destroy)
 */
static void rocblas_stream_destroy(void* handle, void* stream) {
    if (fb_rocblas_trait.stream_destroy && handle) {
        fb_rocblas_trait.stream_destroy(handle, (fb_gpu_stream_t)stream);
    }
}

/**
 * @brief Synchronize GPU stream (wraps fb_rocblas_trait.stream_synchronize)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int rocblas_stream_sync(void* handle, void* stream) {
    if (!fb_rocblas_trait.stream_synchronize || !handle) {
        return -1;
    }
    return fb_rocblas_trait.stream_synchronize(handle, (fb_gpu_stream_t)stream);
}

/**
 * @brief Set active GPU stream
 * @return -1 (not exposed in GPU trait - backends handle internally)
 * @note rocBLAS uses rocblas_set_stream internally, not exposed through trait
 */
static int rocblas_stream_set(void* handle, void* stream) {
    (void)handle;
    (void)stream;
    return -1;  /* Not exposed in GPU trait interface */
}

/* ------------------------------------------------------------------------
 * BACKEND PROPERTIES
 * ------------------------------------------------------------------------ */

/**
 * @brief Get rocBLAS backend capabilities
 * @return Bitfield of FB_CAP_* flags
 */
static uint32_t rocblas_get_capabilities(void* handle) {
    (void)handle;  /* Capabilities are static for rocBLAS */
    
    /* rocBLAS supports:
     * - GPU execution (asynchronous, stream-based)
     * - All BLAS levels (1, 2, 3)
     * - Single and double precision (FP32, FP64)
     * - Complex arithmetic (FP32 complex, FP64 complex)
     * - Mixed precision (FP16, BF16 on supported hardware)
     */
    return FB_CAP_GPU | FB_CAP_ASYNC |
           FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

/**
 * @brief Get number of compute threads (not applicable for GPU)
 * @return -1 (GPU backends don't use CPU threading)
 */
static int rocblas_get_num_threads(void* handle) {
    (void)handle;
    return -1;  /* Not applicable for GPU backends */
}

/**
 * @brief Set number of compute threads (not applicable for GPU)
 * @note No-op for GPU backends
 */
static void rocblas_set_num_threads(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* No-op: GPU backends don't use CPU threading */
}

/* ========================================================================
 * VTABLE POPULATION
 * ======================================================================== */

/**
 * @brief Populate rocBLAS backend vtable with unified interface functions
 * 
 * This function assigns all the wrapper functions to the vtable, bridging
 * the rocBLAS GPU trait to the unified CPU/GPU interface.
 * 
 * @param vtable Pointer to backend vtable to populate
 */
void rocblas_populate_vtable(fb_backend_vtable_t* vtable) {
    if (!vtable) {
        return;
    }
    
    /* BLAS operations are populated separately (in plugin_rocblas.c or via trait) */
    
    /* Unified CPU/GPU Interface - Memory Management */
    vtable->mem_alloc = rocblas_mem_alloc;
    vtable->mem_free = rocblas_mem_free;
    vtable->mem_upload = rocblas_mem_upload;
    vtable->mem_download = rocblas_mem_download;
    vtable->mem_copy = rocblas_mem_copy;
    
    /* Unified CPU/GPU Interface - Stream Management */
    vtable->stream_create = rocblas_stream_create;
    vtable->stream_destroy = rocblas_stream_destroy;
    vtable->stream_sync = rocblas_stream_sync;
    vtable->stream_set = rocblas_stream_set;
    
    /* Unified CPU/GPU Interface - Backend Properties */
    vtable->get_capabilities = rocblas_get_capabilities;
    vtable->get_num_threads = rocblas_get_num_threads;
    vtable->set_num_threads = rocblas_set_num_threads;
}

#ifdef __cplusplus
}
#endif
