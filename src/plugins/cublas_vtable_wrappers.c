/**
 * @file cublas_vtable_wrappers.c
 * @brief cuBLAS GPU trait to unified vtable wrapper functions
 * 
 * This file bridges the GPU trait interface (fb_cublas_trait) to the unified
 * backend vtable interface (fb_backend_vtable_t). It wraps memory management,
 * stream management, and backend property operations.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "../backends/backend_interface.h"
#include "faster-blaster/gpu_backend_trait.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* External reference to cuBLAS GPU trait (defined in gpu_backend_trait.c) */
extern const fb_gpu_backend_trait_t fb_cublas_trait;

/* Global backend handle for wrapper functions
 * TODO: This should be per-device and properly managed through plugin context
 */
static void* g_cublas_backend_handle = NULL;

/**
 * @brief Set the backend handle for wrapper functions
 * @param handle The cuBLAS backend handle to use for operations
 */
void cublas_wrappers_set_handle(void* handle) {
    g_cublas_backend_handle = handle;
}

/* ========================================================================
 * UNIFIED CPU/GPU INTERFACE WRAPPERS FOR cuBLAS
 * ========================================================================
 * 
 * These functions adapt the cuBLAS GPU trait to the unified backend vtable.
 * They handle type conversions (void* ↔ fb_gpu_ptr_t, void* ↔ fb_gpu_stream_t)
 * and delegate to the cuBLAS trait implementation.
 * 
 * For GPU backends:
 * - Memory operations delegate to cudaMalloc/cudaMemcpy
 * - Stream operations delegate to cudaStream* functions
 * - Backend properties return GPU-specific capabilities
 * ======================================================================== */

/* ------------------------------------------------------------------------
 * MEMORY MANAGEMENT OPERATIONS
 * ------------------------------------------------------------------------ */

/**
 * @brief Allocate GPU memory (wraps fb_cublas_trait.malloc)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_mem_alloc(void* handle, void** ptr, size_t size) {
    if (!fb_cublas_trait.malloc || !handle) {
        return -1;
    }
    fb_gpu_ptr_t gpu_ptr;
    int result = fb_cublas_trait.malloc(handle, &gpu_ptr, size);
    if (result == 0) {
        *ptr = (void*)gpu_ptr;  /* Cast GPU pointer to void* for unified interface */
    }
    return result;
}

/**
 * @brief Free GPU memory (wraps fb_cublas_trait.free)
 */
static void cublas_mem_free(void* handle, void* ptr) {
    if (fb_cublas_trait.free && handle) {
        fb_cublas_trait.free(handle, (fb_gpu_ptr_t)ptr);
    }
}

/**
 * @brief Transfer data to GPU (host → device, wraps fb_cublas_trait.memcpy_h2d)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_mem_upload(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_cublas_trait.memcpy_h2d || !handle) {
        return -1;
    }
    return fb_cublas_trait.memcpy_h2d(handle, (fb_gpu_ptr_t)dst, src, size);
}

/**
 * @brief Transfer data from GPU (device → host, wraps fb_cublas_trait.memcpy_d2h)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_mem_download(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_cublas_trait.memcpy_d2h || !handle) {
        return -1;
    }
    return fb_cublas_trait.memcpy_d2h(handle, dst, (fb_gpu_ptr_t)src, size);
}

/**
 * @brief Copy data on GPU (device → device, wraps fb_cublas_trait.memcpy_d2d)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_mem_copy(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_cublas_trait.memcpy_d2d || !handle) {
        return -1;
    }
    return fb_cublas_trait.memcpy_d2d(handle, (fb_gpu_ptr_t)dst, (fb_gpu_ptr_t)src, size);
}

/* ------------------------------------------------------------------------
 * STREAM MANAGEMENT OPERATIONS
 * ------------------------------------------------------------------------ */

/**
 * @brief Create GPU stream (wraps fb_cublas_trait.stream_create)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_stream_create(void* handle, void** stream) {
    if (!fb_cublas_trait.stream_create || !handle) {
        return -1;
    }
    fb_gpu_stream_t gpu_stream;
    int result = fb_cublas_trait.stream_create(handle, &gpu_stream);
    if (result == 0) {
        *stream = (void*)gpu_stream;  /* Cast GPU stream to void* for unified interface */
    }
    return result;
}

/**
 * @brief Destroy GPU stream (wraps fb_cublas_trait.stream_destroy)
 */
static void cublas_stream_destroy(void* handle, void* stream) {
    if (fb_cublas_trait.stream_destroy && handle) {
        fb_cublas_trait.stream_destroy(handle, (fb_gpu_stream_t)stream);
    }
}

/**
 * @brief Synchronize GPU stream (wraps fb_cublas_trait.stream_synchronize)
 * @return 0 on success, -1 on error or if trait function unavailable
 */
static int cublas_stream_sync(void* handle, void* stream) {
    if (!fb_cublas_trait.stream_synchronize || !handle) {
        return -1;
    }
    return fb_cublas_trait.stream_synchronize(handle, (fb_gpu_stream_t)stream);
}

/**
 * @brief Set active GPU stream
 * @return -1 (not exposed in GPU trait - backends handle internally)
 * @note cuBLAS uses cublasSetStream internally, not exposed through trait
 */
static int cublas_stream_set(void* handle, void* stream) {
    (void)handle;
    (void)stream;
    return -1;  /* Not exposed in GPU trait interface */
}

/* ------------------------------------------------------------------------
 * BACKEND PROPERTIES
 * ------------------------------------------------------------------------ */

/**
 * @brief Get cuBLAS backend capabilities
 * @return Bitfield of FB_CAP_* flags
 */
static uint32_t cublas_get_capabilities(void* handle) {
    (void)handle;  /* Capabilities are static for cuBLAS */
    
    /* cuBLAS supports:
     * - GPU execution (asynchronous, stream-based)
     * - All BLAS levels (1, 2, 3)
     * - Single and double precision (FP32, FP64)
     * - Complex arithmetic (FP32 complex, FP64 complex)
     */
    return FB_CAP_GPU | FB_CAP_ASYNC |
           FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

/**
 * @brief Get number of compute threads (not applicable for GPU)
 * @return -1 (GPU backends don't use CPU threading)
 */
static int cublas_get_num_threads(void* handle) {
    (void)handle;
    return -1;  /* Not applicable for GPU backends */
}

/**
 * @brief Set number of compute threads (not applicable for GPU)
 * @note No-op for GPU backends
 */
static void cublas_set_num_threads(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* No-op: GPU backends don't use CPU threading */
}

/* ========================================================================
 * BLAS OPERATION WRAPPERS (GPU TRAIT → CPU VTABLE)
 * ========================================================================
 * 
 * These wrappers bridge GPU trait operations (device pointers, async) to
 * CPU vtable interface (host pointers, synchronous). They:
 * 
 * 1. Allocate device memory for all parameters
 * 2. Transfer host → device (H2D)
 * 3. Call GPU trait operation (synchronously via NULL stream)
 * 4. Transfer device → host (D2H)
 * 5. Free device memory
 * 
 * Note: This approach has H2D/D2H overhead but allows GPU operations to be
 * tested/used through the same interface as CPU backends. For production
 * GPU use, consider direct trait interface access (Option B).
 * ======================================================================== */

/* BLAS Level 2: sgemv (single precision general matrix-vector multiply) */
static void cublas_sgemv_wrapper(
    fb_layout_t layout, fb_transpose_t trans,
    int m, int n,
    float alpha,
    const float* a, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy)
{
    if (!g_cublas_backend_handle || !a || !x || !y || !fb_cublas_trait.sgemv) {
        return;
    }
    
    /* Use global backend handle */
    void* cublas_handle = g_cublas_backend_handle;
    
    /* Convert transpose enum to char */
    char trans_char = (trans == FB_NO_TRANS) ? 'N' : 
                      (trans == FB_TRANS) ? 'T' : 'C';
    
    /* For column-major (standard BLAS), matrix is stored as A[lda*n] */
    size_t size_a = (layout == FB_LAYOUT_COL_MAJOR) ? 
                    (size_t)lda * n : (size_t)lda * m;
    size_t size_x = (size_t)(1 + (n - 1) * abs(incx));
    size_t size_y = (size_t)(1 + (m - 1) * abs(incy));
    
    /* Allocate device memory */
    fb_gpu_ptr_t d_a = NULL, d_x = NULL, d_y = NULL;
    
    if (fb_cublas_trait.malloc(cublas_handle, &d_a, size_a * sizeof(float)) != 0) {
        return;
    }
    if (fb_cublas_trait.malloc(cublas_handle, &d_x, size_x * sizeof(float)) != 0) {
        fb_cublas_trait.free(cublas_handle, d_a);
        return;
    }
    if (fb_cublas_trait.malloc(cublas_handle, &d_y, size_y * sizeof(float)) != 0) {
        fb_cublas_trait.free(cublas_handle, d_a);
        fb_cublas_trait.free(cublas_handle, d_x);
        return;
    }
    
    /* Copy host → device */
    fb_cublas_trait.memcpy_h2d(cublas_handle, d_a, a, size_a * sizeof(float));
    fb_cublas_trait.memcpy_h2d(cublas_handle, d_x, x, size_x * sizeof(float));
    fb_cublas_trait.memcpy_h2d(cublas_handle, d_y, y, size_y * sizeof(float));
    
    /* Call GPU trait sgemv (NULL stream = synchronous) */
    fb_cublas_trait.sgemv(cublas_handle, NULL, trans_char, m, n, alpha,
                          d_a, lda, d_x, incx, beta, d_y, incy);
    
    /* NULL stream is synchronous - no explicit sync needed */
    
    /* Copy device → host */
    fb_cublas_trait.memcpy_d2h(cublas_handle, y, d_y, size_y * sizeof(float));
    
    /* Free device memory */
    fb_cublas_trait.free(cublas_handle, d_a);
    fb_cublas_trait.free(cublas_handle, d_x);
    fb_cublas_trait.free(cublas_handle, d_y);
}

/* ========================================================================
 * FORWARD DECLARATIONS FOR SMART WRAPPER FUNCTIONS
 * ======================================================================== */

/* Level 1 BLAS smart wrappers (from cublas_blas_level1_smart.c) */
extern void cublas_srot_smart_wrapper(int n, float* x, int incx, float* y, int incy, float c, float s);
extern void cublas_drot_smart_wrapper(int n, double* x, int incx, double* y, int incy, double c, double s);
extern void cublas_srotg_smart_wrapper(float* a, float* b, float* c, float* s);
extern void cublas_drotg_smart_wrapper(double* a, double* b, double* c, double* s);
extern void cublas_srotm_smart_wrapper(int n, float* x, int incx, float* y, int incy, const float* param);
extern void cublas_drotm_smart_wrapper(int n, double* x, int incx, double* y, int incy, const double* param);
extern void cublas_srotmg_smart_wrapper(float* d1, float* d2, float* x1, const float* y1, float* param);
extern void cublas_drotmg_smart_wrapper(double* d1, double* d2, double* x1, const double* y1, double* param);

/* Level 2 BLAS smart wrappers (from cublas_blas_level2_smart.c) */
extern void cublas_sgbmv_smart_wrapper(FB_TRANSPOSE trans, int m, int n, int kl, int ku,
                                        float alpha, const float* a, int lda,
                                        const float* x, int incx, float beta, float* y, int incy);
extern void cublas_dgbmv_smart_wrapper(FB_TRANSPOSE trans, int m, int n, int kl, int ku,
                                        double alpha, const double* a, int lda,
                                        const double* x, int incx, double beta, double* y, int incy);
extern void cublas_ssymv_smart_wrapper(char uplo, int n, float alpha,
                                        const float* a, int lda, const float* x, int incx,
                                        float beta, float* y, int incy);
extern void cublas_dsymv_smart_wrapper(char uplo, int n, double alpha,
                                        const double* a, int lda, const double* x, int incx,
                                        double beta, double* y, int incy);
extern void cublas_strmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const float* a, int lda, float* x, int incx);
extern void cublas_dtrmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const double* a, int lda, double* x, int incx);
extern void cublas_strsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const float* a, int lda, float* x, int incx);
extern void cublas_dtrsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const double* a, int lda, double* x, int incx);
extern void cublas_ssyr_smart_wrapper(char uplo, int n, float alpha,
                                       const float* x, int incx, float* a, int lda);
extern void cublas_dsyr_smart_wrapper(char uplo, int n, double alpha,
                                       const double* x, int incx, double* a, int lda);
extern void cublas_ssyr2_smart_wrapper(char uplo, int n, float alpha,
                                        const float* x, int incx, const float* y, int incy,
                                        float* a, int lda);
extern void cublas_dsyr2_smart_wrapper(char uplo, int n, double alpha,
                                        const double* x, int incx, const double* y, int incy,
                                        double* a, int lda);
extern void cublas_ssbmv_smart_wrapper(char uplo, int n, int k, float alpha,
                                        const float* a, int lda, const float* x, int incx,
                                        float beta, float* y, int incy);
extern void cublas_dsbmv_smart_wrapper(char uplo, int n, int k, double alpha,
                                        const double* a, int lda, const double* x, int incx,
                                        double beta, double* y, int incy);
extern void cublas_stbmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, int k, const float* a, int lda, float* x, int incx);
extern void cublas_dtbmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, int k, const double* a, int lda, double* x, int incx);
extern void cublas_stbsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, int k, const float* a, int lda, float* x, int incx);
extern void cublas_dtbsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, int k, const double* a, int lda, double* x, int incx);
extern void cublas_sspmv_smart_wrapper(char uplo, int n, float alpha,
                                        const float* ap, const float* x, int incx,
                                        float beta, float* y, int incy);
extern void cublas_dspmv_smart_wrapper(char uplo, int n, double alpha,
                                        const double* ap, const double* x, int incx,
                                        double beta, double* y, int incy);
extern void cublas_stpmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const float* ap, float* x, int incx);
extern void cublas_dtpmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const double* ap, double* x, int incx);
extern void cublas_stpsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const float* ap, float* x, int incx);
extern void cublas_dtpsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                        int n, const double* ap, double* x, int incx);
extern void cublas_sspr_smart_wrapper(char uplo, int n, float alpha,
                                       const float* x, int incx, float* ap);
extern void cublas_dspr_smart_wrapper(char uplo, int n, double alpha,
                                       const double* x, int incx, double* ap);
extern void cublas_sspr2_smart_wrapper(char uplo, int n, float alpha,
                                        const float* x, int incx, const float* y, int incy, float* ap);
extern void cublas_dspr2_smart_wrapper(char uplo, int n, double alpha,
                                        const double* x, int incx, const double* y, int incy, double* ap);

/* Level 3 BLAS smart wrappers (from cublas_blas_level3_smart.c) */
extern void cublas_ssymm_smart_wrapper(char side, char uplo, int m, int n,
                                        float alpha, const float* a, int lda,
                                        const float* b, int ldb, float beta, float* c, int ldc);
extern void cublas_dsymm_smart_wrapper(char side, char uplo, int m, int n,
                                        double alpha, const double* a, int lda,
                                        const double* b, int ldb, double beta, double* c, int ldc);
extern void cublas_strmm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                        int m, int n, float alpha, const float* a, int lda,
                                        float* b, int ldb);
extern void cublas_dtrmm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                        int m, int n, double alpha, const double* a, int lda,
                                        double* b, int ldb);
extern void cublas_strsm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                        int m, int n, float alpha, const float* a, int lda,
                                        float* b, int ldb);
extern void cublas_dtrsm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                        int m, int n, double alpha, const double* a, int lda,
                                        double* b, int ldb);
extern void cublas_ssyrk_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                        int n, int k, float alpha, const float* a, int lda,
                                        float beta, float* c, int ldc);
extern void cublas_dsyrk_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                        int n, int k, double alpha, const double* a, int lda,
                                        double beta, double* c, int ldc);
extern void cublas_ssyr2k_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                         int n, int k, float alpha, const float* a, int lda,
                                         const float* b, int ldb, float beta, float* c, int ldc);
extern void cublas_dsyr2k_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                         int n, int k, double alpha, const double* a, int lda,
                                         const double* b, int ldb, double beta, double* c, int ldc);

/* ========================================================================
 * VTABLE POPULATION
 * ======================================================================== */

/**
 * @brief Populate cuBLAS backend vtable with unified interface functions
 * 
 * This function assigns all the wrapper functions to the vtable, bridging
 * the cuBLAS GPU trait to the unified CPU/GPU interface.
 * 
 * @param vtable Pointer to backend vtable to populate
 */
void cublas_populate_vtable(fb_backend_vtable_t* vtable) {
    if (!vtable) {
        return;
    }
    
    /* Unified CPU/GPU Interface - Memory Management */
    vtable->mem_alloc = cublas_mem_alloc;
    vtable->mem_free = cublas_mem_free;
    vtable->mem_upload = cublas_mem_upload;
    vtable->mem_download = cublas_mem_download;
    vtable->mem_copy = cublas_mem_copy;
    
    /* Unified CPU/GPU Interface - Stream Management */
    vtable->stream_create = cublas_stream_create;
    vtable->stream_destroy = cublas_stream_destroy;
    vtable->stream_sync = cublas_stream_sync;
    vtable->stream_set = cublas_stream_set;
    
    /* Unified CPU/GPU Interface - Backend Properties */
    vtable->get_capabilities = cublas_get_capabilities;
    vtable->get_num_threads = cublas_get_num_threads;
    vtable->set_num_threads = cublas_set_num_threads;
    
    /* =====================================================================
     * BLAS OPERATIONS - Phase 1 (46 operations)
     * ===================================================================== */
    
    /* Level 1 BLAS - Rotation operations (8 operations) */
    vtable->srot = cublas_srot_smart_wrapper;
    vtable->drot = cublas_drot_smart_wrapper;
    vtable->srotg = cublas_srotg_smart_wrapper;
    vtable->drotg = cublas_drotg_smart_wrapper;
    vtable->srotm = cublas_srotm_smart_wrapper;
    vtable->drotm = cublas_drotm_smart_wrapper;
    vtable->srotmg = cublas_srotmg_smart_wrapper;
    vtable->drotmg = cublas_drotmg_smart_wrapper;
    
    /* Level 2 BLAS - Matrix-vector operations (28 operations) */
    vtable->sgbmv = cublas_sgbmv_smart_wrapper;
    vtable->dgbmv = cublas_dgbmv_smart_wrapper;
    vtable->ssymv = cublas_ssymv_smart_wrapper;
    vtable->dsymv = cublas_dsymv_smart_wrapper;
    vtable->strmv = cublas_strmv_smart_wrapper;
    vtable->dtrmv = cublas_dtrmv_smart_wrapper;
    vtable->strsv = cublas_strsv_smart_wrapper;
    vtable->dtrsv = cublas_dtrsv_smart_wrapper;
    vtable->ssyr = cublas_ssyr_smart_wrapper;
    vtable->dsyr = cublas_dsyr_smart_wrapper;
    vtable->ssyr2 = cublas_ssyr2_smart_wrapper;
    vtable->dsyr2 = cublas_dsyr2_smart_wrapper;
    vtable->ssbmv = cublas_ssbmv_smart_wrapper;
    vtable->dsbmv = cublas_dsbmv_smart_wrapper;
    vtable->stbmv = cublas_stbmv_smart_wrapper;
    vtable->dtbmv = cublas_dtbmv_smart_wrapper;
    vtable->stbsv = cublas_stbsv_smart_wrapper;
    vtable->dtbsv = cublas_dtbsv_smart_wrapper;
    vtable->sspmv = cublas_sspmv_smart_wrapper;
    vtable->dspmv = cublas_dspmv_smart_wrapper;
    vtable->stpmv = cublas_stpmv_smart_wrapper;
    vtable->dtpmv = cublas_dtpmv_smart_wrapper;
    vtable->stpsv = cublas_stpsv_smart_wrapper;
    vtable->dtpsv = cublas_dtpsv_smart_wrapper;
    vtable->sspr = cublas_sspr_smart_wrapper;
    vtable->dspr = cublas_dspr_smart_wrapper;
    vtable->sspr2 = cublas_sspr2_smart_wrapper;
    vtable->dspr2 = cublas_dspr2_smart_wrapper;
    
    /* Level 3 BLAS - Matrix-matrix operations (10 operations) */
    vtable->ssymm = cublas_ssymm_smart_wrapper;
    vtable->dsymm = cublas_dsymm_smart_wrapper;
    vtable->strmm = cublas_strmm_smart_wrapper;
    vtable->dtrmm = cublas_dtrmm_smart_wrapper;
    vtable->strsm = cublas_strsm_smart_wrapper;
    vtable->dtrsm = cublas_dtrsm_smart_wrapper;
    vtable->ssyrk = cublas_ssyrk_smart_wrapper;
    vtable->dsyrk = cublas_dsyrk_smart_wrapper;
    vtable->ssyr2k = cublas_ssyr2k_smart_wrapper;
    vtable->dsyr2k = cublas_dsyr2k_smart_wrapper;
}

#ifdef __cplusplus
}
#endif
