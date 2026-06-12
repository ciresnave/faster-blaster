/**
 * @file gpu_backend_trait.h
 * @brief Unified GPU backend trait interface for vendor-agnostic GPU BLAS/LAPACK
 * 
 * This interface provides a unified abstraction over different GPU BLAS libraries
 * (cuBLAS, rocBLAS, oneMKL, etc.), enabling transparent multi-GPU execution across
 * NVIDIA, AMD, and Intel GPUs with a consistent API.
 * 
 * Key design principles:
 * - Single trait interface for all GPU backends
 * - Backend-agnostic operation signatures
 * - Automatic vendor detection and initialization
 * - Support for multi-GPU workload distribution
 * - Data must remain on single GPU (no automatic transfers)
 */

#ifndef FASTER_BLASTER_GPU_BACKEND_TRAIT_H
#define FASTER_BLASTER_GPU_BACKEND_TRAIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GPU Backend Types and Context
 * ========================================================================== */

/**
 * @brief GPU backend type identifier
 */
typedef enum {
    FB_GPU_BACKEND_NONE = 0,
    
    /* Major GPU Vendors */
    FB_GPU_BACKEND_CUBLAS,       /**< NVIDIA CUDA + cuBLAS + cuSOLVER */
    FB_GPU_BACKEND_ROCBLAS,      /**< AMD ROCm + rocBLAS + rocSOLVER */
    FB_GPU_BACKEND_ONEMKL,       /**< Intel oneAPI + oneMKL (SYCL/DPC++) */
    FB_GPU_BACKEND_ACCELERATE,   /**< Apple Metal + Accelerate Framework */
    
    /* Alternative/Portable Backends */
    FB_GPU_BACKEND_OPENBLAS_GPU, /**< OpenBLAS GPU (if available) */
    FB_GPU_BACKEND_SYCL,         /**< Generic SYCL backend (multi-vendor) */
    FB_GPU_BACKEND_CLBLAST,      /**< CLBlast OpenCL BLAS */
    FB_GPU_BACKEND_CLBLAS,       /**< clBLAS OpenCL BLAS (AMD legacy) */
    
    /* Future/Experimental */
    FB_GPU_BACKEND_VULKAN_COMPUTE, /**< Vulkan compute shaders (portable) */
    FB_GPU_BACKEND_WEBGPU          /**< WebGPU (browser/portable) */
} fb_gpu_backend_type_t;

/**
 * @brief Opaque GPU device pointer (device memory address)
 */
typedef void* fb_gpu_ptr_t;

/**
 * @brief Opaque stream/queue handle for async execution
 */
typedef void* fb_gpu_stream_t;

/**
 * @brief Forward declaration of GPU context
 */
typedef struct fb_gpu_context fb_gpu_context_t;

/**
 * @brief Forward declaration of GPU backend trait
 */
typedef struct fb_gpu_backend_trait fb_gpu_backend_trait_t;

/* ============================================================================
 * GPU Backend Trait Interface (vtable)
 * ========================================================================== */

/**
 * @brief Unified GPU backend trait - virtual function table
 * 
 * This structure contains function pointers for all GPU operations.
 * Each backend (cuBLAS, rocBLAS, etc.) provides its own implementation.
 */
struct fb_gpu_backend_trait {
    /* Backend metadata */
    const char* name;                    /**< Backend name (e.g., "NVIDIA cuBLAS") */
    fb_gpu_backend_type_t type;          /**< Backend type identifier */
    
    /* ========================================================================
     * Lifecycle Management
     * ====================================================================== */
    
    /**
     * @brief Initialize backend for specific device
     * @param device_id Device index (0-based)
     * @param lib_handle Library handle loaded by plugin (DLL/SO handle)
     * @param backend_handle Output: backend-specific handle (cuBLAS/rocBLAS handle)
     * @return 0 on success, non-zero on error
     */
    int (*init)(int device_id, void* lib_handle, void** backend_handle);
    
    /**
     * @brief Shutdown backend and release resources
     * @param backend_handle Backend-specific handle to destroy
     */
    void (*shutdown)(void* backend_handle);
    
    /**
     * @brief Get device properties (compute capability, memory, etc.)
     * @param backend_handle Backend-specific handle
     * @param device_id Device index
     * @param name Output: device name (can be NULL)
     * @param name_len Length of name buffer
     * @param total_memory Output: total device memory in bytes (can be NULL)
     * @return 0 on success, non-zero on error
     */
    int (*get_device_properties)(void* backend_handle, int device_id,
                                  char* name, size_t name_len,
                                  size_t* total_memory);
    
    /* ========================================================================
     * Memory Management
     * ====================================================================== */
    
    /**
     * @brief Allocate device memory
     * @param backend_handle Backend-specific handle
     * @param ptr Output: device pointer
     * @param size Number of bytes to allocate
     * @return 0 on success, non-zero on error
     */
    int (*malloc)(void* backend_handle, fb_gpu_ptr_t* ptr, size_t size);
    
    /**
     * @brief Free device memory
     * @param backend_handle Backend-specific handle
     * @param ptr Device pointer to free
     */
    void (*free)(void* backend_handle, fb_gpu_ptr_t ptr);
    
    /**
     * @brief Copy host memory to device
     * @param backend_handle Backend-specific handle
     * @param dst Device pointer (destination)
     * @param src Host pointer (source)
     * @param size Number of bytes to copy
     * @return 0 on success, non-zero on error
     */
    int (*memcpy_h2d)(void* backend_handle, fb_gpu_ptr_t dst,
                      const void* src, size_t size);
    
    /**
     * @brief Copy device memory to host
     * @param backend_handle Backend-specific handle
     * @param dst Host pointer (destination)
     * @param src Device pointer (source)
     * @param size Number of bytes to copy
     * @return 0 on success, non-zero on error
     */
    int (*memcpy_d2h)(void* backend_handle, void* dst,
                      fb_gpu_ptr_t src, size_t size);
    
    /**
     * @brief Copy device memory to device (same GPU)
     * @param backend_handle Backend-specific handle
     * @param dst Device pointer (destination)
     * @param src Device pointer (source)
     * @param size Number of bytes to copy
     * @return 0 on success, non-zero on error
     */
    int (*memcpy_d2d)(void* backend_handle, fb_gpu_ptr_t dst,
                      fb_gpu_ptr_t src, size_t size);
    
    /* ========================================================================
     * Stream/Queue Management
     * ====================================================================== */
    
    /**
     * @brief Create a new stream/queue for async execution
     * @param backend_handle Backend-specific handle
     * @param stream Output: stream handle
     * @return 0 on success, non-zero on error
     */
    int (*stream_create)(void* backend_handle, fb_gpu_stream_t* stream);
    
    /**
     * @brief Destroy stream/queue
     * @param backend_handle Backend-specific handle
     * @param stream Stream handle to destroy
     */
    void (*stream_destroy)(void* backend_handle, fb_gpu_stream_t stream);
    
    /**
     * @brief Synchronize stream (wait for all operations to complete)
     * @param backend_handle Backend-specific handle
     * @param stream Stream handle (NULL = default stream)
     * @return 0 on success, non-zero on error
     */
    int (*stream_synchronize)(void* backend_handle, fb_gpu_stream_t stream);
    
    /* ========================================================================
     * Enum Conversion Helpers
     * ====================================================================== */
    
    /**
     * @brief Convert BLAS transpose flag to backend-specific enum
     * @param trans 'N', 'T', or 'C'
     * @return Backend-specific transpose enum value
     */
    int (*convert_transpose)(char trans);
    
    /**
     * @brief Convert BLAS uplo flag to backend-specific enum
     * @param uplo 'U' or 'L'
     * @return Backend-specific uplo enum value
     */
    int (*convert_uplo)(char uplo);
    
    /**
     * @brief Convert BLAS diag flag to backend-specific enum
     * @param diag 'N' or 'U'
     * @return Backend-specific diag enum value
     */
    int (*convert_diag)(char diag);
    
    /**
     * @brief Convert BLAS side flag to backend-specific enum
     * @param side 'L' or 'R'
     * @return Backend-specific side enum value
     */
    int (*convert_side)(char side);
    
    /* ========================================================================
     * BLAS Level 1 Operations (54 operations)
     * ====================================================================== */
    
    /* Single/Double precision real operations */
    void (*saxpy)(void* handle, fb_gpu_stream_t stream, int n, float alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*daxpy)(void* handle, fb_gpu_stream_t stream, int n, double alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    void (*sscal)(void* handle, fb_gpu_stream_t stream, int n, float alpha,
                  fb_gpu_ptr_t x, int incx);
    void (*dscal)(void* handle, fb_gpu_stream_t stream, int n, double alpha,
                  fb_gpu_ptr_t x, int incx);
    
    void (*scopy)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*dcopy)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    void (*sswap)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*dswap)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    float (*sdot)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    double (*ddot)(void* handle, fb_gpu_stream_t stream, int n,
                   fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    float (*snrm2)(void* handle, fb_gpu_stream_t stream, int n,
                   fb_gpu_ptr_t x, int incx);
    double (*dnrm2)(void* handle, fb_gpu_stream_t stream, int n,
                    fb_gpu_ptr_t x, int incx);
    
    float (*sasum)(void* handle, fb_gpu_stream_t stream, int n,
                   fb_gpu_ptr_t x, int incx);
    double (*dasum)(void* handle, fb_gpu_stream_t stream, int n,
                    fb_gpu_ptr_t x, int incx);
    
    int (*isamax)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx);
    int (*idamax)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx);
    
    /* Complex operations (26 additional) */
    void (*caxpy)(void* handle, fb_gpu_stream_t stream, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*zaxpy)(void* handle, fb_gpu_stream_t stream, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    void (*cscal)(void* handle, fb_gpu_stream_t stream, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx);
    void (*zscal)(void* handle, fb_gpu_stream_t stream, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx);
    void (*csscal)(void* handle, fb_gpu_stream_t stream, int n, float alpha,
                   fb_gpu_ptr_t x, int incx);
    void (*zdscal)(void* handle, fb_gpu_stream_t stream, int n, double alpha,
                   fb_gpu_ptr_t x, int incx);
    
    void (*ccopy)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*zcopy)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    void (*cswap)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    void (*zswap)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    
    void (*cdotu)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  void* result);
    void (*zdotu)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  void* result);
    void (*cdotc)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  void* result);
    void (*zdotc)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  void* result);
    
    float (*scnrm2)(void* handle, fb_gpu_stream_t stream, int n,
                    fb_gpu_ptr_t x, int incx);
    double (*dznrm2)(void* handle, fb_gpu_stream_t stream, int n,
                     fb_gpu_ptr_t x, int incx);
    
    float (*scasum)(void* handle, fb_gpu_stream_t stream, int n,
                    fb_gpu_ptr_t x, int incx);
    double (*dzasum)(void* handle, fb_gpu_stream_t stream, int n,
                     fb_gpu_ptr_t x, int incx);
    
    int (*icamax)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx);
    int (*izamax)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx);
    
    /* Rotation operations (12 additional) */
    void (*srotg)(void* handle, fb_gpu_stream_t stream,
                  fb_gpu_ptr_t a, fb_gpu_ptr_t b,
                  fb_gpu_ptr_t c, fb_gpu_ptr_t s);
    void (*drotg)(void* handle, fb_gpu_stream_t stream,
                  fb_gpu_ptr_t a, fb_gpu_ptr_t b,
                  fb_gpu_ptr_t c, fb_gpu_ptr_t s);
    
    void (*srot)(void* handle, fb_gpu_stream_t stream, int n,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                 float c, float s);
    void (*drot)(void* handle, fb_gpu_stream_t stream, int n,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                 double c, double s);
    
    void (*srotm)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t param);
    void (*drotm)(void* handle, fb_gpu_stream_t stream, int n,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t param);
    
    void (*srotmg)(void* handle, fb_gpu_stream_t stream,
                   fb_gpu_ptr_t d1, fb_gpu_ptr_t d2,
                   fb_gpu_ptr_t x1, fb_gpu_ptr_t y1,
                   fb_gpu_ptr_t param);
    void (*drotmg)(void* handle, fb_gpu_stream_t stream,
                   fb_gpu_ptr_t d1, fb_gpu_ptr_t d2,
                   fb_gpu_ptr_t x1, fb_gpu_ptr_t y1,
                   fb_gpu_ptr_t param);
    
    /* ========================================================================
     * BLAS Level 2 Operations (70 operations)
     * ====================================================================== */
    
    /* General matrix-vector multiplication */
    void (*sgemv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, float beta,
                  fb_gpu_ptr_t y, int incy);
    void (*dgemv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, double alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, double beta,
                  fb_gpu_ptr_t y, int incy);
    void (*cgemv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    void (*zgemv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    
    /* General banded matrix-vector multiplication */
    void (*sgbmv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, int kl, int ku, float alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  float beta, fb_gpu_ptr_t y, int incy);
    void (*dgbmv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, int kl, int ku, double alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  double beta, fb_gpu_ptr_t y, int incy);
    void (*cgbmv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, int kl, int ku, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    void (*zgbmv)(void* handle, fb_gpu_stream_t stream, char trans,
                  int m, int n, int kl, int ku, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    
    /* Hermitian/symmetric matrix-vector multiplication */
    void (*chemv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    void (*zhemv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    
    void (*ssymv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, float beta,
                  fb_gpu_ptr_t y, int incy);
    void (*dsymv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, double alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, double beta,
                  fb_gpu_ptr_t y, int incy);
    void (*csymv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    void (*zsymv)(void* handle, fb_gpu_stream_t stream, char uplo,
                  int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t x, int incx, const void* beta,
                  fb_gpu_ptr_t y, int incy);
    
    /* Triangular matrix-vector multiplication */
    void (*strmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*dtrmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ctrmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ztrmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    
    /* Triangular solve */
    void (*strsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*dtrsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ctrsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ztrsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    
    /* Rank-1 update */
    void (*sger)(void* handle, fb_gpu_stream_t stream, int m, int n, float alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                 fb_gpu_ptr_t a, int lda);
    void (*dger)(void* handle, fb_gpu_stream_t stream, int m, int n, double alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                 fb_gpu_ptr_t a, int lda);
    void (*cgeru)(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    void (*zgeru)(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    void (*cgerc)(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    void (*zgerc)(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    
    /* Hermitian/symmetric rank-1 update */
    void (*cher)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    void (*zher)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    
    void (*ssyr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    void (*dsyr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    void (*csyr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    void (*zsyr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda);
    
    /* Hermitian/symmetric rank-2 update */
    void (*cher2)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t x, int incx,
                  fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda);
    void (*zher2)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t x, int incx,
                  fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda);
    
    void (*ssyr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    void (*dsyr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                  fb_gpu_ptr_t a, int lda);
    
    /* Banded matrix-vector operations (16 operations) */
    void (*ssbmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  float beta, fb_gpu_ptr_t y, int incy);
    void (*dsbmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k,
                  double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  double beta, fb_gpu_ptr_t y, int incy);
    
    void (*chbmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    void (*zhbmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    
    void (*stbmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*dtbmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ctbmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ztbmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    
    void (*stbsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*dtbsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ctbsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    void (*ztbsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx);
    
    /* Packed matrix operations (16 operations) */
    void (*sspmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  float alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  float beta, fb_gpu_ptr_t y, int incy);
    void (*dspmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  double alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  double beta, fb_gpu_ptr_t y, int incy);
    
    void (*chpmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    void (*zhpmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    
    void (*stpmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*dtpmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*ctpmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*ztpmv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    
    void (*stpsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*dtpsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*ctpsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    void (*ztpsv)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag,
                  int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx);
    
    /* Packed rank update operations (8 operations) */
    void (*sspr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    void (*dspr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    
    void (*chpr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    void (*zhpr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    
    void (*sspr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    void (*dspr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    
    void (*chpr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    void (*zhpr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    
    /* Complex symmetric rank-2 update operations (2 operations) */
    void (*csyr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda);
    void (*zsyr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda);
    
    /* Complex symmetric packed operations (6 operations - rocBLAS only, cuBLAS unsupported) */
    void (*cspmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    void (*zspmv)(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                  const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                  const void* beta, fb_gpu_ptr_t y, int incy);
    
    void (*cspr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    void (*zspr)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                 fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap);
    
    void (*cspr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    void (*zspr2)(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap);
    
    /* ========================================================================
     * BLAS Level 3 Operations (28 operations)
     * ====================================================================== */
    
    /* General matrix-matrix multiplication */
    void (*sgemm)(void* handle, fb_gpu_stream_t stream, char transa, char transb,
                  int m, int n, int k, float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc);
    void (*dgemm)(void* handle, fb_gpu_stream_t stream, char transa, char transb,
                  int m, int n, int k, double alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc);
    void (*cgemm)(void* handle, fb_gpu_stream_t stream, char transa, char transb,
                  int m, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    void (*zgemm)(void* handle, fb_gpu_stream_t stream, char transa, char transb,
                  int m, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Symmetric matrix-matrix multiplication */
    void (*ssymm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc);
    void (*dsymm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, double alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc);
    void (*csymm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    void (*zsymm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Hermitian matrix-matrix multiplication */
    void (*chemm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    void (*zhemm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Triangular matrix-matrix multiplication */
    void (*strmm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, float alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*dtrmm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, double alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*ctrmm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*ztrmm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* Triangular solve with multiple right-hand sides */
    void (*strsm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, float alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*dtrsm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, double alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*ctrsm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    void (*ztrsm)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                  char transa, char diag, int m, int n, const void* alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* Symmetric rank-k update */
    void (*ssyrk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, float alpha, fb_gpu_ptr_t a, int lda,
                  float beta, fb_gpu_ptr_t c, int ldc);
    void (*dsyrk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, double alpha, fb_gpu_ptr_t a, int lda,
                  double beta, fb_gpu_ptr_t c, int ldc);
    void (*csyrk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                  const void* beta, fb_gpu_ptr_t c, int ldc);
    void (*zsyrk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                  const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Hermitian rank-k update */
    void (*cherk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, float alpha, fb_gpu_ptr_t a, int lda,
                  float beta, fb_gpu_ptr_t c, int ldc);
    void (*zherk)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                  int n, int k, double alpha, fb_gpu_ptr_t a, int lda,
                  double beta, fb_gpu_ptr_t c, int ldc);
    
    /* Symmetric rank-2k update */
    void (*ssyr2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, float alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc);
    void (*dsyr2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, double alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc);
    void (*csyr2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    void (*zsyr2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Hermitian rank-2k update */
    void (*cher2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc);
    void (*zher2k)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                   int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda,
                   fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc);
    
    /* ========================================================================
     * LAPACK Operations (via cuSOLVER/rocSOLVER)
     * ====================================================================== */
    
    /* LU Factorization */
    int (*sgetrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*dgetrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*cgetrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*zgetrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    
    /* LU Solve */
    int (*sgetrs)(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*dgetrs)(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*cgetrs)(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*zgetrs)(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    
    /* Cholesky Factorization */
    int (*spotrf)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda);
    int (*dpotrf)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda);
    int (*cpotrf)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda);
    int (*zpotrf)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda);
    
    /* Cholesky Solve */
    int (*spotrs)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*dpotrs)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*cpotrs)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*zpotrs)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* QR Factorization */
    int (*sgeqrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*dgeqrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*cgeqrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*zgeqrf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    
    /* SVD */
    int (*sgesvd)(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*dgesvd)(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*cgesvd)(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*zgesvd)(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    
    /* Eigenvalues (symmetric/Hermitian) */
    int (*ssyev)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*dsyev)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*cheev)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*zheev)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    
    /* ========================================================================
     * Extended LAPACK Operations (cuSOLVER/rocSOLVER/oneMKL)
     * ====================================================================== */
    
    /* Combined LU solve: A*X = B */
    int (*sgesv)(void* handle, fb_gpu_stream_t stream, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*dgesv)(void* handle, fb_gpu_stream_t stream, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*cgesv)(void* handle, fb_gpu_stream_t stream, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    int (*zgesv)(void* handle, fb_gpu_stream_t stream, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb);
    
    /* Combined Cholesky solve: A*X = B */
    int (*sposv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*dposv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*cposv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*zposv)(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* Least squares solve: min ||A*X - B|| */
    int (*sgels)(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*dgels)(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*cgels)(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*zgels)(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int nrhs, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* General eigenvalues (non-symmetric) */
    int (*sgeev)(void* handle, fb_gpu_stream_t stream, char jobvl, char jobvr, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t wr, fb_gpu_ptr_t wi, fb_gpu_ptr_t vl, int ldvl, fb_gpu_ptr_t vr, int ldvr);
    int (*dgeev)(void* handle, fb_gpu_stream_t stream, char jobvl, char jobvr, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t wr, fb_gpu_ptr_t wi, fb_gpu_ptr_t vl, int ldvl, fb_gpu_ptr_t vr, int ldvr);
    int (*cgeev)(void* handle, fb_gpu_stream_t stream, char jobvl, char jobvr, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w, fb_gpu_ptr_t vl, int ldvl, fb_gpu_ptr_t vr, int ldvr);
    int (*zgeev)(void* handle, fb_gpu_stream_t stream, char jobvl, char jobvr, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w, fb_gpu_ptr_t vl, int ldvl, fb_gpu_ptr_t vr, int ldvr);
    
    /* Eigenvalues - divide-and-conquer (faster for large matrices) */
    int (*ssyevd)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*dsyevd)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*cheevd)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*zheevd)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    
    /* Eigenvalues - Jacobi method (better accuracy for small matrices) */
    int (*ssyevj)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*dsyevj)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*cheevj)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    int (*zheevj)(void* handle, fb_gpu_stream_t stream, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w);
    
    /* Generalized symmetric eigenvalue problem: A*x = lambda*B*x */
    int (*ssygvd)(void* handle, fb_gpu_stream_t stream, int itype, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, fb_gpu_ptr_t w);
    int (*dsygvd)(void* handle, fb_gpu_stream_t stream, int itype, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, fb_gpu_ptr_t w);
    int (*chegvd)(void* handle, fb_gpu_stream_t stream, int itype, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, fb_gpu_ptr_t w);
    int (*zhegvd)(void* handle, fb_gpu_stream_t stream, int itype, char jobz, char uplo, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, fb_gpu_ptr_t w);
    
    /* SVD - Jacobi method (better accuracy, slower) */
    int (*sgesvdj)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv);
    int (*dgesvdj)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv);
    int (*cgesvdj)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv);
    int (*zgesvdj)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv);
    
    /* SVD - approximate (faster, less accurate) */
    int (*sgesvda)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv, int rank);
    int (*dgesvda)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv, int rank);
    int (*cgesvda)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv, int rank);
    int (*zgesvda)(void* handle, fb_gpu_stream_t stream, char jobu, char jobv, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t v, int ldv, int rank);
    
    /* SVD - divide-and-conquer (fastest) */
    int (*sgesdd)(void* handle, fb_gpu_stream_t stream, char jobz, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*dgesdd)(void* handle, fb_gpu_stream_t stream, char jobz, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*cgesdd)(void* handle, fb_gpu_stream_t stream, char jobz, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    int (*zgesdd)(void* handle, fb_gpu_stream_t stream, char jobz, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s, fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt);
    
    /* Generate orthogonal/unitary Q matrix from QR factorization */
    int (*sorgqr)(void* handle, fb_gpu_stream_t stream, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*dorgqr)(void* handle, fb_gpu_stream_t stream, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*cungqr)(void* handle, fb_gpu_stream_t stream, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*zungqr)(void* handle, fb_gpu_stream_t stream, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    
    /* Multiply by orthogonal/unitary Q matrix from QR factorization */
    int (*sormqr)(void* handle, fb_gpu_stream_t stream, char side, char trans, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, fb_gpu_ptr_t c, int ldc);
    int (*dormqr)(void* handle, fb_gpu_stream_t stream, char side, char trans, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, fb_gpu_ptr_t c, int ldc);
    int (*cunmqr)(void* handle, fb_gpu_stream_t stream, char side, char trans, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, fb_gpu_ptr_t c, int ldc);
    int (*zunmqr)(void* handle, fb_gpu_stream_t stream, char side, char trans, int m, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, fb_gpu_ptr_t c, int ldc);
    
    /* QR factorization with column pivoting */
    int (*sgeqp3)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t jpvt, fb_gpu_ptr_t tau);
    int (*dgeqp3)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t jpvt, fb_gpu_ptr_t tau);
    int (*cgeqp3)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t jpvt, fb_gpu_ptr_t tau);
    int (*zgeqp3)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t jpvt, fb_gpu_ptr_t tau);
    
    /* LQ factorization */
    int (*sgelqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*dgelqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*cgelqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*zgelqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    
    /* RQ factorization */
    int (*sgerqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*dgerqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*cgerqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*zgerqf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    
    /* QL factorization */
    int (*sgeqlf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*dgeqlf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*cgeqlf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    int (*zgeqlf)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau);
    
    /* Triangular matrix inverse */
    int (*strtri)(void* handle, fb_gpu_stream_t stream, char uplo, char diag, int n, fb_gpu_ptr_t a, int lda);
    int (*dtrtri)(void* handle, fb_gpu_stream_t stream, char uplo, char diag, int n, fb_gpu_ptr_t a, int lda);
    int (*ctrtri)(void* handle, fb_gpu_stream_t stream, char uplo, char diag, int n, fb_gpu_ptr_t a, int lda);
    int (*ztrtri)(void* handle, fb_gpu_stream_t stream, char uplo, char diag, int n, fb_gpu_ptr_t a, int lda);
    
    /* General matrix inverse (via LU) */
    int (*sgetri)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*dgetri)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*cgetri)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    int (*zgetri)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv);
    
    /* Condition number estimation */
    int (*sgecon)(void* handle, fb_gpu_stream_t stream, char norm, int n, fb_gpu_ptr_t a, int lda, float anorm, float* rcond);
    int (*dgecon)(void* handle, fb_gpu_stream_t stream, char norm, int n, fb_gpu_ptr_t a, int lda, double anorm, double* rcond);
    int (*cgecon)(void* handle, fb_gpu_stream_t stream, char norm, int n, fb_gpu_ptr_t a, int lda, float anorm, float* rcond);
    int (*zgecon)(void* handle, fb_gpu_stream_t stream, char norm, int n, fb_gpu_ptr_t a, int lda, double anorm, double* rcond);
    
    int (*spocon)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda, float anorm, float* rcond);
    int (*dpocon)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda, double anorm, double* rcond);
    int (*cpocon)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda, float anorm, float* rcond);
    int (*zpocon)(void* handle, fb_gpu_stream_t stream, char uplo, int n, fb_gpu_ptr_t a, int lda, double anorm, double* rcond);
    
    /* ========================================================================
     * Fused/Multi-Operation Functions (High Performance Combinations)
     * ====================================================================== */
    
    /* Mixed precision GEMM variants */
    int (*gemm_bf16bf16f32)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k,
                            float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                            float beta, fb_gpu_ptr_t c, int ldc);
    int (*gemm_f16f16f32)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k,
                          float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                          float beta, fb_gpu_ptr_t c, int ldc);
    
    /* Complex GEMM optimized (3 multiplies instead of 4) */
    int (*cgemm3m)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k,
                   const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                   const void* beta, fb_gpu_ptr_t c, int ldc);
    int (*zgemm3m)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k,
                   const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                   const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Triangular GEMM (compute only upper/lower triangle) */
    int (*sgemmt)(void* handle, fb_gpu_stream_t stream, char uplo, char transa, char transb, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                  float beta, fb_gpu_ptr_t c, int ldc);
    int (*dgemmt)(void* handle, fb_gpu_stream_t stream, char uplo, char transa, char transb, int n, int k,
                  double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                  double beta, fb_gpu_ptr_t c, int ldc);
    int (*cgemmt)(void* handle, fb_gpu_stream_t stream, char uplo, char transa, char transb, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                  const void* beta, fb_gpu_ptr_t c, int ldc);
    int (*zgemmt)(void* handle, fb_gpu_stream_t stream, char uplo, char transa, char transb, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb,
                  const void* beta, fb_gpu_ptr_t c, int ldc);
    
    /* Out-of-place matrix transpose/copy with scaling: B = alpha * op(A) */
    int (*somatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*domatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*comatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    int (*zomatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb);
    
    /* In-place matrix transpose with scaling: A = alpha * op(A) */
    int (*simatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     float alpha, fb_gpu_ptr_t a, int lda);
    int (*dimatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     double alpha, fb_gpu_ptr_t a, int lda);
    int (*cimatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     const void* alpha, fb_gpu_ptr_t a, int lda);
    int (*zimatcopy)(void* handle, fb_gpu_stream_t stream, char trans, int rows, int cols,
                     const void* alpha, fb_gpu_ptr_t a, int lda);
    
    /* Matrix addition with transpose: C = alpha*op(A) + beta*op(B) */
    int (*sgeam)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n,
                 float alpha, fb_gpu_ptr_t a, int lda, float beta, fb_gpu_ptr_t b, int ldb,
                 fb_gpu_ptr_t c, int ldc);
    int (*dgeam)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n,
                 double alpha, fb_gpu_ptr_t a, int lda, double beta, fb_gpu_ptr_t b, int ldb,
                 fb_gpu_ptr_t c, int ldc);
    int (*cgeam)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n,
                 const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                 fb_gpu_ptr_t c, int ldc);
    int (*zgeam)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n,
                 const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                 fb_gpu_ptr_t c, int ldc);
    
    /* Symmetric rank-k with separate output: C = alpha*A*A^T + beta*B */
    int (*ssyrkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda, float beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    int (*dsyrkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  double alpha, fb_gpu_ptr_t a, int lda, double beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    int (*csyrkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    int (*zsyrkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    
    /* Hermitian rank-k with separate output: C = alpha*A*A^H + beta*B */
    int (*cherkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    int (*zherkx)(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k,
                  const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t b, int ldb,
                  fb_gpu_ptr_t c, int ldc);
    
    /* LU factorization without pivoting (faster when pivoting not needed) */
    int (*sgetrfnp)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda);
    int (*dgetrfnp)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda);
    int (*cgetrfnp)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda);
    int (*zgetrfnp)(void* handle, fb_gpu_stream_t stream, int m, int n, fb_gpu_ptr_t a, int lda);
    
    /* ========================================================================
     * Extended BLAS Operations (Batched, Mixed Precision, Tensor)
     * ====================================================================== */
    
    /* Batched GEMM: C[i] = alpha*A[i]*B[i] + beta*C[i] for all i */
    int (*sgemm_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                         float alpha, fb_gpu_ptr_t* a_array, int lda, fb_gpu_ptr_t* b_array, int ldb, 
                         float beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*dgemm_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                         double alpha, fb_gpu_ptr_t* a_array, int lda, fb_gpu_ptr_t* b_array, int ldb, 
                         double beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*cgemm_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                         const void* alpha, fb_gpu_ptr_t* a_array, int lda, fb_gpu_ptr_t* b_array, int ldb, 
                         const void* beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*zgemm_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                         const void* alpha, fb_gpu_ptr_t* a_array, int lda, fb_gpu_ptr_t* b_array, int ldb, 
                         const void* beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    
    /* Strided batched GEMM: A, B, C are contiguous arrays with strides */
    int (*sgemm_strided_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                                 float alpha, fb_gpu_ptr_t a, int lda, long long stride_a, 
                                 fb_gpu_ptr_t b, int ldb, long long stride_b, 
                                 float beta, fb_gpu_ptr_t c, int ldc, long long stride_c, int batch_count);
    int (*dgemm_strided_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                                 double alpha, fb_gpu_ptr_t a, int lda, long long stride_a, 
                                 fb_gpu_ptr_t b, int ldb, long long stride_b, 
                                 double beta, fb_gpu_ptr_t c, int ldc, long long stride_c, int batch_count);
    int (*cgemm_strided_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                                 const void* alpha, fb_gpu_ptr_t a, int lda, long long stride_a, 
                                 fb_gpu_ptr_t b, int ldb, long long stride_b, 
                                 const void* beta, fb_gpu_ptr_t c, int ldc, long long stride_c, int batch_count);
    int (*zgemm_strided_batched)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, 
                                 const void* alpha, fb_gpu_ptr_t a, int lda, long long stride_a, 
                                 fb_gpu_ptr_t b, int ldb, long long stride_b, 
                                 const void* beta, fb_gpu_ptr_t c, int ldc, long long stride_c, int batch_count);
    
    /* Mixed precision GEMM (supports FP16, BF16, TF32, FP32, FP64) */
    int (*gemm_ex)(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k,
                   const void* alpha, fb_gpu_ptr_t a, int a_type, int lda,
                   fb_gpu_ptr_t b, int b_type, int ldb,
                   const void* beta, fb_gpu_ptr_t c, int c_type, int ldc,
                   int compute_type, int algo);
    
    /* Batched TRSM */
    int (*strsm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo, char transa, char diag, 
                         int m, int n, float alpha, fb_gpu_ptr_t* a_array, int lda, 
                         fb_gpu_ptr_t* b_array, int ldb, int batch_count);
    int (*dtrsm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo, char transa, char diag, 
                         int m, int n, double alpha, fb_gpu_ptr_t* a_array, int lda, 
                         fb_gpu_ptr_t* b_array, int ldb, int batch_count);
    int (*ctrsm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo, char transa, char diag, 
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda, 
                         fb_gpu_ptr_t* b_array, int ldb, int batch_count);
    int (*ztrsm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo, char transa, char diag, 
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda, 
                         fb_gpu_ptr_t* b_array, int ldb, int batch_count);
    
    /* Batched LU factorization */
    int (*sgetrf_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*dgetrf_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*cgetrf_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*zgetrf_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    
    /* Batched matrix inverse (via batched LU) */
    int (*sgetri_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*dgetri_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*cgetri_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    int (*zgetri_batched)(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t* a_array, int lda, 
                          fb_gpu_ptr_t ipiv_array, int* info_array, int batch_count);
    
    /* Batched SYMM/HEMM (symmetric/hermitian matrix-matrix multiplication) */
    int (*ssymm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, float alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, float beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    int (*dsymm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, double alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, double beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    int (*csymm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    int (*zsymm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    
    int (*chemm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    int (*zhemm_batched)(void* handle, fb_gpu_stream_t stream, char side, char uplo,
                         int m, int n, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                         int batch_count);
    
    /* Batched SYRK/HERK (symmetric/hermitian rank-k update) */
    int (*ssyrk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, float alpha, fb_gpu_ptr_t* a_array, int lda,
                         float beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*dsyrk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, double alpha, fb_gpu_ptr_t* a_array, int lda,
                         double beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*csyrk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         const void* beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*zsyrk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                         const void* beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    
    int (*cherk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, float alpha, fb_gpu_ptr_t* a_array, int lda,
                         float beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    int (*zherk_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                         int n, int k, double alpha, fb_gpu_ptr_t* a_array, int lda,
                         double beta, fb_gpu_ptr_t* c_array, int ldc, int batch_count);
    
    /* Batched SYR2K/HER2K (symmetric/hermitian rank-2k update) */
    int (*ssyr2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, float alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, float beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
    int (*dsyr2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, double alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, double beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
    int (*csyr2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
    int (*zsyr2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, const void* beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
    
    int (*cher2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, float beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
    int (*zher2k_batched)(void* handle, fb_gpu_stream_t stream, char uplo, char trans,
                          int n, int k, const void* alpha, fb_gpu_ptr_t* a_array, int lda,
                          fb_gpu_ptr_t* b_array, int ldb, double beta, fb_gpu_ptr_t* c_array, int ldc,
                          int batch_count);
};

/* ============================================================================
 * GPU Context Structure
 * ========================================================================== */

/**
 * @brief GPU execution context
 * 
 * Represents a single GPU device with its backend implementation.
 * Multiple contexts can coexist for multi-GPU systems.
 */
struct fb_gpu_context {
    fb_gpu_backend_type_t backend_type;    /**< Backend type (cuBLAS/rocBLAS/etc.) */
    int device_id;                         /**< Device index (0-based) */
    void* backend_handle;                  /**< Backend-specific handle (cuBLAS/rocBLAS handle) */
    void* device_handle;                   /**< Device handle (CUDA context, HIP device, etc.) */
    const fb_gpu_backend_trait_t* trait;   /**< Function table for this backend */
};

/* ============================================================================
 * Multi-GPU Manager
 * ========================================================================== */

/**
 * @brief Multi-GPU manager
 * 
 * Manages multiple GPU devices across different vendors.
 */
typedef struct {
    int num_devices;                /**< Total number of GPU devices */
    fb_gpu_context_t* contexts;     /**< Array of GPU contexts (one per device) */
} fb_gpu_manager_t;

/**
 * @brief Initialize GPU manager and detect all GPUs
 * @return GPU manager instance, or NULL on error
 */
fb_gpu_manager_t* fb_gpu_manager_init(void);

/**
 * @brief Shutdown GPU manager and release all resources
 * @param mgr GPU manager to shutdown
 */
void fb_gpu_manager_shutdown(fb_gpu_manager_t* mgr);

/**
 * @brief Get context for specific GPU device
 * @param mgr GPU manager
 * @param device_id Device index (0-based)
 * @return GPU context, or NULL if device_id is invalid
 */
fb_gpu_context_t* fb_gpu_get_context(fb_gpu_manager_t* mgr, int device_id);

/**
 * @brief Get number of available GPU devices
 * @param mgr GPU manager
 * @return Number of devices
 */
int fb_gpu_get_device_count(fb_gpu_manager_t* mgr);

/* ============================================================================
 * Backend Trait Instances (defined by each backend implementation)
 * ========================================================================== */

/**
 * @brief NVIDIA cuBLAS backend trait
 */
extern const fb_gpu_backend_trait_t fb_cublas_trait;

/**
 * @brief AMD rocBLAS backend trait
 */
extern const fb_gpu_backend_trait_t fb_rocblas_trait;

/**
 * @brief Intel oneAPI MKL backend trait
 */
extern const fb_gpu_backend_trait_t fb_onemkl_trait;

/**
 * @brief CLBlast OpenCL backend trait
 */
extern const fb_gpu_backend_trait_t fb_clblast_trait;

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_GPU_BACKEND_TRAIT_H */
