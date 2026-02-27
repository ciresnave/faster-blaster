/**
 * @file cublas_trait_impl.c
 * @brief NVIDIA cuBLAS implementation of GPU backend trait
 * 
 * Provides cuBLAS-specific implementation of the unified GPU backend trait interface.
 * This allows transparent use of NVIDIA GPUs alongside AMD or Intel GPUs.
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * cuBLAS-specific Context
 * ========================================================================== */

typedef struct {
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    int device_id;
} cublas_context_t;

/* ============================================================================
 * Lifecycle Management
 * ========================================================================== */

static int cublas_init(int device_id, void* lib_handle, void** backend_handle) {
    /* Note: lib_handle is unused for cuBLAS - CUDA libraries are statically linked */
    (void)lib_handle;  /* Suppress unused parameter warning */
    
    cudaError_t cuda_err;
    cublasStatus_t cublas_status;
    cusolverStatus_t cusolver_status;
    
    // Set device
    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(cuda_err));
        return -1;
    }
    
    // Allocate context
    cublas_context_t* ctx = (cublas_context_t*)malloc(sizeof(cublas_context_t));
    if (!ctx) {
        return -1;
    }
    ctx->device_id = device_id;
    
    // Create cuBLAS handle
    cublas_status = cublasCreate(&ctx->cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS: Failed to create handle: %d\n", cublas_status);
        free(ctx);
        return -1;
    }
    
    // Create cuSOLVER handle
    cusolver_status = cusolverDnCreate(&ctx->cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER: Failed to create handle: %d\n", cusolver_status);
        cublasDestroy(ctx->cublas_handle);
        free(ctx);
        return -1;
    }
    
    *backend_handle = ctx;
    return 0;
}

static void cublas_shutdown(void* backend_handle) {
    if (!backend_handle) {
        return;
    }
    
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    
    if (ctx->cublas_handle) {
        /* Ensure all pending GPU operations complete before destroying cuBLAS handle.
         * Without this, cublasDestroy may crash if async ops are still in flight. */
        cudaSetDevice(ctx->device_id);
        cudaDeviceSynchronize();
        cublasDestroy(ctx->cublas_handle);
    }
    if (ctx->cusolver_handle) {
        cusolverDnDestroy(ctx->cusolver_handle);
    }
    
    free(ctx);
}

static int cublas_get_device_properties(void* backend_handle, int device_id,
                                         char* name, size_t name_len,
                                         size_t* total_memory) {
    cudaError_t err;
    struct cudaDeviceProp prop;
    
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    
    if (total_memory) {
        *total_memory = prop.totalGlobalMem;
    }
    
    return 0;
}

/* ============================================================================
 * Memory Management
 * ========================================================================== */

static int cublas_malloc(void* backend_handle, fb_gpu_ptr_t* ptr, size_t size) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

static void cublas_free(void* backend_handle, fb_gpu_ptr_t ptr) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    
    cudaSetDevice(ctx->device_id);
    cudaFree(ptr);
}

static int cublas_memcpy_h2d(void* backend_handle, fb_gpu_ptr_t dst,
                              const void* src, size_t size) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

static int cublas_memcpy_d2h(void* backend_handle, void* dst,
                              fb_gpu_ptr_t src, size_t size) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

static int cublas_memcpy_d2d(void* backend_handle, fb_gpu_ptr_t dst,
                              fb_gpu_ptr_t src, size_t size) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaMemcpy D2D failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/* ============================================================================
 * Stream/Queue Management
 * ========================================================================== */

static int cublas_stream_create(void* backend_handle, fb_gpu_stream_t* stream) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaStreamCreate((cudaStream_t*)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

static void cublas_stream_destroy(void* backend_handle, fb_gpu_stream_t stream) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    
    cudaSetDevice(ctx->device_id);
    cudaStreamDestroy((cudaStream_t)stream);
}

static int cublas_stream_synchronize(void* backend_handle, fb_gpu_stream_t stream) {
    cublas_context_t* ctx = (cublas_context_t*)backend_handle;
    cudaError_t err;
    
    err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    if (stream) {
        err = cudaStreamSynchronize((cudaStream_t)stream);
    } else {
        err = cudaDeviceSynchronize();
    }
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/* ============================================================================
 * Enum Conversion Helpers
 * ========================================================================== */

static int cublas_convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return CUBLAS_OP_N;
        case 'T': case 't': return CUBLAS_OP_T;
        case 'C': case 'c': return CUBLAS_OP_C;
        default:
            fprintf(stderr, "cuBLAS: Invalid transpose flag '%c'\n", trans);
            return CUBLAS_OP_N;
    }
}

static int cublas_convert_uplo(char uplo) {
    switch (uplo) {
        case 'U': case 'u': return CUBLAS_FILL_MODE_UPPER;
        case 'L': case 'l': return CUBLAS_FILL_MODE_LOWER;
        default:
            fprintf(stderr, "cuBLAS: Invalid uplo flag '%c'\n", uplo);
            return CUBLAS_FILL_MODE_UPPER;
    }
}

static int cublas_convert_diag(char diag) {
    switch (diag) {
        case 'N': case 'n': return CUBLAS_DIAG_NON_UNIT;
        case 'U': case 'u': return CUBLAS_DIAG_UNIT;
        default:
            fprintf(stderr, "cuBLAS: Invalid diag flag '%c'\n", diag);
            return CUBLAS_DIAG_NON_UNIT;
    }
}

static int cublas_convert_side(char side) {
    switch (side) {
        case 'L': case 'l': return CUBLAS_SIDE_LEFT;
        case 'R': case 'r': return CUBLAS_SIDE_RIGHT;
        default:
            fprintf(stderr, "cuBLAS: Invalid side flag '%c'\n", side);
            return CUBLAS_SIDE_LEFT;
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Real (Single/Double Precision)
 * ========================================================================== */

static void cublas_saxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, float alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSaxpy(ctx->cublas_handle, n, &alpha, (const float*)x, incx, (float*)y, incy);
}

static void cublas_daxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, double alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDaxpy(ctx->cublas_handle, n, &alpha, (const double*)x, incx, (double*)y, incy);
}

static void cublas_sscal_impl(void* handle, fb_gpu_stream_t stream,
                               int n, float alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSscal(ctx->cublas_handle, n, &alpha, (float*)x, incx);
}

static void cublas_dscal_impl(void* handle, fb_gpu_stream_t stream,
                               int n, double alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDscal(ctx->cublas_handle, n, &alpha, (double*)x, incx);
}

static void cublas_scopy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasScopy(ctx->cublas_handle, n, (const float*)x, incx, (float*)y, incy);
}

static void cublas_dcopy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDcopy(ctx->cublas_handle, n, (const double*)x, incx, (double*)y, incy);
}

static void cublas_sswap_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSswap(ctx->cublas_handle, n, (float*)x, incx, (float*)y, incy);
}

static void cublas_dswap_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDswap(ctx->cublas_handle, n, (double*)x, incx, (double*)y, incy);
}

static float cublas_sdot_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    float result = 0.0f;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSdot(ctx->cublas_handle, n, (const float*)x, incx, (const float*)y, incy, &result);
    return result;
}

static double cublas_ddot_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    double result = 0.0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDdot(ctx->cublas_handle, n, (const double*)x, incx, (const double*)y, incy, &result);
    return result;
}

static float cublas_snrm2_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    float result = 0.0f;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSnrm2(ctx->cublas_handle, n, (const float*)x, incx, &result);
    return result;
}

static double cublas_dnrm2_impl(void* handle, fb_gpu_stream_t stream,
                                 int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    double result = 0.0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDnrm2(ctx->cublas_handle, n, (const double*)x, incx, &result);
    return result;
}

static float cublas_sasum_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    float result = 0.0f;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSasum(ctx->cublas_handle, n, (const float*)x, incx, &result);
    return result;
}

static double cublas_dasum_impl(void* handle, fb_gpu_stream_t stream,
                                 int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    double result = 0.0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDasum(ctx->cublas_handle, n, (const double*)x, incx, &result);
    return result;
}

static int cublas_isamax_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int result = 0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasIsamax(ctx->cublas_handle, n, (const float*)x, incx, &result);
    return result - 1; // cuBLAS returns 1-based index, convert to 0-based
}

static int cublas_idamax_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int result = 0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasIdamax(ctx->cublas_handle, n, (const double*)x, incx, &result);
    return result - 1; // cuBLAS returns 1-based index, convert to 0-based
}

/* ============================================================================
 * BLAS Level 1 Operations - Complex
 * ========================================================================== */

static void cublas_caxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCaxpy(ctx->cublas_handle, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (cuComplex*)y, incy);
}

static void cublas_zaxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZaxpy(ctx->cublas_handle, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
}

static void cublas_cscal_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCscal(ctx->cublas_handle, n, (const cuComplex*)alpha, (cuComplex*)x, incx);
}

static void cublas_zscal_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZscal(ctx->cublas_handle, n, (const cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx);
}

static void cublas_csscal_impl(void* handle, fb_gpu_stream_t stream,
                                int n, float alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsscal(ctx->cublas_handle, n, &alpha, (cuComplex*)x, incx);
}

static void cublas_zdscal_impl(void* handle, fb_gpu_stream_t stream,
                                int n, double alpha, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZdscal(ctx->cublas_handle, n, &alpha, (cuDoubleComplex*)x, incx);
}

static void cublas_ccopy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCcopy(ctx->cublas_handle, n, (const cuComplex*)x, incx, (cuComplex*)y, incy);
}

static void cublas_zcopy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZcopy(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
}

static void cublas_cswap_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCswap(ctx->cublas_handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy);
}

static void cublas_zswap_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZswap(ctx->cublas_handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
}

static void cublas_cdotu_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, void* result) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCdotu(ctx->cublas_handle, n, (const cuComplex*)x, incx,
                (const cuComplex*)y, incy, (cuComplex*)result);
}

static void cublas_zdotu_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, void* result) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZdotu(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)y, incy, (cuDoubleComplex*)result);
}

static void cublas_cdotc_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, void* result) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCdotc(ctx->cublas_handle, n, (const cuComplex*)x, incx,
                (const cuComplex*)y, incy, (cuComplex*)result);
}

static void cublas_zdotc_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, void* result) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZdotc(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)y, incy, (cuDoubleComplex*)result);
}

static float cublas_scnrm2_impl(void* handle, fb_gpu_stream_t stream,
                                 int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    float result = 0.0f;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasScnrm2(ctx->cublas_handle, n, (const cuComplex*)x, incx, &result);
    return result;
}

static double cublas_dznrm2_impl(void* handle, fb_gpu_stream_t stream,
                                  int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    double result = 0.0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDznrm2(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx, &result);
    return result;
}

static float cublas_scasum_impl(void* handle, fb_gpu_stream_t stream,
                                 int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    float result = 0.0f;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasScasum(ctx->cublas_handle, n, (const cuComplex*)x, incx, &result);
    return result;
}

static double cublas_dzasum_impl(void* handle, fb_gpu_stream_t stream,
                                  int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    double result = 0.0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDzasum(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx, &result);
    return result;
}

static int cublas_icamax_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int result = 0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasIcamax(ctx->cublas_handle, n, (const cuComplex*)x, incx, &result);
    return result - 1; // cuBLAS returns 1-based index
}

static int cublas_izamax_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int result = 0;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasIzamax(ctx->cublas_handle, n, (const cuDoubleComplex*)x, incx, &result);
    return result - 1; // cuBLAS returns 1-based index
}

/* ============================================================================
 * BLAS Level 1 Operations - Rotation
 * ========================================================================== */

static void cublas_srotg_impl(void* handle, fb_gpu_stream_t stream,
                               fb_gpu_ptr_t a, fb_gpu_ptr_t b,
                               fb_gpu_ptr_t c, fb_gpu_ptr_t s) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSrotg(ctx->cublas_handle, (float*)a, (float*)b, (float*)c, (float*)s);
}

static void cublas_drotg_impl(void* handle, fb_gpu_stream_t stream,
                               fb_gpu_ptr_t a, fb_gpu_ptr_t b,
                               fb_gpu_ptr_t c, fb_gpu_ptr_t s) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDrotg(ctx->cublas_handle, (double*)a, (double*)b, (double*)c, (double*)s);
}

static void cublas_srot_impl(void* handle, fb_gpu_stream_t stream, int n,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                              float c, float s) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSrot(ctx->cublas_handle, n, (float*)x, incx, (float*)y, incy, &c, &s);
}

static void cublas_drot_impl(void* handle, fb_gpu_stream_t stream, int n,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                              double c, double s) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDrot(ctx->cublas_handle, n, (double*)x, incx, (double*)y, incy, &c, &s);
}

static void cublas_srotm_impl(void* handle, fb_gpu_stream_t stream, int n,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t param) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSrotm(ctx->cublas_handle, n, (float*)x, incx, (float*)y, incy, (const float*)param);
}

static void cublas_drotm_impl(void* handle, fb_gpu_stream_t stream, int n,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t param) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDrotm(ctx->cublas_handle, n, (double*)x, incx, (double*)y, incy, (const double*)param);
}

static void cublas_srotmg_impl(void* handle, fb_gpu_stream_t stream,
                                fb_gpu_ptr_t d1, fb_gpu_ptr_t d2,
                                fb_gpu_ptr_t x1, fb_gpu_ptr_t y1,
                                fb_gpu_ptr_t param) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSrotmg(ctx->cublas_handle, (float*)d1, (float*)d2, (float*)x1,
                 (const float*)y1, (float*)param);
}

static void cublas_drotmg_impl(void* handle, fb_gpu_stream_t stream,
                                fb_gpu_ptr_t d1, fb_gpu_ptr_t d2,
                                fb_gpu_ptr_t x1, fb_gpu_ptr_t y1,
                                fb_gpu_ptr_t param) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDrotmg(ctx->cublas_handle, (double*)d1, (double*)d2, (double*)x1,
                 (const double*)y1, (double*)param);
}

/* ============================================================================
 * BLAS Level 2 Operations - General Matrix-Vector Multiplication
 * ========================================================================== */

static void cublas_sgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, float beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSgemv(ctx->cublas_handle, op, m, n, &alpha,
                (const float*)a, lda, (const float*)x, incx,
                &beta, (float*)y, incy);
}

static void cublas_dgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, double alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, double beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDgemv(ctx->cublas_handle, op, m, n, &alpha,
                (const double*)a, lda, (const double*)x, incx,
                &beta, (double*)y, incy);
}

static void cublas_cgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCgemv(ctx->cublas_handle, op, m, n, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZgemv(ctx->cublas_handle, op, m, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

/* ============================================================================
 * BLAS Level 2 Operations - General Banded Matrix-Vector Multiplication
 * ========================================================================== */

static void cublas_sgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, int kl, int ku, float alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               float beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSgbmv(ctx->cublas_handle, op, m, n, kl, ku, &alpha,
                (const float*)a, lda, (const float*)x, incx,
                &beta, (float*)y, incy);
}

static void cublas_dgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, int kl, int ku, double alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               double beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDgbmv(ctx->cublas_handle, op, m, n, kl, ku, &alpha,
                (const double*)a, lda, (const double*)x, incx,
                &beta, (double*)y, incy);
}

static void cublas_cgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, int kl, int ku, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCgbmv(ctx->cublas_handle, op, m, n, kl, ku, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int m, int n, int kl, int ku, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZgbmv(ctx->cublas_handle, op, m, n, kl, ku, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

/* ============================================================================
 * BLAS Level 2 Operations - Hermitian/Symmetric Matrix-Vector Multiplication
 * ========================================================================== */

static void cublas_chemv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasChemv(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zhemv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZhemv(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

static void cublas_ssymv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, float beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsymv(ctx->cublas_handle, fill, n, &alpha,
                (const float*)a, lda, (const float*)x, incx,
                &beta, (float*)y, incy);
}

static void cublas_dsymv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, double alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, double beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsymv(ctx->cublas_handle, fill, n, &alpha,
                (const double*)a, lda, (const double*)x, incx,
                &beta, (double*)y, incy);
}

static void cublas_csymv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsymv(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zsymv_impl(void* handle, fb_gpu_stream_t stream, char uplo,
                               int n, const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx, const void* beta,
                               fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZsymv(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

/* ============================================================================
 * BLAS Level 2 Operations - Triangular Matrix-Vector Multiplication
 * ========================================================================== */

static void cublas_strmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasStrmv(ctx->cublas_handle, fill, op, diag_type, n,
                (const float*)a, lda, (float*)x, incx);
}

static void cublas_dtrmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDtrmv(ctx->cublas_handle, fill, op, diag_type, n,
                (const double*)a, lda, (double*)x, incx);
}

static void cublas_ctrmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCtrmv(ctx->cublas_handle, fill, op, diag_type, n,
                (const cuComplex*)a, lda, (cuComplex*)x, incx);
}

static void cublas_ztrmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZtrmv(ctx->cublas_handle, fill, op, diag_type, n,
                (const cuDoubleComplex*)a, lda, (cuDoubleComplex*)x, incx);
}

/* ============================================================================
 * BLAS Level 2 Operations - Triangular Solve
 * ========================================================================== */

static void cublas_strsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasStrsv(ctx->cublas_handle, fill, op, diag_type, n,
                (const float*)a, lda, (float*)x, incx);
}

static void cublas_dtrsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDtrsv(ctx->cublas_handle, fill, op, diag_type, n,
                (const double*)a, lda, (double*)x, incx);
}

static void cublas_ctrsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCtrsv(ctx->cublas_handle, fill, op, diag_type, n,
                (const cuComplex*)a, lda, (cuComplex*)x, incx);
}

static void cublas_ztrsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag,
                               int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZtrsv(ctx->cublas_handle, fill, op, diag_type, n,
                (const cuDoubleComplex*)a, lda, (cuDoubleComplex*)x, incx);
}

/* ============================================================================
 * BLAS Level 2 Operations - Rank-1 Update
 * ========================================================================== */

static void cublas_sger_impl(void* handle, fb_gpu_stream_t stream,
                              int m, int n, float alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                              fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSger(ctx->cublas_handle, m, n, &alpha,
               (const float*)x, incx, (const float*)y, incy,
               (float*)a, lda);
}

static void cublas_dger_impl(void* handle, fb_gpu_stream_t stream,
                              int m, int n, double alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                              fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDger(ctx->cublas_handle, m, n, &alpha,
               (const double*)x, incx, (const double*)y, incy,
               (double*)a, lda);
}

static void cublas_cgeru_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCgeru(ctx->cublas_handle, m, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (const cuComplex*)y, incy,
                (cuComplex*)a, lda);
}

static void cublas_zgeru_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZgeru(ctx->cublas_handle, m, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy,
                (cuDoubleComplex*)a, lda);
}

static void cublas_cgerc_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCgerc(ctx->cublas_handle, m, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (const cuComplex*)y, incy,
                (cuComplex*)a, lda);
}

static void cublas_zgerc_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZgerc(ctx->cublas_handle, m, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy,
                (cuDoubleComplex*)a, lda);
}

/* ============================================================================
 * BLAS Level 2 Operations - Hermitian/Symmetric Rank-1 Update
 * ========================================================================== */

static void cublas_cher_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, float alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCher(ctx->cublas_handle, fill, n, &alpha,
               (const cuComplex*)x, incx, (cuComplex*)a, lda);
}

static void cublas_zher_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, double alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZher(ctx->cublas_handle, fill, n, &alpha,
               (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)a, lda);
}

static void cublas_ssyr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, float alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsyr(ctx->cublas_handle, fill, n, &alpha,
               (const float*)x, incx, (float*)a, lda);
}

static void cublas_dsyr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, double alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsyr(ctx->cublas_handle, fill, n, &alpha,
               (const double*)x, incx, (double*)a, lda);
}

static void cublas_csyr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, const void* alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsyr(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
               (const cuComplex*)x, incx, (cuComplex*)a, lda);
}

static void cublas_zsyr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, const void* alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZsyr(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
               (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)a, lda);
}

/* ============================================================================
 * BLAS Level 2 Operations - Hermitian/Symmetric Rank-2 Update
 * ========================================================================== */

static void cublas_cher2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCher2(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (const cuComplex*)y, incy,
                (cuComplex*)a, lda);
}

static void cublas_zher2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZher2(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy,
                (cuDoubleComplex*)a, lda);
}

static void cublas_ssyr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, float alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsyr2(ctx->cublas_handle, fill, n, &alpha,
                (const float*)x, incx, (const float*)y, incy,
                (float*)a, lda);
}

static void cublas_dsyr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, double alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsyr2(ctx->cublas_handle, fill, n, &alpha,
                (const double*)x, incx, (const double*)y, incy,
                (double*)a, lda);
}

/* ============================================================================
 * BLAS Level 2 Operations - Banded Matrix Operations (New)
 * ========================================================================== */

static void cublas_ssbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int k, float alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               float beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasSsbmv(ctx->cublas_handle, fill, n, k, &alpha,
                (const float*)a, lda, (const float*)x, incx,
                &beta, (float*)y, incy);
}

static void cublas_dsbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int k, double alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               double beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDsbmv(ctx->cublas_handle, fill, n, k, &alpha,
                (const double*)a, lda, (const double*)x, incx,
                &beta, (double*)y, incy);
}

static void cublas_chbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int k, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasChbmv(ctx->cublas_handle, fill, n, k, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zhbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int k, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZhbmv(ctx->cublas_handle, fill, n, k, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

static void cublas_stbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasStbmv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const float*)a, lda, (float*)x, incx);
}

static void cublas_dtbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDtbmv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const double*)a, lda, (double*)x, incx);
}

static void cublas_ctbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasCtbmv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const cuComplex*)a, lda, (cuComplex*)x, incx);
}

static void cublas_ztbmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZtbmv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const cuDoubleComplex*)a, lda, (cuDoubleComplex*)x, incx);
}

static void cublas_stbsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasStbsv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const float*)a, lda, (float*)x, incx);
}

static void cublas_dtbsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDtbsv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const double*)a, lda, (double*)x, incx);
}

static void cublas_ctbsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasCtbsv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const cuComplex*)a, lda, (cuComplex*)x, incx);
}

static void cublas_ztbsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n, int k,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZtbsv(ctx->cublas_handle, fill, op, diagtype, n, k,
                (const cuDoubleComplex*)a, lda, (cuDoubleComplex*)x, incx);
}

/* ============================================================================
 * BLAS Level 2 Operations - Packed Matrix Operations
 * ========================================================================== */

static void cublas_sspmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, float alpha,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                               float beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasSspmv(ctx->cublas_handle, fill, n, &alpha,
                (const float*)ap, (const float*)x, incx,
                &beta, (float*)y, incy);
}

static void cublas_dspmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, double alpha,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                               double beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDspmv(ctx->cublas_handle, fill, n, &alpha,
                (const double*)ap, (const double*)x, incx,
                &beta, (double*)y, incy);
}

static void cublas_chpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasChpmv(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)ap, (const cuComplex*)x, incx,
                (const cuComplex*)beta, (cuComplex*)y, incy);
}

static void cublas_zhpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx,
                               const void* beta, fb_gpu_ptr_t y, int incy) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZhpmv(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)ap, (const cuDoubleComplex*)x, incx,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
}

static void cublas_stpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasStpmv(ctx->cublas_handle, fill, op, diagtype, n,
                (const float*)ap, (float*)x, incx);
}

static void cublas_dtpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDtpmv(ctx->cublas_handle, fill, op, diagtype, n,
                (const double*)ap, (double*)x, incx);
}

static void cublas_ctpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasCtpmv(ctx->cublas_handle, fill, op, diagtype, n,
                (const cuComplex*)ap, (cuComplex*)x, incx);
}

static void cublas_ztpmv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZtpmv(ctx->cublas_handle, fill, op, diagtype, n,
                (const cuDoubleComplex*)ap, (cuDoubleComplex*)x, incx);
}

static void cublas_stpsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasStpsv(ctx->cublas_handle, fill, op, diagtype, n,
                (const float*)ap, (float*)x, incx);
}

static void cublas_dtpsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDtpsv(ctx->cublas_handle, fill, op, diagtype, n,
                (const double*)ap, (double*)x, incx);
}

static void cublas_ctpsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasCtpsv(ctx->cublas_handle, fill, op, diagtype, n,
                (const cuComplex*)ap, (cuComplex*)x, incx);
}

static void cublas_ztpsv_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, char diag, int n,
                               fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    cublasDiagType_t diagtype = (cublasDiagType_t)cublas_convert_diag(diag);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZtpsv(ctx->cublas_handle, fill, op, diagtype, n,
                (const cuDoubleComplex*)ap, (cuDoubleComplex*)x, incx);
}

/* ============================================================================
 * BLAS Level 2 Operations - Packed Rank Updates
 * ========================================================================== */

static void cublas_sspr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, float alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasSspr(ctx->cublas_handle, fill, n, &alpha,
               (const float*)x, incx, (float*)ap);
}

static void cublas_dspr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, double alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDspr(ctx->cublas_handle, fill, n, &alpha,
               (const double*)x, incx, (double*)ap);
}

static void cublas_chpr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, float alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasChpr(ctx->cublas_handle, fill, n, &alpha,
               (const cuComplex*)x, incx, (cuComplex*)ap);
}

static void cublas_zhpr_impl(void* handle, fb_gpu_stream_t stream,
                              char uplo, int n, double alpha,
                              fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZhpr(ctx->cublas_handle, fill, n, &alpha,
               (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)ap);
}

static void cublas_sspr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, float alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasSspr2(ctx->cublas_handle, fill, n, &alpha,
                (const float*)x, incx, (const float*)y, incy, (float*)ap);
}

static void cublas_dspr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, double alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasDspr2(ctx->cublas_handle, fill, n, &alpha,
                (const double*)x, incx, (const double*)y, incy, (double*)ap);
}

static void cublas_chpr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasChpr2(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (const cuComplex*)y, incy,
                (cuComplex*)ap);
}

static void cublas_zhpr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t ap) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZhpr2(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy,
                (cuDoubleComplex*)ap);
}

/* ============================================================================
 * BLAS Level 2 Operations - Complex Symmetric Rank-2 Update
 * ========================================================================== */

static void cublas_csyr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasCsyr2(ctx->cublas_handle, fill, n, (const cuComplex*)alpha,
                (const cuComplex*)x, incx, (const cuComplex*)y, incy,
                (cuComplex*)a, lda);
}

static void cublas_zsyr2_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, const void* alpha,
                               fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    if (stream) cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    
    cublasZsyr2(ctx->cublas_handle, fill, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy,
                (cuDoubleComplex*)a, lda);
}

/* ============================================================================
 * BLAS Level 3 Operations - General Matrix-Matrix Multiplication
 * ========================================================================== */

static void cublas_sgemm_impl(void* handle, fb_gpu_stream_t stream,
                               char transa, char transb, int m, int n, int k,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, float beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t opA = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasOperation_t opB = (cublasOperation_t)cublas_convert_transpose(transb);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSgemm(ctx->cublas_handle, opA, opB, m, n, k, &alpha,
                (const float*)a, lda, (const float*)b, ldb,
                &beta, (float*)c, ldc);
}

static void cublas_dgemm_impl(void* handle, fb_gpu_stream_t stream,
                               char transa, char transb, int m, int n, int k,
                               double alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, double beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t opA = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasOperation_t opB = (cublasOperation_t)cublas_convert_transpose(transb);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDgemm(ctx->cublas_handle, opA, opB, m, n, k, &alpha,
                (const double*)a, lda, (const double*)b, ldb,
                &beta, (double*)c, ldc);
}

static void cublas_cgemm_impl(void* handle, fb_gpu_stream_t stream,
                               char transa, char transb, int m, int n, int k,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t opA = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasOperation_t opB = (cublasOperation_t)cublas_convert_transpose(transb);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCgemm(ctx->cublas_handle, opA, opB, m, n, k, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)b, ldb,
                (const cuComplex*)beta, (cuComplex*)c, ldc);
}

static void cublas_zgemm_impl(void* handle, fb_gpu_stream_t stream,
                               char transa, char transb, int m, int n, int k,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t opA = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasOperation_t opB = (cublasOperation_t)cublas_convert_transpose(transb);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZgemm(ctx->cublas_handle, opA, opB, m, n, k, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)b, ldb,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)c, ldc);
}

/* ============================================================================
 * BLAS Level 3 Operations - Symmetric Matrix-Matrix Multiplication
 * ========================================================================== */

static void cublas_ssymm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, float beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsymm(ctx->cublas_handle, side_mode, fill, m, n, &alpha,
                (const float*)a, lda, (const float*)b, ldb,
                &beta, (float*)c, ldc);
}

static void cublas_dsymm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               double alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, double beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsymm(ctx->cublas_handle, side_mode, fill, m, n, &alpha,
                (const double*)a, lda, (const double*)b, ldb,
                &beta, (double*)c, ldc);
}

static void cublas_csymm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsymm(ctx->cublas_handle, side_mode, fill, m, n, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)b, ldb,
                (const cuComplex*)beta, (cuComplex*)c, ldc);
}

static void cublas_zsymm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZsymm(ctx->cublas_handle, side_mode, fill, m, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)b, ldb,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)c, ldc);
}

/* ============================================================================
 * BLAS Level 3 Operations - Hermitian Matrix-Matrix Multiplication
 * ========================================================================== */

static void cublas_chemm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasChemm(ctx->cublas_handle, side_mode, fill, m, n, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)b, ldb,
                (const cuComplex*)beta, (cuComplex*)c, ldc);
}

static void cublas_zhemm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, int m, int n,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, const void* beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZhemm(ctx->cublas_handle, side_mode, fill, m, n, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)b, ldb,
                (const cuDoubleComplex*)beta, (cuDoubleComplex*)c, ldc);
}

/* ============================================================================
 * BLAS Level 3 Operations - Triangular Matrix-Matrix Multiplication
 * ========================================================================== */

static void cublas_strmm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, float alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasStrmm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n, &alpha,
                (const float*)a, lda, (float*)b, ldb, (float*)b, ldb);
}

static void cublas_dtrmm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, double alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDtrmm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n, &alpha,
                (const double*)a, lda, (double*)b, ldb, (double*)b, ldb);
}

static void cublas_ctrmm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCtrmm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n,
                (const cuComplex*)alpha, (const cuComplex*)a, lda,
                (cuComplex*)b, ldb, (cuComplex*)b, ldb);
}

static void cublas_ztrmm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZtrmm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n,
                (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)a, lda,
                (cuDoubleComplex*)b, ldb, (cuDoubleComplex*)b, ldb);
}

/* ============================================================================
 * BLAS Level 3 Operations - Triangular Solve with Multiple RHS
 * ========================================================================== */

static void cublas_strsm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, float alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasStrsm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n, &alpha,
                (const float*)a, lda, (float*)b, ldb);
}

static void cublas_dtrsm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, double alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDtrsm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n, &alpha,
                (const double*)a, lda, (double*)b, ldb);
}

static void cublas_ctrsm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCtrsm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n,
                (const cuComplex*)alpha, (const cuComplex*)a, lda,
                (cuComplex*)b, ldb);
}

static void cublas_ztrsm_impl(void* handle, fb_gpu_stream_t stream,
                               char side, char uplo, char transa, char diag,
                               int m, int n, const void* alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasSideMode_t side_mode = (cublasSideMode_t)cublas_convert_side(side);
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(transa);
    cublasDiagType_t diag_type = (cublasDiagType_t)cublas_convert_diag(diag);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZtrsm(ctx->cublas_handle, side_mode, fill, op, diag_type, m, n,
                (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)a, lda,
                (cuDoubleComplex*)b, ldb);
}

/* ============================================================================
 * BLAS Level 3 Operations - Symmetric/Hermitian Rank-k Update
 * ========================================================================== */

static void cublas_ssyrk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               float beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsyrk(ctx->cublas_handle, fill, op, n, k, &alpha,
                (const float*)a, lda, &beta, (float*)c, ldc);
}

static void cublas_dsyrk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               double alpha, fb_gpu_ptr_t a, int lda,
                               double beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsyrk(ctx->cublas_handle, fill, op, n, k, &alpha,
                (const double*)a, lda, &beta, (double*)c, ldc);
}

static void cublas_csyrk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               const void* beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsyrk(ctx->cublas_handle, fill, op, n, k, (const cuComplex*)alpha,
                (const cuComplex*)a, lda, (const cuComplex*)beta, (cuComplex*)c, ldc);
}

static void cublas_zsyrk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               const void* alpha, fb_gpu_ptr_t a, int lda,
                               const void* beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZsyrk(ctx->cublas_handle, fill, op, n, k, (const cuDoubleComplex*)alpha,
                (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)beta,
                (cuDoubleComplex*)c, ldc);
}

static void cublas_cherk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               float beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCherk(ctx->cublas_handle, fill, op, n, k, &alpha,
                (const cuComplex*)a, lda, &beta, (cuComplex*)c, ldc);
}

static void cublas_zherk_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, char trans, int n, int k,
                               double alpha, fb_gpu_ptr_t a, int lda,
                               double beta, fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZherk(ctx->cublas_handle, fill, op, n, k, &alpha,
                (const cuDoubleComplex*)a, lda, &beta, (cuDoubleComplex*)c, ldc);
}

/* ============================================================================
 * BLAS Level 3 Operations - Symmetric/Hermitian Rank-2k Update
 * ========================================================================== */

static void cublas_ssyr2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                float alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, float beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasSsyr2k(ctx->cublas_handle, fill, op, n, k, &alpha,
                 (const float*)a, lda, (const float*)b, ldb,
                 &beta, (float*)c, ldc);
}

static void cublas_dsyr2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                double alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, double beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasDsyr2k(ctx->cublas_handle, fill, op, n, k, &alpha,
                 (const double*)a, lda, (const double*)b, ldb,
                 &beta, (double*)c, ldc);
}

static void cublas_csyr2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                const void* alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, const void* beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCsyr2k(ctx->cublas_handle, fill, op, n, k, (const cuComplex*)alpha,
                 (const cuComplex*)a, lda, (const cuComplex*)b, ldb,
                 (const cuComplex*)beta, (cuComplex*)c, ldc);
}

static void cublas_zsyr2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                const void* alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, const void* beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZsyr2k(ctx->cublas_handle, fill, op, n, k, (const cuDoubleComplex*)alpha,
                 (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)b, ldb,
                 (const cuDoubleComplex*)beta, (cuDoubleComplex*)c, ldc);
}

static void cublas_cher2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                const void* alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, float beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasCher2k(ctx->cublas_handle, fill, op, n, k, (const cuComplex*)alpha,
                 (const cuComplex*)a, lda, (const cuComplex*)b, ldb,
                 &beta, (cuComplex*)c, ldc);
}

static void cublas_zher2k_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                const void* alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, double beta,
                                fb_gpu_ptr_t c, int ldc) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    
    if (stream) {
        cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);
    }
    
    cublasZher2k(ctx->cublas_handle, fill, op, n, k, (const cuDoubleComplex*)alpha,
                 (const cuDoubleComplex*)a, lda, (const cuDoubleComplex*)b, ldb,
                 &beta, (cuDoubleComplex*)c, ldc);
}

/* ============================================================================
 * cuSOLVER LAPACK - Linear Systems (LU Factorization)
 * ========================================================================== */

static int cublas_sgetrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    float* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    // Query workspace size
    status = cusolverDnSgetrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (float*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    // Allocate workspace and device info
    cudaMalloc((void**)&workspace, lwork * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    // Perform LU factorization
    status = cusolverDnSgetrf(ctx->cusolver_handle, m, n, (float*)a, lda,
                               workspace, (int*)ipiv, devInfo);
    
    // Check for errors
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dgetrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    double* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnDgetrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (double*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDgetrf(ctx->cusolver_handle, m, n, (double*)a, lda,
                               workspace, (int*)ipiv, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cgetrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnCgetrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (cuComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCgetrf(ctx->cusolver_handle, m, n, (cuComplex*)a, lda,
                               workspace, (int*)ipiv, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zgetrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuDoubleComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnZgetrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (cuDoubleComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZgetrf(ctx->cusolver_handle, m, n, (cuDoubleComplex*)a, lda,
                               workspace, (int*)ipiv, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* LU Solve (getrs) */
static int cublas_sgetrs_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSgetrs(ctx->cusolver_handle, op, n, nrhs,
                              (const float*)a, lda, (const int*)ipiv,
                              (float*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dgetrs_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDgetrs(ctx->cusolver_handle, op, n, nrhs,
                              (const double*)a, lda, (const int*)ipiv,
                              (double*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cgetrs_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCgetrs(ctx->cusolver_handle, op, n, nrhs,
                              (const cuComplex*)a, lda, (const int*)ipiv,
                              (cuComplex*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zgetrs_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasOperation_t op = (cublasOperation_t)cublas_convert_transpose(trans);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZgetrs(ctx->cusolver_handle, op, n, nrhs,
                              (const cuDoubleComplex*)a, lda, (const int*)ipiv,
                              (cuDoubleComplex*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* ============================================================================
 * cuSOLVER LAPACK - Cholesky Factorization
 * ========================================================================== */

static int cublas_spotrf_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int lwork;
    float* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnSpotrf_bufferSize(ctx->cusolver_handle, fill, n,
                                         (float*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSpotrf(ctx->cusolver_handle, fill, n, (float*)a, lda,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dpotrf_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int lwork;
    double* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnDpotrf_bufferSize(ctx->cusolver_handle, fill, n,
                                         (double*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDpotrf(ctx->cusolver_handle, fill, n, (double*)a, lda,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cpotrf_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int lwork;
    cuComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnCpotrf_bufferSize(ctx->cusolver_handle, fill, n,
                                         (cuComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCpotrf(ctx->cusolver_handle, fill, n, (cuComplex*)a, lda,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zpotrf_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, fb_gpu_ptr_t a, int lda) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int lwork;
    cuDoubleComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnZpotrf_bufferSize(ctx->cusolver_handle, fill, n,
                                         (cuDoubleComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZpotrf(ctx->cusolver_handle, fill, n, (cuDoubleComplex*)a, lda,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* Cholesky Solve (potrs) */
static int cublas_spotrs_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSpotrs(ctx->cusolver_handle, fill, n, nrhs,
                              (const float*)a, lda, (float*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dpotrs_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDpotrs(ctx->cusolver_handle, fill, n, nrhs,
                              (const double*)a, lda, (double*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cpotrs_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCpotrs(ctx->cusolver_handle, fill, n, nrhs,
                              (const cuComplex*)a, lda, (cuComplex*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zpotrs_impl(void* handle, fb_gpu_stream_t stream,
                               char uplo, int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZpotrs(ctx->cusolver_handle, fill, n, nrhs,
                              (const cuDoubleComplex*)a, lda,
                              (cuDoubleComplex*)b, ldb, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* ============================================================================
 * cuSOLVER LAPACK - QR Factorization
 * ========================================================================== */

static int cublas_sgeqrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t tau) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    float* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnSgeqrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (float*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSgeqrf(ctx->cusolver_handle, m, n, (float*)a, lda,
                              (float*)tau, workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dgeqrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t tau) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    double* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnDgeqrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (double*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDgeqrf(ctx->cusolver_handle, m, n, (double*)a, lda,
                              (double*)tau, workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cgeqrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t tau) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnCgeqrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (cuComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCgeqrf(ctx->cusolver_handle, m, n, (cuComplex*)a, lda,
                              (cuComplex*)tau, workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zgeqrf_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t tau) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuDoubleComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnZgeqrf_bufferSize(ctx->cusolver_handle, m, n,
                                         (cuDoubleComplex*)a, lda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZgeqrf(ctx->cusolver_handle, m, n, (cuDoubleComplex*)a, lda,
                              (cuDoubleComplex*)tau, workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* ============================================================================
 * cuSOLVER LAPACK - Singular Value Decomposition (SVD)
 * ========================================================================== */

static int cublas_sgesvd_impl(void* handle, fb_gpu_stream_t stream,
                               char jobu, char jobvt, int m, int n,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                               fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    float* workspace;
    int* devInfo;
    float* rwork = NULL;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnSgesvd_bufferSize(ctx->cusolver_handle, m, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSgesvd(ctx->cusolver_handle, jobu, jobvt, m, n,
                              (float*)a, lda, (float*)s,
                              (float*)u, ldu, (float*)vt, ldvt,
                              workspace, lwork, rwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dgesvd_impl(void* handle, fb_gpu_stream_t stream,
                               char jobu, char jobvt, int m, int n,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                               fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    double* workspace;
    int* devInfo;
    double* rwork = NULL;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnDgesvd_bufferSize(ctx->cusolver_handle, m, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDgesvd(ctx->cusolver_handle, jobu, jobvt, m, n,
                              (double*)a, lda, (double*)s,
                              (double*)u, ldu, (double*)vt, ldvt,
                              workspace, lwork, rwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cgesvd_impl(void* handle, fb_gpu_stream_t stream,
                               char jobu, char jobvt, int m, int n,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                               fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuComplex* workspace;
    int* devInfo;
    float* rwork;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnCgesvd_bufferSize(ctx->cusolver_handle, m, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    int minmn = (m < n) ? m : n;
    cudaMalloc((void**)&workspace, lwork * sizeof(cuComplex));
    cudaMalloc((void**)&rwork, 5 * minmn * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCgesvd(ctx->cusolver_handle, jobu, jobvt, m, n,
                              (cuComplex*)a, lda, (float*)s,
                              (cuComplex*)u, ldu, (cuComplex*)vt, ldvt,
                              workspace, lwork, rwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(rwork);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zgesvd_impl(void* handle, fb_gpu_stream_t stream,
                               char jobu, char jobvt, int m, int n,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                               fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    int lwork;
    cuDoubleComplex* workspace;
    int* devInfo;
    double* rwork;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnZgesvd_bufferSize(ctx->cusolver_handle, m, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    int minmn = (m < n) ? m : n;
    cudaMalloc((void**)&workspace, lwork * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&rwork, 5 * minmn * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZgesvd(ctx->cusolver_handle, jobu, jobvt, m, n,
                              (cuDoubleComplex*)a, lda, (double*)s,
                              (cuDoubleComplex*)u, ldu, (cuDoubleComplex*)vt, ldvt,
                              workspace, lwork, rwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(rwork);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* ============================================================================
 * cuSOLVER LAPACK - Eigenvalue Decomposition (Symmetric/Hermitian)
 * ========================================================================== */

static int cublas_ssyev_impl(void* handle, fb_gpu_stream_t stream,
                              char jobz, char uplo, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cusolverEigMode_t jobmode = (jobz == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    int lwork;
    float* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnSsyevd_bufferSize(ctx->cusolver_handle, jobmode, fill, n,
                                         (float*)a, lda, (float*)w, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnSsyevd(ctx->cusolver_handle, jobmode, fill, n,
                              (float*)a, lda, (float*)w,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_dsyev_impl(void* handle, fb_gpu_stream_t stream,
                              char jobz, char uplo, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cusolverEigMode_t jobmode = (jobz == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    int lwork;
    double* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnDsyevd_bufferSize(ctx->cusolver_handle, jobmode, fill, n,
                                         (double*)a, lda, (double*)w, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(double));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnDsyevd(ctx->cusolver_handle, jobmode, fill, n,
                              (double*)a, lda, (double*)w,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_cheev_impl(void* handle, fb_gpu_stream_t stream,
                              char jobz, char uplo, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cusolverEigMode_t jobmode = (jobz == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    int lwork;
    cuComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnCheevd_bufferSize(ctx->cusolver_handle, jobmode, fill, n,
                                         (cuComplex*)a, lda, (float*)w, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnCheevd(ctx->cusolver_handle, jobmode, fill, n,
                              (cuComplex*)a, lda, (float*)w,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

static int cublas_zheev_impl(void* handle, fb_gpu_stream_t stream,
                              char jobz, char uplo, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    cublas_context_t* ctx = (cublas_context_t*)handle;
    cublasFillMode_t fill = (cublasFillMode_t)cublas_convert_uplo(uplo);
    cusolverEigMode_t jobmode = (jobz == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    int lwork;
    cuDoubleComplex* workspace;
    int* devInfo;
    cusolverStatus_t status;
    
    if (stream) {
        cusolverDnSetStream(ctx->cusolver_handle, (cudaStream_t)stream);
    }
    
    status = cusolverDnZheevd_bufferSize(ctx->cusolver_handle, jobmode, fill, n,
                                         (cuDoubleComplex*)a, lda, (double*)w, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;
    
    cudaMalloc((void**)&workspace, lwork * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devInfo, sizeof(int));
    
    status = cusolverDnZheevd(ctx->cusolver_handle, jobmode, fill, n,
                              (cuDoubleComplex*)a, lda, (double*)w,
                              workspace, lwork, devInfo);
    
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(devInfo);
    
    return (status == CUSOLVER_STATUS_SUCCESS && info == 0) ? 0 : -1;
}

/* Note: Core LAPACK operations implemented above demonstrate the complete pattern:
 * - LU factorization (getrf/getrs) - 8 functions
 * - Cholesky factorization (potrf/potrs) - 8 functions  
 * - QR factorization (geqrf) - 4 functions
 * - SVD (gesvd) - 4 functions
 * - Eigenvalue decomposition (syev/heev) - 4 functions
 * Total: 28 core LAPACK operations implemented
 * 
 * Additional operations (gesv, orgqr, ormqr, getri, potri, etc.) follow identical patterns
 * using cusolverDn API with workspace buffer queries.
 */

/* ============================================================================
 * cuBLAS Backend Trait Instance
 * ========================================================================== */

const fb_gpu_backend_trait_t fb_cublas_trait = {
    .name = "NVIDIA cuBLAS",
    .type = FB_GPU_BACKEND_CUBLAS,
    
    // Lifecycle
    .init = cublas_init,
    .shutdown = cublas_shutdown,
    .get_device_properties = cublas_get_device_properties,
    
    // Memory
    .malloc = cublas_malloc,
    .free = cublas_free,
    .memcpy_h2d = cublas_memcpy_h2d,
    .memcpy_d2h = cublas_memcpy_d2h,
    .memcpy_d2d = cublas_memcpy_d2d,
    
    // Streams
    .stream_create = cublas_stream_create,
    .stream_destroy = cublas_stream_destroy,
    .stream_synchronize = cublas_stream_synchronize,
    
    // Enum conversions
    .convert_transpose = cublas_convert_transpose,
    .convert_uplo = cublas_convert_uplo,
    .convert_diag = cublas_convert_diag,
    .convert_side = cublas_convert_side,
    
    // Level 1 BLAS - Real
    .saxpy = cublas_saxpy_impl,
    .daxpy = cublas_daxpy_impl,
    .sscal = cublas_sscal_impl,
    .dscal = cublas_dscal_impl,
    .scopy = cublas_scopy_impl,
    .dcopy = cublas_dcopy_impl,
    .sswap = cublas_sswap_impl,
    .dswap = cublas_dswap_impl,
    .sdot = cublas_sdot_impl,
    .ddot = cublas_ddot_impl,
    .snrm2 = cublas_snrm2_impl,
    .dnrm2 = cublas_dnrm2_impl,
    .sasum = cublas_sasum_impl,
    .dasum = cublas_dasum_impl,
    .isamax = cublas_isamax_impl,
    .idamax = cublas_idamax_impl,
    
    // Level 1 BLAS - Complex
    .caxpy = cublas_caxpy_impl,
    .zaxpy = cublas_zaxpy_impl,
    .cscal = cublas_cscal_impl,
    .zscal = cublas_zscal_impl,
    .csscal = cublas_csscal_impl,
    .zdscal = cublas_zdscal_impl,
    .ccopy = cublas_ccopy_impl,
    .zcopy = cublas_zcopy_impl,
    .cswap = cublas_cswap_impl,
    .zswap = cublas_zswap_impl,
    .cdotu = cublas_cdotu_impl,
    .zdotu = cublas_zdotu_impl,
    .cdotc = cublas_cdotc_impl,
    .zdotc = cublas_zdotc_impl,
    .scnrm2 = cublas_scnrm2_impl,
    .dznrm2 = cublas_dznrm2_impl,
    .scasum = cublas_scasum_impl,
    .dzasum = cublas_dzasum_impl,
    .icamax = cublas_icamax_impl,
    .izamax = cublas_izamax_impl,
    
    // Level 1 BLAS - Rotation
    .srotg = cublas_srotg_impl,
    .drotg = cublas_drotg_impl,
    .srot = cublas_srot_impl,
    .drot = cublas_drot_impl,
    .srotm = cublas_srotm_impl,
    .drotm = cublas_drotm_impl,
    .srotmg = cublas_srotmg_impl,
    .drotmg = cublas_drotmg_impl,
    
    // Level 2 BLAS - Matrix-vector operations
    .sgemv = cublas_sgemv_impl,
    .dgemv = cublas_dgemv_impl,
    .cgemv = cublas_cgemv_impl,
    .zgemv = cublas_zgemv_impl,
    .chemv = cublas_chemv_impl,
    .zhemv = cublas_zhemv_impl,
    .ssymv = cublas_ssymv_impl,
    .dsymv = cublas_dsymv_impl,
    .strmv = cublas_strmv_impl,
    .dtrmv = cublas_dtrmv_impl,
    .ctrmv = cublas_ctrmv_impl,
    .ztrmv = cublas_ztrmv_impl,
    .strsv = cublas_strsv_impl,
    .dtrsv = cublas_dtrsv_impl,
    .ctrsv = cublas_ctrsv_impl,
    .ztrsv = cublas_ztrsv_impl,
    
    // Level 2 BLAS - Rank updates
    .sger = cublas_sger_impl,
    .dger = cublas_dger_impl,
    .cgeru = cublas_cgeru_impl,
    .zgeru = cublas_zgeru_impl,
    .cgerc = cublas_cgerc_impl,
    .zgerc = cublas_zgerc_impl,
    .cher = cublas_cher_impl,
    .zher = cublas_zher_impl,
    .ssyr = cublas_ssyr_impl,
    .dsyr = cublas_dsyr_impl,
    .cher2 = cublas_cher2_impl,
    .zher2 = cublas_zher2_impl,
    .ssyr2 = cublas_ssyr2_impl,
    .dsyr2 = cublas_dsyr2_impl,
    
    // Level 2 BLAS - Banded matrix operations
    .sgbmv = cublas_sgbmv_impl,
    .dgbmv = cublas_dgbmv_impl,
    .cgbmv = cublas_cgbmv_impl,
    .zgbmv = cublas_zgbmv_impl,
    .ssbmv = cublas_ssbmv_impl,
    .dsbmv = cublas_dsbmv_impl,
    .chbmv = cublas_chbmv_impl,
    .zhbmv = cublas_zhbmv_impl,
    .stbmv = cublas_stbmv_impl,
    .dtbmv = cublas_dtbmv_impl,
    .ctbmv = cublas_ctbmv_impl,
    .ztbmv = cublas_ztbmv_impl,
    .stbsv = cublas_stbsv_impl,
    .dtbsv = cublas_dtbsv_impl,
    .ctbsv = cublas_ctbsv_impl,
    .ztbsv = cublas_ztbsv_impl,
    
    // Level 2 BLAS - Packed matrix operations
    .sspmv = cublas_sspmv_impl,
    .dspmv = cublas_dspmv_impl,
    .chpmv = cublas_chpmv_impl,
    .zhpmv = cublas_zhpmv_impl,
    .stpmv = cublas_stpmv_impl,
    .dtpmv = cublas_dtpmv_impl,
    .ctpmv = cublas_ctpmv_impl,
    .ztpmv = cublas_ztpmv_impl,
    .stpsv = cublas_stpsv_impl,
    .dtpsv = cublas_dtpsv_impl,
    .ctpsv = cublas_ctpsv_impl,
    .ztpsv = cublas_ztpsv_impl,
    
    // Level 2 BLAS - Packed rank updates
    .sspr = cublas_sspr_impl,
    .dspr = cublas_dspr_impl,
    .chpr = cublas_chpr_impl,
    .zhpr = cublas_zhpr_impl,
    .sspr2 = cublas_sspr2_impl,
    .dspr2 = cublas_dspr2_impl,
    .chpr2 = cublas_chpr2_impl,
    .zhpr2 = cublas_zhpr2_impl,
    
    // Level 2 BLAS - Complex symmetric operations
    .csymv = cublas_csymv_impl,
    .zsymv = cublas_zsymv_impl,
    .csyr = cublas_csyr_impl,
    .zsyr = cublas_zsyr_impl,
    .csyr2 = cublas_csyr2_impl,
    .zsyr2 = cublas_zsyr2_impl,
    
    // Complex symmetric packed operations (NOT supported by cuBLAS - rocBLAS only)
    .cspmv = NULL,
    .zspmv = NULL,
    .cspr = NULL,
    .zspr = NULL,
    .cspr2 = NULL,
    .zspr2 = NULL,
    
    // Level 3 BLAS - Matrix-matrix operations
    .sgemm = cublas_sgemm_impl,
    .dgemm = cublas_dgemm_impl,
    .cgemm = cublas_cgemm_impl,
    .zgemm = cublas_zgemm_impl,
    .ssymm = cublas_ssymm_impl,
    .dsymm = cublas_dsymm_impl,
    .csymm = cublas_csymm_impl,
    .zsymm = cublas_zsymm_impl,
    .chemm = cublas_chemm_impl,
    .zhemm = cublas_zhemm_impl,
    .strmm = cublas_strmm_impl,
    .dtrmm = cublas_dtrmm_impl,
    .ctrmm = cublas_ctrmm_impl,
    .ztrmm = cublas_ztrmm_impl,
    .strsm = cublas_strsm_impl,
    .dtrsm = cublas_dtrsm_impl,
    .ctrsm = cublas_ctrsm_impl,
    .ztrsm = cublas_ztrsm_impl,
    .ssyrk = cublas_ssyrk_impl,
    .dsyrk = cublas_dsyrk_impl,
    .csyrk = cublas_csyrk_impl,
    .zsyrk = cublas_zsyrk_impl,
    .cherk = cublas_cherk_impl,
    .zherk = cublas_zherk_impl,
    .ssyr2k = cublas_ssyr2k_impl,
    .dsyr2k = cublas_dsyr2k_impl,
    .csyr2k = cublas_csyr2k_impl,
    .zsyr2k = cublas_zsyr2k_impl,
    .cher2k = cublas_cher2k_impl,
    .zher2k = cublas_zher2k_impl,
    
    // LAPACK - LU factorization
    .sgetrf = cublas_sgetrf_impl,
    .dgetrf = cublas_dgetrf_impl,
    .cgetrf = cublas_cgetrf_impl,
    .zgetrf = cublas_zgetrf_impl,
    .sgetrs = cublas_sgetrs_impl,
    .dgetrs = cublas_dgetrs_impl,
    .cgetrs = cublas_cgetrs_impl,
    .zgetrs = cublas_zgetrs_impl,
    
    // LAPACK - Cholesky factorization
    .spotrf = cublas_spotrf_impl,
    .dpotrf = cublas_dpotrf_impl,
    .cpotrf = cublas_cpotrf_impl,
    .zpotrf = cublas_zpotrf_impl,
    .spotrs = cublas_spotrs_impl,
    .dpotrs = cublas_dpotrs_impl,
    .cpotrs = cublas_cpotrs_impl,
    .zpotrs = cublas_zpotrs_impl,
    
    // LAPACK - QR factorization
    .sgeqrf = cublas_sgeqrf_impl,
    .dgeqrf = cublas_dgeqrf_impl,
    .cgeqrf = cublas_cgeqrf_impl,
    .zgeqrf = cublas_zgeqrf_impl,
    
    // LAPACK - SVD
    .sgesvd = cublas_sgesvd_impl,
    .dgesvd = cublas_dgesvd_impl,
    .cgesvd = cublas_cgesvd_impl,
    .zgesvd = cublas_zgesvd_impl,
    
    // LAPACK - Eigenvalues
    .ssyev = cublas_ssyev_impl,
    .dsyev = cublas_dsyev_impl,
    .cheev = cublas_cheev_impl,
    .zheev = cublas_zheev_impl,
    
    // Note: Banded/packed Level 2 operations and additional LAPACK operations
    // (gesv, orgqr, ormqr, getri, potri, etc.) can be added following the same patterns
};
