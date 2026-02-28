/**
 * @file cublas_backend.c
 * @brief NVIDIA cuBLAS backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "cublas_backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* Forward declare CUDA/cuBLAS types to avoid hard dependency */
typedef struct CUstream_st* cudaStream_t;
typedef struct cublasContext* cublasHandle_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED,
    CUBLAS_STATUS_ALLOC_FAILED,
    CUBLAS_STATUS_INVALID_VALUE,
    CUBLAS_STATUS_ARCH_MISMATCH,
    CUBLAS_STATUS_MAPPING_ERROR,
    CUBLAS_STATUS_EXECUTION_FAILED,
    CUBLAS_STATUS_INTERNAL_ERROR
} cublasStatus_t;

/* cuBLAS operation mode and transpose enums */
typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

/* Function pointer types for dynamic loading */
typedef cublasStatus_t (*cublasCreate_t)(cublasHandle_t*);
typedef cublasStatus_t (*cublasDestroy_t)(cublasHandle_t);
typedef cublasStatus_t (*cublasSetStream_t)(cublasHandle_t, cudaStream_t);

/* cuBLAS Level 1 BLAS function pointers */
typedef cublasStatus_t (*cublasSasum_t)(cublasHandle_t, int, const float*, int, float*);
typedef cublasStatus_t (*cublasDasum_t)(cublasHandle_t, int, const double*, int, double*);
typedef cublasStatus_t (*cublasSaxpy_t)(cublasHandle_t, int, const float*, const float*, int, float*, int);
typedef cublasStatus_t (*cublasDaxpy_t)(cublasHandle_t, int, const double*, const double*, int, double*, int);
typedef cublasStatus_t (*cublasSdot_t)(cublasHandle_t, int, const float*, int, const float*, int, float*);
typedef cublasStatus_t (*cublasDdot_t)(cublasHandle_t, int, const double*, int, const double*, int, double*);
typedef cublasStatus_t (*cublasScopy_t)(cublasHandle_t, int, const float*, int, float*, int);
typedef cublasStatus_t (*cublasDcopy_t)(cublasHandle_t, int, const double*, int, double*, int);
typedef cublasStatus_t (*cublasSscal_t)(cublasHandle_t, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDscal_t)(cublasHandle_t, int, const double*, double*, int);
typedef cublasStatus_t (*cublasSnrm2_t)(cublasHandle_t, int, const float*, int, float*);
typedef cublasStatus_t (*cublasDnrm2_t)(cublasHandle_t, int, const double*, int, double*);
typedef cublasStatus_t (*cublasSswap_t)(cublasHandle_t, int, float*, int, float*, int);
typedef cublasStatus_t (*cublasDswap_t)(cublasHandle_t, int, double*, int, double*, int);
typedef cublasStatus_t (*cublasIsamax_t)(cublasHandle_t, int, const float*, int, int*);
typedef cublasStatus_t (*cublasIdamax_t)(cublasHandle_t, int, const double*, int, int*);

/* cuBLAS Level 2 BLAS function pointers */
typedef cublasStatus_t (*cublasSgemv_t)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDgemv_t)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);

/* cuBLAS Level 3 BLAS function pointers */
typedef cublasStatus_t (*cublasSgemm_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDgemm_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
typedef cublasStatus_t (*cublasSsymm_t)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDsymm_t)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
typedef cublasStatus_t (*cublasStrsm_t)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, float*, int);
typedef cublasStatus_t (*cublasDtrsm_t)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, double*, int);
typedef cublasStatus_t (*cublasSsyrk_t)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, float*, int);
typedef cublasStatus_t (*cublasDsyrk_t)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, double*, int);
typedef cublasStatus_t (*cublasSgemm_strided_batched_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, long long, const float*, int, long long, const float*, float*, int, long long, int);
typedef cublasStatus_t (*cublasDgemm_strided_batched_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, long long, const double*, int, long long, const double*, double*, int, long long, int);

/* CUDA runtime function pointers */
typedef int (*cudaGetDeviceCount_t)(int*);
typedef int (*cudaSetDevice_t)(int);
typedef int (*cudaGetDevice_t)(int*);
typedef int (*cudaMalloc_t)(void**, size_t);
typedef int (*cudaFree_t)(void*);
typedef int (*cudaMemcpy_t)(void*, const void*, size_t, int);
typedef int (*cudaMemset_t)(void*, int, size_t);
typedef int (*cudaDeviceSynchronize_t)(void);
typedef int (*cudaStreamCreate_t)(cudaStream_t*);
typedef int (*cudaStreamDestroy_t)(cudaStream_t);
typedef int (*cudaStreamSynchronize_t)(cudaStream_t);

/* Global function pointers (loaded dynamically) */
static struct {
    void* cuda_handle;
    void* cublas_handle;
    
    /* cuBLAS lifecycle */
    cublasCreate_t cublasCreate;
    cublasDestroy_t cublasDestroy;
    cublasSetStream_t cublasSetStream;

    /* BLAS Level 1 */
    cublasSasum_t  Sasum;
    cublasDasum_t  Dasum;
    cublasSaxpy_t  Saxpy;
    cublasDaxpy_t  Daxpy;
    cublasSdot_t   Sdot;
    cublasDdot_t   Ddot;
    cublasScopy_t  Scopy;
    cublasDcopy_t  Dcopy;
    cublasSscal_t  Sscal;
    cublasDscal_t  Dscal;
    cublasSnrm2_t  Snrm2;
    cublasDnrm2_t  Dnrm2;
    cublasSswap_t  Sswap;
    cublasDswap_t  Dswap;
    cublasIsamax_t Isamax;
    cublasIdamax_t Idamax;

    /* BLAS Level 2 */
    cublasSgemv_t  Sgemv;
    cublasDgemv_t  Dgemv;

    /* BLAS Level 3 */
    cublasSgemm_t  Sgemm;
    cublasDgemm_t  Dgemm;
    cublasSsymm_t  Ssymm;
    cublasDsymm_t  Dsymm;
    cublasStrsm_t  Strsm;
    cublasDtrsm_t  Dtrsm;
    cublasSsyrk_t  Ssyrk;
    cublasDsyrk_t  Dsyrk;
    cublasSgemm_strided_batched_t  Sgemm_strided_batched;
    cublasDgemm_strided_batched_t  Dgemm_strided_batched;

    /* CUDA runtime functions */
    cudaGetDeviceCount_t cudaGetDeviceCount;
    cudaSetDevice_t cudaSetDevice;
    cudaGetDevice_t cudaGetDevice;
    cudaMalloc_t cudaMalloc;
    cudaFree_t cudaFree;
    cudaMemcpy_t cudaMemcpy;
    cudaMemset_t cudaMemset;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaStreamCreate_t cudaStreamCreate;
    cudaStreamDestroy_t cudaStreamDestroy;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    
    bool initialized;
} g_cublas_api;

/* Default per-process cuBLAS handle (device 0).  Created once in
 * cublas_load_api() and used by the vtable shim functions. */
static cublasHandle_t g_cublas_default_handle = NULL;

/* cuBLAS context structure */
typedef struct {
    cublasHandle_t handle;
    int device_id;
} cublas_context_t;

static int cublas_load_api(void) {
    if (g_cublas_api.initialized) {
        return 0;
    }
    
#ifdef _WIN32
    /* Try CUDA 13 first, then 12, then 11 */
    g_cublas_api.cuda_handle = LoadLibraryA("cudart64_13.dll");
    if (!g_cublas_api.cuda_handle) {
        g_cublas_api.cuda_handle = LoadLibraryA("cudart64_12.dll");
    }
    if (!g_cublas_api.cuda_handle) {
        g_cublas_api.cuda_handle = LoadLibraryA("cudart64_11.dll");
    }
    
    g_cublas_api.cublas_handle = LoadLibraryA("cublas64_13.dll");
    if (!g_cublas_api.cublas_handle) {
        g_cublas_api.cublas_handle = LoadLibraryA("cublas64_12.dll");
    }
    if (!g_cublas_api.cublas_handle) {
        g_cublas_api.cublas_handle = LoadLibraryA("cublas64_11.dll");
    }
    
    #define GET_PROC(lib, name) GetProcAddress(lib, #name)
#else
    #include <dlfcn.h>
    g_cublas_api.cuda_handle = dlopen("libcudart.so.13", RTLD_LAZY);
    if (!g_cublas_api.cuda_handle) {
        g_cublas_api.cuda_handle = dlopen("libcudart.so.12", RTLD_LAZY);
    }
    if (!g_cublas_api.cuda_handle) {
        g_cublas_api.cuda_handle = dlopen("libcudart.so.11", RTLD_LAZY);
    }
    
    g_cublas_api.cublas_handle = dlopen("libcublas.so.13", RTLD_LAZY);
    if (!g_cublas_api.cublas_handle) {
        g_cublas_api.cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY);
    }
    if (!g_cublas_api.cublas_handle) {
        g_cublas_api.cublas_handle = dlopen("libcublas.so.11", RTLD_LAZY);
    }
    
    #define GET_PROC(lib, name) dlsym(lib, #name)
#endif
    
    if (!g_cublas_api.cuda_handle || !g_cublas_api.cublas_handle) {
        return -1;
    }
    
    /* Load function pointers */
    g_cublas_api.cublasCreate    = (cublasCreate_t)   GET_PROC(g_cublas_api.cublas_handle, cublasCreate_v2);
    g_cublas_api.cublasDestroy   = (cublasDestroy_t)  GET_PROC(g_cublas_api.cublas_handle, cublasDestroy_v2);
    g_cublas_api.cublasSetStream = (cublasSetStream_t)GET_PROC(g_cublas_api.cublas_handle, cublasSetStream_v2);

    /* BLAS Level 1 */
    g_cublas_api.Sasum   = (cublasSasum_t) GET_PROC(g_cublas_api.cublas_handle, cublasSasum_v2);
    g_cublas_api.Dasum   = (cublasDasum_t) GET_PROC(g_cublas_api.cublas_handle, cublasDasum_v2);
    g_cublas_api.Saxpy   = (cublasSaxpy_t) GET_PROC(g_cublas_api.cublas_handle, cublasSaxpy_v2);
    g_cublas_api.Daxpy   = (cublasDaxpy_t) GET_PROC(g_cublas_api.cublas_handle, cublasDaxpy_v2);
    g_cublas_api.Sdot    = (cublasSdot_t)  GET_PROC(g_cublas_api.cublas_handle, cublasSdot_v2);
    g_cublas_api.Ddot    = (cublasDdot_t)  GET_PROC(g_cublas_api.cublas_handle, cublasDdot_v2);
    g_cublas_api.Scopy   = (cublasScopy_t) GET_PROC(g_cublas_api.cublas_handle, cublasScopy_v2);
    g_cublas_api.Dcopy   = (cublasDcopy_t) GET_PROC(g_cublas_api.cublas_handle, cublasDcopy_v2);
    g_cublas_api.Sscal   = (cublasSscal_t) GET_PROC(g_cublas_api.cublas_handle, cublasSscal_v2);
    g_cublas_api.Dscal   = (cublasDscal_t) GET_PROC(g_cublas_api.cublas_handle, cublasDscal_v2);
    g_cublas_api.Snrm2   = (cublasSnrm2_t) GET_PROC(g_cublas_api.cublas_handle, cublasSnrm2_v2);
    g_cublas_api.Dnrm2   = (cublasDnrm2_t) GET_PROC(g_cublas_api.cublas_handle, cublasDnrm2_v2);
    g_cublas_api.Sswap   = (cublasSswap_t) GET_PROC(g_cublas_api.cublas_handle, cublasSswap_v2);
    g_cublas_api.Dswap   = (cublasDswap_t) GET_PROC(g_cublas_api.cublas_handle, cublasDswap_v2);
    g_cublas_api.Isamax  = (cublasIsamax_t)GET_PROC(g_cublas_api.cublas_handle, cublasIsamax_v2);
    g_cublas_api.Idamax  = (cublasIdamax_t)GET_PROC(g_cublas_api.cublas_handle, cublasIdamax_v2);

    /* BLAS Level 2 */
    g_cublas_api.Sgemv   = (cublasSgemv_t) GET_PROC(g_cublas_api.cublas_handle, cublasSgemv_v2);
    g_cublas_api.Dgemv   = (cublasDgemv_t) GET_PROC(g_cublas_api.cublas_handle, cublasDgemv_v2);

    /* BLAS Level 3 */
    g_cublas_api.Sgemm   = (cublasSgemm_t) GET_PROC(g_cublas_api.cublas_handle, cublasSgemm_v2);
    g_cublas_api.Dgemm   = (cublasDgemm_t) GET_PROC(g_cublas_api.cublas_handle, cublasDgemm_v2);
    g_cublas_api.Ssymm   = (cublasSsymm_t) GET_PROC(g_cublas_api.cublas_handle, cublasSsymm_v2);
    g_cublas_api.Dsymm   = (cublasDsymm_t) GET_PROC(g_cublas_api.cublas_handle, cublasDsymm_v2);
    g_cublas_api.Strsm   = (cublasStrsm_t) GET_PROC(g_cublas_api.cublas_handle, cublasStrsm_v2);
    g_cublas_api.Dtrsm   = (cublasDtrsm_t) GET_PROC(g_cublas_api.cublas_handle, cublasDtrsm_v2);
    g_cublas_api.Ssyrk   = (cublasSsyrk_t) GET_PROC(g_cublas_api.cublas_handle, cublasSsyrk_v2);
    g_cublas_api.Dsyrk   = (cublasDsyrk_t) GET_PROC(g_cublas_api.cublas_handle, cublasDsyrk_v2);
    g_cublas_api.Sgemm_strided_batched =
        (cublasSgemm_strided_batched_t)GET_PROC(g_cublas_api.cublas_handle, cublasSgemmStridedBatched);
    g_cublas_api.Dgemm_strided_batched =
        (cublasDgemm_strided_batched_t)GET_PROC(g_cublas_api.cublas_handle, cublasDgemmStridedBatched);
    
    g_cublas_api.cudaGetDeviceCount = (cudaGetDeviceCount_t)GET_PROC(g_cublas_api.cuda_handle, cudaGetDeviceCount);
    g_cublas_api.cudaSetDevice = (cudaSetDevice_t)GET_PROC(g_cublas_api.cuda_handle, cudaSetDevice);
    g_cublas_api.cudaGetDevice = (cudaGetDevice_t)GET_PROC(g_cublas_api.cuda_handle, cudaGetDevice);
    g_cublas_api.cudaMalloc = (cudaMalloc_t)GET_PROC(g_cublas_api.cuda_handle, cudaMalloc);
    g_cublas_api.cudaFree = (cudaFree_t)GET_PROC(g_cublas_api.cuda_handle, cudaFree);
    g_cublas_api.cudaMemcpy = (cudaMemcpy_t)GET_PROC(g_cublas_api.cuda_handle, cudaMemcpy);
    g_cublas_api.cudaMemset = (cudaMemset_t)GET_PROC(g_cublas_api.cuda_handle, cudaMemset);
    g_cublas_api.cudaDeviceSynchronize = (cudaDeviceSynchronize_t)GET_PROC(g_cublas_api.cuda_handle, cudaDeviceSynchronize);
    g_cublas_api.cudaStreamCreate = (cudaStreamCreate_t)GET_PROC(g_cublas_api.cuda_handle, cudaStreamCreate);
    g_cublas_api.cudaStreamDestroy = (cudaStreamDestroy_t)GET_PROC(g_cublas_api.cuda_handle, cudaStreamDestroy);
    g_cublas_api.cudaStreamSynchronize = (cudaStreamSynchronize_t)GET_PROC(g_cublas_api.cuda_handle, cudaStreamSynchronize);
    
    g_cublas_api.initialized = true;

    /* Create the default handle used by vtable shims (device 0). */
    if (g_cublas_api.cublasCreate) {
        if (g_cublas_api.cublasCreate(&g_cublas_default_handle) != CUBLAS_STATUS_SUCCESS) {
            g_cublas_default_handle = NULL;
        }
    }

    return 0;
}

bool fb_cublas_is_available(void) {
    return cublas_load_api() == 0;
}

int fb_cublas_device_count(void) {
    if (cublas_load_api() != 0) {
        return 0;
    }
    
    int count = 0;
    if (g_cublas_api.cudaGetDeviceCount(&count) != 0) {
        return 0;
    }
    
    return count;
}

int fb_cublas_init(int device_id, fb_gpu_context_t* ctx) {
    if (!ctx) {
        return -1;
    }
    
    if (cublas_load_api() != 0) {
        return -1;
    }
    
    /* Set device */
    if (device_id >= 0) {
        if (g_cublas_api.cudaSetDevice(device_id) != 0) {
            return -1;
        }
    } else {
        /* Auto-select device 0 */
        device_id = 0;
        g_cublas_api.cudaSetDevice(0);
    }
    
    /* Create cuBLAS handle */
    cublas_context_t* cublas_ctx = (cublas_context_t*)malloc(sizeof(cublas_context_t));
    if (!cublas_ctx) {
        return -1;
    }
    
    if (g_cublas_api.cublasCreate(&cublas_ctx->handle) != CUBLAS_STATUS_SUCCESS) {
        free(cublas_ctx);
        return -1;
    }
    
    cublas_ctx->device_id = device_id;
    
    ctx->device_id = device_id;
    ctx->backend_context = cublas_ctx;
    ctx->default_stream = NULL;
    ctx->synchronous = false;
    
    return 0;
}

void fb_cublas_shutdown(fb_gpu_context_t* ctx) {
    if (!ctx || !ctx->backend_context) {
        return;
    }
    
    cublas_context_t* cublas_ctx = (cublas_context_t*)ctx->backend_context;
    
    if (g_cublas_api.cublasDestroy) {
        g_cublas_api.cublasDestroy(cublas_ctx->handle);
    }
    
    free(cublas_ctx);
    ctx->backend_context = NULL;
}

static int cublas_malloc(fb_gpu_ptr_t* ptr, size_t size) {
    if (g_cublas_api.cudaMalloc(ptr, size) != 0) {
        return -1;
    }
    return 0;
}

static void cublas_free(fb_gpu_ptr_t ptr) {
    if (g_cublas_api.cudaFree) {
        g_cublas_api.cudaFree(ptr);
    }
}

static int cublas_memcpy_h2d(fb_gpu_ptr_t dst, const void* src, size_t size, fb_gpu_stream_t stream) {
    /* cudaMemcpyKind: cudaMemcpyHostToDevice = 1 */
    return g_cublas_api.cudaMemcpy(dst, src, size, 1);
}

static int cublas_memcpy_d2h(void* dst, fb_gpu_ptr_t src, size_t size, fb_gpu_stream_t stream) {
    /* cudaMemcpyKind: cudaMemcpyDeviceToHost = 2 */
    return g_cublas_api.cudaMemcpy(dst, src, size, 2);
}

static int cublas_memcpy_d2d(fb_gpu_ptr_t dst, fb_gpu_ptr_t src, size_t size, fb_gpu_stream_t stream) {
    /* cudaMemcpyKind: cudaMemcpyDeviceToDevice = 3 */
    return g_cublas_api.cudaMemcpy(dst, src, size, 3);
}

static int cublas_device_synchronize(void) {
    return g_cublas_api.cudaDeviceSynchronize();
}

static int cublas_backend_init(int device_id, fb_gpu_context_t* ctx) {
    return fb_cublas_init(device_id, ctx);
}

static void cublas_backend_shutdown(fb_gpu_context_t* ctx) {
    fb_cublas_shutdown(ctx);
}

static int cublas_set_device(int device_id) {
    return g_cublas_api.cudaSetDevice(device_id);
}

static int cublas_get_device(void) {
    int device_id = -1;
    g_cublas_api.cudaGetDevice(&device_id);
    return device_id;
}

static int cublas_backend_device_count(void) {
    return fb_cublas_device_count();
}

static int cublas_stream_create(fb_gpu_stream_t* stream) {
    return g_cublas_api.cudaStreamCreate((cudaStream_t*)stream);
}

static void cublas_stream_destroy(fb_gpu_stream_t stream) {
    if (g_cublas_api.cudaStreamDestroy) {
        g_cublas_api.cudaStreamDestroy((cudaStream_t)stream);
    }
}

static int cublas_stream_synchronize(fb_gpu_stream_t stream) {
    return g_cublas_api.cudaStreamSynchronize((cudaStream_t)stream);
}

static int cublas_memset_impl(fb_gpu_ptr_t ptr, int value, size_t size, fb_gpu_stream_t stream) {
    return g_cublas_api.cudaMemset(ptr, value, size);
}

/* cuBLAS GPU backend operations */
static const fb_gpu_backend_t g_cublas_backend = {
    .init = cublas_backend_init,
    .shutdown = cublas_backend_shutdown,
    .set_device = cublas_set_device,
    .get_device = cublas_get_device,
    .device_count = cublas_backend_device_count,
    .stream_create = cublas_stream_create,
    .stream_destroy = cublas_stream_destroy,
    .stream_synchronize = cublas_stream_synchronize,
    .malloc = cublas_malloc,
    .free = cublas_free,
    .memcpy_h2d = cublas_memcpy_h2d,
    .memcpy_d2h = cublas_memcpy_d2h,
    .memcpy_d2d = cublas_memcpy_d2d,
    .memset = cublas_memset_impl,
    .device_synchronize = cublas_device_synchronize,
    .blas_vtable = NULL  /* filled in fb_cublas_get_vtable() */
};

/* -------------------------------------------------------------------------
 * CBLAS-style shims
 * Each shim captures g_cublas_default_handle (device 0) and adapts the
 * CBLAS signature expected by fb_backend_vtable_t to the cuBLAS v2 API
 * which requires a handle as its first argument.
 * ------------------------------------------------------------------------- */

static float cublas_sasum(int n, const float* x, int incx) {
    float r = 0.0f;
    if (g_cublas_api.Sasum) g_cublas_api.Sasum(g_cublas_default_handle, n, x, incx, &r);
    return r;
}
static double cublas_dasum(int n, const double* x, int incx) {
    double r = 0.0;
    if (g_cublas_api.Dasum) g_cublas_api.Dasum(g_cublas_default_handle, n, x, incx, &r);
    return r;
}
static void cublas_saxpy(int n, float a, const float* x, int incx, float* y, int incy) {
    if (g_cublas_api.Saxpy) g_cublas_api.Saxpy(g_cublas_default_handle, n, &a, x, incx, y, incy);
}
static void cublas_daxpy(int n, double a, const double* x, int incx, double* y, int incy) {
    if (g_cublas_api.Daxpy) g_cublas_api.Daxpy(g_cublas_default_handle, n, &a, x, incx, y, incy);
}
static float cublas_sdot(int n, const float* x, int incx, const float* y, int incy) {
    float r = 0.0f;
    if (g_cublas_api.Sdot) g_cublas_api.Sdot(g_cublas_default_handle, n, x, incx, y, incy, &r);
    return r;
}
static double cublas_ddot(int n, const double* x, int incx, const double* y, int incy) {
    double r = 0.0;
    if (g_cublas_api.Ddot) g_cublas_api.Ddot(g_cublas_default_handle, n, x, incx, y, incy, &r);
    return r;
}
static void cublas_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_cublas_api.Scopy) g_cublas_api.Scopy(g_cublas_default_handle, n, x, incx, y, incy);
}
static void cublas_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_cublas_api.Dcopy) g_cublas_api.Dcopy(g_cublas_default_handle, n, x, incx, y, incy);
}
static void cublas_sscal(int n, float a, float* x, int incx) {
    if (g_cublas_api.Sscal) g_cublas_api.Sscal(g_cublas_default_handle, n, &a, x, incx);
}
static void cublas_dscal(int n, double a, double* x, int incx) {
    if (g_cublas_api.Dscal) g_cublas_api.Dscal(g_cublas_default_handle, n, &a, x, incx);
}
static float cublas_snrm2(int n, const float* x, int incx) {
    float r = 0.0f;
    if (g_cublas_api.Snrm2) g_cublas_api.Snrm2(g_cublas_default_handle, n, x, incx, &r);
    return r;
}
static double cublas_dnrm2(int n, const double* x, int incx) {
    double r = 0.0;
    if (g_cublas_api.Dnrm2) g_cublas_api.Dnrm2(g_cublas_default_handle, n, x, incx, &r);
    return r;
}
static void cublas_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_cublas_api.Sswap) g_cublas_api.Sswap(g_cublas_default_handle, n, x, incx, y, incy);
}
static void cublas_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_cublas_api.Dswap) g_cublas_api.Dswap(g_cublas_default_handle, n, x, incx, y, incy);
}
static int cublas_isamax(int n, const float* x, int incx) {
    int r = 0;
    if (g_cublas_api.Isamax) g_cublas_api.Isamax(g_cublas_default_handle, n, x, incx, &r);
    return r - 1; /* cuBLAS returns 1-based index; CBLAS is 0-based */
}
static int cublas_idamax(int n, const double* x, int incx) {
    int r = 0;
    if (g_cublas_api.Idamax) g_cublas_api.Idamax(g_cublas_default_handle, n, x, incx, &r);
    return r - 1;
}

/* Level 2 */
static void cublas_sgemv(int order, int transA, int m, int n,
                          float alpha, const float* A, int lda,
                          const float* x, int incx, float beta, float* y, int incy) {
    if (!g_cublas_api.Sgemv) return;
    /* CBLAS CblasRowMajor = 101; cuBLAS is column-major — swap dimensions & transpose */
    cublasOperation_t op = (transA == 111) ? CUBLAS_OP_T : CUBLAS_OP_N; /* 111=NoTrans */
    int cm = (order == 102) ? m : n; /* 102=ColMajor */
    int cn = (order == 102) ? n : m;
    if (order != 102) { int tmp = cm; cm = cn; cn = tmp; }
    g_cublas_api.Sgemv(g_cublas_default_handle, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
static void cublas_dgemv(int order, int transA, int m, int n,
                          double alpha, const double* A, int lda,
                          const double* x, int incx, double beta, double* y, int incy) {
    if (!g_cublas_api.Dgemv) return;
    cublasOperation_t op = (transA == 111) ? CUBLAS_OP_T : CUBLAS_OP_N;
    g_cublas_api.Dgemv(g_cublas_default_handle, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

/* Level 3 */
static void cublas_sgemm(int order, int transA, int transB, int m, int n, int k,
                          float alpha, const float* A, int lda,
                          const float* B, int ldb, float beta, float* C, int ldc) {
    if (!g_cublas_api.Sgemm) return;
    cublasOperation_t opA = (transA == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transB == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    /* Row-major: swap A/B and m/n to convert to column-major */
    if (order == 101) { /* CblasRowMajor */
        g_cublas_api.Sgemm(g_cublas_default_handle, opB, opA, n, m, k,
                            &alpha, B, ldb, A, lda, &beta, C, ldc);
    } else {
        g_cublas_api.Sgemm(g_cublas_default_handle, opA, opB, m, n, k,
                            &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
}
static void cublas_dgemm(int order, int transA, int transB, int m, int n, int k,
                          double alpha, const double* A, int lda,
                          const double* B, int ldb, double beta, double* C, int ldc) {
    if (!g_cublas_api.Dgemm) return;
    cublasOperation_t opA = (transA == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transB == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (order == 101) {
        g_cublas_api.Dgemm(g_cublas_default_handle, opB, opA, n, m, k,
                            &alpha, B, ldb, A, lda, &beta, C, ldc);
    } else {
        g_cublas_api.Dgemm(g_cublas_default_handle, opA, opB, m, n, k,
                            &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
}
static void cublas_strsm(int order, int side, int uplo, int transA, int diag,
                          int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
    if (!g_cublas_api.Strsm) return;
    cublasSideMode_t s = (side == 141) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; /* 141=Left */
    cublasFillMode_t u = (uplo == 121) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t op = (transA == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDiagType_t dt = (diag == 131) ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
    g_cublas_api.Strsm(g_cublas_default_handle, s, u, op, dt, m, n, &alpha, A, lda, B, ldb);
}
static void cublas_dtrsm(int order, int side, int uplo, int transA, int diag,
                          int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
    if (!g_cublas_api.Dtrsm) return;
    cublasSideMode_t s = (side == 141) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t u = (uplo == 121) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t op = (transA == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDiagType_t dt = (diag == 131) ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
    g_cublas_api.Dtrsm(g_cublas_default_handle, s, u, op, dt, m, n, &alpha, A, lda, B, ldb);
}
static void cublas_ssyrk(int order, int uplo, int trans, int n, int k,
                          float alpha, const float* A, int lda, float beta, float* C, int ldc) {
    if (!g_cublas_api.Ssyrk) return;
    cublasFillMode_t u = (uplo == 121) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t op = (trans == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    g_cublas_api.Ssyrk(g_cublas_default_handle, u, op, n, k, &alpha, A, lda, &beta, C, ldc);
}
static void cublas_dsyrk(int order, int uplo, int trans, int n, int k,
                          double alpha, const double* A, int lda, double beta, double* C, int ldc) {
    if (!g_cublas_api.Dsyrk) return;
    cublasFillMode_t u = (uplo == 121) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t op = (trans == 111) ? CUBLAS_OP_N : CUBLAS_OP_T;
    g_cublas_api.Dsyrk(g_cublas_default_handle, u, op, n, k, &alpha, A, lda, &beta, C, ldc);
}

/* Memory management — forward to CUDA runtime */
static int cublas_mem_alloc(void* bh, void** ptr, size_t sz) {
    (void)bh;
    return (g_cublas_api.cudaMalloc && g_cublas_api.cudaMalloc(ptr, sz) == 0) ? 0 : -1;
}
static void cublas_mem_free(void* bh, void* ptr) {
    (void)bh;
    if (g_cublas_api.cudaFree) g_cublas_api.cudaFree(ptr);
}
static int cublas_mem_upload(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh;
    return g_cublas_api.cudaMemcpy ? g_cublas_api.cudaMemcpy(dst, src, sz, 1) : -1; /* H2D=1 */
}
static int cublas_mem_download(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh;
    return g_cublas_api.cudaMemcpy ? g_cublas_api.cudaMemcpy(dst, src, sz, 2) : -1; /* D2H=2 */
}
static int cublas_mem_copy_dev(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh;
    return g_cublas_api.cudaMemcpy ? g_cublas_api.cudaMemcpy(dst, src, sz, 3) : -1; /* D2D=3 */
}

/* Stream management */
static int cublas_stream_create_vt(void* bh, void** stream) {
    (void)bh;
    return g_cublas_api.cudaStreamCreate((cudaStream_t*)stream);
}
static void cublas_stream_destroy_vt(void* bh, void* stream) {
    (void)bh;
    if (g_cublas_api.cudaStreamDestroy) g_cublas_api.cudaStreamDestroy((cudaStream_t)stream);
}
static int cublas_stream_sync_vt(void* bh, void* stream) {
    (void)bh;
    return g_cublas_api.cudaStreamSynchronize ? g_cublas_api.cudaStreamSynchronize((cudaStream_t)stream) : 0;
}
static int cublas_stream_set_vt(void* bh, void* stream) {
    (void)bh;
    return (g_cublas_default_handle && g_cublas_api.cublasSetStream)
               ? (int)g_cublas_api.cublasSetStream(g_cublas_default_handle, (cudaStream_t)stream)
               : -1;
}

static uint32_t cublas_get_capabilities(void* bh) {
    (void)bh;
    return FB_CAP_GPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3
         | FB_CAP_SINGLE | FB_CAP_DOUBLE;
}

/* The static vtable that gets_vtable() returns */
static fb_backend_vtable_t g_cublas_vtable = {
    /* BLAS L1 */
    .sasum  = cublas_sasum,
    .dasum  = cublas_dasum,
    .saxpy  = cublas_saxpy,
    .daxpy  = cublas_daxpy,
    .sdot   = cublas_sdot,
    .ddot   = cublas_ddot,
    .scopy  = cublas_scopy,
    .dcopy  = cublas_dcopy,
    .sscal  = cublas_sscal,
    .dscal  = cublas_dscal,
    .snrm2  = cublas_snrm2,
    .dnrm2  = cublas_dnrm2,
    .sswap  = cublas_sswap,
    .dswap  = cublas_dswap,
    .isamax = cublas_isamax,
    .idamax = cublas_idamax,

    /* BLAS L2 */
    .sgemv  = cublas_sgemv,
    .dgemv  = cublas_dgemv,

    /* BLAS L3 */
    .sgemm  = cublas_sgemm,
    .dgemm  = cublas_dgemm,
    .strsm  = cublas_strsm,
    .dtrsm  = cublas_dtrsm,
    .ssyrk  = cublas_ssyrk,
    .dsyrk  = cublas_dsyrk,

    /* Memory / streams */
    .mem_alloc    = cublas_mem_alloc,
    .mem_free     = cublas_mem_free,
    .mem_upload   = cublas_mem_upload,
    .mem_download = cublas_mem_download,
    .mem_copy     = cublas_mem_copy_dev,
    .stream_create  = cublas_stream_create_vt,
    .stream_destroy = cublas_stream_destroy_vt,
    .stream_sync    = cublas_stream_sync_vt,
    .stream_set     = cublas_stream_set_vt,

    /* Backend properties */
    .get_capabilities = cublas_get_capabilities,
    .get_num_threads  = NULL,  /* GPU kernels have their own parallelism */
    .set_num_threads  = NULL,
};

const fb_gpu_backend_trait_t* fb_cublas_get_backend(void) {
    if (!fb_cublas_is_available()) {
        return NULL;
    }
    return NULL; /* fb_gpu_backend_trait_t layer not yet implemented */
}

const fb_backend_vtable_t* fb_cublas_get_vtable(void) {
    if (!fb_cublas_is_available() || !g_cublas_default_handle) {
        return NULL;
    }
    /* ext_ops[] NOTE: cuBLAS v2 symbols take cublasHandle_t as first arg,
     * which is incompatible with the no-handle CBLAS signature expected by
     * fb_auto_populate_ext_ops().  Extended GPU ops (cuSOLVER, batched GEMM
     * variants, etc.) will be added via purpose-built handles when needed. */
    return &g_cublas_vtable;
}

int fb_cublas_get_device_info(int device_id, fb_gpu_device_info_t* info) {
    /* TODO: Query CUDA device properties via cuDeviceGetAttribute */
    (void)device_id; (void)info;    return -1;
}