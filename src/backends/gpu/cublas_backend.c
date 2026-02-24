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
    
    /* cuBLAS functions */
    cublasCreate_t cublasCreate;
    cublasDestroy_t cublasDestroy;
    cublasSetStream_t cublasSetStream;
    
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
    g_cublas_api.cublasCreate = (cublasCreate_t)GET_PROC(g_cublas_api.cublas_handle, cublasCreate_v2);
    g_cublas_api.cublasDestroy = (cublasDestroy_t)GET_PROC(g_cublas_api.cublas_handle, cublasDestroy_v2);
    g_cublas_api.cublasSetStream = (cublasSetStream_t)GET_PROC(g_cublas_api.cublas_handle, cublasSetStream_v2);
    
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
    .blas_vtable = NULL  /* TODO: Implement cuBLAS BLAS wrappers */
};

const fb_gpu_backend_trait_t* fb_cublas_get_backend(void) {
    if (!fb_cublas_is_available()) {
        return NULL;
    }
    /* TODO: Return proper trait implementation */
    return NULL;
}

const fb_backend_vtable_t* fb_cublas_get_vtable(void) {
    static fb_backend_vtable_t cublas_vtable = {0};
    
    /* Return empty vtable - actual BLAS operations implemented by dispatch system */
    /* Old vtable_wrappers implementation disabled to use pure dynamic loading */
    return &cublas_vtable;
    
    return &cublas_vtable;
}

int fb_cublas_get_device_info(int device_id, fb_gpu_device_info_t* info) {
    /* TODO: Query CUDA device properties */
    return -1;
}
