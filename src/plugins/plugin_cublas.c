/**
 * @file plugin_cublas.c
 * @brief NVIDIA cuBLAS GPU plugin implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Try to include CUDA headers if available */
#ifdef FB_ENABLE_CUDA
    #include <cuda_runtime.h>
    #define CUBLAS_AVAILABLE 1
    #pragma message("FB_ENABLE_CUDA is defined - CUBLAS_AVAILABLE = 1")
#else
    #define CUBLAS_AVAILABLE 0
    /* Fallback type stubs when CUDA is not available */
    #ifndef CUDA_SUCCESS
    #define CUDA_SUCCESS 0
    typedef int cudaError_t;
    typedef void* cudaDeviceProp_t;
    #endif
    #pragma message("FB_ENABLE_CUDA is NOT defined - CUBLAS_AVAILABLE = 0")
#endif

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* External functions for cuBLAS smart wrappers */
extern void cublas_populate_vtable(fb_backend_vtable_t* vtable);
extern int fb_cublas_smart_init(int device_id, void* lib_handle);
extern void fb_cublas_smart_shutdown(void);
extern void cublas_wrappers_set_handle(void* handle);
extern void* g_cublas_handle;  /* Initialized by fb_cublas_smart_init */

/* GPU vendor detection */
static int is_nvidia_gpu_available(void) {
#if CUBLAS_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        /* Verify it's actually an NVIDIA GPU */
        struct cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err == cudaSuccess) {
            /* NVIDIA GPUs have compute capability */
            return (prop.major > 0);
        }
    }
#endif
    return 0;
}

/* Plugin context - minimal for now since cuBLAS uses trait system */
typedef struct {
    fb_lib_handle_t lib_handle;
    int device_count;
} cublas_plugin_context_t;

static fb_backend_vtable_t g_cublas_vtable;
static cublas_plugin_context_t* g_cublas_context = NULL;

/* ============================================================================
 * GPU Memory Management Wrappers
 * ========================================================================= */

static int fb_cublas_mem_alloc(void* handle, void** ptr, size_t size) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? 0 : -1;
#else
    *ptr = NULL;
    return -1;
#endif
}

static int fb_cublas_mem_free(void* handle, void* ptr) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

static int fb_cublas_mem_upload(void* handle, void* dst, const void* src, size_t size) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

static int fb_cublas_mem_download(void* handle, void* dst, const void* src, size_t size) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

static int fb_cublas_mem_copy(void* handle, void* dst, const void* src, size_t size) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

/* ============================================================================
 * GPU Stream Management Wrappers
 * ========================================================================= */

static int fb_cublas_stream_create(void* handle, void** stream) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
    return (err == cudaSuccess) ? 0 : -1;
#else
    *stream = NULL;
    return -1;
#endif
}

static int fb_cublas_stream_destroy(void* handle, void* stream) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

static int fb_cublas_stream_sync(void* handle, void* stream) {
    (void)handle;
#if CUBLAS_AVAILABLE
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

static int fb_cublas_stream_set(void* handle, void* stream) {
    (void)handle;
    (void)stream;
    /* cuBLAS uses cublasSetStream() - would need context to implement */
    return 0;  /* Return success for now */
}

/* ============================================================================
 * Capability and Property Wrappers
 * ========================================================================= */

static uint32_t fb_cublas_get_capabilities_wrapper(void* handle) {
    (void)handle;
    return FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
           FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

static int fb_cublas_get_num_threads_wrapper(void* handle) {
    (void)handle;
    return -1; /* Not applicable for GPU backends */
}

static void fb_cublas_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads; /* Not applicable for GPU backends */
}

static const fb_plugin_metadata_t g_cublas_metadata = {
    .name = "cublas",
    .version = "12.0",
    .vendor = "NVIDIA Corporation",
    .description = "NVIDIA CUDA cuBLAS - GPU-accelerated BLAS for NVIDIA GPUs",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_THREADSAFE
};

/* Note: GPU backends currently use the trait interface, not direct vtable wrappers */
/* These would need to be implemented to fully integrate with the plugin system */

static fb_plugin_probe_result_t cublas_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#if !CUBLAS_AVAILABLE
    result.score = 0;
    result.reason = "cuBLAS not enabled in build (FB_ENABLE_CUDA not defined)";
    return result;
#else
    
    /* Check if NVIDIA GPU is available */
    if (!is_nvidia_gpu_available()) {
        result.score = 0;
        result.reason = "No NVIDIA GPU detected";
        return result;
    }
    
    /* Try to find CUDA runtime library */
    const char* lib_names[] = {
#ifdef _WIN32
        "cudart64_13.dll",
        "cudart64_12.dll",
        "cudart64_110.dll",
        "cudart64_11.dll",
#elif defined(__APPLE__)
        "libcudart.dylib",
#else
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\bin",
#elif defined(__APPLE__)
        "/usr/local/cuda/lib",
#else
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* NVIDIA GPU detected and CUDA runtime available */
        result.score = 95; /* High score for NVIDIA GPU */
        result.library_path = NULL;
        result.reason = "Found NVIDIA GPU with CUDA runtime";
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "NVIDIA GPU detected but CUDA runtime not found";
    }
    
    return result;
#endif
}

static int cublas_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    fprintf(stderr, "[cuBLAS Plugin] Init called\n");
    
    if (!ctx_out) {
        fprintf(stderr, "[cuBLAS Plugin] ctx_out is NULL\n");
        return -1;
    }
    
#if !CUBLAS_AVAILABLE
    fprintf(stderr, "[cuBLAS Plugin] CUBLAS_AVAILABLE is 0 (CUDA not enabled)\n");
    return -1; /* Not available */
#else
    fprintf(stderr, "[cuBLAS Plugin] CUBLAS_AVAILABLE is 1, proceeding\n");
    
    cublas_plugin_context_t* ctx = (cublas_plugin_context_t*)calloc(1, sizeof(cublas_plugin_context_t));
    if (!ctx) {
        fprintf(stderr, "[cuBLAS Plugin] Failed to allocate context\n");
        return -2;
    }
    
    ctx->lib_handle = lib_handle;
    
    /* Get device count */
    cudaError_t err = cudaGetDeviceCount(&ctx->device_count);
    fprintf(stderr, "[cuBLAS Plugin] cudaGetDeviceCount returned %d, count=%d\n", err, ctx->device_count);
    if (err != cudaSuccess || ctx->device_count == 0) {
        fprintf(stderr, "[cuBLAS Plugin] No CUDA devices or cudaGetDeviceCount failed\n");
        free(ctx);
        return -3;
    }
    
    g_cublas_context = ctx;
    fprintf(stderr, "[cuBLAS Plugin] Context created, populating vtable\n");
    
    /* Zero out vtable first to ensure clean state */
    memset(&g_cublas_vtable, 0, sizeof(fb_backend_vtable_t));
    
    /* Populate vtable with all cuBLAS operations (memory, streams, BLAS ops) */
    cublas_populate_vtable(&g_cublas_vtable);
    fprintf(stderr, "[cuBLAS Plugin] Vtable populated, initializing smart wrappers\n");
    
    /* Initialize cuBLAS smart wrappers (sets up backend handle and device memory manager) */
    if (fb_cublas_smart_init(0, lib_handle) != 0) {
        fprintf(stderr, "[cuBLAS Plugin] Failed to initialize smart wrappers\n");
        free(ctx);
        return -4;
    }
    
    /* Set the backend handle for vtable wrapper functions (mem_alloc, mem_free, etc.)
     * g_cublas_handle is set by fb_cublas_smart_init via fb_cublas_trait.init */
    cublas_wrappers_set_handle(g_cublas_handle);
    fprintf(stderr, "[cuBLAS Plugin] cublas_wrappers_set_handle called with %p\n", g_cublas_handle);
    
    fprintf(stderr, "[cuBLAS Plugin] Init completed successfully\n");
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
#endif
}

static const fb_backend_vtable_t* cublas_get_vtable(fb_plugin_context_t* ctx) {
    /* GPU backends use trait interface, not CPU vtable */
    /* This would need GPU-specific vtable or trait integration */
    return &g_cublas_vtable;
}

static void* cublas_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void cublas_shutdown(fb_plugin_context_t* ctx) {
    /* Shutdown cuBLAS smart wrappers */
    fb_cublas_smart_shutdown();
    
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_cublas_plugin = {
    .metadata = &g_cublas_metadata,
    .probe = cublas_probe,
    .init = cublas_init,
    .get_vtable = cublas_get_vtable,
    .get_context = cublas_get_context,
    .shutdown = cublas_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function */
void fb_register_cublas_plugin(void) {
    fb_register_plugin(&g_cublas_plugin);
}

#ifdef __cplusplus
}
#endif
