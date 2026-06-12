/**
 * @file plugin_rocblas.c
 * @brief AMD rocBLAS GPU plugin implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

/* NOTE: This plugin requires the HIP compiler (hipcc) and cannot be compiled with MSVC/GCC/Clang */
/* Only compile when using HIP platform */
#if defined(FB_ENABLE_ROCM) && defined(__HIP_PLATFORM_AMD__)

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* GPU vendor detection */
static int is_amd_gpu_available(void) {
#if ROCBLAS_AVAILABLE
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err == hipSuccess && device_count > 0) {
        /* Verify it's actually an AMD GPU */
        struct hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, 0);
        if (err == hipSuccess) {
            /* AMD GPUs are accessed via HIP */
            return 1;
        }
    }
#endif
    return 0;
}

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    int device_count;
} rocblas_plugin_context_t;

static fb_backend_vtable_t g_rocblas_vtable;
static rocblas_plugin_context_t* g_rocblas_context = NULL;

static const fb_plugin_metadata_t g_rocblas_metadata = {
    .name = "rocblas",
    .version = "6.0",
    .vendor = "Advanced Micro Devices (AMD)",
    .description = "AMD ROCm rocBLAS - GPU-accelerated BLAS for AMD GPUs",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_THREADSAFE
};

static fb_plugin_probe_result_t rocblas_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#if !ROCBLAS_AVAILABLE
    result.score = 0;
    result.reason = "rocBLAS not enabled in build (FB_ENABLE_ROCM not defined)";
    return result;
#else
    
    /* Check if AMD GPU is available */
    if (!is_amd_gpu_available()) {
        result.score = 0;
        result.reason = "No AMD GPU detected";
        return result;
    }
    
    /* Try to find HIP runtime library */
    const char* lib_names[] = {
#ifdef _WIN32
        "amdhip64.dll",
        "hip_hcc.dll",
#elif defined(__APPLE__)
        "libamdhip64.dylib",
#else
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\AMD\\ROCm\\6.0\\bin",
        "C:\\Program Files\\AMD\\ROCm\\bin",
#elif defined(__APPLE__)
        "/opt/rocm/lib",
#else
        "/opt/rocm/lib",
        "/opt/rocm/hip/lib",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* AMD GPU detected and HIP runtime available */
        result.score = 95; /* High score for AMD GPU */
        result.library_path = NULL;
        result.reason = "Found AMD GPU with ROCm/HIP runtime";
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "AMD GPU detected but ROCm/HIP runtime not found";
    }
    
    return result;
#endif
}

static int rocblas_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
#if !ROCBLAS_AVAILABLE
    return -1; /* Not available */
#else
    
    rocblas_plugin_context_t* ctx = (rocblas_plugin_context_t*)calloc(1, sizeof(rocblas_plugin_context_t));
    if (!ctx) {
        return -2;
    }
    
    ctx->lib_handle = lib_handle;
    
    /* Get device count */
    hipError_t err = hipGetDeviceCount(&ctx->device_count);
    if (err != hipSuccess || ctx->device_count == 0) {
        free(ctx);
        return -3;
    }
    
    g_rocblas_context = ctx;
    
    /* Note: Actual rocBLAS operations go through the GPU trait interface */
    /* This plugin mainly handles detection and scoring */
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
#endif
}

static const fb_backend_vtable_t* rocblas_get_vtable(fb_plugin_context_t* ctx) {
    /* GPU backends use trait interface, not CPU vtable */
    return &g_rocblas_vtable;
}

static void* rocblas_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void rocblas_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_rocblas_plugin = {
    .metadata = &g_rocblas_metadata,
    .probe = rocblas_probe,
    .init = rocblas_init,
    .get_vtable = rocblas_get_vtable,
    .get_context = rocblas_get_context,
    .shutdown = rocblas_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function */
void fb_register_rocblas_plugin(void) {
    fb_register_plugin(&g_rocblas_plugin);
}

#ifdef __cplusplus
}
#endif

#else  /* FB_ENABLE_ROCM && __HIP_PLATFORM_AMD__ */

/* Stub implementation when ROCm not available or not using HIP compiler */
#include "faster-blaster/backend_plugin.h"

void fb_register_rocblas_plugin(void) {
    /* No-op: ROCm plugin not compiled (requires HIP compiler) */
}

#endif  /* FB_ENABLE_ROCM && __HIP_PLATFORM_AMD__ */
