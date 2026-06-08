/**
 * @file plugin_clblas.c
 * @brief clBLAS OpenCL BLAS Plugin
 * 
 * clBLAS is AMD's OpenCL BLAS library (now in maintenance mode).
 * Supports AMD and other OpenCL-capable GPUs.
 * 
 * Homepage: https://github.com/clMathLibraries/clBLAS
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* clBLAS availability check */
#if defined(FB_ENABLE_OPENCL)
#define CLBLAS_AVAILABLE 1
#else
#define CLBLAS_AVAILABLE 0
#endif

#if CLBLAS_AVAILABLE
#include <CL/cl.h>
#endif

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* OpenCL GPU detection */
static int is_opencl_gpu_available(void) {
#if CLBLAS_AVAILABLE
    cl_platform_id platform_id = NULL;
    cl_uint num_platforms = 0;
    
    /* Try to get at least one OpenCL platform */
    cl_int err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err == CL_SUCCESS && num_platforms > 0) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        return (err == CL_SUCCESS && num_devices > 0);
    }
#endif
    return 0;
}

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    int device_count;
} clblas_plugin_context_t;

static fb_backend_vtable_t g_clblas_vtable = {0};
static clblas_plugin_context_t* g_clblas_context = NULL;

static const fb_plugin_metadata_t g_clblas_metadata = {
    .name = "clblas",
    .version = "2.12",
    .vendor = "AMD (clMath)",
    .description = "clBLAS - AMD OpenCL BLAS library",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC
};

/* Probe function */
static fb_plugin_probe_result_t clblas_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#if !CLBLAS_AVAILABLE
    result.score = 0;
    result.reason = "clBLAS not enabled in build (FB_ENABLE_OPENCL not defined)";
    return result;
#else
    
    /* Check if OpenCL GPU is available */
    if (!is_opencl_gpu_available()) {
        result.score = 0;
        result.reason = "No OpenCL GPU detected";
        return result;
    }
    
    /* Try to find clBLAS library */
    const char* lib_names[] = {
#ifdef _WIN32
        "clBLAS.dll",
#elif defined(__APPLE__)
        "libclBLAS.dylib",
#else
        "libclBLAS.so",
        "libclBLAS.so.2",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\clBLAS\\lib",
        "C:\\Program Files (x86)\\clBLAS\\lib",
#elif defined(__APPLE__)
        "/usr/local/lib",
        "/opt/homebrew/lib",
#else
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        result.score = 80; /* Slightly lower than CLBlast (maintenance mode) */
        result.library_path = NULL;
        result.reason = "Found OpenCL GPU with clBLAS library";
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "OpenCL GPU detected but clBLAS library not found";
    }
    
    return result;
#endif
}

static int clblas_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
#if !CLBLAS_AVAILABLE
    return -1;
#else
    
    clblas_plugin_context_t* ctx = (clblas_plugin_context_t*)calloc(1, sizeof(clblas_plugin_context_t));
    if (!ctx) {
        return -1;
    }
    
    ctx->lib_handle = lib_handle;
    ctx->device_count = 0; /* Initialize as needed */
    
    /* Note: Actual clBLAS operations would be loaded here */
    /* clBLAS requires clblasSetup() call */
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
#endif
}

static const fb_backend_vtable_t* clblas_get_vtable(fb_plugin_context_t* ctx) {
    /* GPU backends use trait interface or GPU-specific vtable */
    return &g_clblas_vtable;
}

static void* clblas_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void clblas_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_clblas_plugin = {
    .metadata = &g_clblas_metadata,
    .probe = clblas_probe,
    .init = clblas_init,
    .get_vtable = clblas_get_vtable,
    .get_context = clblas_get_context,
    .shutdown = clblas_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function */
void fb_register_clblas_plugin(void) {
    fb_register_plugin(&g_clblas_plugin);
}

#ifdef __cplusplus
}
#endif
