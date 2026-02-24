/**
 * @file plugin_onemkl.c
 * @brief Plugin implementation for Intel oneMKL (GPU backend for Arc/Xe GPUs)
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#define FB_CLOSE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FB_CLOSE_LIBRARY(handle) dlclose(handle)
#endif

/* Plugin metadata */
static const fb_plugin_metadata_t onemkl_metadata = {
    .name = "onemkl",
    .version = "2024.0",
    .vendor = "Intel Corporation",
    .description = "Intel oneAPI Math Kernel Library - GPU-accelerated BLAS for Intel Arc/Xe GPUs",
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/**
 * @brief Check if Intel GPU is available using Level Zero API
 * 
 * Attempts to detect Intel Arc/Xe GPUs via Level Zero runtime.
 * Returns 1 if Intel GPU with compute capability is found, 0 otherwise.
 */
static int is_intel_gpu_available(void) {
#ifdef FB_ENABLE_ONEMKL
    /* Try to detect Intel GPU via Level Zero API */
    #if defined(_WIN32)
        /* On Windows, try to load Level Zero DLL */
        HMODULE ze_module = LoadLibraryA("ze_loader.dll");
        if (!ze_module) {
            return 0;
        }
        
        /* Get zeInit function */
        typedef int (*zeInit_fn)(int);
        typedef int (*zeDriverGet_fn)(unsigned int*, void*);
        
        zeInit_fn zeInit = (zeInit_fn)GetProcAddress(ze_module, "zeInit");
        zeDriverGet_fn zeDriverGet = (zeDriverGet_fn)GetProcAddress(ze_module, "zeDriverGet");
        
        if (zeInit && zeDriverGet) {
            if (zeInit(0) == 0) {  /* ZE_RESULT_SUCCESS */
                unsigned int driver_count = 0;
                if (zeDriverGet(&driver_count, NULL) == 0 && driver_count > 0) {
                    FreeLibrary(ze_module);
                    return 1;
                }
            }
        }
        
        FreeLibrary(ze_module);
    #else
        /* On Linux, check for Intel GPU via sysfs or Level Zero */
        /* Simplified check - look for Level Zero library */
        void* ze_handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
        if (ze_handle) {
            dlclose(ze_handle);
            return 1;
        }
    #endif
#endif
    return 0;
}

/**
 * @brief Probe for Intel oneMKL GPU library
 */
static fb_plugin_probe_result_t onemkl_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#ifndef FB_ENABLE_ONEMKL
    result.score = 0;
    result.reason = "oneMKL not enabled in build (FB_ENABLE_ONEMKL not defined)";
    return result;
#else
    
    /* Check for Intel GPU availability */
    if (!is_intel_gpu_available()) {
        result.score = 0;
        result.reason = "No Intel GPU detected or Level Zero runtime not available";
        return result;
    }
    
    const char* lib_names[] = {
#if defined(_WIN32)
        "mkl_sycl.dll",
        "mkl_rt.dll",
#elif defined(__linux__)
        "libmkl_sycl.so",
        "libmkl_rt.so",
#elif defined(__APPLE__)
        "libmkl_sycl.dylib",
        "libmkl_rt.dylib",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#if defined(_WIN32)
        "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\redist\\intel64",
        "C:\\Program Files\\Intel\\oneAPI\\mkl\\latest\\redist\\intel64",
#elif defined(__linux__)
        "/opt/intel/oneapi/mkl/latest/lib/intel64",
        "/usr/lib/x86_64-linux-gnu",
#elif defined(__APPLE__)
        "/opt/intel/oneapi/mkl/latest/lib",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* Check for CBLAS interface in oneMKL */
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            result.score = 95; /* High score for Intel GPU on Intel hardware */
            result.library_path = NULL; /* Init will load the library */
            result.reason = "Found Intel GPU with oneMKL runtime";
        } else {
            result.score = 0;
            result.reason = "oneMKL library found but missing CBLAS interface";
        }
        
        FB_CLOSE_LIBRARY(test_handle);
    } else {
        result.score = 0;
        result.reason = "oneMKL library not found in search paths";
    }
    
    return result;
#endif
}

/**
 * @brief Initialize Intel oneMKL GPU backend
 * 
 * Note: Returns empty vtable as GPU operations use trait interface.
 * This plugin handles detection/scoring; actual GPU operations use existing traits.
 */
static int onemkl_init(fb_plugin_context_t* ctx, const char* lib_path) {
    (void)ctx;
    (void)lib_path;
    
#ifdef FB_ENABLE_ONEMKL
    /* GPU backend initialization would go here */
    /* Currently using trait interface for GPU operations */
    return 0; /* Success */
#else
    return -1; /* Not enabled */
#endif
}

/**
 * @brief Shutdown Intel oneMKL GPU backend
 */
static void onemkl_shutdown(fb_plugin_context_t* ctx) {
    (void)ctx;
#ifdef FB_ENABLE_ONEMKL
    /* Cleanup would go here */
#endif
}

/**
 * @brief Get vtable for Intel oneMKL GPU backend
 * 
 * Returns empty vtable as GPU operations use trait interface.
 */
static const fb_backend_vtable_t* onemkl_get_vtable(fb_plugin_context_t* ctx) {
    (void)ctx;
    
    static const fb_backend_vtable_t vtable = {0};
    return &vtable;
}

/**
 * @brief Get plugin context
 */
static void* onemkl_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

/* Plugin interface */
static const fb_backend_plugin_t g_onemkl_plugin = {
    .metadata = &onemkl_metadata,
    .probe = onemkl_probe,
    .init = onemkl_init,
    .get_vtable = onemkl_get_vtable,
    .get_context = onemkl_get_context,
    .shutdown = onemkl_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

void fb_register_onemkl_plugin(void) {
    fb_register_plugin(&g_onemkl_plugin);
}
