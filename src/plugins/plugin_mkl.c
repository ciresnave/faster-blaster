/**
 * @file plugin_mkl.c
 * @brief Intel MKL plugin implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include "../backends/backend_auto_detect.h"   /* fb_enumerate_and_populate  */
#include "faster-blaster/vtable_autofill.h"    /* fb_vtable_sync_ext_ops     */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #include <cpuid.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* CPU vendor detection */
static int is_intel_cpu(void) {
    int cpu_info[4];
    char vendor[13];
    
#ifdef _WIN32
    __cpuid(cpu_info, 0);
#else
    __cpuid(0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif
    
    memcpy(vendor, &cpu_info[1], 4);
    memcpy(vendor + 4, &cpu_info[3], 4);
    memcpy(vendor + 8, &cpu_info[2], 4);
    vendor[12] = '\0';
    
    return strcmp(vendor, "GenuineIntel") == 0;
}

/* MKL specific functions */
typedef void (*mkl_set_num_threads_t)(int num_threads);
typedef int (*mkl_get_max_threads_t)(void);

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    mkl_set_num_threads_t set_num_threads;
    mkl_get_max_threads_t get_max_threads;
} mkl_plugin_context_t;

static fb_backend_vtable_t g_mkl_vtable;
static mkl_plugin_context_t* g_mkl_context = NULL;

static const fb_plugin_metadata_t g_mkl_metadata = {
    .name = "mkl",
    .version = "2024.0",
    .vendor = "Intel Corporation",
    .description = "Intel Math Kernel Library - highly optimized for Intel CPUs",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_THREADSAFE
};

/* Wrapper functions */
/* Backend capability functions */
static uint32_t fb_mkl_get_capabilities_wrapper(void* handle) {
    (void)handle;
    return FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
           FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

static int fb_mkl_get_num_threads_wrapper(void* handle) {
    (void)handle;
    /* MKL has mkl_get_max_threads - use context if available */
    if (g_mkl_context && g_mkl_context->get_max_threads) {
        return g_mkl_context->get_max_threads();
    }
    return 1;
}

static void fb_mkl_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    /* MKL has mkl_set_num_threads - use context if available */
    if (g_mkl_context && g_mkl_context->set_num_threads) {
        g_mkl_context->set_num_threads(num_threads);
    }
}

static fb_plugin_probe_result_t mkl_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
    const char* lib_names[] = {
#ifdef _WIN32
        "mkl_rt.dll",
        "mkl_rt.2.dll",
#elif defined(__APPLE__)
        "libmkl_rt.dylib",
        "libmkl_rt.2.dylib",
#else
        "libmkl_rt.so",
        "libmkl_rt.so.2",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\redist\\intel64",
        "C:\\Program Files\\Intel\\MKL\\redist\\intel64",
        "C:\\libraries\\mkl\\bin",
#elif defined(__APPLE__)
        "/opt/intel/oneapi/mkl/latest/lib",
        "/usr/local/lib",
#else
        "/opt/intel/oneapi/mkl/latest/lib/intel64",
        "/opt/intel/mkl/lib/intel64",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            /* Adjust score based on CPU vendor */
            if (is_intel_cpu()) {
                result.score = 95; /* MKL is highly optimized for Intel CPUs */
                result.reason = "Found Intel MKL with CBLAS interface (Intel CPU detected)";
            } else {
                result.score = 70; /* Lower priority on non-Intel CPUs */
                result.reason = "Found Intel MKL with CBLAS interface (non-Intel CPU - not optimal)";
            }
            result.library_path = NULL; /* Init will load the library */
        } else {
            result.score = 0;
            result.reason = "Library found but missing required CBLAS symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "Intel MKL library not found in search paths";
    }
    
    return result;
}

static int mkl_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
    /* Load the library if not already provided */
    if (!lib_handle) {
        const char* lib_names[] = {
#ifdef _WIN32
            "mkl_rt.dll",
            "mkl_rt.2.dll",
#elif defined(__APPLE__)
            "libmkl_rt.dylib",
            "libmkl_rt.2.dylib",
#else
            "libmkl_rt.so",
            "libmkl_rt.so.2",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\redist\\intel64",
            "C:\\Program Files\\Intel\\MKL\\redist\\intel64",
            "C:\\libraries\\mkl\\bin",
#elif defined(__APPLE__)
            "/opt/intel/oneapi/mkl/latest/lib",
            "/usr/local/lib",
#else
            "/opt/intel/oneapi/mkl/latest/lib/intel64",
            "/opt/intel/mkl/lib/intel64",
            "/usr/lib/x86_64-linux-gnu",
#endif
            NULL
        };
        
        lib_handle = fb_plugin_load_library(lib_names, default_paths);
        if (!lib_handle) {
            return -1; /* Library not found */
        }
    }
    
    mkl_plugin_context_t* ctx = (mkl_plugin_context_t*)calloc(1, sizeof(mkl_plugin_context_t));
    if (!ctx) {
        return -2;
    }
    
    ctx->lib_handle = lib_handle;

    /* Auto-populate ext_ops[op][conv] for all exported MKL BLAS/LAPACK symbols. */
    fb_enumerate_and_populate(&g_mkl_vtable, lib_handle);
    fb_vtable_sync_ext_ops(&g_mkl_vtable);

    /* Load MKL-specific functions */
    ctx->set_num_threads = (mkl_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "MKL_Set_Num_Threads");
    ctx->get_max_threads = (mkl_get_max_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "MKL_Get_Max_Threads");

    if (g_mkl_vtable.sdot == NULL || g_mkl_vtable.sgemm == NULL) {
        free(ctx);
        return -3;
    }

    g_mkl_context = ctx;

    /* CPU backend properties */
    g_mkl_vtable.mem_alloc = NULL;
    g_mkl_vtable.mem_free = NULL;
    g_mkl_vtable.mem_upload = NULL;
    g_mkl_vtable.mem_download = NULL;
    g_mkl_vtable.mem_copy = NULL;
    g_mkl_vtable.stream_create = NULL;
    g_mkl_vtable.stream_destroy = NULL;
    g_mkl_vtable.stream_sync = NULL;
    g_mkl_vtable.stream_set = NULL;
    g_mkl_vtable.get_capabilities = fb_mkl_get_capabilities_wrapper;
    g_mkl_vtable.get_num_threads = fb_mkl_get_num_threads_wrapper;
    g_mkl_vtable.set_num_threads = fb_mkl_set_num_threads_wrapper;
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
}

static const fb_backend_vtable_t* mkl_get_vtable(fb_plugin_context_t* ctx) {
    return &g_mkl_vtable;
}

static void* mkl_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void mkl_set_num_threads(fb_plugin_context_t* ctx, int num_threads) {
    if (g_mkl_context && g_mkl_context->set_num_threads) {
        g_mkl_context->set_num_threads(num_threads);
    }
}

static int mkl_get_num_threads(fb_plugin_context_t* ctx) {
    if (g_mkl_context && g_mkl_context->get_max_threads) {
        return g_mkl_context->get_max_threads();
    }
    return -1;
}

static void mkl_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_mkl_plugin = {
    .metadata = &g_mkl_metadata,
    .probe = mkl_probe,
    .init = mkl_init,
    .get_vtable = mkl_get_vtable,
    .get_context = mkl_get_context,
    .shutdown = mkl_shutdown,
    .set_num_threads = mkl_set_num_threads,
    .get_num_threads = mkl_get_num_threads
};

/* Plugin registration function - must be called explicitly */
void fb_register_mkl_plugin(void) {
    fb_register_plugin(&g_mkl_plugin);
}
