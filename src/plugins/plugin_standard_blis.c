/**
 * @file plugin_standard_blis.c
 * @brief Standard BLIS plugin with CBLAS fallback for AOCL compatibility
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
#include <stdbool.h>

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
} blis_plugin_context_t;

static fb_backend_vtable_t g_blis_vtable;
static blis_plugin_context_t* g_blis_context = NULL;

static const fb_plugin_metadata_t g_blis_metadata = {
    .name = "standard-blis",
    .version = "0.9.0",
    .vendor = "BLIS Project",
    .description = "BLAS-like Library Instantiation Software",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/* BLAS operations are dispatched directly via ext_ops[op][conv] populated by
 * fb_enumerate_and_populate() — no per-operation wrapper functions needed. */

/* Backend capability functions */
static uint32_t fb_blis_get_capabilities_wrapper(void* handle) {
    (void)handle;
    /* BLIS only provides Level 1 BLAS operations */
    return FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

/* BLIS doesn't expose thread control in standard interface */
static int fb_blis_get_num_threads_wrapper(void* handle) {
    (void)handle;
    return 1;
}

static void fb_blis_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* No-op */
}

static fb_plugin_probe_result_t blis_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
    const char* lib_names[] = {
#ifdef _WIN32
        "blis.dll",
        "libblis.dll",
        "AOCL-LibBlis-Win-dll.dll", /* May find AOCL BLIS */
#elif defined(__APPLE__)
        "libblis.dylib",
#else
        "libblis.so",
        "libblis.so.4",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\BLIS\\lib",
        "C:\\libraries\\blis\\lib",
#elif defined(__APPLE__)
        "/usr/local/lib",
        "/opt/homebrew/lib",
#else
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* Check for CBLAS symbols first (AOCL BLIS) */
        void* cblas_sdot = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* bli_sdotv = FB_GET_PROC_ADDRESS(test_handle, "bli_sdotv");
        
        if (cblas_sdot) {
            result.score = 85; /* Slightly lower than dedicated AOCL plugin */
            result.library_path = NULL; /* Init will load the library */
            result.reason = "Found BLIS with CBLAS interface (likely AOCL)";
        } else if (bli_sdotv) {
            result.score = 80; /* Standard BLIS native API */
            result.library_path = NULL; /* Init will load the library */
            result.reason = "Found standard BLIS with native API";
        } else {
            result.score = 0;
            result.reason = "Library found but missing required symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "BLIS library not found in search paths";
    }
    
    return result;
}

static int blis_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
    /* Load the library if not provided */
    if (!lib_handle) {
        const char* lib_names[] = {
#ifdef _WIN32
            "blis.dll",
            "libblis.dll",
            "AOCL-LibBlis-Win-dll.dll", /* May find AOCL BLIS */
#elif defined(__APPLE__)
            "libblis.dylib",
#else
            "libblis.so",
            "libblis.so.4",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            "C:\\Program Files\\BLIS\\lib",
            "C:\\libraries\\blis\\lib",
#elif defined(__APPLE__)
            "/usr/local/lib",
            "/opt/homebrew/lib",
#else
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
#endif
            NULL
        };
        
        lib_handle = fb_plugin_load_library(lib_names, default_paths);
        if (!lib_handle) {
            return -1;
        }
    }
    
    blis_plugin_context_t* ctx = (blis_plugin_context_t*)calloc(1, sizeof(blis_plugin_context_t));
    if (!ctx) {
        return -2;
    }
    
    ctx->lib_handle = lib_handle;

    /* Enumerate all exported BLAS/LAPACK symbols and fill ext_ops[op][conv]. */
    fb_enumerate_and_populate(&g_blis_vtable, lib_handle);
    fb_vtable_sync_ext_ops(&g_blis_vtable);

    /* Require sdot to be present as basic sanity check. */
    if (g_blis_vtable.sdot == NULL) {
        free(ctx);
        return -3;
    }
    g_blis_context = ctx;

    /* CPU backend properties */
    g_blis_vtable.mem_alloc = NULL;
    g_blis_vtable.mem_free = NULL;
    g_blis_vtable.mem_upload = NULL;
    g_blis_vtable.mem_download = NULL;
    g_blis_vtable.mem_copy = NULL;
    g_blis_vtable.stream_create = NULL;
    g_blis_vtable.stream_destroy = NULL;
    g_blis_vtable.stream_sync = NULL;
    g_blis_vtable.stream_set = NULL;
    g_blis_vtable.get_capabilities = fb_blis_get_capabilities_wrapper;
    g_blis_vtable.get_num_threads = fb_blis_get_num_threads_wrapper;
    g_blis_vtable.set_num_threads = fb_blis_set_num_threads_wrapper;
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
}

static const fb_backend_vtable_t* blis_get_vtable(fb_plugin_context_t* ctx) {
    return &g_blis_vtable;
}

static void* blis_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void blis_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_blis_plugin = {
    .metadata = &g_blis_metadata,
    .probe = blis_probe,
    .init = blis_init,
    .get_vtable = blis_get_vtable,
    .get_context = blis_get_context,
    .shutdown = blis_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function - must be called explicitly */
void fb_register_standard_blis_plugin(void) {
    fb_register_plugin(&g_blis_plugin);
}
