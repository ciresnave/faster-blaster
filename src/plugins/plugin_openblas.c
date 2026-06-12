/**
 * @file plugin_openblas.c
 * @brief OpenBLAS plugin implementation
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
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* OpenBLAS thread control function types */
typedef int (*openblas_get_num_threads_t)(void);
typedef void (*openblas_set_num_threads_t)(int num_threads);

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    openblas_get_num_threads_t get_num_threads;
    openblas_set_num_threads_t set_num_threads;
    int is_single_threaded;
} openblas_plugin_context_t;

static fb_backend_vtable_t g_openblas_vtable;
static openblas_plugin_context_t* g_openblas_context = NULL;

static const fb_plugin_metadata_t g_openblas_metadata = {
    .name = "openblas",
    .version = "0.3.30",
    .vendor = "OpenBLAS Project",
    .description = "Optimized BLAS library based on GotoBLAS2",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/* BLAS operations are dispatched directly via ext_ops[op][conv] populated by
 * fb_enumerate_and_populate() — no per-operation wrapper functions needed. */

/* Backend capability functions */
static uint32_t fb_openblas_get_capabilities_wrapper(void* handle) {
    (void)handle;
    uint32_t caps = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                    FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
    
    /* Add THREADSAFE only if threading actually works */
    if (g_openblas_context && !g_openblas_context->is_single_threaded) {
        caps |= FB_PLUGIN_CAP_THREADSAFE;
    }
    
    return caps;
}

static int fb_openblas_get_num_threads_wrapper(void* handle) {
    (void)handle;
    if (g_openblas_context && g_openblas_context->get_num_threads) {
        int threads = g_openblas_context->get_num_threads();
        printf("[OpenBLAS] get_num_threads() returned: %d\n", threads);
        return threads;
    }
    printf("[OpenBLAS] get_num_threads() not available, returning 1\n");
    return 1;  /* Fallback to single-threaded */
}

static void fb_openblas_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    printf("[OpenBLAS] set_num_threads(%d) called\n", num_threads);
    if (g_openblas_context && g_openblas_context->set_num_threads) {
        g_openblas_context->set_num_threads(num_threads);
        printf("[OpenBLAS] set_num_threads executed\n");
        /* Verify it worked */
        if (g_openblas_context->get_num_threads) {
            int actual = g_openblas_context->get_num_threads();
            printf("[OpenBLAS] Verification: threads now = %d\n", actual);
        }
    } else {
        printf("[OpenBLAS] set_num_threads() not available\n");
    }
}

static fb_plugin_probe_result_t openblas_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
    const char* lib_names[] = {
#ifdef _WIN32
        "openblas.dll",
        "libopenblas.dll",
#elif defined(__APPLE__)
        "libopenblas.dylib",
        "libopenblas.0.dylib",
#else
        "libopenblas.so",
        "libopenblas.so.0",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\libraries\\vcpkg\\installed\\x64-windows\\bin",
        "C:\\Program Files\\OpenBLAS\\bin",
        "C:\\libraries\\openblas\\bin",
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
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            result.score = 80; /* Good general-purpose BLAS */
            result.library_path = NULL; /* Init will load the library */
            result.reason = "Found OpenBLAS with CBLAS interface";
        } else {
            result.score = 0;
            result.reason = "Library found but missing required CBLAS symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "OpenBLAS library not found in search paths";
    }
    
    return result;
}

static int openblas_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
    /* Load the library if not already provided */
    if (!lib_handle) {
        const char* lib_names[] = {
#ifdef _WIN32
            "openblas.dll",
            "libopenblas.dll",
#elif defined(__APPLE__)
            "libopenblas.dylib",
            "libopenblas.0.dylib",
#else
            "libopenblas.so",
            "libopenblas.so.0",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            "C:\\libraries\\vcpkg\\installed\\x64-windows\\bin",
            "C:\\Program Files\\OpenBLAS\\bin",
            "C:\\libraries\\openblas\\bin",
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
            return -1; /* Library not found */
        }
    }
    
    openblas_plugin_context_t* ctx = (openblas_plugin_context_t*)calloc(1, sizeof(openblas_plugin_context_t));
    if (!ctx) {
        return -2;
    }
    
    ctx->lib_handle = lib_handle;

    /* Enumerate all exported BLAS/LAPACK symbols and fill ext_ops[op][conv]. */
    fb_enumerate_and_populate(&g_openblas_vtable, lib_handle);

    /* Load OpenBLAS thread control functions (optional) */
    /* Try multiple possible symbol names */
    ctx->get_num_threads = (openblas_get_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "openblas_get_num_threads");
    if (!ctx->get_num_threads) {
        /* Try with underscore suffix (some builds) */
        ctx->get_num_threads = (openblas_get_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "openblas_get_num_threads_");
    }
    if (!ctx->get_num_threads) {
        /* Try goto_get_num_procs (older GotoBLAS-style name) */
        ctx->get_num_threads = (openblas_get_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "goto_get_num_procs");
    }
    
    ctx->set_num_threads = (openblas_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "openblas_set_num_threads");
    if (!ctx->set_num_threads) {
        /* Try with underscore suffix */
        ctx->set_num_threads = (openblas_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "openblas_set_num_threads_");
    }
    if (!ctx->set_num_threads) {
        /* Try omp_set_num_threads as fallback */
        ctx->set_num_threads = (openblas_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "omp_set_num_threads");
    }
    
    #ifdef _WIN32
    /* Windows builds might have different export names */
    if (!ctx->get_num_threads) {
        ctx->get_num_threads = (openblas_get_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "OPENBLAS_GET_NUM_THREADS");
    }
    if (!ctx->set_num_threads) {
        ctx->set_num_threads = (openblas_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "OPENBLAS_SET_NUM_THREADS");
    }
    #endif
    
    printf("[OpenBLAS] Thread control: get=%p, set=%p\n", 
           (void*)ctx->get_num_threads, (void*)ctx->set_num_threads);
    
    /* Test if threading actually works */
    ctx->is_single_threaded = 0;
    if (ctx->get_num_threads && ctx->set_num_threads) {
        int original_threads = ctx->get_num_threads();
        int test_threads = (original_threads == 1) ? 2 : 1;
        ctx->set_num_threads(test_threads);
        int actual_threads = ctx->get_num_threads();
        
        if (actual_threads != test_threads) {
            printf("[OpenBLAS] WARNING: Threading API present but non-functional\n");
            printf("[OpenBLAS] This OpenBLAS build was compiled without threading support\n");
            ctx->is_single_threaded = 1;
        } else {
            printf("[OpenBLAS] Threading capability verified: %d threads\n", actual_threads);
        }
        
        /* Restore original thread count */
        ctx->set_num_threads(original_threads);
    } else if (!ctx->get_num_threads || !ctx->set_num_threads) {
        printf("[OpenBLAS] Threading API not available (expected for vcpkg builds)\n");
        ctx->is_single_threaded = 1;
    }
    
    fb_vtable_sync_ext_ops(&g_openblas_vtable);
    if (g_openblas_vtable.sdot == NULL || g_openblas_vtable.sgemm == NULL) {
        printf("[OpenBLAS] WARNING: Missing sdot/sgemm named aliases after vtable sync; continuing with ext_ops-populated backend\n");
    }
    g_openblas_context = ctx;

    /* CPU backend properties */
    g_openblas_vtable.mem_alloc = NULL;
    g_openblas_vtable.mem_free = NULL;
    g_openblas_vtable.mem_upload = NULL;
    g_openblas_vtable.mem_download = NULL;
    g_openblas_vtable.mem_copy = NULL;
    g_openblas_vtable.stream_create = NULL;
    g_openblas_vtable.stream_destroy = NULL;
    g_openblas_vtable.stream_sync = NULL;
    g_openblas_vtable.stream_set = NULL;
    g_openblas_vtable.get_capabilities = fb_openblas_get_capabilities_wrapper;
    g_openblas_vtable.get_num_threads = fb_openblas_get_num_threads_wrapper;
    g_openblas_vtable.set_num_threads = fb_openblas_set_num_threads_wrapper;
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
}

static const fb_backend_vtable_t* openblas_get_vtable(fb_plugin_context_t* ctx) {
    (void)ctx;
    return &g_openblas_vtable;
}

static void* openblas_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void openblas_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_openblas_plugin = {
    .metadata = &g_openblas_metadata,
    .probe = openblas_probe,
    .init = openblas_init,
    .get_vtable = openblas_get_vtable,
    .get_context = openblas_get_context,
    .shutdown = openblas_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function - must be called explicitly */
void fb_register_openblas_plugin(void) {
    fb_register_plugin(&g_openblas_plugin);
}
