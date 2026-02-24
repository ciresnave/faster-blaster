/**
 * @file plugin_standard_blis.c
 * @brief Standard BLIS plugin with CBLAS fallback for AOCL compatibility
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* CBLAS enums */
typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

typedef CBLAS_ORDER CBLAS_LAYOUT;

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* CBLAS function pointer typedefs (fallback for AOCL BLIS) */
typedef float (*cblas_sasum_t)(const int n, const float* x, const int incx);
typedef void (*cblas_saxpy_t)(const int n, const float alpha, const float* x, const int incx, float* y, const int incy);
typedef float (*cblas_sdot_t)(const int n, const float* x, const int incx, const float* y, const int incy);
typedef void (*cblas_scopy_t)(const int n, const float* x, const int incx, float* y, const int incy);
typedef void (*cblas_sscal_t)(const int n, const float alpha, float* x, const int incx);
typedef float (*cblas_snrm2_t)(const int n, const float* x, const int incx);
typedef void (*cblas_sswap_t)(const int n, float* x, const int incx, float* y, const int incy);
typedef int (*cblas_isamax_t)(const int n, const float* x, const int incx);

/* Native BLIS function pointer typedefs */
typedef void (*bli_saxpyv_t)(int conjalpha, int n, const float* alpha, const float* x, int incx, float* y, int incy);
typedef void (*bli_sdotv_t)(int conjx, int conjy, int n, const float* x, int incx, const float* y, int incy, float* rho);
typedef void (*bli_snormfv_t)(int n, const float* x, int incx, float* norm);
typedef void (*bli_scopyv_t)(int n, const float* x, int incx, float* y, int incy);
typedef void (*bli_sscalv_t)(int conjalpha, int n, const float* alpha, float* x, int incx);
typedef void (*bli_sswapv_t)(int n, float* x, int incx, float* y, int incy);

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    bool use_cblas; /* true if CBLAS available (AOCL), false for native BLIS */
    
    /* CBLAS functions (fallback for AOCL) */
    cblas_sasum_t cblas_sasum;
    cblas_saxpy_t cblas_saxpy;
    cblas_sdot_t cblas_sdot;
    cblas_scopy_t cblas_scopy;
    cblas_sscal_t cblas_sscal;
    cblas_snrm2_t cblas_snrm2;
    cblas_sswap_t cblas_sswap;
    cblas_isamax_t cblas_isamax;
    
    /* Native BLIS functions */
    bli_saxpyv_t bli_saxpyv;
    bli_sdotv_t bli_sdotv;
    bli_snormfv_t bli_snormfv;
    bli_scopyv_t bli_scopyv;
    bli_sscalv_t bli_sscalv;
    bli_sswapv_t bli_sswapv;
    
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

/* Wrapper functions */
static float blis_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!g_blis_context) return 0.0f;
    
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis_context->use_cblas && g_blis_context->cblas_sdot) {
        return g_blis_context->cblas_sdot(n, x, incx, y, incy);
    }
    
    /* Fall back to native BLIS */
    if (g_blis_context->bli_sdotv) {
        float result;
        g_blis_context->bli_sdotv(0, 0, n, x, incx, y, incy, &result);
        return result;
    }
    
    return 0.0f;
}

static float blis_snrm2_wrapper(int n, const float* x, int incx) {
    
    if (!g_blis_context) return 0.0f;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_snrm2) {
        return g_blis_context->cblas_snrm2(n, x, incx);
    }
    
    if (g_blis_context->bli_snormfv) {
        float result;
        g_blis_context->bli_snormfv(n, x, incx, &result);
        return result;
    }
    
    return 0.0f;
}

static float blis_sasum_wrapper(int n, const float* x, int incx) {
    
    if (!g_blis_context) return 0.0f;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_sasum) {
        return g_blis_context->cblas_sasum(n, x, incx);
    }
    
    /* BLIS doesn't have asum in native API - would need to implement */
    return 0.0f;
}

static int blis_isamax_wrapper(int n, const float* x, int incx) {
    
    if (!g_blis_context) return 0;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_isamax) {
        return g_blis_context->cblas_isamax(n, x, incx);
    }
    
    return 0;
}

static void blis_saxpy_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    
    if (!g_blis_context) return;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_saxpy) {
        g_blis_context->cblas_saxpy(n, alpha, x, incx, y, incy);
        return;
    }
    
    if (g_blis_context->bli_saxpyv) {
        g_blis_context->bli_saxpyv(0, n, &alpha, x, incx, y, incy);
    }
}

static void blis_scopy_wrapper(int n, const float* x, int incx, float* y, int incy) {
    
    if (!g_blis_context) return;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_scopy) {
        g_blis_context->cblas_scopy(n, x, incx, y, incy);
        return;
    }
    
    if (g_blis_context->bli_scopyv) {
        g_blis_context->bli_scopyv(n, x, incx, y, incy);
    }
}

static void blis_sscal_wrapper(int n, float alpha, float* x, int incx) {
    
    if (!g_blis_context) return;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_sscal) {
        g_blis_context->cblas_sscal(n, alpha, x, incx);
        return;
    }
    
    if (g_blis_context->bli_sscalv) {
        g_blis_context->bli_sscalv(0, n, &alpha, x, incx);
    }
}

static void blis_sswap_wrapper(int n, float* x, int incx, float* y, int incy) {
    
    if (!g_blis_context) return;
    
    if (g_blis_context->use_cblas && g_blis_context->cblas_sswap) {
        g_blis_context->cblas_sswap(n, x, incx, y, incy);
        return;
    }
    
    if (g_blis_context->bli_sswapv) {
        g_blis_context->bli_sswapv(n, x, incx, y, incy);
    }
}

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

static fb_plugin_probe_result_t blis_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
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
    
    /* Try loading CBLAS functions first */
    ctx->cblas_sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sasum");
    ctx->cblas_saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_saxpy");
    ctx->cblas_sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sdot");
    ctx->cblas_scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_scopy");
    ctx->cblas_sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sscal");
    ctx->cblas_snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_snrm2");
    ctx->cblas_sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sswap");
    ctx->cblas_isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_isamax");
    
    ctx->use_cblas = (ctx->cblas_sdot != NULL);
    
    /* Load native BLIS functions as fallback */
    ctx->bli_saxpyv = (bli_saxpyv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_saxpyv");
    ctx->bli_sdotv = (bli_sdotv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_sdotv");
    ctx->bli_snormfv = (bli_snormfv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_snormfv");
    ctx->bli_scopyv = (bli_scopyv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_scopyv");
    ctx->bli_sscalv = (bli_sscalv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_sscalv");
    ctx->bli_sswapv = (bli_sswapv_t)FB_GET_PROC_ADDRESS(lib_handle, "bli_sswapv");
    
    /* Need at least one working interface */
    if (!ctx->cblas_sdot && !ctx->bli_sdotv) {
        free(ctx);
        return -3;
    }
    
    /* Populate vtable */
    g_blis_context = ctx;
    g_blis_vtable.saxpy = blis_saxpy_wrapper;
    g_blis_vtable.sdot = blis_sdot_wrapper;
    g_blis_vtable.snrm2 = blis_snrm2_wrapper;
    g_blis_vtable.sasum = blis_sasum_wrapper;
    g_blis_vtable.isamax = blis_isamax_wrapper;
    g_blis_vtable.scopy = blis_scopy_wrapper;
    g_blis_vtable.sscal = blis_sscal_wrapper;
    g_blis_vtable.sswap = blis_sswap_wrapper;
    
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
