/**
 * @file plugin_accelerate.c
 * @brief Apple Accelerate framework plugin implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#ifdef __APPLE__
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#else
    /* Accelerate is macOS-only, but provide stubs for other platforms */
    #define FB_GET_PROC_ADDRESS(handle, name) NULL
#endif

/* CBLAS function pointer typedefs */
typedef float (*cblas_sasum_t)(const int n, const float* x, const int incx);
typedef void (*cblas_saxpy_t)(const int n, const float alpha, const float* x, const int incx, float* y, const int incy);
typedef float (*cblas_sdot_t)(const int n, const float* x, const int incx, const float* y, const int incy);
typedef void (*cblas_scopy_t)(const int n, const float* x, const int incx, float* y, const int incy);
typedef void (*cblas_sscal_t)(const int n, const float alpha, float* x, const int incx);
typedef float (*cblas_snrm2_t)(const int n, const float* x, const int incx);
typedef void (*cblas_sswap_t)(const int n, float* x, const int incx, float* y, const int incy);
typedef int (*cblas_isamax_t)(const int n, const float* x, const int incx);
typedef void (*cblas_sgemv_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, const int M, const int N,
                               const float alpha, const float* A, const int lda,
                               const float* X, const int incX, const float beta,
                               float* Y, const int incY);
typedef void (*cblas_sgemm_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const float alpha, const float* A, const int lda,
                               const float* B, const int ldb, const float beta,
                               float* C, const int ldc);

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    
    /* CBLAS Level 1 */
    cblas_sasum_t sasum;
    cblas_saxpy_t saxpy;
    cblas_sdot_t sdot;
    cblas_scopy_t scopy;
    cblas_sscal_t sscal;
    cblas_snrm2_t snrm2;
    cblas_sswap_t sswap;
    cblas_isamax_t isamax;
    
    /* CBLAS Level 2 */
    cblas_sgemv_t sgemv;
    
    /* CBLAS Level 3 */
    cblas_sgemm_t sgemm;
    
} accelerate_plugin_context_t;

static fb_backend_vtable_t g_accelerate_vtable;
static accelerate_plugin_context_t* g_accelerate_context = NULL;

static const fb_plugin_metadata_t g_accelerate_metadata = {
    .name = "accelerate",
    .version = "1.0",
    .vendor = "Apple Inc.",
    .description = "Apple Accelerate framework - optimized for Apple Silicon and Intel Macs",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_THREADSAFE
};

/* Wrapper functions */
static float accelerate_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!g_accelerate_context || !g_accelerate_context->sdot) return 0.0f;
    return g_accelerate_context->sdot(n, x, incx, y, incy);
}

static float accelerate_snrm2_wrapper(int n, const float* x, int incx) {
    if (!g_accelerate_context || !g_accelerate_context->snrm2) return 0.0f;
    return g_accelerate_context->snrm2(n, x, incx);
}

static float accelerate_sasum_wrapper(int n, const float* x, int incx) {
    if (!g_accelerate_context || !g_accelerate_context->sasum) return 0.0f;
    return g_accelerate_context->sasum(n, x, incx);
}

static int accelerate_isamax_wrapper(int n, const float* x, int incx) {
    if (!g_accelerate_context || !g_accelerate_context->isamax) return 0;
    return g_accelerate_context->isamax(n, x, incx);
}

static void accelerate_saxpy_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_accelerate_context && g_accelerate_context->saxpy) {
        g_accelerate_context->saxpy(n, alpha, x, incx, y, incy);
    }
}

static void accelerate_scopy_wrapper(int n, const float* x, int incx, float* y, int incy) {
    if (g_accelerate_context && g_accelerate_context->scopy) {
        g_accelerate_context->scopy(n, x, incx, y, incy);
    }
}

static void accelerate_sscal_wrapper(int n, float alpha, float* x, int incx) {
    if (g_accelerate_context && g_accelerate_context->sscal) {
        g_accelerate_context->sscal(n, alpha, x, incx);
    }
}

static void accelerate_sswap_wrapper(int n, float* x, int incx, float* y, int incy) {
    if (g_accelerate_context && g_accelerate_context->sswap) {
        g_accelerate_context->sswap(n, x, incx, y, incy);
    }
}

static void accelerate_sgemv_wrapper(char trans, int m, int n, float alpha,
                                     const float* a, int lda, const float* x, int incx,
                                     float beta, float* y, int incy) {
    if (!g_accelerate_context || !g_accelerate_context->sgemv) return;
    
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_accelerate_context->sgemv(CblasColMajor, cblas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void accelerate_sgemm_wrapper(char transa, char transb, int m, int n, int k,
                                     float alpha, const float* a, int lda,
                                     const float* b, int ldb, float beta,
                                     float* c, int ldc) {
    if (!g_accelerate_context || !g_accelerate_context->sgemm) return;
    
    CBLAS_TRANSPOSE cblas_transa = (transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb = (transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans;
    g_accelerate_context->sgemm(CblasColMajor, cblas_transa, cblas_transb, m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
}

static fb_plugin_probe_result_t accelerate_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#ifndef __APPLE__
    result.score = 0;
    result.reason = "Accelerate framework only available on macOS";
    return result;
#else
    
    const char* lib_names[] = {
        "/System/Library/Frameworks/Accelerate.framework/Accelerate",
        "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
        NULL
    };
    
    const char* default_paths[] = { NULL }; /* Use absolute paths */
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            result.score = 98; /* Accelerate is the best choice on macOS */
            result.library_path = NULL; /* Init will load the library */
            result.reason = "Found Apple Accelerate framework with CBLAS interface";
        } else {
            result.score = 0;
            result.reason = "Framework found but missing required CBLAS symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "Accelerate framework not found";
    }
    
    return result;
#endif
}

static int accelerate_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
#ifndef __APPLE__
    return -1; /* Not available on non-Apple platforms */
#else
    
    /* Load the library if not already provided */
    if (!lib_handle) {
        const char* lib_names[] = {
            "/System/Library/Frameworks/Accelerate.framework/Accelerate",
            "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
            NULL
        };
        
        const char* default_paths[] = { NULL };
        
        lib_handle = fb_plugin_load_library(lib_names, default_paths);
        if (!lib_handle) {
            return -1; /* Library not found */
        }
    }
    
    accelerate_plugin_context_t* ctx = (accelerate_plugin_context_t*)calloc(1, sizeof(accelerate_plugin_context_t));
    if (!ctx) {
        return -2;
    }
    
    ctx->lib_handle = lib_handle;
    
    /* Load CBLAS Level 1 functions */
    ctx->sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sasum");
    ctx->saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_saxpy");
    ctx->sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sdot");
    ctx->scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_scopy");
    ctx->sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sscal");
    ctx->snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_snrm2");
    ctx->sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sswap");
    ctx->isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_isamax");
    
    /* Load CBLAS Level 2 functions */
    ctx->sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sgemv");
    
    /* Load CBLAS Level 3 functions */
    ctx->sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sgemm");
    
    if (!ctx->sdot || !ctx->sgemm) {
        free(ctx);
        return -3;
    }
    
    /* Populate vtable */
    g_accelerate_context = ctx;
    g_accelerate_vtable.saxpy = accelerate_saxpy_wrapper;
    g_accelerate_vtable.sdot = accelerate_sdot_wrapper;
    g_accelerate_vtable.snrm2 = accelerate_snrm2_wrapper;
    g_accelerate_vtable.sasum = accelerate_sasum_wrapper;
    g_accelerate_vtable.isamax = accelerate_isamax_wrapper;
    g_accelerate_vtable.scopy = accelerate_scopy_wrapper;
    g_accelerate_vtable.sscal = accelerate_sscal_wrapper;
    g_accelerate_vtable.sswap = accelerate_sswap_wrapper;
    g_accelerate_vtable.sgemv = accelerate_sgemv_wrapper;
    g_accelerate_vtable.sgemm = accelerate_sgemm_wrapper;
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
#endif
}

static const fb_backend_vtable_t* accelerate_get_vtable(fb_plugin_context_t* ctx) {
    return &g_accelerate_vtable;
}

static void* accelerate_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void accelerate_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

static const fb_backend_plugin_t g_accelerate_plugin = {
    .metadata = &g_accelerate_metadata,
    .probe = accelerate_probe,
    .init = accelerate_init,
    .get_vtable = accelerate_get_vtable,
    .get_context = accelerate_get_context,
    .shutdown = accelerate_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function - must be called explicitly */
void fb_register_accelerate_plugin(void) {
    fb_register_plugin(&g_accelerate_plugin);
}
