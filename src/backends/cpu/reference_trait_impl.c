/**
 * @file reference_trait_impl.c
 * @brief Reference CPU backend trait implementation
 * 
 * Provides reference implementation of the unified CPU backend trait interface.
 * This wraps the existing reference backend to provide trait-based access.
 */

#include "faster-blaster/cpu_backend_trait.h"
#include "../reference.h"
#include "../backend_interface.h"
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * Reference-specific Context
 * ========================================================================== */

typedef struct {
    void* lib_handle;  /* Library handle (if provided by plugin) */
    const fb_backend_vtable_t* vtable;  /* Reference backend vtable */
    int initialized;
} reference_trait_context_t;

/* ============================================================================
 * Lifecycle Management
 * ========================================================================== */

static int reference_trait_init(void* lib_handle, void** backend_handle) {
    /* Note: lib_handle could be used if plugin pre-loaded the library,
     * but reference backend is built-in (no library loading needed) */
    (void)lib_handle;  /* Unused - Reference is built-in */
    
    /* Get reference backend vtable (no init needed, it's stateless) */
    const fb_backend_vtable_t* vtable = fb_reference_backend();
    if (!vtable) {
        fprintf(stderr, "Reference trait: Failed to get reference vtable\n");
        return -1;
    }
    
    /* Create context */
    reference_trait_context_t* ctx = (reference_trait_context_t*)malloc(sizeof(reference_trait_context_t));
    if (!ctx) {
        return -1;
    }
    
    ctx->lib_handle = lib_handle;
    ctx->vtable = vtable;
    ctx->initialized = 1;
    
    *backend_handle = ctx;
    return 0;
}

static void reference_trait_shutdown(void* backend_handle) {
    if (!backend_handle) {
        return;
    }
    
    reference_trait_context_t* ctx = (reference_trait_context_t*)backend_handle;
    
    /* Reference backend has no shutdown (it's stateless) */
    ctx->initialized = 0;
    free(ctx);
}

static int reference_trait_get_num_threads(void* backend_handle) {
    (void)backend_handle;
    /* Reference implementation is single-threaded */
    return 1;
}

static void reference_trait_set_num_threads(void* backend_handle, int num_threads) {
    (void)backend_handle;
    (void)num_threads;
    /* Reference implementation ignores thread setting (always single-threaded) */
}

/* ============================================================================
 * BLAS Level 1 - Wrapper implementations
 * ========================================================================== */

static void reference_trait_saxpy(void* handle, int n, float alpha, const float* x, int incx, float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->saxpy) {
        ctx->vtable->saxpy((int)n, alpha, x, (int)incx, y, (int)incy);
    }
}

static void reference_trait_daxpy(void* handle, int n, double alpha, const double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->daxpy) {
        ctx->vtable->daxpy((int)n, alpha, x, (int)incx, y, (int)incy);
    }
}

static void reference_trait_sscal(void* handle, int n, float alpha, float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sscal) {
        ctx->vtable->sscal((int)n, alpha, x, (int)incx);
    }
}

static void reference_trait_dscal(void* handle, int n, double alpha, double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dscal) {
        ctx->vtable->dscal((int)n, alpha, x, (int)incx);
    }
}

static void reference_trait_scopy(void* handle, int n, const float* x, int incx, float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->scopy) {
        ctx->vtable->scopy((int)n, x, (int)incx, y, (int)incy);
    }
}

static void reference_trait_dcopy(void* handle, int n, const double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dcopy) {
        ctx->vtable->dcopy((int)n, x, (int)incx, y, (int)incy);
    }
}

static void reference_trait_sswap(void* handle, int n, float* x, int incx, float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sswap) {
        ctx->vtable->sswap((int)n, x, (int)incx, y, (int)incy);
    }
}

static void reference_trait_dswap(void* handle, int n, double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dswap) {
        ctx->vtable->dswap((int)n, x, (int)incx, y, (int)incy);
    }
}

static float reference_trait_sdot(void* handle, int n, const float* x, int incx, const float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sdot) {
        return ctx->vtable->sdot((int)n, x, (int)incx, y, (int)incy);
    }
    return 0.0f;
}

static double reference_trait_ddot(void* handle, int n, const double* x, int incx, const double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->ddot) {
        return ctx->vtable->ddot((int)n, x, (int)incx, y, (int)incy);
    }
    return 0.0;
}

static float reference_trait_snrm2(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->snrm2) {
        return ctx->vtable->snrm2((int)n, x, (int)incx);
    }
    return 0.0f;
}

static double reference_trait_dnrm2(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dnrm2) {
        return ctx->vtable->dnrm2((int)n, x, (int)incx);
    }
    return 0.0;
}

static float reference_trait_sasum(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sasum) {
        return ctx->vtable->sasum((int)n, x, (int)incx);
    }
    return 0.0f;
}

static double reference_trait_dasum(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dasum) {
        return ctx->vtable->dasum((int)n, x, (int)incx);
    }
    return 0.0;
}

static int reference_trait_isamax(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->isamax) {
        return (int)ctx->vtable->isamax((int)n, x, (int)incx);
    }
    return 0;
}

static int reference_trait_idamax(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->idamax) {
        return (int)ctx->vtable->idamax((int)n, x, (int)incx);
    }
    return 0;
}

/* ============================================================================
 * BLAS Level 2 - Wrapper implementations
 * ========================================================================== */

static void reference_trait_sgemv(void* handle, char trans, int m, int n,
                                   float alpha, const float* a, int lda,
                                   const float* x, int incx, float beta,
                                   float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemv) {
        fb_transpose_t trans_e = (trans=='T'||trans=='t') ? FB_TRANS : (trans=='C'||trans=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        ctx->vtable->sgemv(FB_LAYOUT_ROW_MAJOR, trans_e,
                          m, n, alpha, a, lda,
                          x, incx, beta, y, incy);
    }
}

static void reference_trait_dgemv(void* handle, char trans, int m, int n,
                                   double alpha, const double* a, int lda,
                                   const double* x, int incx, double beta,
                                   double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemv) {
        fb_transpose_t trans_e = (trans=='T'||trans=='t') ? FB_TRANS : (trans=='C'||trans=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        ctx->vtable->dgemv(FB_LAYOUT_ROW_MAJOR, trans_e,
                          m, n, alpha, a, lda,
                          x, incx, beta, y, incy);
    }
}

/* ============================================================================
 * BLAS Level 3 - Wrapper implementations
 * ========================================================================== */

static void reference_trait_sgemm(void* handle, char transa, char transb,
                                   int m, int n, int k, float alpha,
                                   const float* a, int lda, const float* b, int ldb,
                                   float beta, float* c, int ldc) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemm) {
        fb_transpose_t ta = (transa=='T'||transa=='t') ? FB_TRANS : (transa=='C'||transa=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        fb_transpose_t tb = (transb=='T'||transb=='t') ? FB_TRANS : (transb=='C'||transb=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        ctx->vtable->sgemm(FB_LAYOUT_ROW_MAJOR, ta, tb,
                          m, n, k, alpha,
                          a, lda, b, ldb, beta, c, ldc);
    }
}

static void reference_trait_dgemm(void* handle, char transa, char transb,
                                   int m, int n, int k, double alpha,
                                   const double* a, int lda, const double* b, int ldb,
                                   double beta, double* c, int ldc) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemm) {
        fb_transpose_t ta = (transa=='T'||transa=='t') ? FB_TRANS : (transa=='C'||transa=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        fb_transpose_t tb = (transb=='T'||transb=='t') ? FB_TRANS : (transb=='C'||transb=='c') ? FB_CONJ_TRANS : FB_NO_TRANS;
        ctx->vtable->dgemm(FB_LAYOUT_ROW_MAJOR, ta, tb,
                          m, n, k, alpha,
                          a, lda, b, ldb, beta, c, ldc);
    }
}

/* ============================================================================
 * Trait Export
 * ========================================================================== */

const fb_cpu_backend_trait_t fb_reference_trait = {
    .name = "Reference",
    .type = FB_CPU_BACKEND_NETLIB,
    
    /* Lifecycle */
    .init = reference_trait_init,
    .shutdown = reference_trait_shutdown,
    .get_num_threads = reference_trait_get_num_threads,
    .set_num_threads = reference_trait_set_num_threads,
    
    /* Level 1 BLAS */
    .saxpy = reference_trait_saxpy,
    .daxpy = reference_trait_daxpy,
    .sscal = reference_trait_sscal,
    .dscal = reference_trait_dscal,
    .scopy = reference_trait_scopy,
    .dcopy = reference_trait_dcopy,
    .sswap = reference_trait_sswap,
    .dswap = reference_trait_dswap,
    .sdot = reference_trait_sdot,
    .ddot = reference_trait_ddot,
    .snrm2 = reference_trait_snrm2,
    .dnrm2 = reference_trait_dnrm2,
    .sasum = reference_trait_sasum,
    .dasum = reference_trait_dasum,
    .isamax = reference_trait_isamax,
    .idamax = reference_trait_idamax,
    
    /* Level 2 BLAS */
    .sgemv = reference_trait_sgemv,
    .dgemv = reference_trait_dgemv,
    
    /* Level 3 BLAS */
    .sgemm = reference_trait_sgemm,
    .dgemm = reference_trait_dgemm,
    
    /* TODO: Add remaining BLAS operations as needed */
};
