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
    const fb_backend_vtable_t* vtable = fb_reference_get_vtable();
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
        ctx->vtable->saxpy((int64_t)n, alpha, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void reference_trait_daxpy(void* handle, int n, double alpha, const double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->daxpy) {
        ctx->vtable->daxpy((int64_t)n, alpha, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void reference_trait_sscal(void* handle, int n, float alpha, float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sscal) {
        ctx->vtable->sscal((int64_t)n, alpha, x, (int64_t)incx);
    }
}

static void reference_trait_dscal(void* handle, int n, double alpha, double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dscal) {
        ctx->vtable->dscal((int64_t)n, alpha, x, (int64_t)incx);
    }
}

static void reference_trait_scopy(void* handle, int n, const float* x, int incx, float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->scopy) {
        ctx->vtable->scopy((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void reference_trait_dcopy(void* handle, int n, const double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dcopy) {
        ctx->vtable->dcopy((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void reference_trait_sswap(void* handle, int n, float* x, int incx, float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sswap) {
        ctx->vtable->sswap((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void reference_trait_dswap(void* handle, int n, double* x, int incx, double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dswap) {
        ctx->vtable->dswap((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static float reference_trait_sdot(void* handle, int n, const float* x, int incx, const float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sdot) {
        return ctx->vtable->sdot((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
    return 0.0f;
}

static double reference_trait_ddot(void* handle, int n, const double* x, int incx, const double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->ddot) {
        return ctx->vtable->ddot((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
    return 0.0;
}

static float reference_trait_snrm2(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->snrm2) {
        return ctx->vtable->snrm2((int64_t)n, x, (int64_t)incx);
    }
    return 0.0f;
}

static double reference_trait_dnrm2(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dnrm2) {
        return ctx->vtable->dnrm2((int64_t)n, x, (int64_t)incx);
    }
    return 0.0;
}

static float reference_trait_sasum(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sasum) {
        return ctx->vtable->sasum((int64_t)n, x, (int64_t)incx);
    }
    return 0.0f;
}

static double reference_trait_dasum(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dasum) {
        return ctx->vtable->dasum((int64_t)n, x, (int64_t)incx);
    }
    return 0.0;
}

static int reference_trait_isamax(void* handle, int n, const float* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->isamax) {
        return (int)ctx->vtable->isamax((int64_t)n, x, (int64_t)incx);
    }
    return 0;
}

static int reference_trait_idamax(void* handle, int n, const double* x, int incx) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->idamax) {
        return (int)ctx->vtable->idamax((int64_t)n, x, (int64_t)incx);
    }
    return 0;
}

/* ============================================================================
 * BLAS Level 2 - Wrapper implementations
 * ========================================================================== */

static void reference_trait_sgemv(void* handle, int order, int trans, int m, int n,
                                   float alpha, const float* a, int lda,
                                   const float* x, int incx, float beta,
                                   float* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemv) {
        ctx->vtable->sgemv((fb_layout_t)order, (fb_transpose_t)trans,
                          (int64_t)m, (int64_t)n, alpha, a, (int64_t)lda,
                          x, (int64_t)incx, beta, y, (int64_t)incy);
    }
}

static void reference_trait_dgemv(void* handle, int order, int trans, int m, int n,
                                   double alpha, const double* a, int lda,
                                   const double* x, int incx, double beta,
                                   double* y, int incy) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemv) {
        ctx->vtable->dgemv((fb_layout_t)order, (fb_transpose_t)trans,
                          (int64_t)m, (int64_t)n, alpha, a, (int64_t)lda,
                          x, (int64_t)incx, beta, y, (int64_t)incy);
    }
}

/* ============================================================================
 * BLAS Level 3 - Wrapper implementations
 * ========================================================================== */

static void reference_trait_sgemm(void* handle, int order, int transa, int transb,
                                   int m, int n, int k, float alpha,
                                   const float* a, int lda, const float* b, int ldb,
                                   float beta, float* c, int ldc) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemm) {
        ctx->vtable->sgemm((fb_layout_t)order, (fb_transpose_t)transa, (fb_transpose_t)transb,
                          (int64_t)m, (int64_t)n, (int64_t)k, alpha,
                          a, (int64_t)lda, b, (int64_t)ldb, beta, c, (int64_t)ldc);
    }
}

static void reference_trait_dgemm(void* handle, int order, int transa, int transb,
                                   int m, int n, int k, double alpha,
                                   const double* a, int lda, const double* b, int ldb,
                                   double beta, double* c, int ldc) {
    reference_trait_context_t* ctx = (reference_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemm) {
        ctx->vtable->dgemm((fb_layout_t)order, (fb_transpose_t)transa, (fb_transpose_t)transb,
                          (int64_t)m, (int64_t)n, (int64_t)k, alpha,
                          a, (int64_t)lda, b, (int64_t)ldb, beta, c, (int64_t)ldc);
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
