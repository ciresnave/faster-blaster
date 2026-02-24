/**
 * @file openblas_trait_impl.c
 * @brief OpenBLAS CPU backend trait implementation
 * 
 * Provides OpenBLAS-specific implementation of the unified CPU backend trait interface.
 * This wraps the existing OpenBLAS backend to provide trait-based access.
 */

#include "faster-blaster/cpu_backend_trait.h"
#include "../openblas_backend.h"
#include "../backend_interface.h"
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * OpenBLAS-specific Context
 * ========================================================================== */

typedef struct {
    void* lib_handle;  /* Library handle (if provided by plugin) */
    const fb_backend_vtable_t* vtable;  /* OpenBLAS backend vtable */
    int initialized;
} openblas_trait_context_t;

/* ============================================================================
 * Lifecycle Management
 * ========================================================================== */

static int openblas_trait_init(void* lib_handle, void** backend_handle) {
    /* Note: lib_handle could be used if plugin pre-loaded the library,
     * but OpenBLAS backend currently handles its own loading */
    (void)lib_handle;  /* Unused - OpenBLAS loads its own library */
    
    /* Initialize OpenBLAS backend */
    int result = fb_openblas_init();
    if (result != 0) {
        fprintf(stderr, "OpenBLAS trait: Failed to initialize OpenBLAS backend\n");
        return -1;
    }
    
    /* Get OpenBLAS backend vtable */
    const fb_backend_vtable_t* vtable = fb_openblas_get_vtable();
    if (!vtable) {
        fprintf(stderr, "OpenBLAS trait: Failed to get OpenBLAS vtable\n");
        fb_openblas_shutdown();
        return -1;
    }
    
    /* Create context */
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)malloc(sizeof(openblas_trait_context_t));
    if (!ctx) {
        fb_openblas_shutdown();
        return -1;
    }
    
    ctx->lib_handle = lib_handle;
    ctx->vtable = vtable;
    ctx->initialized = 1;
    
    *backend_handle = ctx;
    return 0;
}

static void openblas_trait_shutdown(void* backend_handle) {
    if (!backend_handle) {
        return;
    }
    
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)backend_handle;
    
    if (ctx->initialized) {
        fb_openblas_shutdown();
        ctx->initialized = 0;
    }
    
    free(ctx);
}

static int openblas_trait_get_num_threads(void* backend_handle) {
    (void)backend_handle;
    return fb_openblas_get_num_threads();
}

static void openblas_trait_set_num_threads(void* backend_handle, int num_threads) {
    (void)backend_handle;
    fb_openblas_set_num_threads(num_threads);
}

/* ============================================================================
 * BLAS Level 1 - Wrapper implementations
 * ========================================================================== */

static void openblas_trait_saxpy(void* handle, int n, float alpha, const float* x, int incx, float* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->saxpy) {
        ctx->vtable->saxpy((int64_t)n, alpha, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void openblas_trait_daxpy(void* handle, int n, double alpha, const double* x, int incx, double* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->daxpy) {
        ctx->vtable->daxpy((int64_t)n, alpha, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void openblas_trait_sscal(void* handle, int n, float alpha, float* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sscal) {
        ctx->vtable->sscal((int64_t)n, alpha, x, (int64_t)incx);
    }
}

static void openblas_trait_dscal(void* handle, int n, double alpha, double* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dscal) {
        ctx->vtable->dscal((int64_t)n, alpha, x, (int64_t)incx);
    }
}

static void openblas_trait_scopy(void* handle, int n, const float* x, int incx, float* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->scopy) {
        ctx->vtable->scopy((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void openblas_trait_dcopy(void* handle, int n, const double* x, int incx, double* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dcopy) {
        ctx->vtable->dcopy((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void openblas_trait_sswap(void* handle, int n, float* x, int incx, float* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sswap) {
        ctx->vtable->sswap((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static void openblas_trait_dswap(void* handle, int n, double* x, int incx, double* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dswap) {
        ctx->vtable->dswap((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
}

static float openblas_trait_sdot(void* handle, int n, const float* x, int incx, const float* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sdot) {
        return ctx->vtable->sdot((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
    return 0.0f;
}

static double openblas_trait_ddot(void* handle, int n, const double* x, int incx, const double* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->ddot) {
        return ctx->vtable->ddot((int64_t)n, x, (int64_t)incx, y, (int64_t)incy);
    }
    return 0.0;
}

static float openblas_trait_snrm2(void* handle, int n, const float* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->snrm2) {
        return ctx->vtable->snrm2((int64_t)n, x, (int64_t)incx);
    }
    return 0.0f;
}

static double openblas_trait_dnrm2(void* handle, int n, const double* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dnrm2) {
        return ctx->vtable->dnrm2((int64_t)n, x, (int64_t)incx);
    }
    return 0.0;
}

static float openblas_trait_sasum(void* handle, int n, const float* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sasum) {
        return ctx->vtable->sasum((int64_t)n, x, (int64_t)incx);
    }
    return 0.0f;
}

static double openblas_trait_dasum(void* handle, int n, const double* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dasum) {
        return ctx->vtable->dasum((int64_t)n, x, (int64_t)incx);
    }
    return 0.0;
}

static int openblas_trait_isamax(void* handle, int n, const float* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->isamax) {
        return (int)ctx->vtable->isamax((int64_t)n, x, (int64_t)incx);
    }
    return 0;
}

static int openblas_trait_idamax(void* handle, int n, const double* x, int incx) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->idamax) {
        return (int)ctx->vtable->idamax((int64_t)n, x, (int64_t)incx);
    }
    return 0;
}

/* ============================================================================
 * BLAS Level 2 - Wrapper implementations
 * ========================================================================== */

static void openblas_trait_sgemv(void* handle, int order, int trans, int m, int n,
                                  float alpha, const float* a, int lda,
                                  const float* x, int incx, float beta,
                                  float* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemv) {
        ctx->vtable->sgemv((fb_layout_t)order, (fb_transpose_t)trans,
                          (int64_t)m, (int64_t)n, alpha, a, (int64_t)lda,
                          x, (int64_t)incx, beta, y, (int64_t)incy);
    }
}

static void openblas_trait_dgemv(void* handle, int order, int trans, int m, int n,
                                  double alpha, const double* a, int lda,
                                  const double* x, int incx, double beta,
                                  double* y, int incy) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemv) {
        ctx->vtable->dgemv((fb_layout_t)order, (fb_transpose_t)trans,
                          (int64_t)m, (int64_t)n, alpha, a, (int64_t)lda,
                          x, (int64_t)incx, beta, y, (int64_t)incy);
    }
}

/* ============================================================================
 * BLAS Level 3 - Wrapper implementations
 * ========================================================================== */

static void openblas_trait_sgemm(void* handle, int order, int transa, int transb,
                                  int m, int n, int k, float alpha,
                                  const float* a, int lda, const float* b, int ldb,
                                  float beta, float* c, int ldc) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->sgemm) {
        ctx->vtable->sgemm((fb_layout_t)order, (fb_transpose_t)transa, (fb_transpose_t)transb,
                          (int64_t)m, (int64_t)n, (int64_t)k, alpha,
                          a, (int64_t)lda, b, (int64_t)ldb, beta, c, (int64_t)ldc);
    }
}

static void openblas_trait_dgemm(void* handle, int order, int transa, int transb,
                                  int m, int n, int k, double alpha,
                                  const double* a, int lda, const double* b, int ldb,
                                  double beta, double* c, int ldc) {
    openblas_trait_context_t* ctx = (openblas_trait_context_t*)handle;
    if (ctx && ctx->vtable && ctx->vtable->dgemm) {
        ctx->vtable->dgemm((fb_layout_t)order, (fb_transpose_t)transa, (fb_transpose_t)transb,
                          (int64_t)m, (int64_t)n, (int64_t)k, alpha,
                          a, (int64_t)lda, b, (int64_t)ldb, beta, c, (int64_t)ldc);
    }
}

/* ============================================================================
 * Trait Export
 * ========================================================================== */

const fb_cpu_backend_trait_t fb_openblas_trait = {
    .name = "OpenBLAS",
    .type = FB_CPU_BACKEND_OPENBLAS,
    
    /* Lifecycle */
    .init = openblas_trait_init,
    .shutdown = openblas_trait_shutdown,
    .get_num_threads = openblas_trait_get_num_threads,
    .set_num_threads = openblas_trait_set_num_threads,
    
    /* Level 1 BLAS */
    .saxpy = openblas_trait_saxpy,
    .daxpy = openblas_trait_daxpy,
    .sscal = openblas_trait_sscal,
    .dscal = openblas_trait_dscal,
    .scopy = openblas_trait_scopy,
    .dcopy = openblas_trait_dcopy,
    .sswap = openblas_trait_sswap,
    .dswap = openblas_trait_dswap,
    .sdot = openblas_trait_sdot,
    .ddot = openblas_trait_ddot,
    .snrm2 = openblas_trait_snrm2,
    .dnrm2 = openblas_trait_dnrm2,
    .sasum = openblas_trait_sasum,
    .dasum = openblas_trait_dasum,
    .isamax = openblas_trait_isamax,
    .idamax = openblas_trait_idamax,
    
    /* Level 2 BLAS */
    .sgemv = openblas_trait_sgemv,
    .dgemv = openblas_trait_dgemv,
    
    /* Level 3 BLAS */
    .sgemm = openblas_trait_sgemm,
    .dgemm = openblas_trait_dgemm,
    
    /* TODO: Add remaining BLAS operations as needed */
};
