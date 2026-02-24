/**
 * @file plugin_openblas.c
 * @brief OpenBLAS plugin implementation
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

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
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

/* Phase 1 BLAS typedefs */
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;

typedef void (*cblas_srot_t)(const int n, float* x, const int incx, float* y, const int incy, const float c, const float s);
typedef void (*cblas_drot_t)(const int n, double* x, const int incx, double* y, const int incy, const double c, const double s);
typedef void (*cblas_srotg_t)(float* a, float* b, float* c, float* s);
typedef void (*cblas_drotg_t)(double* a, double* b, double* c, double* s);
typedef void (*cblas_srotm_t)(const int n, float* x, const int incx, float* y, const int incy, const float* param);
typedef void (*cblas_drotm_t)(const int n, double* x, const int incx, double* y, const int incy, const double* param);
typedef void (*cblas_srotmg_t)(float* d1, float* d2, float* x1, const float y1, float* param);
typedef void (*cblas_drotmg_t)(double* d1, double* d2, double* x1, const double y1, double* param);
typedef void (*cblas_ssymv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const float alpha,
                              const float* a, const int lda, const float* x, const int incx,
                              const float beta, float* y, const int incy);
typedef void (*cblas_dsymv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const double alpha,
                              const double* a, const int lda, const double* x, const int incx,
                              const double beta, double* y, const int incy);
typedef void (*cblas_sger_t)(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                             const float* x, const int incx, const float* y, const int incy,
                             float* a, const int lda);
typedef void (*cblas_dger_t)(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                             const double* x, const int incx, const double* y, const int incy,
                             double* a, const int lda);
typedef void (*cblas_ssyr_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const float alpha,
                             const float* x, const int incx, float* a, const int lda);
typedef void (*cblas_dsyr_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const double alpha,
                             const double* x, const int incx, double* a, const int lda);
typedef void (*cblas_ssyr2_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const float alpha,
                              const float* x, const int incx, const float* y, const int incy,
                              float* a, const int lda);
typedef void (*cblas_dsyr2_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, const int n, const double alpha,
                              const double* x, const int incx, const double* y, const int incy,
                              double* a, const int lda);
typedef void (*cblas_strmv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                              const int n, const float* a, const int lda, float* x, const int incx);
typedef void (*cblas_dtrmv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                              const int n, const double* a, const int lda, double* x, const int incx);
typedef void (*cblas_strsv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                              const int n, const float* a, const int lda, float* x, const int incx);
typedef void (*cblas_dtrsv_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                              const int n, const double* a, const int lda, double* x, const int incx);
typedef void (*cblas_strmm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              CBLAS_DIAG diag, const int m, const int n, const float alpha,
                              const float* a, const int lda, float* b, const int ldb);
typedef void (*cblas_dtrmm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              CBLAS_DIAG diag, const int m, const int n, const double alpha,
                              const double* a, const int lda, double* b, const int ldb);
typedef void (*cblas_strsm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              CBLAS_DIAG diag, const int m, const int n, const float alpha,
                              const float* a, const int lda, float* b, const int ldb);
typedef void (*cblas_dtrsm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              CBLAS_DIAG diag, const int m, const int n, const double alpha,
                              const double* a, const int lda, double* b, const int ldb);
typedef void (*cblas_ssymm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, const int m, const int n,
                              const float alpha, const float* a, const int lda, const float* b, const int ldb,
                              const float beta, float* c, const int ldc);
typedef void (*cblas_dsymm_t)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, const int m, const int n,
                              const double alpha, const double* a, const int lda, const double* b, const int ldb,
                              const double beta, double* c, const int ldc);
typedef void (*cblas_ssyrk_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n, const int k,
                              const float alpha, const float* a, const int lda, const float beta, float* c, const int ldc);
typedef void (*cblas_dsyrk_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n, const int k,
                              const double alpha, const double* a, const int lda, const double beta, double* c, const int ldc);
typedef void (*cblas_ssyr2k_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n, const int k,
                               const float alpha, const float* a, const int lda, const float* b, const int ldb,
                               const float beta, float* c, const int ldc);
typedef void (*cblas_dsyr2k_t)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n, const int k,
                               const double alpha, const double* a, const int lda, const double* b, const int ldb,
                               const double beta, double* c, const int ldc);

/* OpenBLAS thread control function types */
typedef int (*openblas_get_num_threads_t)(void);
typedef void (*openblas_set_num_threads_t)(int num_threads);

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
    
    /* Phase 1 operations */
    cblas_srot_t srot;
    cblas_drot_t drot;
    cblas_srotg_t srotg;
    cblas_drotg_t drotg;
    cblas_srotm_t srotm;
    cblas_drotm_t drotm;
    cblas_srotmg_t srotmg;
    cblas_drotmg_t drotmg;
    cblas_ssymv_t ssymv;
    cblas_dsymv_t dsymv;
    cblas_sger_t sger;
    cblas_dger_t dger;
    cblas_ssyr_t ssyr;
    cblas_dsyr_t dsyr;
    cblas_ssyr2_t ssyr2;
    cblas_dsyr2_t dsyr2;
    cblas_strmv_t strmv;
    cblas_dtrmv_t dtrmv;
    cblas_strsv_t strsv;
    cblas_dtrsv_t dtrsv;
    cblas_strmm_t strmm;
    cblas_dtrmm_t dtrmm;
    cblas_strsm_t strsm;
    cblas_dtrsm_t dtrsm;
    cblas_ssymm_t ssymm;
    cblas_dsymm_t dsymm;
    cblas_ssyrk_t ssyrk;
    cblas_dsyrk_t dsyrk;
    cblas_ssyr2k_t ssyr2k;
    cblas_dsyr2k_t dsyr2k;
    
    /* Thread control */
    openblas_get_num_threads_t get_num_threads;
    openblas_set_num_threads_t set_num_threads;
    int is_single_threaded;  /* 1 if threading is not supported/working */
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

/* Wrapper functions */
static float openblas_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!g_openblas_context || !g_openblas_context->sdot) return 0.0f;
    return g_openblas_context->sdot(n, x, incx, y, incy);
}

static float openblas_snrm2_wrapper(int n, const float* x, int incx) {
    
    if (!g_openblas_context || !g_openblas_context->snrm2) return 0.0f;
    return g_openblas_context->snrm2(n, x, incx);
}

static float openblas_sasum_wrapper(int n, const float* x, int incx) {
    
    if (!g_openblas_context || !g_openblas_context->sasum) return 0.0f;
    return g_openblas_context->sasum(n, x, incx);
}

static int openblas_isamax_wrapper(int n, const float* x, int incx) {
    
    if (!g_openblas_context || !g_openblas_context->isamax) return 0;
    return g_openblas_context->isamax(n, x, incx);
}

static void openblas_saxpy_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    
    if (g_openblas_context && g_openblas_context->saxpy) {
        g_openblas_context->saxpy(n, alpha, x, incx, y, incy);
    }
}

static void openblas_scopy_wrapper(int n, const float* x, int incx, float* y, int incy) {
    
    if (g_openblas_context && g_openblas_context->scopy) {
        g_openblas_context->scopy(n, x, incx, y, incy);
    }
}

static void openblas_sscal_wrapper(int n, float alpha, float* x, int incx) {
    
    if (g_openblas_context && g_openblas_context->sscal) {
        g_openblas_context->sscal(n, alpha, x, incx);
    }
}

static void openblas_sswap_wrapper(int n, float* x, int incx, float* y, int incy) {
    
    if (g_openblas_context && g_openblas_context->sswap) {
        g_openblas_context->sswap(n, x, incx, y, incy);
    }
}

static void openblas_sgemv_wrapper(fb_layout_t layout, fb_transpose_t trans, int m, int n, float alpha,
                                   const float* a, int lda, const float* x, int incx,
                                   float beta, float* y, int incy) {
    
    if (!g_openblas_context || !g_openblas_context->sgemv) return;
    
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->sgemv(cblas_layout, cblas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void openblas_sgemm_wrapper(fb_layout_t layout, fb_transpose_t transa, fb_transpose_t transb, int m, int n, int k,
                                   float alpha, const float* a, int lda,
                                   const float* b, int ldb, float beta,
                                   float* c, int ldc) {
    
    if (!g_openblas_context || !g_openblas_context->sgemm) return;
    
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_transa = (transa == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb = (transb == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->sgemm(cblas_layout, cblas_transa, cblas_transb, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

/* Phase 1 BLAS wrapper functions */
static void openblas_srot_wrapper(int n, float* x, int incx, float* y, int incy, float c, float s) {
    if (g_openblas_context && g_openblas_context->srot) {
        g_openblas_context->srot(n, x, incx, y, incy, c, s);
    }
}

static void openblas_drot_wrapper(int n, double* x, int incx, double* y, int incy, double c, double s) {
    if (g_openblas_context && g_openblas_context->drot) {
        g_openblas_context->drot(n, x, incx, y, incy, c, s);
    }
}

static void openblas_srotg_wrapper(float* a, float* b, float* c, float* s) {
    if (g_openblas_context && g_openblas_context->srotg) {
        g_openblas_context->srotg(a, b, c, s);
    }
}

static void openblas_drotg_wrapper(double* a, double* b, double* c, double* s) {
    if (g_openblas_context && g_openblas_context->drotg) {
        g_openblas_context->drotg(a, b, c, s);
    }
}

static void openblas_srotm_wrapper(int n, float* x, int incx, float* y, int incy, const float* param) {
    if (g_openblas_context && g_openblas_context->srotm) {
        g_openblas_context->srotm(n, x, incx, y, incy, param);
    }
}

static void openblas_drotm_wrapper(int n, double* x, int incx, double* y, int incy, const double* param) {
    if (g_openblas_context && g_openblas_context->drotm) {
        g_openblas_context->drotm(n, x, incx, y, incy, param);
    }
}

static void openblas_srotmg_wrapper(float* d1, float* d2, float* x1, float y1, float* param) {
    if (g_openblas_context && g_openblas_context->srotmg) {
        g_openblas_context->srotmg(d1, d2, x1, y1, param);
    }
}

static void openblas_drotmg_wrapper(double* d1, double* d2, double* x1, double y1, double* param) {
    if (g_openblas_context && g_openblas_context->drotmg) {
        g_openblas_context->drotmg(d1, d2, x1, y1, param);
    }
}

static void openblas_ssymv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                                   const float* a, int lda, const float* x, int incx,
                                   float beta, float* y, int incy) {
    if (!g_openblas_context || !g_openblas_context->ssymv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->ssymv(cblas_layout, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void openblas_dsymv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                                   const double* a, int lda, const double* x, int incx,
                                   double beta, double* y, int incy) {
    if (!g_openblas_context || !g_openblas_context->dsymv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->dsymv(cblas_layout, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void openblas_sger_wrapper(fb_layout_t layout, int m, int n, float alpha,
                                  const float* x, int incx, const float* y, int incy,
                                  float* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->sger) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    g_openblas_context->sger(cblas_layout, m, n, alpha, x, incx, y, incy, a, lda);
}

static void openblas_dger_wrapper(fb_layout_t layout, int m, int n, double alpha,
                                  const double* x, int incx, const double* y, int incy,
                                  double* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->dger) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    g_openblas_context->dger(cblas_layout, m, n, alpha, x, incx, y, incy, a, lda);
}

static void openblas_ssyr_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                                  const float* x, int incx, float* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->ssyr) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->ssyr(cblas_layout, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void openblas_dsyr_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                                  const double* x, int incx, double* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->dsyr) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->dsyr(cblas_layout, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void openblas_ssyr2_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                                   const float* x, int incx, const float* y, int incy,
                                   float* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->ssyr2) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->ssyr2(cblas_layout, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void openblas_dsyr2_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                                   const double* x, int incx, const double* y, int incy,
                                   double* a, int lda) {
    if (!g_openblas_context || !g_openblas_context->dsyr2) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->dsyr2(cblas_layout, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void openblas_strmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   fb_diag_t diag, int n, const float* a, int lda, float* x, int incx) {
    if (!g_openblas_context || !g_openblas_context->strmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->strmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void openblas_dtrmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   fb_diag_t diag, int n, const double* a, int lda, double* x, int incx) {
    if (!g_openblas_context || !g_openblas_context->dtrmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->dtrmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void openblas_strsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   fb_diag_t diag, int n, const float* a, int lda, float* x, int incx) {
    if (!g_openblas_context || !g_openblas_context->strsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->strsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void openblas_dtrsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   fb_diag_t diag, int n, const double* a, int lda, double* x, int incx) {
    if (!g_openblas_context || !g_openblas_context->dtrsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->dtrsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void openblas_strmm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   fb_transpose_t trans, fb_diag_t diag, int m, int n,
                                   float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_openblas_context || !g_openblas_context->strmm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->strmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void openblas_dtrmm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   fb_transpose_t trans, fb_diag_t diag, int m, int n,
                                   double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_openblas_context || !g_openblas_context->dtrmm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->dtrmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void openblas_strsm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   fb_transpose_t trans, fb_diag_t diag, int m, int n,
                                   float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_openblas_context || !g_openblas_context->strsm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->strsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void openblas_dtrsm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   fb_transpose_t trans, fb_diag_t diag, int m, int n,
                                   double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_openblas_context || !g_openblas_context->dtrsm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_UNIT) ? CblasUnit : CblasNonUnit;
    g_openblas_context->dtrsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void openblas_ssymm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   int m, int n, float alpha, const float* a, int lda,
                                   const float* b, int ldb, float beta, float* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->ssymm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->ssymm(cblas_layout, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void openblas_dsymm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo,
                                   int m, int n, double alpha, const double* a, int lda,
                                   const double* b, int ldb, double beta, double* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->dsymm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_openblas_context->dsymm(cblas_layout, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void openblas_ssyrk_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   int n, int k, float alpha, const float* a, int lda,
                                   float beta, float* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->ssyrk) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->ssyrk(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void openblas_dsyrk_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                   int n, int k, double alpha, const double* a, int lda,
                                   double beta, double* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->dsyrk) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->dsyrk(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void openblas_ssyr2k_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                    int n, int k, float alpha, const float* a, int lda,
                                    const float* b, int ldb, float beta, float* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->ssyr2k) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->ssyr2k(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void openblas_dsyr2k_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                    int n, int k, double alpha, const double* a, int lda,
                                    const double* b, int ldb, double beta, double* c, int ldc) {
    if (!g_openblas_context || !g_openblas_context->dsyr2k) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_openblas_context->dsyr2k(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

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

static fb_plugin_probe_result_t openblas_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
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
    
    /* Load Phase 1 BLAS functions */
    ctx->srot = (cblas_srot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srot");
    ctx->drot = (cblas_drot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drot");
    ctx->srotg = (cblas_srotg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotg");
    ctx->drotg = (cblas_drotg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotg");
    ctx->srotm = (cblas_srotm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotm");
    ctx->drotm = (cblas_drotm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotm");
    ctx->srotmg = (cblas_srotmg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotmg");
    ctx->drotmg = (cblas_drotmg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotmg");
    ctx->ssymv = (cblas_ssymv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssymv");
    ctx->dsymv = (cblas_dsymv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsymv");
    ctx->sger = (cblas_sger_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sger");
    ctx->dger = (cblas_dger_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dger");
    ctx->ssyr = (cblas_ssyr_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr");
    ctx->dsyr = (cblas_dsyr_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr");
    ctx->ssyr2 = (cblas_ssyr2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr2");
    ctx->dsyr2 = (cblas_dsyr2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr2");
    ctx->strmv = (cblas_strmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strmv");
    ctx->dtrmv = (cblas_dtrmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrmv");
    ctx->strsv = (cblas_strsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strsv");
    ctx->dtrsv = (cblas_dtrsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrsv");
    ctx->strmm = (cblas_strmm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strmm");
    ctx->dtrmm = (cblas_dtrmm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrmm");
    ctx->strsm = (cblas_strsm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strsm");
    ctx->dtrsm = (cblas_dtrsm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrsm");
    ctx->ssymm = (cblas_ssymm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssymm");
    ctx->dsymm = (cblas_dsymm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsymm");
    ctx->ssyrk = (cblas_ssyrk_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyrk");
    ctx->dsyrk = (cblas_dsyrk_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyrk");
    ctx->ssyr2k = (cblas_ssyr2k_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr2k");
    ctx->dsyr2k = (cblas_dsyr2k_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr2k");
    
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
    
    if (!ctx->sdot || !ctx->sgemm) {
        free(ctx);
        return -3;
    }
    
    /* Populate vtable */
    g_openblas_context = ctx;
    g_openblas_vtable.saxpy = openblas_saxpy_wrapper;
    g_openblas_vtable.sdot = openblas_sdot_wrapper;
    g_openblas_vtable.snrm2 = openblas_snrm2_wrapper;
    g_openblas_vtable.sasum = openblas_sasum_wrapper;
    g_openblas_vtable.isamax = openblas_isamax_wrapper;
    g_openblas_vtable.scopy = openblas_scopy_wrapper;
    g_openblas_vtable.sscal = openblas_sscal_wrapper;
    g_openblas_vtable.sswap = openblas_sswap_wrapper;
    g_openblas_vtable.sgemv = openblas_sgemv_wrapper;
    g_openblas_vtable.sgemm = openblas_sgemm_wrapper;
    
    /* Phase 1 BLAS operations */
    g_openblas_vtable.srot = openblas_srot_wrapper;
    g_openblas_vtable.drot = openblas_drot_wrapper;
    g_openblas_vtable.srotg = openblas_srotg_wrapper;
    g_openblas_vtable.drotg = openblas_drotg_wrapper;
    g_openblas_vtable.srotm = openblas_srotm_wrapper;
    g_openblas_vtable.drotm = openblas_drotm_wrapper;
    g_openblas_vtable.srotmg = openblas_srotmg_wrapper;
    g_openblas_vtable.drotmg = openblas_drotmg_wrapper;
    g_openblas_vtable.ssymv = openblas_ssymv_wrapper;
    g_openblas_vtable.dsymv = openblas_dsymv_wrapper;
    g_openblas_vtable.sger = openblas_sger_wrapper;
    g_openblas_vtable.dger = openblas_dger_wrapper;
    g_openblas_vtable.ssyr = openblas_ssyr_wrapper;
    g_openblas_vtable.dsyr = openblas_dsyr_wrapper;
    g_openblas_vtable.ssyr2 = openblas_ssyr2_wrapper;
    g_openblas_vtable.dsyr2 = openblas_dsyr2_wrapper;
    g_openblas_vtable.strmv = openblas_strmv_wrapper;
    g_openblas_vtable.dtrmv = openblas_dtrmv_wrapper;
    g_openblas_vtable.strsv = openblas_strsv_wrapper;
    g_openblas_vtable.dtrsv = openblas_dtrsv_wrapper;
    g_openblas_vtable.strmm = openblas_strmm_wrapper;
    g_openblas_vtable.dtrmm = openblas_dtrmm_wrapper;
    g_openblas_vtable.strsm = openblas_strsm_wrapper;
    g_openblas_vtable.dtrsm = openblas_dtrsm_wrapper;
    g_openblas_vtable.ssymm = openblas_ssymm_wrapper;
    g_openblas_vtable.dsymm = openblas_dsymm_wrapper;
    g_openblas_vtable.ssyrk = openblas_ssyrk_wrapper;
    g_openblas_vtable.dsyrk = openblas_dsyrk_wrapper;
    g_openblas_vtable.ssyr2k = openblas_ssyr2k_wrapper;
    g_openblas_vtable.dsyr2k = openblas_dsyr2k_wrapper;
    
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
