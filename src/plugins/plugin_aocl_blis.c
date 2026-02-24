/**
 * @file plugin_aocl_blis.c
 * @brief AMD AOCL BLIS plugin implementation
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

/* CBLAS enums (AOCL uses CBLAS interface) */
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
    #include <intrin.h>
    #define FB_LOAD_LIBRARY(path) LoadLibraryA(path)
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #include <cpuid.h>
    #define FB_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* CPU vendor detection */
static int is_amd_cpu(void) {
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
    
    return strcmp(vendor, "AuthenticAMD") == 0;
}

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

/* PHASE2: CBLAS typedefs for symmetric rank updates and band operations */
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;

typedef void (*cblas_ssyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dsyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_ssyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double, double*, const int);
typedef void (*cblas_sgbmv_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, const int, const int, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dgbmv_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, const int, const int, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);

/* PHASE1: Rotation operations */
typedef void (*cblas_srot_t)(const int, float*, const int, float*, const int, const float, const float);
typedef void (*cblas_drot_t)(const int, double*, const int, double*, const int, const double, const double);
typedef void (*cblas_srotg_t)(float*, float*, float*, float*);
typedef void (*cblas_drotg_t)(double*, double*, double*, double*);
typedef void (*cblas_srotm_t)(const int, float*, const int, float*, const int, const float*);
typedef void (*cblas_drotm_t)(const int, double*, const int, double*, const int, const double*);
typedef void (*cblas_srotmg_t)(float*, float*, float*, const float, float*);
typedef void (*cblas_drotmg_t)(double*, double*, double*, const double, double*);

/* PHASE1: Symmetric matrix operations */
typedef void (*cblas_ssymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_sger_t)(CBLAS_LAYOUT, const int, const int, const float, const float*, const int, const float*, const int, float*, const int);
typedef void (*cblas_dger_t)(CBLAS_LAYOUT, const int, const int, const double, const double*, const int, const double*, const int, double*, const int);
typedef void (*cblas_ssyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, float*, const int);
typedef void (*cblas_dsyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, double*, const int);
typedef void (*cblas_ssbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);

/* PHASE1: Triangular matrix operations */
typedef void (*cblas_strmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);
typedef void (*cblas_strsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);
typedef void (*cblas_stbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double*, const int, double*, const int);
typedef void (*cblas_stbsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtbsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double*, const int, double*, const int);

/* PHASE1: Level 3 symmetric/triangular operations */
typedef void (*cblas_ssymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_ssyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_strmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_strsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);

/* Plugin context - holds loaded function pointers and library handle */
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
    
    /* PHASE2: Symmetric rank updates and band operations */
    cblas_ssyr_t ssyr;
    cblas_dsyr_t dsyr;
    cblas_ssyrk_t ssyrk;
    cblas_dsyrk_t dsyrk;
    cblas_sgbmv_t sgbmv;
    cblas_dgbmv_t dgbmv;
    
    /* PHASE1: Rotation operations */
    cblas_srot_t srot;
    cblas_drot_t drot;
    cblas_srotg_t srotg;
    cblas_drotg_t drotg;
    cblas_srotm_t srotm;
    cblas_drotm_t drotm;
    cblas_srotmg_t srotmg;
    cblas_drotmg_t drotmg;
    
    /* PHASE1: Symmetric matrix operations */
    cblas_ssymv_t ssymv;
    cblas_dsymv_t dsymv;
    cblas_sger_t sger;
    cblas_dger_t dger;
    cblas_ssyr2_t ssyr2;
    cblas_dsyr2_t dsyr2;
    cblas_ssbmv_t ssbmv;
    cblas_dsbmv_t dsbmv;
    
    /* PHASE1: Triangular matrix operations */
    cblas_strmv_t strmv;
    cblas_dtrmv_t dtrmv;
    cblas_strsv_t strsv;
    cblas_dtrsv_t dtrsv;
    cblas_stbmv_t stbmv;
    cblas_dtbmv_t dtbmv;
    cblas_stbsv_t stbsv;
    cblas_dtbsv_t dtbsv;
    
    /* PHASE1: Level 3 symmetric/triangular operations */
    cblas_ssymm_t ssymm;
    cblas_dsymm_t dsymm;
    cblas_ssyr2k_t ssyr2k;
    cblas_dsyr2k_t dsyr2k;
    cblas_strmm_t strmm;
    cblas_dtrmm_t dtrmm;
    cblas_strsm_t strsm;
    cblas_dtrsm_t dtrsm;
    
} aocl_plugin_context_t;

/* Static backend vtable that will be populated during init */
static fb_backend_vtable_t g_aocl_vtable;

/* File-scoped context - set during init */
static aocl_plugin_context_t* g_aocl_context = NULL;

/* Plugin metadata */
static const fb_plugin_metadata_t g_aocl_metadata = {
    .name = "aocl-blis",
    .version = "4.2.1",
    .vendor = "AMD",
    .description = "AMD Optimizing CPU Libraries - BLIS (CBLAS interface)",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | 
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/* Wrapper functions to adapt CBLAS signatures to vtable signatures */
static float aocl_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->sdot) return 0.0f;
    return g_aocl_context->sdot(n, x, incx, y, incy);
}

static float aocl_snrm2_wrapper(int n, const float* x, int incx) {
    
    if (!g_aocl_context || !g_aocl_context->snrm2) return 0.0f;
    return g_aocl_context->snrm2(n, x, incx);
}

static float aocl_sasum_wrapper(int n, const float* x, int incx) {
    
    if (!g_aocl_context || !g_aocl_context->sasum) return 0.0f;
    return g_aocl_context->sasum(n, x, incx);
}

static int aocl_isamax_wrapper(int n, const float* x, int incx) {
    
    if (!g_aocl_context || !g_aocl_context->isamax) return 0;
    return g_aocl_context->isamax(n, x, incx);
}

static void aocl_saxpy_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    
    if (g_aocl_context && g_aocl_context->saxpy) {
        g_aocl_context->saxpy(n, alpha, x, incx, y, incy);
    }
}

static void aocl_scopy_wrapper(int n, const float* x, int incx, float* y, int incy) {
    
    if (g_aocl_context && g_aocl_context->scopy) {
        g_aocl_context->scopy(n, x, incx, y, incy);
    }
}

static void aocl_sscal_wrapper(int n, float alpha, float* x, int incx) {
    
    if (g_aocl_context && g_aocl_context->sscal) {
        g_aocl_context->sscal(n, alpha, x, incx);
    }
}

static void aocl_sswap_wrapper(int n, float* x, int incx, float* y, int incy) {
    
    if (g_aocl_context && g_aocl_context->sswap) {
        g_aocl_context->sswap(n, x, incx, y, incy);
    }
}

static void aocl_sgemv_wrapper(fb_layout_t layout, fb_transpose_t trans, int m, int n, float alpha,
                               const float* a, int lda, const float* x, int incx,
                               float beta, float* y, int incy) {
    
    if (!g_aocl_context || !g_aocl_context->sgemv) return;
    
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->sgemv(cblas_layout, cblas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void aocl_sgemm_wrapper(fb_layout_t layout, fb_transpose_t transa, fb_transpose_t transb, int m, int n, int k,
                               float alpha, const float* a, int lda,
                               const float* b, int ldb, float beta,
                               float* c, int ldc) {
    
    if (!g_aocl_context || !g_aocl_context->sgemm) return;
    
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_transa = (transa == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb = (transb == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->sgemm(cblas_layout, cblas_transa, cblas_transb, m, n, k,
                           alpha, a, lda, b, ldb, beta, c, ldc);
}

/* PHASE2 wrappers */
static void aocl_ssyr_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                              const float* x, int incx, float* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->ssyr) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->ssyr(cblas_layout, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void aocl_dsyr_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                              const double* x, int incx, double* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->dsyr) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->dsyr(cblas_layout, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void aocl_ssyrk_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                               int n, int k, float alpha, const float* a, int lda,
                               float beta, float* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->ssyrk) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->ssyrk(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void aocl_dsyrk_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                               int n, int k, double alpha, const double* a, int lda,
                               double beta, double* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->dsyrk) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->dsyrk(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void aocl_sgbmv_wrapper(fb_layout_t layout, fb_transpose_t trans, int m, int n, int kl, int ku,
                               float alpha, const float* a, int lda, const float* x, int incx,
                               float beta, float* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->sgbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->sgbmv(cblas_layout, cblas_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static void aocl_dgbmv_wrapper(fb_layout_t layout, fb_transpose_t trans, int m, int n, int kl, int ku,
                               double alpha, const double* a, int lda, const double* x, int incx,
                               double beta, double* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->dgbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->dgbmv(cblas_layout, cblas_trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

/* PHASE1 wrappers - Rotation operations */
static void aocl_srot_wrapper(int n, float* x, int incx, float* y, int incy, float c, float s) {
    if (g_aocl_context && g_aocl_context->srot) {
        g_aocl_context->srot(n, x, incx, y, incy, c, s);
    }
}

static void aocl_drot_wrapper(int n, double* x, int incx, double* y, int incy, double c, double s) {
    if (g_aocl_context && g_aocl_context->drot) {
        g_aocl_context->drot(n, x, incx, y, incy, c, s);
    }
}

static void aocl_srotg_wrapper(float* a, float* b, float* c, float* s) {
    if (g_aocl_context && g_aocl_context->srotg) {
        g_aocl_context->srotg(a, b, c, s);
    }
}

static void aocl_drotg_wrapper(double* a, double* b, double* c, double* s) {
    if (g_aocl_context && g_aocl_context->drotg) {
        g_aocl_context->drotg(a, b, c, s);
    }
}

static void aocl_srotm_wrapper(int n, float* x, int incx, float* y, int incy, const float* param) {
    if (g_aocl_context && g_aocl_context->srotm) {
        g_aocl_context->srotm(n, x, incx, y, incy, param);
    }
}

static void aocl_drotm_wrapper(int n, double* x, int incx, double* y, int incy, const double* param) {
    if (g_aocl_context && g_aocl_context->drotm) {
        g_aocl_context->drotm(n, x, incx, y, incy, param);
    }
}

static void aocl_srotmg_wrapper(float* d1, float* d2, float* x1, float y1, float* param) {
    if (g_aocl_context && g_aocl_context->srotmg) {
        g_aocl_context->srotmg(d1, d2, x1, y1, param);
    }
}

static void aocl_drotmg_wrapper(double* d1, double* d2, double* x1, double y1, double* param) {
    if (g_aocl_context && g_aocl_context->drotmg) {
        g_aocl_context->drotmg(d1, d2, x1, y1, param);
    }
}

/* PHASE1 wrappers - Symmetric matrix operations */
static void aocl_ssymv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                               const float* a, int lda, const float* x, int incx,
                               float beta, float* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->ssymv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->ssymv(cblas_layout, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void aocl_dsymv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                               const double* a, int lda, const double* x, int incx,
                               double beta, double* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->dsymv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->dsymv(cblas_layout, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void aocl_sger_wrapper(fb_layout_t layout, int m, int n, float alpha,
                              const float* x, int incx, const float* y, int incy,
                              float* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->sger) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    g_aocl_context->sger(cblas_layout, m, n, alpha, x, incx, y, incy, a, lda);
}

static void aocl_dger_wrapper(fb_layout_t layout, int m, int n, double alpha,
                              const double* x, int incx, const double* y, int incy,
                              double* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->dger) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    g_aocl_context->dger(cblas_layout, m, n, alpha, x, incx, y, incy, a, lda);
}

static void aocl_ssyr2_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, float alpha,
                               const float* x, int incx, const float* y, int incy,
                               float* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->ssyr2) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->ssyr2(cblas_layout, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void aocl_dsyr2_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, double alpha,
                               const double* x, int incx, const double* y, int incy,
                               double* a, int lda) {
    if (!g_aocl_context || !g_aocl_context->dsyr2) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->dsyr2(cblas_layout, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void aocl_ssbmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, int k, float alpha,
                               const float* a, int lda, const float* x, int incx,
                               float beta, float* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->ssbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->ssbmv(cblas_layout, cblas_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static void aocl_dsbmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, int n, int k, double alpha,
                               const double* a, int lda, const double* x, int incx,
                               double beta, double* y, int incy) {
    if (!g_aocl_context || !g_aocl_context->dsbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->dsbmv(cblas_layout, cblas_uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/* PHASE1 wrappers - Triangular matrix operations */
static void aocl_strmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, const float* a, int lda, float* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->strmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->strmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void aocl_dtrmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, const double* a, int lda, double* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->dtrmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtrmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void aocl_strsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, const float* a, int lda, float* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->strsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->strsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void aocl_dtrsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, const double* a, int lda, double* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->dtrsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtrsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void aocl_stbmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, int k, const float* a, int lda, float* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->stbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->stbmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, k, a, lda, x, incx);
}

static void aocl_dtbmv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, int k, const double* a, int lda, double* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->dtbmv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtbmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, k, a, lda, x, incx);
}

static void aocl_stbsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, int k, const float* a, int lda, float* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->stbsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->stbsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, k, a, lda, x, incx);
}

static void aocl_dtbsv_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int n, int k, const double* a, int lda, double* x, int incx) {
    if (!g_aocl_context || !g_aocl_context->dtbsv) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtbsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, k, a, lda, x, incx);
}

/* PHASE1 wrappers - Level 3 symmetric/triangular operations */
static void aocl_ssymm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, int m, int n,
                               float alpha, const float* a, int lda, const float* b, int ldb,
                               float beta, float* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->ssymm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->ssymm(cblas_layout, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void aocl_dsymm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, int m, int n,
                               double alpha, const double* a, int lda, const double* b, int ldb,
                               double beta, double* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->dsymm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl_context->dsymm(cblas_layout, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void aocl_ssyr2k_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                int n, int k, float alpha, const float* a, int lda,
                                const float* b, int ldb, float beta, float* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->ssyr2k) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->ssyr2k(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void aocl_dsyr2k_wrapper(fb_layout_t layout, fb_uplo_t uplo, fb_transpose_t trans,
                                int n, int k, double alpha, const double* a, int lda,
                                const double* b, int ldb, double beta, double* c, int ldc) {
    if (!g_aocl_context || !g_aocl_context->dsyr2k) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    g_aocl_context->dsyr2k(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void aocl_strmm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_aocl_context || !g_aocl_context->strmm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->strmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void aocl_dtrmm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_aocl_context || !g_aocl_context->dtrmm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtrmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void aocl_strsm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_aocl_context || !g_aocl_context->strsm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->strsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void aocl_dtrsm_wrapper(fb_layout_t layout, fb_side_t side, fb_uplo_t uplo, fb_transpose_t trans, fb_diag_t diag,
                               int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_aocl_context || !g_aocl_context->dtrsm) return;
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    g_aocl_context->dtrsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

/* Backend capability functions */
static uint32_t fb_aocl_get_capabilities_wrapper(void* handle) {
    (void)handle;
    return FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
           FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

static int fb_aocl_get_num_threads_wrapper(void* handle) {
    (void)handle;
    /* AOCL BLIS may support thread control - return default */
    return 1;
}

static void fb_aocl_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* TODO: Call AOCL thread control if available */
}

/* Plugin probe function - search for AOCL BLIS and return compatibility score */
static fb_plugin_probe_result_t aocl_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
    const char* lib_names[] = {
#ifdef _WIN32
        "AOCL-LibBlis-Win-dll.dll",
        "AOCL-LibBlis-Win-MT-dll.dll",
#elif defined(__APPLE__)
        "libaocl_blis.dylib",
#else
        "libaocl_blis.so",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib",
#elif defined(__APPLE__)
        "/opt/AMD/aocl/lib",
        "/usr/local/lib",
#else
        "/opt/AMD/aocl/lib",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    /* Try to find the library */
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names, 
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* Check for CBLAS symbols to verify it's AOCL BLIS */
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            /* Adjust score based on CPU vendor */
            if (is_amd_cpu()) {
                result.score = 95; /* AOCL is highly optimized for AMD CPUs */
                result.reason = "Found AOCL BLIS with CBLAS interface (AMD CPU detected)";
            } else {
                result.score = 70; /* Lower priority on non-AMD CPUs */
                result.reason = "Found AOCL BLIS with CBLAS interface (non-AMD CPU - not optimal)";
            }
            result.library_path = NULL; /* Init will load the library */
        } else {
            result.score = 0;
            result.reason = "Library found but missing required CBLAS symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "AOCL BLIS library not found in search paths";
    }
    
    return result;
}

/* Plugin init function - load library and map function pointers */
static int aocl_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    printf("[AOCL] Init called\n");
    if (!ctx_out) {
        printf("[AOCL] ctx_out is NULL\n");
        return -1;
    }
    
    /* Load the library if not provided */
    if (!lib_handle) {
        printf("[AOCL] Loading library...\n");
        const char* lib_names[] = {
#ifdef _WIN32
            /* Prioritize multi-threaded version */
            "AOCL-LibBlis-Win-MT-dll.dll",
            "AOCL-LibBlis-Win-dll.dll",
#elif defined(__APPLE__)
            "libaocl_blis.dylib",
#else
            "libaocl_blis.so",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            /* Try MT version paths first */
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib",
#elif defined(__APPLE__)
            "/opt/AMD/aocl/lib",
            "/usr/local/lib",
#else
            "/opt/AMD/aocl/lib",
            "/usr/lib/x86_64-linux-gnu",
#endif
            NULL
        };
        
        /* Try to load with full paths for MT version first */
#ifdef _WIN32
        const char* mt_paths[] = {
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-MT-dll.dll",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64\\AOCL-LibBlis-Win-MT-dll.dll",
            NULL
        };
        for (int i = 0; mt_paths[i]; i++) {
            lib_handle = FB_LOAD_LIBRARY(mt_paths[i]);
            if (lib_handle) {
                printf("[AOCL] Loaded MT version from: %s\n", mt_paths[i]);
                break;
            }
        }
#endif
        
        if (!lib_handle) {
            lib_handle = fb_plugin_load_library(lib_names, default_paths);
        }
        printf("[AOCL] Library load result: %p\n", lib_handle);
        if (!lib_handle) {
            printf("[AOCL] Failed to load library\n");
            return -1;
        }
    }
    
    printf("[AOCL] Allocating context...\n");
    /* Allocate plugin context */
    aocl_plugin_context_t* ctx = (aocl_plugin_context_t*)calloc(1, sizeof(aocl_plugin_context_t));
    printf("[AOCL] Context allocated at %p\n", ctx);
    if (!ctx) {
        printf("[AOCL] Allocation failed!\n");
        return -2; /* Out of memory */
    }
    
    printf("[AOCL] Setting lib_handle...\n");
    ctx->lib_handle = lib_handle;
    
    printf("[AOCL] Loading CBLAS functions...\n");
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
    
    /* Load PHASE2 functions */
    ctx->ssyr = (cblas_ssyr_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr");
    ctx->dsyr = (cblas_dsyr_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr");
    ctx->ssyrk = (cblas_ssyrk_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyrk");
    ctx->dsyrk = (cblas_dsyrk_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyrk");
    ctx->sgbmv = (cblas_sgbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sgbmv");
    ctx->dgbmv = (cblas_dgbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dgbmv");
    
    /* Load PHASE1 rotation functions */
    ctx->srot = (cblas_srot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srot");
    ctx->drot = (cblas_drot_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drot");
    ctx->srotg = (cblas_srotg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotg");
    ctx->drotg = (cblas_drotg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotg");
    ctx->srotm = (cblas_srotm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotm");
    ctx->drotm = (cblas_drotm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotm");
    ctx->srotmg = (cblas_srotmg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_srotmg");
    ctx->drotmg = (cblas_drotmg_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_drotmg");
    
    /* Load PHASE1 symmetric matrix functions */
    ctx->ssymv = (cblas_ssymv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssymv");
    ctx->dsymv = (cblas_dsymv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsymv");
    ctx->sger = (cblas_sger_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_sger");
    ctx->dger = (cblas_dger_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dger");
    ctx->ssyr2 = (cblas_ssyr2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr2");
    ctx->dsyr2 = (cblas_dsyr2_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr2");
    ctx->ssbmv = (cblas_ssbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssbmv");
    ctx->dsbmv = (cblas_dsbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsbmv");
    
    /* Load PHASE1 triangular matrix functions */
    ctx->strmv = (cblas_strmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strmv");
    ctx->dtrmv = (cblas_dtrmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrmv");
    ctx->strsv = (cblas_strsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strsv");
    ctx->dtrsv = (cblas_dtrsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrsv");
    ctx->stbmv = (cblas_stbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_stbmv");
    ctx->dtbmv = (cblas_dtbmv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtbmv");
    ctx->stbsv = (cblas_stbsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_stbsv");
    ctx->dtbsv = (cblas_dtbsv_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtbsv");
    
    /* Load PHASE1 level 3 symmetric/triangular functions */
    ctx->ssymm = (cblas_ssymm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssymm");
    ctx->dsymm = (cblas_dsymm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsymm");
    ctx->ssyr2k = (cblas_ssyr2k_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_ssyr2k");
    ctx->dsyr2k = (cblas_dsyr2k_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dsyr2k");
    ctx->strmm = (cblas_strmm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strmm");
    ctx->dtrmm = (cblas_dtrmm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrmm");
    ctx->strsm = (cblas_strsm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_strsm");
    ctx->dtrsm = (cblas_dtrsm_t)FB_GET_PROC_ADDRESS(lib_handle, "cblas_dtrsm");
    
    printf("[AOCL] Checking critical functions...\n");
    /* Check critical functions */
    if (!ctx->sdot || !ctx->sgemm) {
        printf("[AOCL] Missing critical symbols!\n");
        free(ctx);
        return -3; /* Missing critical symbols */
    }
    
    printf("[AOCL] Populating vtable...\n");
    /* Populate vtable with wrappers */
    g_aocl_context = ctx;
    g_aocl_vtable.saxpy = aocl_saxpy_wrapper;
    g_aocl_vtable.sdot = aocl_sdot_wrapper;
    g_aocl_vtable.snrm2 = aocl_snrm2_wrapper;
    g_aocl_vtable.sasum = aocl_sasum_wrapper;
    g_aocl_vtable.isamax = aocl_isamax_wrapper;
    g_aocl_vtable.scopy = aocl_scopy_wrapper;
    g_aocl_vtable.sscal = aocl_sscal_wrapper;
    g_aocl_vtable.sswap = aocl_sswap_wrapper;
    g_aocl_vtable.sgemv = aocl_sgemv_wrapper;
    g_aocl_vtable.sgemm = aocl_sgemm_wrapper;
    
    /* Populate PHASE2 operations */
    g_aocl_vtable.ssyr = aocl_ssyr_wrapper;
    g_aocl_vtable.dsyr = aocl_dsyr_wrapper;
    g_aocl_vtable.ssyrk = aocl_ssyrk_wrapper;
    g_aocl_vtable.dsyrk = aocl_dsyrk_wrapper;
    g_aocl_vtable.sgbmv = aocl_sgbmv_wrapper;
    g_aocl_vtable.dgbmv = aocl_dgbmv_wrapper;
    
    /* Populate PHASE1 rotation operations */
    g_aocl_vtable.srot = aocl_srot_wrapper;
    g_aocl_vtable.drot = aocl_drot_wrapper;
    g_aocl_vtable.srotg = aocl_srotg_wrapper;
    g_aocl_vtable.drotg = aocl_drotg_wrapper;
    g_aocl_vtable.srotm = aocl_srotm_wrapper;
    g_aocl_vtable.drotm = aocl_drotm_wrapper;
    g_aocl_vtable.srotmg = aocl_srotmg_wrapper;
    g_aocl_vtable.drotmg = aocl_drotmg_wrapper;
    
    /* Populate PHASE1 symmetric matrix operations */
    g_aocl_vtable.ssymv = aocl_ssymv_wrapper;
    g_aocl_vtable.dsymv = aocl_dsymv_wrapper;
    g_aocl_vtable.sger = aocl_sger_wrapper;
    g_aocl_vtable.dger = aocl_dger_wrapper;
    g_aocl_vtable.ssyr2 = aocl_ssyr2_wrapper;
    g_aocl_vtable.dsyr2 = aocl_dsyr2_wrapper;
    g_aocl_vtable.ssbmv = aocl_ssbmv_wrapper;
    g_aocl_vtable.dsbmv = aocl_dsbmv_wrapper;
    
    /* Populate PHASE1 triangular matrix operations */
    g_aocl_vtable.strmv = aocl_strmv_wrapper;
    g_aocl_vtable.dtrmv = aocl_dtrmv_wrapper;
    g_aocl_vtable.strsv = aocl_strsv_wrapper;
    g_aocl_vtable.dtrsv = aocl_dtrsv_wrapper;
    g_aocl_vtable.stbmv = aocl_stbmv_wrapper;
    g_aocl_vtable.dtbmv = aocl_dtbmv_wrapper;
    g_aocl_vtable.stbsv = aocl_stbsv_wrapper;
    g_aocl_vtable.dtbsv = aocl_dtbsv_wrapper;
    
    /* Populate PHASE1 level 3 symmetric/triangular operations */
    g_aocl_vtable.ssymm = aocl_ssymm_wrapper;
    g_aocl_vtable.dsymm = aocl_dsymm_wrapper;
    g_aocl_vtable.ssyr2k = aocl_ssyr2k_wrapper;
    g_aocl_vtable.dsyr2k = aocl_dsyr2k_wrapper;
    g_aocl_vtable.strmm = aocl_strmm_wrapper;
    g_aocl_vtable.dtrmm = aocl_dtrmm_wrapper;
    g_aocl_vtable.strsm = aocl_strsm_wrapper;
    g_aocl_vtable.dtrsm = aocl_dtrsm_wrapper;
    
    /* CPU backend properties - null operations for memory/stream management */
    g_aocl_vtable.mem_alloc = NULL;
    g_aocl_vtable.mem_free = NULL;
    g_aocl_vtable.mem_upload = NULL;
    g_aocl_vtable.mem_download = NULL;
    g_aocl_vtable.mem_copy = NULL;
    g_aocl_vtable.stream_create = NULL;
    g_aocl_vtable.stream_destroy = NULL;
    g_aocl_vtable.stream_sync = NULL;
    g_aocl_vtable.stream_set = NULL;
    
    /* Backend capabilities */
    g_aocl_vtable.get_capabilities = fb_aocl_get_capabilities_wrapper;
    g_aocl_vtable.get_num_threads = fb_aocl_get_num_threads_wrapper;
    g_aocl_vtable.set_num_threads = fb_aocl_set_num_threads_wrapper;
    
    /* Return initialized context */
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0; /* Success */
}

/* Shutdown plugin */
static void aocl_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        aocl_plugin_context_t* aocl_ctx = (aocl_plugin_context_t*)ctx;
        /* Library handle will be freed by plugin registry */
        free(aocl_ctx);
    }
}

/* Get backend vtable */
static const fb_backend_vtable_t* aocl_get_vtable(fb_plugin_context_t* ctx) {
    (void)ctx;
    return &g_aocl_vtable;
}

/* Get plugin context */
static void* aocl_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

/* Plugin interface */
static const fb_backend_plugin_t g_aocl_plugin = {
    .metadata = &g_aocl_metadata,
    .probe = aocl_probe,
    .init = aocl_init,
    .get_vtable = aocl_get_vtable,
    .get_context = aocl_get_context,
    .shutdown = aocl_shutdown,
    .set_num_threads = NULL, /* TODO: Implement if AOCL supports threading control */
    .get_num_threads = NULL
};

/* Plugin registration function - must be called explicitly */
void fb_register_aocl_plugin(void) {
    fb_register_plugin(&g_aocl_plugin);
}
