/**
 * @file plugin_mkl.c
 * @brief Intel MKL plugin implementation
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

/* CBLAS enums for Phase 1 operations */
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;

/* PHASE1: Rotation operations */
typedef void (*cblas_srot_t)(const int, float*, const int, float*, const int, const float, const float);
typedef void (*cblas_drot_t)(const int, double*, const int, double*, const int, const double, const double);
typedef void (*cblas_srotg_t)(float*, float*, float*, float*);
typedef void (*cblas_drotg_t)(double*, double*, double*, double*);
typedef void (*cblas_srotm_t)(const int, float*, const int, float*, const int, const float*);
typedef void (*cblas_drotm_t)(const int, double*, const int, double*, const int, const double*);
typedef void (*cblas_srotmg_t)(float*, float*, float*, const float, float*);
typedef void (*cblas_drotmg_t)(double*, double*, double*, const double, double*);

/* PHASE1: Symmetric/triangular matrix operations */
typedef void (*cblas_ssymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_sger_t)(CBLAS_LAYOUT, const int, const int, const float, const float*, const int, const float*, const int, float*, const int);
typedef void (*cblas_dger_t)(CBLAS_LAYOUT, const int, const int, const double, const double*, const int, const double*, const int, double*, const int);
typedef void (*cblas_ssyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dsyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_ssyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, float*, const int);
typedef void (*cblas_dsyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, double*, const int);
typedef void (*cblas_strmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);
typedef void (*cblas_strsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);
typedef void (*cblas_strmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_strsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_ssymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_ssyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double, double*, const int);
typedef void (*cblas_ssyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);

/* MKL specific functions */
typedef void (*mkl_set_num_threads_t)(int num_threads);
typedef int (*mkl_get_max_threads_t)(void);

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
    
    /* PHASE1 operations */
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
    
    /* MKL specific */
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
static float mkl_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!g_mkl_context || !g_mkl_context->sdot) return 0.0f;
    return g_mkl_context->sdot(n, x, incx, y, incy);
}

static float mkl_snrm2_wrapper(int n, const float* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->snrm2) return 0.0f;
    return g_mkl_context->snrm2(n, x, incx);
}

static float mkl_sasum_wrapper(int n, const float* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->sasum) return 0.0f;
    return g_mkl_context->sasum(n, x, incx);
}

static int mkl_isamax_wrapper(int n, const float* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->isamax) return 0;
    return g_mkl_context->isamax(n, x, incx);
}

static void mkl_saxpy_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_mkl_context && g_mkl_context->saxpy) {
        g_mkl_context->saxpy(n, alpha, x, incx, y, incy);
    }
}

static void mkl_scopy_wrapper(int n, const float* x, int incx, float* y, int incy) {
    if (g_mkl_context && g_mkl_context->scopy) {
        g_mkl_context->scopy(n, x, incx, y, incy);
    }
}

static void mkl_sscal_wrapper(int n, float alpha, float* x, int incx) {
    if (g_mkl_context && g_mkl_context->sscal) {
        g_mkl_context->sscal(n, alpha, x, incx);
    }
}

static void mkl_sswap_wrapper(int n, float* x, int incx, float* y, int incy) {
    if (g_mkl_context && g_mkl_context->sswap) {
        g_mkl_context->sswap(n, x, incx, y, incy);
    }
}

static void mkl_sgemv_wrapper(char trans, int m, int n, float alpha,
                              const float* a, int lda, const float* x, int incx,
                              float beta, float* y, int incy) {
    if (!g_mkl_context || !g_mkl_context->sgemv) return;
    
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->sgemv(CblasColMajor, cblas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void mkl_sgemm_wrapper(char transa, char transb, int m, int n, int k,
                              float alpha, const float* a, int lda,
                              const float* b, int ldb, float beta,
                              float* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->sgemm) return;
    
    CBLAS_TRANSPOSE cblas_transa = (transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb = (transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->sgemm(CblasColMajor, cblas_transa, cblas_transb, m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
}

/* Phase 1 BLAS wrapper functions */
static void mkl_srot_wrapper(int n, float* x, int incx, float* y, int incy, float c, float s) {
    if (g_mkl_context && g_mkl_context->srot) {
        g_mkl_context->srot(n, x, incx, y, incy, c, s);
    }
}

static void mkl_drot_wrapper(int n, double* x, int incx, double* y, int incy, double c, double s) {
    if (g_mkl_context && g_mkl_context->drot) {
        g_mkl_context->drot(n, x, incx, y, incy, c, s);
    }
}

static void mkl_srotg_wrapper(float* a, float* b, float* c, float* s) {
    if (g_mkl_context && g_mkl_context->srotg) {
        g_mkl_context->srotg(a, b, c, s);
    }
}

static void mkl_drotg_wrapper(double* a, double* b, double* c, double* s) {
    if (g_mkl_context && g_mkl_context->drotg) {
        g_mkl_context->drotg(a, b, c, s);
    }
}

static void mkl_srotm_wrapper(int n, float* x, int incx, float* y, int incy, const float* param) {
    if (g_mkl_context && g_mkl_context->srotm) {
        g_mkl_context->srotm(n, x, incx, y, incy, param);
    }
}

static void mkl_drotm_wrapper(int n, double* x, int incx, double* y, int incy, const double* param) {
    if (g_mkl_context && g_mkl_context->drotm) {
        g_mkl_context->drotm(n, x, incx, y, incy, param);
    }
}

static void mkl_srotmg_wrapper(float* d1, float* d2, float* x1, float y1, float* param) {
    if (g_mkl_context && g_mkl_context->srotmg) {
        g_mkl_context->srotmg(d1, d2, x1, y1, param);
    }
}

static void mkl_drotmg_wrapper(double* d1, double* d2, double* x1, double y1, double* param) {
    if (g_mkl_context && g_mkl_context->drotmg) {
        g_mkl_context->drotmg(d1, d2, x1, y1, param);
    }
}

static void mkl_ssymv_wrapper(char uplo, int n, float alpha, const float* a, int lda,
                              const float* x, int incx, float beta, float* y, int incy) {
    if (!g_mkl_context || !g_mkl_context->ssymv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->ssymv(CblasColMajor, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void mkl_dsymv_wrapper(char uplo, int n, double alpha, const double* a, int lda,
                              const double* x, int incx, double beta, double* y, int incy) {
    if (!g_mkl_context || !g_mkl_context->dsymv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->dsymv(CblasColMajor, cblas_uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

static void mkl_sger_wrapper(int m, int n, float alpha, const float* x, int incx,
                             const float* y, int incy, float* a, int lda) {
    if (g_mkl_context && g_mkl_context->sger) {
        g_mkl_context->sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void mkl_dger_wrapper(int m, int n, double alpha, const double* x, int incx,
                             const double* y, int incy, double* a, int lda) {
    if (g_mkl_context && g_mkl_context->dger) {
        g_mkl_context->dger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void mkl_ssyr_wrapper(char uplo, int n, float alpha, const float* x, int incx,
                             float* a, int lda) {
    if (!g_mkl_context || !g_mkl_context->ssyr) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->ssyr(CblasColMajor, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void mkl_dsyr_wrapper(char uplo, int n, double alpha, const double* x, int incx,
                             double* a, int lda) {
    if (!g_mkl_context || !g_mkl_context->dsyr) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->dsyr(CblasColMajor, cblas_uplo, n, alpha, x, incx, a, lda);
}

static void mkl_ssyr2_wrapper(char uplo, int n, float alpha, const float* x, int incx,
                              const float* y, int incy, float* a, int lda) {
    if (!g_mkl_context || !g_mkl_context->ssyr2) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->ssyr2(CblasColMajor, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void mkl_dsyr2_wrapper(char uplo, int n, double alpha, const double* x, int incx,
                              const double* y, int incy, double* a, int lda) {
    if (!g_mkl_context || !g_mkl_context->dsyr2) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->dsyr2(CblasColMajor, cblas_uplo, n, alpha, x, incx, y, incy, a, lda);
}

static void mkl_strmv_wrapper(char uplo, char trans, char diag, int n,
                              const float* a, int lda, float* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->strmv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->strmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void mkl_dtrmv_wrapper(char uplo, char trans, char diag, int n,
                              const double* a, int lda, double* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->dtrmv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->dtrmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void mkl_strsv_wrapper(char uplo, char trans, char diag, int n,
                              const float* a, int lda, float* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->strsv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->strsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void mkl_dtrsv_wrapper(char uplo, char trans, char diag, int n,
                              const double* a, int lda, double* x, int incx) {
    if (!g_mkl_context || !g_mkl_context->dtrsv) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->dtrsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag, n, a, lda, x, incx);
}

static void mkl_strmm_wrapper(char side, char uplo, char trans, char diag, int m, int n,
                              float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_mkl_context || !g_mkl_context->strmm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->strmm(CblasColMajor, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void mkl_dtrmm_wrapper(char side, char uplo, char trans, char diag, int m, int n,
                              double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_mkl_context || !g_mkl_context->dtrmm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->dtrmm(CblasColMajor, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void mkl_strsm_wrapper(char side, char uplo, char trans, char diag, int m, int n,
                              float alpha, const float* a, int lda, float* b, int ldb) {
    if (!g_mkl_context || !g_mkl_context->strsm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->strsm(CblasColMajor, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void mkl_dtrsm_wrapper(char side, char uplo, char trans, char diag, int m, int n,
                              double alpha, const double* a, int lda, double* b, int ldb) {
    if (!g_mkl_context || !g_mkl_context->dtrsm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;
    g_mkl_context->dtrsm(CblasColMajor, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, a, lda, b, ldb);
}

static void mkl_ssymm_wrapper(char side, char uplo, int m, int n, float alpha,
                              const float* a, int lda, const float* b, int ldb,
                              float beta, float* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->ssymm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->ssymm(CblasColMajor, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void mkl_dsymm_wrapper(char side, char uplo, int m, int n, double alpha,
                              const double* a, int lda, const double* b, int ldb,
                              double beta, double* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->dsymm) return;
    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    g_mkl_context->dsymm(CblasColMajor, cblas_side, cblas_uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void mkl_ssyrk_wrapper(char uplo, char trans, int n, int k, float alpha,
                              const float* a, int lda, float beta, float* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->ssyrk) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->ssyrk(CblasColMajor, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void mkl_dsyrk_wrapper(char uplo, char trans, int n, int k, double alpha,
                              const double* a, int lda, double beta, double* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->dsyrk) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->dsyrk(CblasColMajor, cblas_uplo, cblas_trans, n, k, alpha, a, lda, beta, c, ldc);
}

static void mkl_ssyr2k_wrapper(char uplo, char trans, int n, int k, float alpha,
                               const float* a, int lda, const float* b, int ldb,
                               float beta, float* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->ssyr2k) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->ssyr2k(CblasColMajor, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void mkl_dsyr2k_wrapper(char uplo, char trans, int n, int k, double alpha,
                               const double* a, int lda, const double* b, int ldb,
                               double beta, double* c, int ldc) {
    if (!g_mkl_context || !g_mkl_context->dsyr2k) return;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'N' || trans == 'n') ? CblasNoTrans : CblasTrans;
    g_mkl_context->dsyr2k(CblasColMajor, cblas_uplo, cblas_trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

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

static fb_plugin_probe_result_t mkl_probe(fb_plugin_context_t* unused_ctx, const char** search_paths) {
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
    
    /* Load MKL-specific functions */
    ctx->set_num_threads = (mkl_set_num_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "MKL_Set_Num_Threads");
    ctx->get_max_threads = (mkl_get_max_threads_t)FB_GET_PROC_ADDRESS(lib_handle, "MKL_Get_Max_Threads");
    
    if (!ctx->sdot || !ctx->sgemm) {
        free(ctx);
        return -3;
    }
    
    /* Populate vtable */
    g_mkl_context = ctx;
    g_mkl_vtable.saxpy = mkl_saxpy_wrapper;
    g_mkl_vtable.sdot = mkl_sdot_wrapper;
    g_mkl_vtable.snrm2 = mkl_snrm2_wrapper;
    g_mkl_vtable.sasum = mkl_sasum_wrapper;
    g_mkl_vtable.isamax = mkl_isamax_wrapper;
    g_mkl_vtable.scopy = mkl_scopy_wrapper;
    g_mkl_vtable.sscal = mkl_sscal_wrapper;
    g_mkl_vtable.sswap = mkl_sswap_wrapper;
    g_mkl_vtable.sgemv = mkl_sgemv_wrapper;
    g_mkl_vtable.sgemm = mkl_sgemm_wrapper;
    
    /* Phase 1 BLAS operations */
    g_mkl_vtable.srot = mkl_srot_wrapper;
    g_mkl_vtable.drot = mkl_drot_wrapper;
    g_mkl_vtable.srotg = mkl_srotg_wrapper;
    g_mkl_vtable.drotg = mkl_drotg_wrapper;
    g_mkl_vtable.srotm = mkl_srotm_wrapper;
    g_mkl_vtable.drotm = mkl_drotm_wrapper;
    g_mkl_vtable.srotmg = mkl_srotmg_wrapper;
    g_mkl_vtable.drotmg = mkl_drotmg_wrapper;
    g_mkl_vtable.ssymv = mkl_ssymv_wrapper;
    g_mkl_vtable.dsymv = mkl_dsymv_wrapper;
    g_mkl_vtable.sger = mkl_sger_wrapper;
    g_mkl_vtable.dger = mkl_dger_wrapper;
    g_mkl_vtable.ssyr = mkl_ssyr_wrapper;
    g_mkl_vtable.dsyr = mkl_dsyr_wrapper;
    g_mkl_vtable.ssyr2 = mkl_ssyr2_wrapper;
    g_mkl_vtable.dsyr2 = mkl_dsyr2_wrapper;
    g_mkl_vtable.strmv = mkl_strmv_wrapper;
    g_mkl_vtable.dtrmv = mkl_dtrmv_wrapper;
    g_mkl_vtable.strsv = mkl_strsv_wrapper;
    g_mkl_vtable.dtrsv = mkl_dtrsv_wrapper;
    g_mkl_vtable.strmm = mkl_strmm_wrapper;
    g_mkl_vtable.dtrmm = mkl_dtrmm_wrapper;
    g_mkl_vtable.strsm = mkl_strsm_wrapper;
    g_mkl_vtable.dtrsm = mkl_dtrsm_wrapper;
    g_mkl_vtable.ssymm = mkl_ssymm_wrapper;
    g_mkl_vtable.dsymm = mkl_dsymm_wrapper;
    g_mkl_vtable.ssyrk = mkl_ssyrk_wrapper;
    g_mkl_vtable.dsyrk = mkl_dsyrk_wrapper;
    g_mkl_vtable.ssyr2k = mkl_ssyr2k_wrapper;
    g_mkl_vtable.dsyr2k = mkl_dsyr2k_wrapper;
    
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

static int mkl_set_num_threads(fb_plugin_context_t* ctx, int num_threads) {
    if (g_mkl_context && g_mkl_context->set_num_threads) {
        g_mkl_context->set_num_threads(num_threads);
        return 0;
    }
    return -1;
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
