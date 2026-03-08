/**
 * @file aocl_backend.c
 * @brief Intel aocl backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "aocl_backend.h"
#include "backend_auto_detect.h"     /* fb_auto_populate_ext_ops       */
#include "sym_tables/sym_tables.h"     /* k_lapacke_symbols              */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Define CBLAS enums only if not already defined */
#ifndef CBLAS_ORDER
typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;
#endif

#ifndef CBLAS_TRANSPOSE
typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;
#endif

#ifndef CBLAS_UPLO
typedef enum {
    CblasUpper = 121,
    CblasLower = 122
} CBLAS_UPLO;
#endif

#ifndef CBLAS_DIAG
typedef enum {
    CblasNonUnit = 131,
    CblasUnit = 132
} CBLAS_DIAG;
#endif

#ifndef CBLAS_SIDE
typedef enum {
    CblasLeft = 141,
    CblasRight = 142
} CBLAS_SIDE;
#endif

/* CBLAS_LAYOUT is the same as CBLAS_ORDER */
#ifndef CBLAS_LAYOUT
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif

#ifdef _WIN32
#include <windows.h>
#define FB_LOAD_LIBRARY(path) LoadLibraryA(path)
#define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#define FB_FREE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
typedef HMODULE fb_lib_handle_t;
#else
#include <dlfcn.h>
#define FB_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FB_FREE_LIBRARY(handle) dlclose(handle)
typedef void* fb_lib_handle_t;
#endif

/* aocl uses CBLAS interface - same signatures as OpenBLAS */
typedef float (*cblas_sasum_t)(const int n, const float* x, const int incx);
typedef double (*cblas_dasum_t)(const int n, const double* x, const int incx);
typedef void (*cblas_saxpy_t)(const int n, const float alpha, const float* x, const int incx, float* y, const int incy);
typedef void (*cblas_daxpy_t)(const int n, const double alpha, const double* x, const int incx, double* y, const int incy);
typedef float (*cblas_sdot_t)(const int n, const float* x, const int incx, const float* y, const int incy);
typedef double (*cblas_ddot_t)(const int n, const double* x, const int incx, const double* y, const int incy);
typedef void (*cblas_scopy_t)(const int n, const float* x, const int incx, float* y, const int incy);
typedef void (*cblas_dcopy_t)(const int n, const double* x, const int incx, double* y, const int incy);
typedef void (*cblas_sscal_t)(const int n, const float alpha, float* x, const int incx);
typedef void (*cblas_dscal_t)(const int n, const double alpha, double* x, const int incx);
typedef float (*cblas_snrm2_t)(const int n, const float* x, const int incx);
typedef double (*cblas_dnrm2_t)(const int n, const double* x, const int incx);
typedef void (*cblas_sswap_t)(const int n, float* x, const int incx, float* y, const int incy);
typedef void (*cblas_dswap_t)(const int n, double* x, const int incx, double* y, const int incy);
typedef int (*cblas_isamax_t)(const int n, const float* x, const int incx);
typedef int (*cblas_idamax_t)(const int n, const double* x, const int incx);

typedef void (*cblas_sgemv_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, const int M, const int N,
                               const float alpha, const float* A, const int lda,
                               const float* X, const int incX, const float beta,
                               float* Y, const int incY);
typedef void (*cblas_dgemv_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, const int M, const int N,
                               const double alpha, const double* A, const int lda,
                               const double* X, const int incX, const double beta,
                               double* Y, const int incY);
typedef void (*cblas_sgemm_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const float alpha, const float* A, const int lda,
                               const float* B, const int ldb, const float beta,
                               float* C, const int ldc);
typedef void (*cblas_dgemm_t)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const double alpha, const double* A, const int lda,
                               const double* B, const int ldb, const double beta,
                               double* C, const int ldc);

/* Level 2 BLAS typedefs */
typedef void (*cblas_sger_t)(CBLAS_LAYOUT, const int, const int, const float, const float*, const int, const float*, const int, float*, const int);
typedef void (*cblas_dger_t)(CBLAS_LAYOUT, const int, const int, const double, const double*, const int, const double*, const int, double*, const int);
typedef void (*cblas_ssymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_strmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);
typedef void (*cblas_strsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const int, float*, const int);
typedef void (*cblas_dtrsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const int, double*, const int);

/* Level 3 BLAS typedefs */
typedef void (*cblas_ssymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_dsymm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_strmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrmm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);
typedef void (*cblas_strsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float, const float*, const int, float*, const int);
typedef void (*cblas_dtrsm_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double, const double*, const int, double*, const int);

/* PHASE2: Level 2 BLAS typedefs */
typedef void (*cblas_dgbmv_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, const int, const int, const int, const int, const double, const double*, const int, const double*, const int, const double, const double*, const int);
typedef void (*cblas_dsbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const int, const double, const double*, const int, const double*, const int, const double, const double*, const int);
typedef void (*cblas_dspmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, double*, const double*, const int, const double, const double*, const int);
typedef void (*cblas_dspr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, double*);
typedef void (*cblas_dsyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int);
typedef void (*cblas_dsyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, const double*, const int);
typedef void (*cblas_dtbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double*, const int, const double*, const int);
typedef void (*cblas_dtbsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const double*, const int, const double*, const int);
typedef void (*cblas_sgbmv_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, const int, const int, const int, const int, const float, const float*, const int, const float*, const int, const float, const float*, const int);
typedef void (*cblas_ssbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const int, const float, const float*, const int, const float*, const int, const float, const float*, const int);
typedef void (*cblas_sspmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, float*, const float*, const int, const float, const float*, const int);
typedef void (*cblas_sspr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, float*);
typedef void (*cblas_ssyr_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int);
typedef void (*cblas_ssyr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, const float*, const int);
typedef void (*cblas_stbmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float*, const int, const float*, const int);
typedef void (*cblas_stbsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const int, const float*, const int, const float*, const int);
typedef void (*cblas_stpmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const float*, const int);
typedef void (*cblas_dtpmv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const double*, const int);
typedef void (*cblas_stpsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const float*, const float*, const int);
typedef void (*cblas_dtpsv_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, const int, const double*, const double*, const int);
typedef void (*cblas_sspr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const float, const float*, const int, const float*, const int, float*);
typedef void (*cblas_dspr2_t)(CBLAS_LAYOUT, CBLAS_UPLO, const int, const double, const double*, const int, const double*, const int, double*);

/* PHASE2: Level 3 BLAS typedefs */
typedef void (*cblas_dsyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int);
typedef void (*cblas_dsyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const double, const double*, const int, const double, double*, const int);
typedef void (*cblas_ssyr2k_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float*, const int, const float, float*, const int);
typedef void (*cblas_ssyrk_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, const int, const int, const float, const float*, const int, const float, float*, const int);

/* BLIS threading API (correct function names per AOCL documentation) */
typedef void (*bli_thread_set_num_threads_t)(int num_threads);
typedef int (*bli_thread_get_num_threads_t)(void);

/* aocl-specific functions */
typedef void (*aocl_get_version_t)(int* major, int* minor, int* update);
typedef int (*aocl_set_threading_layer_t)(int layer);
typedef void (*aocl_verbose_t)(int enable);

/* Global aocl API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    
    /* BLIS threading control */
    bli_thread_set_num_threads_t set_num_threads;
    bli_thread_get_num_threads_t get_num_threads;
    
    /* aocl control functions */
    aocl_get_version_t get_version;
    aocl_set_threading_layer_t set_threading_layer;
    aocl_verbose_t verbose;
    
    /* CBLAS Level 1 */
    cblas_sasum_t sasum;
    cblas_dasum_t dasum;
    cblas_saxpy_t saxpy;
    cblas_daxpy_t daxpy;
    cblas_sdot_t sdot;
    cblas_ddot_t ddot;
    cblas_scopy_t scopy;
    cblas_dcopy_t dcopy;
    cblas_sscal_t sscal;
    cblas_dscal_t dscal;
    cblas_snrm2_t snrm2;
    cblas_dnrm2_t dnrm2;
    cblas_sswap_t sswap;
    cblas_dswap_t dswap;
    cblas_isamax_t isamax;
    cblas_idamax_t idamax;
    
    /* CBLAS Level 2 */
    cblas_sgemv_t sgemv;
    cblas_dgemv_t dgemv;
    cblas_sger_t sger;
    cblas_dger_t dger;
    cblas_ssymv_t ssymv;
    cblas_dsymv_t dsymv;
    cblas_strmv_t strmv;
    cblas_dtrmv_t dtrmv;
    cblas_strsv_t strsv;
    cblas_dtrsv_t dtrsv;
    
    /* CBLAS Level 3 */
    cblas_sgemm_t sgemm;
    cblas_dgemm_t dgemm;
    cblas_ssymm_t ssymm;
    cblas_dsymm_t dsymm;
    cblas_strmm_t strmm;
    cblas_dtrmm_t dtrmm;
    cblas_strsm_t strsm;
    cblas_dtrsm_t dtrsm;
    
    /* PHASE2: Level 2 BLAS */
    cblas_dgbmv_t dgbmv;
    cblas_dsbmv_t dsbmv;
    cblas_dspmv_t dspmv;
    cblas_dspr_t dspr;
    cblas_dsyr_t dsyr;
    cblas_dsyr2_t dsyr2;
    cblas_dtbmv_t dtbmv;
    cblas_dtbsv_t dtbsv;
    cblas_sgbmv_t sgbmv;
    cblas_ssbmv_t ssbmv;
    cblas_sspmv_t sspmv;
    cblas_sspr_t sspr;
    cblas_ssyr_t ssyr;
    cblas_ssyr2_t ssyr2;
    cblas_stbmv_t stbmv;
    cblas_stbsv_t stbsv;
    cblas_stpmv_t stpmv;
    cblas_dtpmv_t dtpmv;
    cblas_stpsv_t stpsv;
    cblas_dtpsv_t dtpsv;
    cblas_sspr2_t sspr2;
    cblas_dspr2_t dspr2;

    /* PHASE2: Level 3 BLAS */
    cblas_dsyr2k_t dsyr2k;
    cblas_dsyrk_t dsyrk;
    cblas_ssyr2k_t ssyr2k;
    cblas_ssyrk_t ssyrk;
} g_aocl;

/* aocl threading layer constants */
enum aocl_THREADING_LAYER {
    aocl_THREADING_INTEL = 0,
    aocl_THREADING_SEQUENTIAL = 1,
    aocl_THREADING_TBB = 2,
    aocl_THREADING_GNU = 3
};

static int load_aocl_library(void) {
    if (g_aocl.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        /* Try multi-threaded versions first */
        "AOCL-LibBlis-Win-MT-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-MT-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64\\AOCL-LibBlis-Win-MT-dll.dll",
        /* Fall back to single-threaded versions */
        "AOCL-LibBlis-Win-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-dll.dll",
        "aocl_rt.2.dll",
        "aocl_rt.dll"
#elif defined(__APPLE__)
        "libaocl_rt.dylib",
        "libaocl_rt.2.dylib",
        "/opt/intel/oneapi/aocl/latest/lib/libaocl_rt.dylib"
#else
        "libaocl_rt.so",
        "libaocl_rt.so.2",
        "/opt/intel/oneapi/aocl/latest/lib/intel64/libaocl_rt.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_aocl.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_aocl.handle) {
            printf("[INFO] AOCL: Successfully loaded: %s\n", lib_names[i]);
            break;
        }
    }
    
    if (!g_aocl.handle) {
        return -1;
    }
    
    /* Load BLIS threading API (per AOCL BLIS documentation) */
    g_aocl.set_num_threads = (bli_thread_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_aocl.handle, "bli_thread_set_num_threads");
    g_aocl.get_num_threads = (bli_thread_get_num_threads_t)
        FB_GET_PROC_ADDRESS(g_aocl.handle, "bli_thread_get_num_threads");
    
    if (!g_aocl.set_num_threads) {
        fprintf(stderr, "[INFO] AOCL: bli_thread_set_num_threads not found, trying bli_set_num_threads\n");
        g_aocl.set_num_threads = (bli_thread_set_num_threads_t)
            FB_GET_PROC_ADDRESS(g_aocl.handle, "bli_set_num_threads");
    }
    if (!g_aocl.get_num_threads) {
        fprintf(stderr, "[INFO] AOCL: bli_thread_get_num_threads not found, trying bli_get_num_threads\n");
        g_aocl.get_num_threads = (bli_thread_get_num_threads_t)
            FB_GET_PROC_ADDRESS(g_aocl.handle, "bli_get_num_threads");
    }
    g_aocl.get_version = (aocl_get_version_t)
        FB_GET_PROC_ADDRESS(g_aocl.handle, "aocl_get_version");
    g_aocl.set_threading_layer = (aocl_set_threading_layer_t)
        FB_GET_PROC_ADDRESS(g_aocl.handle, "aocl_set_threading_layer");
    g_aocl.verbose = (aocl_verbose_t)
        FB_GET_PROC_ADDRESS(g_aocl.handle, "aocl_verbose");
    
    /* Load CBLAS functions (same names as OpenBLAS) */
    g_aocl.sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sasum");
    g_aocl.dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dasum");
    g_aocl.saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_saxpy");
    g_aocl.daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_daxpy");
    g_aocl.sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sdot");
    g_aocl.ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ddot");
    g_aocl.scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_scopy");
    g_aocl.dcopy = (cblas_dcopy_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dcopy");
    g_aocl.sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sscal");
    g_aocl.dscal = (cblas_dscal_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dscal");
    g_aocl.snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_snrm2");
    g_aocl.dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dnrm2");
    g_aocl.sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sswap");
    g_aocl.dswap = (cblas_dswap_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dswap");
    g_aocl.isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_isamax");
    g_aocl.idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_idamax");
    
    g_aocl.sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sgemv");
    g_aocl.dgemv = (cblas_dgemv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dgemv");
    g_aocl.sger = (cblas_sger_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sger");
    g_aocl.dger = (cblas_dger_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dger");
    g_aocl.ssymv = (cblas_ssymv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssymv");
    g_aocl.dsymv = (cblas_dsymv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsymv");
    g_aocl.strmv = (cblas_strmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_strmv");
    g_aocl.dtrmv = (cblas_dtrmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtrmv");
    g_aocl.strsv = (cblas_strsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_strsv");
    g_aocl.dtrsv = (cblas_dtrsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtrsv");
    
    g_aocl.sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sgemm");
    g_aocl.dgemm = (cblas_dgemm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dgemm");
    g_aocl.ssymm = (cblas_ssymm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssymm");
    g_aocl.dsymm = (cblas_dsymm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsymm");
    g_aocl.strmm = (cblas_strmm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_strmm");
    g_aocl.dtrmm = (cblas_dtrmm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtrmm");
    g_aocl.strsm = (cblas_strsm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_strsm");
    g_aocl.dtrsm = (cblas_dtrsm_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtrsm");
    
    /* PHASE2: Level 2 BLAS */
    g_aocl.dgbmv = (cblas_dgbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dgbmv");
    g_aocl.dsbmv = (cblas_dsbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsbmv");
    g_aocl.dspmv = (cblas_dspmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dspmv");
    g_aocl.dspr = (cblas_dspr_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dspr");
    g_aocl.dsyr = (cblas_dsyr_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsyr");
    g_aocl.dsyr2 = (cblas_dsyr2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsyr2");
    g_aocl.dtbmv = (cblas_dtbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtbmv");
    g_aocl.dtbsv = (cblas_dtbsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtbsv");
    g_aocl.sgbmv = (cblas_sgbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sgbmv");
    g_aocl.ssbmv = (cblas_ssbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssbmv");
    g_aocl.sspmv = (cblas_sspmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sspmv");
    g_aocl.sspr = (cblas_sspr_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sspr");
    g_aocl.ssyr = (cblas_ssyr_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssyr");
    g_aocl.ssyr2 = (cblas_ssyr2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssyr2");
    g_aocl.stbmv = (cblas_stbmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_stbmv");
    g_aocl.stbsv = (cblas_stbsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_stbsv");
    g_aocl.stpmv = (cblas_stpmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_stpmv");
    g_aocl.dtpmv = (cblas_dtpmv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtpmv");
    g_aocl.stpsv = (cblas_stpsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_stpsv");
    g_aocl.dtpsv = (cblas_dtpsv_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dtpsv");
    g_aocl.sspr2 = (cblas_sspr2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_sspr2");
    g_aocl.dspr2 = (cblas_dspr2_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dspr2");

    /* PHASE2: Level 3 BLAS */
    g_aocl.dsyr2k = (cblas_dsyr2k_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsyr2k");
    g_aocl.dsyrk = (cblas_dsyrk_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_dsyrk");
    g_aocl.ssyr2k = (cblas_ssyr2k_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssyr2k");
    g_aocl.ssyrk = (cblas_ssyrk_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_ssyrk");

    /* Debug: Log PHASE2 operation loading status */
    printf("[DEBUG] PHASE2 Operations Loading Status:\n");
    printf("  ssyr: %s\n", g_aocl.ssyr ? "OK" : "FAILED");
    printf("  dsyr: %s\n", g_aocl.dsyr ? "OK" : "FAILED");
    printf("  ssyrk: %s\n", g_aocl.ssyrk ? "OK" : "FAILED");
    printf("  dsyrk: %s\n", g_aocl.dsyrk ? "OK" : "FAILED");
    printf("  sgbmv: %s\n", g_aocl.sgbmv ? "OK" : "FAILED");
    printf("  dgbmv: %s\n", g_aocl.dgbmv ? "OK" : "FAILED");
    printf("  sspmv: %s\n", g_aocl.sspmv ? "OK" : "FAILED");
    printf("  dspmv: %s\n", g_aocl.dspmv ? "OK" : "FAILED");
    printf("  sspr: %s\n", g_aocl.sspr ? "OK" : "FAILED");
    printf("  dspr: %s\n", g_aocl.dspr ? "OK" : "FAILED");

    g_aocl.initialized = true;
    return 0;
}

bool fb_aocl_is_available(void) {
    return load_aocl_library() == 0;
}

int fb_aocl_init(void) {
    return load_aocl_library();
}

void fb_aocl_shutdown(void) {
    if (g_aocl.initialized && g_aocl.handle) {
        FB_FREE_LIBRARY(g_aocl.handle);
        g_aocl.handle = NULL;
        g_aocl.initialized = false;
    }
}

int fb_aocl_get_version(int* major, int* minor, int* update) {
    if (!g_aocl.initialized || !g_aocl.get_version) {
        return -1;
    }
    g_aocl.get_version(major, minor, update);
    return 0;
}

void fb_aocl_set_num_threads(int num_threads) {
    if (g_aocl.set_num_threads) {
        g_aocl.set_num_threads(num_threads);
    }
}

int fb_aocl_get_num_threads(void) {
    if (g_aocl.get_num_threads) {
        return g_aocl.get_num_threads();
    }
    return 1;
}

int fb_aocl_set_threading_layer(const char* layer) {
    if (!g_aocl.set_threading_layer) {
        return -1;
    }
    
    int aocl_layer = aocl_THREADING_INTEL;
    if (strcmp(layer, "sequential") == 0) {
        aocl_layer = aocl_THREADING_SEQUENTIAL;
    } else if (strcmp(layer, "tbb") == 0) {
        aocl_layer = aocl_THREADING_TBB;
    } else if (strcmp(layer, "gnu") == 0) {
        aocl_layer = aocl_THREADING_GNU;
    }
    
    return g_aocl.set_threading_layer(aocl_layer);
}

void fb_aocl_set_verbose(bool enable) {
    if (g_aocl.verbose) {
        g_aocl.verbose(enable ? 1 : 0);
    }
}

/* Backend property functions for unified interface */

static uint32_t aocl_get_capabilities(void* handle) {
    (void)handle;
    return FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

static int aocl_get_num_threads_vtable(void* handle) {
    (void)handle;
    return fb_aocl_get_num_threads();
}

static void aocl_set_num_threads_vtable(void* handle, int num_threads) {
    (void)handle;
    fb_aocl_set_num_threads(num_threads);
}

/* Wrapper functions (nearly identical to OpenBLAS wrappers) */

static void aocl_sasum(int n, const float* x, int incx, float* result) {
    if (g_aocl.sasum) {
        *result = g_aocl.sasum(n, x, incx);
    }
}

static void aocl_dasum(int n, const double* x, int incx, double* result) {
    if (g_aocl.dasum) {
        *result = g_aocl.dasum(n, x, incx);
    }
}

static void aocl_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_aocl.saxpy) {
        g_aocl.saxpy(n, alpha, x, incx, y, incy);
    }
}

static void aocl_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_aocl.daxpy) {
        g_aocl.daxpy(n, alpha, x, incx, y, incy);
    }
}

static void aocl_sdot(int n, const float* x, int incx, const float* y, int incy, float* result) {
    if (g_aocl.sdot) {
        *result = g_aocl.sdot(n, x, incx, y, incy);
    }
}

static void aocl_ddot(int n, const double* x, int incx, const double* y, int incy, double* result) {
    if (g_aocl.ddot) {
        *result = g_aocl.ddot(n, x, incx, y, incy);
    }
}

static void aocl_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_aocl.scopy) {
        g_aocl.scopy(n, x, incx, y, incy);
    }
}

static void aocl_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_aocl.dcopy) {
        g_aocl.dcopy(n, x, incx, y, incy);
    }
}

static void aocl_sscal(int n, float alpha, float* x, int incx) {
    if (g_aocl.sscal) {
        g_aocl.sscal(n, alpha, x, incx);
    }
}

static void aocl_dscal(int n, double alpha, double* x, int incx) {
    if (g_aocl.dscal) {
        g_aocl.dscal(n, alpha, x, incx);
    }
}

static void aocl_snrm2(int n, const float* x, int incx, float* result) {
    if (g_aocl.snrm2) {
        *result = g_aocl.snrm2(n, x, incx);
    }
}

static void aocl_dnrm2(int n, const double* x, int incx, double* result) {
    if (g_aocl.dnrm2) {
        *result = g_aocl.dnrm2(n, x, incx);
    }
}

static void aocl_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_aocl.sswap) {
        g_aocl.sswap(n, x, incx, y, incy);
    }
}

static void aocl_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_aocl.dswap) {
        g_aocl.dswap(n, x, incx, y, incy);
    }
}

static void aocl_isamax(int n, const float* x, int incx, int* result) {
    if (g_aocl.isamax) {
        *result = g_aocl.isamax(n, x, incx);
    }
}

static void aocl_idamax(int n, const double* x, int incx, int* result) {
    if (g_aocl.idamax) {
        *result = g_aocl.idamax(n, x, incx);
    }
}

/* ============================================================================
 * Vtable Adapter Wrappers
 * 
 * These wrappers adapt from the vtable interface (which returns values)
 * to the AOCL backend implementation (which uses output pointers).
 * ========================================================================= */

static float aocl_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    float result = 0.0f;
    aocl_sdot(n, x, incx, y, incy, &result);
    return result;
}

static double aocl_ddot_wrapper(int n, const double* x, int incx, const double* y, int incy) {
    double result = 0.0;
    aocl_ddot(n, x, incx, y, incy, &result);
    return result;
}

static float aocl_snrm2_wrapper(int n, const float* x, int incx) {
    float result = 0.0f;
    aocl_snrm2(n, x, incx, &result);
    return result;
}

static double aocl_dnrm2_wrapper(int n, const double* x, int incx) {
    double result = 0.0;
    aocl_dnrm2(n, x, incx, &result);
    return result;
}

static float aocl_sasum_wrapper(int n, const float* x, int incx) {
    float result = 0.0f;
    aocl_sasum(n, x, incx, &result);
    return result;
}

static double aocl_dasum_wrapper(int n, const double* x, int incx) {
    double result = 0.0;
    aocl_dasum(n, x, incx, &result);
    return result;
}

static int aocl_isamax_wrapper(int n, const float* x, int incx) {
    int result = 0;
    aocl_isamax(n, x, incx, &result);
    return result;
}

static int aocl_idamax_wrapper(int n, const double* x, int incx) {
    int result = 0;
    aocl_idamax(n, x, incx, &result);
    return result;
}

static int transpose_to_cblas(char trans) {
    switch (trans) {
        case 'N': case 'n': return CblasNoTrans;
        case 'T': case 't': return CblasTrans;
        case 'C': case 'c': return CblasConjTrans;
        default: return CblasNoTrans;
    }
}

static int uplo_to_cblas(char uplo) {
    switch (uplo) {
        case 'U': case 'u': return CblasUpper;
        case 'L': case 'l': return CblasLower;
        default: return CblasUpper;
    }
}

static int diag_to_cblas(char diag) {
    switch (diag) {
        case 'N': case 'n': return CblasNonUnit;
        case 'U': case 'u': return CblasUnit;
        default: return CblasNonUnit;
    }
}

static int side_to_cblas(char side) {
    switch (side) {
        case 'L': case 'l': return CblasLeft;
        case 'R': case 'r': return CblasRight;
        default: return CblasLeft;
    }
}

#define FB_LOAD_SYMBOL FB_GET_PROC_ADDRESS

/* Complex type Level 1 BLAS */

static void aocl_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void aocl_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void aocl_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void aocl_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void aocl_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void aocl_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void aocl_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void aocl_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void aocl_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void aocl_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void aocl_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void aocl_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void aocl_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void aocl_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void aocl_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void aocl_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void aocl_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void aocl_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void aocl_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void aocl_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void aocl_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void aocl_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void aocl_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void aocl_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void aocl_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void aocl_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void aocl_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void aocl_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Level 2 BLAS operations */

static void aocl_sgemv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const float alpha,
                      const float* a, const int lda, const float* x, const int incx,
                      const float beta, float* y, const int incy) {
    if (g_aocl.sgemv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.sgemv(cblas_layout, cblas_trans,
                    (int)m, (int)n, alpha, a, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

static void aocl_dgemv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const double alpha,
                      const double* a, const int lda, const double* x, const int incx,
                      const double beta, double* y, const int incy) {
    if (g_aocl.dgemv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.dgemv(cblas_layout, cblas_trans,
                    (int)m, (int)n, alpha, a, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

static void aocl_sgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, const int m, const int n, const int k,
                      const float alpha, const float* a, const int lda,
                      const float* b, const int ldb, const float beta,
                      float* c, const int ldc) {
    if (g_aocl.sgemm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_transa = (transa == FB_NO_TRANS) ? CblasNoTrans : (transa == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_TRANSPOSE cblas_transb = (transb == FB_NO_TRANS) ? CblasNoTrans : (transb == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.sgemm(cblas_layout, cblas_transa, cblas_transb,
                    (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
    }
}

static void aocl_dgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, const int m, const int n, const int k,
                      const double alpha, const double* a, const int lda,
                      const double* b, const int ldb, const double beta,
                      double* c, const int ldc) {
    if (g_aocl.dgemm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_transa = (transa == FB_NO_TRANS) ? CblasNoTrans : (transa == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_TRANSPOSE cblas_transb = (transb == FB_NO_TRANS) ? CblasNoTrans : (transb == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.dgemm(cblas_layout, cblas_transa, cblas_transb,
                    (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - NEW
 * ========================================================================== */

/* General rank-1 update: A = alpha*x*y' + A */
static void aocl_sger(const fb_layout_t layout, const int m, const int n,
                      const float alpha, const float* x, const int incx,
                      const float* y, const int incy, float* A, const int lda) {
    if (g_aocl.sger) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        g_aocl.sger(cblas_layout, (int)m, (int)n, alpha, x, (int)incx, y, (int)incy, A, (int)lda);
    }
}

static void aocl_dger(const fb_layout_t layout, const int m, const int n,
                      const double alpha, const double* x, const int incx,
                      const double* y, const int incy, double* A, const int lda) {
    if (g_aocl.dger) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        g_aocl.dger(cblas_layout, (int)m, (int)n, alpha, x, (int)incx, y, (int)incy, A, (int)lda);
    }
}

/* Symmetric matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_ssymv(const fb_layout_t layout, const fb_uplo_t uplo, const int n,
                       const float alpha, const float* A, const int lda,
                       const float* x, const int incx, const float beta,
                       float* y, const int incy) {
    if (!g_aocl.ssymv) {
        fprintf(stderr, "aocl_ssymv: Function pointer is NULL\n");
        return;
    }
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl.ssymv(cblas_layout, cblas_uplo, (int)n, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

static void aocl_dsymv(const fb_layout_t layout, const fb_uplo_t uplo, const int n,
                       const double alpha, const double* A, const int lda,
                       const double* x, const int incx, const double beta,
                       double* y, const int incy) {
    if (!g_aocl.dsymv) {
        fprintf(stderr, "aocl_dsymv: Function pointer is NULL\n");
        return;
    }
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    g_aocl.dsymv(cblas_layout, cblas_uplo, (int)n, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

/* Triangular matrix-vector multiply: x = A*x */
static void aocl_strmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans,
                       const fb_diag_t diag, const int n, const float* A, const int lda,
                       float* x, const int incx) {
    if (g_aocl.strmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.strmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, (int)lda, x, (int)incx);
    }
}

static void aocl_dtrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans,
                       const fb_diag_t diag, const int n, const double* A, const int lda,
                       double* x, const int incx) {
    if (g_aocl.dtrmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtrmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, (int)lda, x, (int)incx);
    }
}

/* Triangular system solve: A*x = b */
static void aocl_strsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans,
                       const fb_diag_t diag, const int n, const float* A, const int lda,
                       float* x, const int incx) {
    if (g_aocl.strsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.strsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, (int)lda, x, (int)incx);
    }
}

static void aocl_dtrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans,
                       const fb_diag_t diag, const int n, const double* A, const int lda,
                       double* x, const int incx) {
    if (g_aocl.dtrsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtrsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, (int)lda, x, (int)incx);
    }
}

/* ============================================================================
 * BLAS Level 3 Operations - NEW
 * ========================================================================== */

/* Symmetric matrix-matrix multiply: C = alpha*A*B + beta*C */
static void aocl_ssymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const int m, const int n, const float alpha,
                       const float* A, const int lda, const float* B, const int ldb,
                       const float beta, float* C, const int ldc) {
    if (g_aocl.ssymm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.ssymm(cblas_layout, cblas_side, cblas_uplo, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    }
}

static void aocl_dsymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const int m, const int n, const double alpha,
                       const double* A, const int lda, const double* B, const int ldb,
                       const double beta, double* C, const int ldc) {
    if (g_aocl.dsymm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dsymm(cblas_layout, cblas_side, cblas_uplo, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    }
}

/* Triangular matrix-matrix multiply: B = alpha*A*B */
static void aocl_strmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int m, const int n,
                       const float alpha, const float* A, const int lda,
                       float* B, const int ldb) {
    if (g_aocl.strmm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.strmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb);
    }
}

static void aocl_dtrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int m, const int n,
                       const double alpha, const double* A, const int lda,
                       double* B, const int ldb) {
    if (g_aocl.dtrmm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtrmm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb);
    }
}

/* Triangular system solve with multiple RHS: A*X = alpha*B */
static void aocl_strsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int m, const int n,
                       const float alpha, const float* A, const int lda,
                       float* B, const int ldb) {
    if (g_aocl.strsm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.strsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb);
    }
}

static void aocl_dtrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int m, const int n,
                       const double alpha, const double* A, const int lda,
                       double* B, const int ldb) {
    if (g_aocl.dtrsm) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtrsm(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, (int)m, (int)n, alpha, A, (int)lda, B, (int)ldb);
    }
}

/* ============================================================================
 * PHASE2: New BLAS Operations with Correct Signatures
 * Note: Old wrappers below use incorrect signatures and should be deprecated
 * ========================================================================== */

/* General band matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_dgbmv_new(const fb_layout_t layout,
                       const fb_transpose_t trans,
                       const int m,
                       const int n,
                       const int kl,
                       const int ku,
                       const double alpha,
                       const double* A,
                       const int lda,
                       const double* x,
                       const int incx,
                       const double beta,
                       double* y,
                       const int incy) {
    if (g_aocl.dgbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.dgbmv(cblas_layout, cblas_trans, (int)m, (int)n, (int)kl, (int)ku, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric band matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_dsbmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const int k,
                       const double alpha,
                       const double* A,
                       const int lda,
                       const double* x,
                       const int incx,
                       const double beta,
                       double* y,
                       const int incy) {
    if (g_aocl.dsbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dsbmv(cblas_layout, cblas_uplo, (int)n, (int)k, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric packed matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_dspmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const double alpha,
                       const double* Ap,
                       const double* x,
                       const int incx,
                       const double beta,
                       double* y,
                       const int incy) {
    if (g_aocl.dspmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dspmv(cblas_layout, cblas_uplo, (int)n, alpha, (double*)Ap, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric packed rank-1 update: A = alpha*x*x' + A */
static void aocl_dspr_new(const fb_layout_t layout,
                      const fb_uplo_t uplo,
                      const int n,
                      const double alpha,
                      const double* x,
                      const int incx,
                      double* Ap) {
    if (g_aocl.dspr) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dspr(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, Ap);
    }
}

/* Symmetric rank-1 update: A = alpha*x*x' + A */
static void aocl_dsyr_new(const fb_layout_t layout,
                      const fb_uplo_t uplo,
                      const int n,
                      const double alpha,
                      const double* x,
                      const int incx,
                      double* A,
                      const int lda) {
    if (g_aocl.dsyr) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dsyr(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, A, (int)lda);
    }
}

/* Symmetric rank-2 update: A = alpha*x*y' + alpha*y*x' + A */
static void aocl_dsyr2_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const double alpha,
                       const double* x,
                       const int incx,
                       const double* y,
                       const int incy,
                       double* A,
                       const int lda) {
    if (g_aocl.dsyr2) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dsyr2(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, y, (int)incy, A, (int)lda);
    }
}

/* Triangular band matrix-vector multiply: x = A*x */
static void aocl_dtbmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const int k,
                       const double* A,
                       const int lda,
                       double* x,
                       const int incx) {
    if (g_aocl.dtbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtbmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, (int)k, A, (int)lda, x, (int)incx);
    }
}

/* Triangular band system solve: A*x = b */
static void aocl_dtbsv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const int k,
                       const double* A,
                       const int lda,
                       double* x,
                       const int incx) {
    if (g_aocl.dtbsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtbsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, (int)k, A, (int)lda, x, (int)incx);
    }
}

/* General band matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_sgbmv_new(const fb_layout_t layout,
                       const fb_transpose_t trans,
                       const int m,
                       const int n,
                       const int kl,
                       const int ku,
                       const float alpha,
                       const float* A,
                       const int lda,
                       const float* x,
                       const int incx,
                       const float beta,
                       float* y,
                       const int incy) {
    if (g_aocl.sgbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.sgbmv(cblas_layout, cblas_trans, (int)m, (int)n, (int)kl, (int)ku, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric band matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_ssbmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const int k,
                       const float alpha,
                       const float* A,
                       const int lda,
                       const float* x,
                       const int incx,
                       const float beta,
                       float* y,
                       const int incy) {
    if (g_aocl.ssbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.ssbmv(cblas_layout, cblas_uplo, (int)n, (int)k, alpha, A, (int)lda, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric packed matrix-vector multiply: y = alpha*A*x + beta*y */
static void aocl_sspmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const float alpha,
                       const float* Ap,
                       const float* x,
                       const int incx,
                       const float beta,
                       float* y,
                       const int incy) {
    if (g_aocl.sspmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.sspmv(cblas_layout, cblas_uplo, (int)n, alpha, (float*)Ap, x, (int)incx, beta, y, (int)incy);
    }
}

/* Symmetric packed rank-1 update: A = alpha*x*x' + A */
static void aocl_sspr_new(const fb_layout_t layout,
                      const fb_uplo_t uplo,
                      const int n,
                      const float alpha,
                      const float* x,
                      const int incx,
                      float* Ap) {
    if (g_aocl.sspr) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.sspr(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, Ap);
    }
}

/* Symmetric rank-1 update: A = alpha*x*x' + A */
static void aocl_ssyr_new(const fb_layout_t layout,
                      const fb_uplo_t uplo,
                      const int n,
                      const float alpha,
                      const float* x,
                      const int incx,
                      float* A,
                      const int lda) {
    if (g_aocl.ssyr) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.ssyr(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, A, (int)lda);
    }
}

/* Symmetric rank-2 update: A = alpha*x*y' + alpha*y*x' + A */
static void aocl_ssyr2_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const float alpha,
                       const float* x,
                       const int incx,
                       const float* y,
                       const int incy,
                       float* A,
                       const int lda) {
    if (g_aocl.ssyr2) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.ssyr2(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, y, (int)incy, A, (int)lda);
    }
}

/* Triangular band matrix-vector multiply: x = A*x */
static void aocl_stbmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const int k,
                       const float* A,
                       const int lda,
                       float* x,
                       const int incx) {
    if (g_aocl.stbmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.stbmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, (int)k, A, (int)lda, x, (int)incx);
    }
}

/* Triangular band system solve: A*x = b */
static void aocl_stbsv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const int k,
                       const float* A,
                       const int lda,
                       float* x,
                       const int incx) {
    if (g_aocl.stbsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.stbsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, (int)k, A, (int)lda, x, (int)incx);
    }
}

/* Triangular packed matrix-vector multiply: x = A*x */
static void aocl_stpmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const float* A,
                       float* x,
                       const int incx) {
    if (g_aocl.stpmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.stpmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, x, (int)incx);
    }
}

static void aocl_dtpmv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const double* A,
                       double* x,
                       const int incx) {
    if (g_aocl.dtpmv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtpmv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, x, (int)incx);
    }
}

/* Triangular packed solve: A*x = b, overwrite x with solution */
static void aocl_stpsv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const float* A,
                       float* x,
                       const int incx) {
    if (g_aocl.stpsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.stpsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, x, (int)incx);
    }
}

static void aocl_dtpsv_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const fb_diag_t diag,
                       const int n,
                       const double* A,
                       double* x,
                       const int incx) {
    if (g_aocl.dtpsv) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
        g_aocl.dtpsv(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, (int)n, A, x, (int)incx);
    }
}

/* Symmetric rank-2 update: A = alpha*x*y' + alpha*y*x' + A */
static void aocl_sspr2_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const float alpha,
                       const float* x,
                       const int incx,
                       const float* y,
                       const int incy,
                       float* Ap) {
    if (g_aocl.sspr2) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.sspr2(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, y, (int)incy, Ap);
    }
}

static void aocl_dspr2_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const int n,
                       const double alpha,
                       const double* x,
                       const int incx,
                       const double* y,
                       const int incy,
                       double* Ap) {
    if (g_aocl.dspr2) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        g_aocl.dspr2(cblas_layout, cblas_uplo, (int)n, alpha, x, (int)incx, y, (int)incy, Ap);
    }
}

/* Symmetric rank-2k update: C = alpha*A*B' + alpha*B*A' + beta*C */
static void aocl_dsyr2k_new(const fb_layout_t layout,
                        const fb_uplo_t uplo,
                        const fb_transpose_t trans,
                        const int n,
                        const int k,
                        const double alpha,
                        const double* A,
                        const int lda,
                        const double* B,
                        const int ldb,
                        const double beta,
                        double* C,
                        const int ldc) {
    if (g_aocl.dsyr2k) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.dsyr2k(cblas_layout, cblas_uplo, cblas_trans, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    }
}

/* Symmetric rank-k update: C = alpha*A*A' + beta*C */
static void aocl_dsyrk_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const int n,
                       const int k,
                       const double alpha,
                       const double* A,
                       const int lda,
                       const double beta,
                       double* C,
                       const int ldc) {
    if (g_aocl.dsyrk) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.dsyrk(cblas_layout, cblas_uplo, cblas_trans, (int)n, (int)k, alpha, A, (int)lda, beta, C, (int)ldc);
    }
}

/* Symmetric rank-2k update: C = alpha*A*B' + alpha*B*A' + beta*C */
static void aocl_ssyr2k_new(const fb_layout_t layout,
                        const fb_uplo_t uplo,
                        const fb_transpose_t trans,
                        const int n,
                        const int k,
                        const float alpha,
                        const float* A,
                        const int lda,
                        const float* B,
                        const int ldb,
                        const float beta,
                        float* C,
                        const int ldc) {
    if (g_aocl.ssyr2k) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.ssyr2k(cblas_layout, cblas_uplo, cblas_trans, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    }
}

/* Symmetric rank-k update: C = alpha*A*A' + beta*C */
static void aocl_ssyrk_new(const fb_layout_t layout,
                       const fb_uplo_t uplo,
                       const fb_transpose_t trans,
                       const int n,
                       const int k,
                       const float alpha,
                       const float* A,
                       const int lda,
                       const float beta,
                       float* C,
                       const int ldc) {
    if (g_aocl.ssyrk) {
        CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
        g_aocl.ssyrk(cblas_layout, cblas_uplo, cblas_trans, (int)n, (int)k, alpha, A, (int)lda, beta, C, (int)ldc);
    }
}

/* Level 2 BLAS - Complex GEMV */
static void aocl_cgemv(char trans, int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_zgemv(char trans, int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - GBMV (banded matrix-vector) */
static void aocl_sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_sgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_cgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_zgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HEMV (Hermitian matrix-vector) */
static void aocl_chemv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_zhemv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HBMV (Hermitian banded matrix-vector) */
static void aocl_chbmv(char uplo, int n, int k, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_zhbmv(char uplo, int n, int k, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HPMV (Hermitian packed matrix-vector) */
static void aocl_chpmv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* ap,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void aocl_zhpmv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* ap,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void aocl_csymv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_zsymv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SBMV (symmetric banded matrix-vector) */
static void aocl_ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ssbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void aocl_dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dsbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SPMV (symmetric packed matrix-vector) */
static void aocl_sspmv(char uplo, int n, float alpha, const float* ap,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_sspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void aocl_dspmv(char uplo, int n, double alpha, const double* ap,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void aocl_ctrmv(char uplo, char trans, char diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void aocl_ztrmv(char uplo, char trans, char diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBMV (triangular banded matrix-vector) */
static void aocl_stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_stbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dtbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_ctbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_ztbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPMV (triangular packed matrix-vector) */
static void aocl_stpmv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_stpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_dtpmv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dtpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_ctpmv(char uplo, char trans, char diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_ztpmv(char uplo, char trans, char diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - TRSV (triangular solve) */
static void aocl_ctrsv(char uplo, char trans, char diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void aocl_ztrsv(char uplo, char trans, char diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBSV (triangular banded solve) */
static void aocl_stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_stbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dtbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_ctbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void aocl_ztbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPSV (triangular packed solve) */
static void aocl_stpsv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_stpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_dtpsv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dtpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_ctpsv(char uplo, char trans, char diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_ztpsv(char uplo, char trans, char diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void aocl_cgeru(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void aocl_zgeru(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void aocl_cgerc(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void aocl_zgerc(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HER (Hermitian rank-1 update) */
static void aocl_cher(char uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void aocl_zher(char uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - HPR (Hermitian packed rank-1 update) */
static void aocl_chpr(char uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void aocl_zhpr(char uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - HER2 (Hermitian rank-2 update) */
static void aocl_cher2(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void aocl_zher2(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HPR2 (Hermitian packed rank-2 update) */
static void aocl_chpr2(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void aocl_zhpr2(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 2 BLAS - SYR (symmetric rank-1 update) */
static void aocl_ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ssyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void aocl_dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void aocl_csyr(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void aocl_zsyr(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - SPR (symmetric packed rank-1 update) */
static void aocl_sspr(char uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_sspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void aocl_dspr(char uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - SYR2 (symmetric rank-2 update) */
static void aocl_ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ssyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void aocl_dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dsyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - SPR2 (symmetric packed rank-2 update) */
static void aocl_sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_sspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void aocl_dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 3 BLAS - GEMM (complex) */
static void aocl_cgemm(char transa, char transb, int m, int n, int k, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_zgemm(char transa, char transb, int m, int n, int k, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_csymm(char side, char uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_zsymm(char side, char uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HEMM (Hermitian matrix-matrix) */
static void aocl_chemm(char side, char uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_chemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_zhemm(char side, char uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zhemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYRK (symmetric rank-k update) */
static void aocl_ssyrk(char uplo, char trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ssyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void aocl_dsyrk(char uplo, char trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void aocl_csyrk(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void aocl_zsyrk(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - HERK (Hermitian rank-k update) */
static void aocl_cherk(char uplo, char trans, int n, int k, float alpha, const fb_complex_float_t* a, int lda, float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void aocl_zherk(char uplo, char trans, int n, int k, double alpha, const fb_complex_double_t* a, int lda, double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - SYR2K (symmetric rank-2k update) */
static void aocl_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float* a, int lda,
                       const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ssyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_dsyr2k(char uplo, char trans, int n, int k, double alpha, const double* a, int lda,
                       const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_dsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_csyr2k(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_csyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_zsyr2k(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HER2K (Hermitian rank-2k update) */
static void aocl_cher2k(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_cher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_zher2k(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_zher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void aocl_ctrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void aocl_ztrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* Level 3 BLAS - TRSM (triangular solve matrix) */
static void aocl_ctrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ctrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void aocl_ztrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_aocl.handle, "cblas_ztrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* aocl vtable */
static fb_backend_vtable_t g_aocl_vtable = {
    /* Level 1 BLAS */
    .sasum = aocl_sasum_wrapper,
    .dasum = aocl_dasum_wrapper,
    .saxpy = aocl_saxpy,
    .daxpy = aocl_daxpy,
    .sdot = aocl_sdot_wrapper,
    .ddot = aocl_ddot_wrapper,
    .scopy = aocl_scopy,
    .dcopy = aocl_dcopy,
    .sscal = aocl_sscal,
    .dscal = aocl_dscal,
    .snrm2 = aocl_snrm2_wrapper,
    .dnrm2 = aocl_dnrm2_wrapper,
    .sswap = aocl_sswap,
    .dswap = aocl_dswap,
    .isamax = aocl_isamax_wrapper,
    .idamax = aocl_idamax_wrapper,
    
    /* Level 2 BLAS */
    .sgemv = aocl_sgemv,
    .dgemv = aocl_dgemv,
    .sger = aocl_sger,
    .dger = aocl_dger,
    .ssymv = aocl_ssymv,
    .dsymv = aocl_dsymv,
    .strmv = aocl_strmv,
    .dtrmv = aocl_dtrmv,
    .strsv = aocl_strsv,
    .dtrsv = aocl_dtrsv,
    /* Additional Level 2 operations (using _new wrappers) */
    .sgbmv = aocl_sgbmv_new,
    .dgbmv = aocl_dgbmv_new,
    .ssbmv = aocl_ssbmv_new,
    .dsbmv = aocl_dsbmv_new,
    .sspmv = aocl_sspmv_new,
    .dspmv = aocl_dspmv_new,
    .stbmv = aocl_stbmv_new,
    .dtbmv = aocl_dtbmv_new,
    .stpmv = aocl_stpmv_new,
    .dtpmv = aocl_dtpmv_new,
    .stbsv = aocl_stbsv_new,
    .dtbsv = aocl_dtbsv_new,
    .stpsv = aocl_stpsv_new,
    .dtpsv = aocl_dtpsv_new,
    .ssyr = aocl_ssyr_new,
    .dsyr = aocl_dsyr_new,
    .sspr = aocl_sspr_new,
    .dspr = aocl_dspr_new,
    .ssyr2 = aocl_ssyr2_new,
    .dsyr2 = aocl_dsyr2_new,
    .sspr2 = aocl_sspr2_new,
    .dspr2 = aocl_dspr2_new,
    
    /* Level 3 BLAS */
    .sgemm = aocl_sgemm,
    .dgemm = aocl_dgemm,
    .ssymm = aocl_ssymm,
    .dsymm = aocl_dsymm,
    .strmm = aocl_strmm,
    .dtrmm = aocl_dtrmm,
    .strsm = aocl_strsm,
    .dtrsm = aocl_dtrsm,
    /* Additional Level 3 operations (using _new wrappers) */
    .ssyrk = aocl_ssyrk_new,
    .dsyrk = aocl_dsyrk_new,
    .ssyr2k = aocl_ssyr2k_new,
    .dsyr2k = aocl_dsyr2k_new,
    
    /* Unified CPU/GPU Interface - Memory Management (No-ops for CPU) */
    .mem_alloc = NULL,
    .mem_free = NULL,
    .mem_upload = NULL,
    .mem_download = NULL,
    .mem_copy = NULL,
    
    /* Unified CPU/GPU Interface - Stream Management (No-ops for CPU) */
    .stream_create = NULL,
    .stream_destroy = NULL,
    .stream_sync = NULL,
    .stream_set = NULL,
    
    /* Unified CPU/GPU Interface - Backend Properties */
    .get_capabilities = aocl_get_capabilities,
    .get_num_threads = aocl_get_num_threads_vtable,
    .set_num_threads = aocl_set_num_threads_vtable
};

const fb_backend_vtable_t* fb_aocl_get_vtable(void) {
    if (!fb_aocl_is_available()) {
        return NULL;
    }
    /* AOCL includes libFLAME which exposes the LAPACKE_* interface.
     * Lazily fill ext_ops[] for all LAPACK ops via dlsym; typed wrappers win. */
    static bool g_ext_ops_populated = false;
    if (!g_ext_ops_populated) {
        fb_auto_populate_ext_ops(&g_aocl_vtable, g_aocl.handle,
                                 k_lapacke_symbols, k_lapacke_symbols_count);
        g_ext_ops_populated = true;
    }
    return &g_aocl_vtable;
}
