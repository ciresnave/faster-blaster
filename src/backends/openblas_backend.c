/**
 * @file openblas_backend.c
 * @brief OpenBLAS backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "openblas_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CBLAS enum constant definitions */
#ifndef CblasRowMajor
#define CblasRowMajor 101
#define CblasColMajor 102
#define CblasNoTrans 111
#define CblasTrans 112
#define CblasConjTrans 113
#define CblasUpper 121
#define CblasLower 122
#define CblasNonUnit 131
#define CblasUnit 132
#define CblasLeft 141
#define CblasRight 142
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

/* OpenBLAS CBLAS function pointer types - Level 1 */
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

/* OpenBLAS CBLAS function pointer types - Level 2 */
typedef void (*cblas_sgemv_t)(const int Order, const int TransA, const int M, const int N,
                               const float alpha, const float* A, const int lda,
                               const float* X, const int incX, const float beta,
                               float* Y, const int incY);
typedef void (*cblas_dgemv_t)(const int Order, const int TransA, const int M, const int N,
                               const double alpha, const double* A, const int lda,
                               const double* X, const int incX, const double beta,
                               double* Y, const int incY);
typedef void (*cblas_sger_t)(const int Order, const int M, const int N, const float alpha,
                              const float* X, const int incX, const float* Y, const int incY,
                              float* A, const int lda);
typedef void (*cblas_dger_t)(const int Order, const int M, const int N, const double alpha,
                              const double* X, const int incX, const double* Y, const int incY,
                              double* A, const int lda);

/* OpenBLAS CBLAS function pointer types - Level 3 */
typedef void (*cblas_sgemm_t)(const int Order, const int TransA, const int TransB,
                               const int M, const int N, const int K,
                               const float alpha, const float* A, const int lda,
                               const float* B, const int ldb, const float beta,
                               float* C, const int ldc);
typedef void (*cblas_dgemm_t)(const int Order, const int TransA, const int TransB,
                               const int M, const int N, const int K,
                               const double alpha, const double* A, const int lda,
                               const double* B, const int ldb, const double beta,
                               double* C, const int ldc);

/* OpenBLAS threading control */
typedef void (*openblas_set_num_threads_t)(int num_threads);
typedef int (*openblas_get_num_threads_t)(void);
typedef char* (*openblas_get_config_t)(void);

/* Global OpenBLAS API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    char version[64];
    
    /* Threading control */
    openblas_set_num_threads_t set_num_threads;
    openblas_get_num_threads_t get_num_threads;
    openblas_get_config_t get_config;
    
    /* Level 1 BLAS */
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
    
    /* Level 2 BLAS */
    cblas_sgemv_t sgemv;
    cblas_dgemv_t dgemv;
    cblas_sger_t sger;
    cblas_dger_t dger;
    
    /* Level 3 BLAS */
    cblas_sgemm_t sgemm;
    cblas_dgemm_t dgemm;
    
    /* TODO: Add more function pointers as needed */
} g_openblas;

static int load_openblas_library(void) {
    if (g_openblas.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        "openblas.dll",
        "libopenblas.dll",
        /* Try build directory for locally built OpenBLAS */
        "..\\backends-install\\openblas\\bin\\openblas.dll",
        "..\\..\\backends-install\\openblas\\bin\\openblas.dll",
        "backends-install\\openblas\\bin\\openblas.dll",
        "C:\\libraries\\vcpkg\\installed\\x64-windows\\bin\\openblas.dll"
#elif defined(__APPLE__)
        "libopenblas.dylib",
        "libopenblas.0.dylib",
        "/opt/homebrew/lib/libopenblas.dylib",
        "/usr/local/lib/libopenblas.dylib"
#else
        "libopenblas.so",
        "libopenblas.so.0",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_openblas.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_openblas.handle) {
            break;
        }
    }
    
    if (!g_openblas.handle) {
        return -1;
    }
    
    /* Load threading control functions */
    g_openblas.set_num_threads = (openblas_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_openblas.handle, "openblas_set_num_threads");
    g_openblas.get_num_threads = (openblas_get_num_threads_t)
        FB_GET_PROC_ADDRESS(g_openblas.handle, "openblas_get_num_threads");
    g_openblas.get_config = (openblas_get_config_t)
        FB_GET_PROC_ADDRESS(g_openblas.handle, "openblas_get_config");
    
    /* Load Level 1 BLAS functions */
    g_openblas.sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sasum");
    g_openblas.dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dasum");
    g_openblas.saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_saxpy");
    g_openblas.daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_daxpy");
    g_openblas.sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sdot");
    g_openblas.ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ddot");
    g_openblas.scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_scopy");
    g_openblas.dcopy = (cblas_dcopy_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dcopy");
    g_openblas.sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sscal");
    g_openblas.dscal = (cblas_dscal_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dscal");
    g_openblas.snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_snrm2");
    g_openblas.dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dnrm2");
    g_openblas.sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sswap");
    g_openblas.dswap = (cblas_dswap_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dswap");
    g_openblas.isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_isamax");
    g_openblas.idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_idamax");
    
    /* Load Level 2 BLAS functions */
    g_openblas.sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sgemv");
    g_openblas.dgemv = (cblas_dgemv_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dgemv");
    g_openblas.sger = (cblas_sger_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sger");
    g_openblas.dger = (cblas_dger_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dger");
    
    /* Load Level 3 BLAS functions */
    g_openblas.sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sgemm");
    g_openblas.dgemm = (cblas_dgemm_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dgemm");
    
    /* Get version info */
    if (g_openblas.get_config) {
        char* config = g_openblas.get_config();
        if (config) {
            snprintf(g_openblas.version, sizeof(g_openblas.version), "%s", config);
        }
    }
    
    g_openblas.initialized = true;
    return 0;
}

bool fb_openblas_is_available(void) {
    return load_openblas_library() == 0;
}

int fb_openblas_init(void) {
    return load_openblas_library();
}

void fb_openblas_shutdown(void) {
    if (g_openblas.initialized && g_openblas.handle) {
        FB_FREE_LIBRARY(g_openblas.handle);
        g_openblas.handle = NULL;
        g_openblas.initialized = false;
    }
}

const char* fb_openblas_get_version(void) {
    if (!g_openblas.initialized) {
        return NULL;
    }
    return g_openblas.version[0] ? g_openblas.version : "Unknown";
}

void fb_openblas_set_num_threads(int num_threads) {
    if (g_openblas.set_num_threads) {
        g_openblas.set_num_threads(num_threads);
    }
}

int fb_openblas_get_num_threads(void) {
    if (g_openblas.get_num_threads) {
        return g_openblas.get_num_threads();
    }
    return 1;
}

/* Backend property functions for unified interface */

static uint32_t openblas_get_capabilities(void* handle) {
    (void)handle;  /* OpenBLAS context not needed for capabilities */
    return FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

static int openblas_get_num_threads_vtable(void* handle) {
    (void)handle;
    return fb_openblas_get_num_threads();
}

static void openblas_set_num_threads_vtable(void* handle, int num_threads) {
    (void)handle;
    fb_openblas_set_num_threads(num_threads);
}

/* Wrapper functions - Level 1 BLAS */

static void openblas_sasum(int n, const float* x, int incx, float* result) {
    if (g_openblas.sasum) {
        *result = g_openblas.sasum(n, x, incx);
    }
}

static void openblas_dasum(int n, const double* x, int incx, double* result) {
    if (g_openblas.dasum) {
        *result = g_openblas.dasum(n, x, incx);
    }
}

static void openblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_openblas.saxpy) {
        g_openblas.saxpy(n, alpha, x, incx, y, incy);
    }
}

static void openblas_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_openblas.daxpy) {
        g_openblas.daxpy(n, alpha, x, incx, y, incy);
    }
}

static void openblas_sdot(int n, const float* x, int incx, const float* y, int incy, float* result) {
    if (g_openblas.sdot) {
        *result = g_openblas.sdot(n, x, incx, y, incy);
    }
}

static void openblas_ddot(int n, const double* x, int incx, const double* y, int incy, double* result) {
    if (g_openblas.ddot) {
        *result = g_openblas.ddot(n, x, incx, y, incy);
    }
}

static void openblas_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_openblas.scopy) {
        g_openblas.scopy(n, x, incx, y, incy);
    }
}

static void openblas_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_openblas.dcopy) {
        g_openblas.dcopy(n, x, incx, y, incy);
    }
}

static void openblas_sscal(int n, float alpha, float* x, int incx) {
    if (g_openblas.sscal) {
        g_openblas.sscal(n, alpha, x, incx);
    }
}

static void openblas_dscal(int n, double alpha, double* x, int incx) {
    if (g_openblas.dscal) {
        g_openblas.dscal(n, alpha, x, incx);
    }
}

static void openblas_snrm2(int n, const float* x, int incx, float* result) {
    if (g_openblas.snrm2) {
        *result = g_openblas.snrm2(n, x, incx);
    }
}

static void openblas_dnrm2(int n, const double* x, int incx, double* result) {
    if (g_openblas.dnrm2) {
        *result = g_openblas.dnrm2(n, x, incx);
    }
}

static void openblas_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_openblas.sswap) {
        g_openblas.sswap(n, x, incx, y, incy);
    }
}

static void openblas_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_openblas.dswap) {
        g_openblas.dswap(n, x, incx, y, incy);
    }
}

static void openblas_isamax(int n, const float* x, int incx, int* result) {
    if (g_openblas.isamax) {
        *result = g_openblas.isamax(n, x, incx);
    }
}

static void openblas_idamax(int n, const double* x, int incx, int* result) {
    if (g_openblas.idamax) {
        *result = g_openblas.idamax(n, x, incx);
    }
}

/* Complex type Level 1 BLAS */

static void openblas_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void openblas_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void openblas_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void openblas_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void openblas_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void openblas_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void openblas_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void openblas_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void openblas_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void openblas_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void openblas_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void openblas_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void openblas_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void openblas_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void openblas_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void openblas_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void openblas_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void openblas_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void openblas_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void openblas_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Wrapper functions - Level 2 BLAS */

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

static void openblas_sgemv(char trans, int m, int n, float alpha, 
                           const float* a, int lda, const float* x, int incx,
                           float beta, float* y, int incy) {
    if (g_openblas.sgemv) {
        g_openblas.sgemv(CblasColMajor, transpose_to_cblas(trans), 
                         m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dgemv(char trans, int m, int n, double alpha,
                           const double* a, int lda, const double* x, int incx,
                           double beta, double* y, int incy) {
    if (g_openblas.dgemv) {
        g_openblas.dgemv(CblasColMajor, transpose_to_cblas(trans),
                         m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_sger(int m, int n, float alpha, const float* x, int incx,
                          const float* y, int incy, float* a, int lda) {
    if (g_openblas.sger) {
        g_openblas.sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_dger(int m, int n, double alpha, const double* x, int incx,
                          const double* y, int incy, double* a, int lda) {
    if (g_openblas.dger) {
        g_openblas.dger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

/* Complex GEMV */
static void openblas_cgemv(char trans, int m, int n, const void* alpha,
                           const void* a, int lda, const void* x, int incx,
                           const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgemv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_zgemv(char trans, int m, int n, const void* alpha,
                           const void* a, int lda, const void* x, int incx,
                           const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgemv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* GBMV - Banded matrix-vector multiply */
static void openblas_sgbmv(char trans, int m, int n, int kl, int ku, float alpha,
                           const float* a, int lda, const float* x, int incx,
                           float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sgbmv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dgbmv(char trans, int m, int n, int kl, int ku, double alpha,
                           const double* a, int lda, const double* x, int incx,
                           double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dgbmv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_cgbmv(char trans, int m, int n, int kl, int ku, const void* alpha,
                           const void* a, int lda, const void* x, int incx,
                           const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgbmv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_zgbmv(char trans, int m, int n, int kl, int ku, const void* alpha,
                           const void* a, int lda, const void* x, int incx,
                           const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgbmv");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* HEMV - Hermitian matrix-vector multiply */
static void openblas_chemv(char uplo, int n, const void* alpha, const void* a, int lda,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chemv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_zhemv(char uplo, int n, const void* alpha, const void* a, int lda,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhemv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* HBMV - Hermitian banded matrix-vector multiply */
static void openblas_chbmv(char uplo, int n, int k, const void* alpha, const void* a, int lda,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_zhbmv(char uplo, int n, int k, const void* alpha, const void* a, int lda,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* HPMV - Hermitian packed matrix-vector multiply */
static void openblas_chpmv(char uplo, int n, const void* alpha, const void* ap,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy);
    }
}

static void openblas_zhpmv(char uplo, int n, const void* alpha, const void* ap,
                           const void* x, int incx, const void* beta, void* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy);
    }
}

/* SYMV - Symmetric matrix-vector multiply */
static void openblas_ssymv(char uplo, int n, float alpha, const float* a, int lda,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssymv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dsymv(char uplo, int n, double alpha, const double* a, int lda,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsymv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* SBMV - Symmetric banded matrix-vector multiply */
static void openblas_ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* SPMV - Symmetric packed matrix-vector multiply */
static void openblas_sspmv(char uplo, int n, float alpha, const float* ap,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy);
    }
}

static void openblas_dspmv(char uplo, int n, double alpha, const double* ap,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy);
    }
}

/* TRMV - Triangular matrix-vector multiply */
static void openblas_strmv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_ctrmv(char uplo, char trans, char diag, int n, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_ztrmv(char uplo, char trans, char diag, int n, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

/* TBMV - Triangular banded matrix-vector multiply */
static void openblas_stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_ctbmv(char uplo, char trans, char diag, int n, int k, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_ztbmv(char uplo, char trans, char diag, int n, int k, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztbmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

/* TPMV - Triangular packed matrix-vector multiply */
static void openblas_stpmv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_dtpmv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_ctpmv(char uplo, char trans, char diag, int n, const void* ap, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_ztpmv(char uplo, char trans, char diag, int n, const void* ap, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztpmv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

/* TRSV - Triangular solve */
static void openblas_strsv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_ctrsv(char uplo, char trans, char diag, int n, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

static void openblas_ztrsv(char uplo, char trans, char diag, int n, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx);
    }
}

/* TBSV - Triangular banded solve */
static void openblas_stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stbsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtbsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_ctbsv(char uplo, char trans, char diag, int n, int k, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctbsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

static void openblas_ztbsv(char uplo, char trans, char diag, int n, int k, const void* a, int lda, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztbsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx);
    }
}

/* TPSV - Triangular packed solve */
static void openblas_stpsv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stpsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_dtpsv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtpsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_ctpsv(char uplo, char trans, char diag, int n, const void* ap, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctpsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

static void openblas_ztpsv(char uplo, char trans, char diag, int n, const void* ap, void* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztpsv");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx);
    }
}

/* Complex GER variants */
static void openblas_cgeru(int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgeru");
    if (fn) {
        fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zgeru(int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgeru");
    if (fn) {
        fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_cgerc(int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgerc");
    if (fn) {
        fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zgerc(int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgerc");
    if (fn) {
        fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

/* HER - Hermitian rank-1 update */
static void openblas_cher(char uplo, int n, float alpha, const void* x, int incx, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda);
    }
}

static void openblas_zher(char uplo, int n, double alpha, const void* x, int incx, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda);
    }
}

/* HPR - Hermitian packed rank-1 update */
static void openblas_chpr(char uplo, int n, float alpha, const void* x, int incx, void* ap) {
    typedef void (*fn_t)(int, int, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap);
    }
}

static void openblas_zhpr(char uplo, int n, double alpha, const void* x, int incx, void* ap) {
    typedef void (*fn_t)(int, int, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap);
    }
}

/* HER2 - Hermitian rank-2 update */
static void openblas_cher2(char uplo, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zher2(char uplo, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda);
    }
}

/* HPR2 - Hermitian packed rank-2 update */
static void openblas_chpr2(char uplo, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* ap) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap);
    }
}

static void openblas_zhpr2(char uplo, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* ap) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap);
    }
}

/* SYR - Symmetric rank-1 update */
static void openblas_ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda);
    }
}

static void openblas_dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda);
    }
}

/* SPR - Symmetric packed rank-1 update */
static void openblas_sspr(char uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap);
    }
}

static void openblas_dspr(char uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspr");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap);
    }
}

/* SYR2 - Symmetric rank-2 update */
static void openblas_ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda);
    }
}

/* SPR2 - Symmetric packed rank-2 update */
static void openblas_sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap);
    }
}

static void openblas_dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspr2");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap);
    }
}

/* Wrapper functions - Level 3 BLAS */

static void openblas_sgemm(char transa, char transb, int m, int n, int k,
                           float alpha, const float* a, int lda,
                           const float* b, int ldb, float beta,
                           float* c, int ldc) {
    if (g_openblas.sgemm) {
        g_openblas.sgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                         m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dgemm(char transa, char transb, int m, int n, int k,
                           double alpha, const double* a, int lda,
                           const double* b, int ldb, double beta,
                           double* c, int ldc) {
    if (g_openblas.dgemm) {
        g_openblas.dgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                         m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* Complex GEMM */
static void openblas_cgemm(char transa, char transb, int m, int n, int k,
                           const void* alpha, const void* a, int lda,
                           const void* b, int ldb, const void* beta,
                           void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgemm");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
           m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zgemm(char transa, char transb, int m, int n, int k,
                           const void* alpha, const void* a, int lda,
                           const void* b, int ldb, const void* beta,
                           void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgemm");
    if (fn) {
        fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
           m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* SYMM - Symmetric matrix-matrix multiply */
static void openblas_ssymm(char side, char uplo, int m, int n, float alpha,
                           const float* a, int lda, const float* b, int ldb,
                           float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssymm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dsymm(char side, char uplo, int m, int n, double alpha,
                           const double* a, int lda, const double* b, int ldb,
                           double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsymm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_csymm(char side, char uplo, int m, int n, const void* alpha,
                           const void* a, int lda, const void* b, int ldb,
                           const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csymm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zsymm(char side, char uplo, int m, int n, const void* alpha,
                           const void* a, int lda, const void* b, int ldb,
                           const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsymm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* HEMM - Hermitian matrix-matrix multiply */
static void openblas_chemm(char side, char uplo, int m, int n, const void* alpha,
                           const void* a, int lda, const void* b, int ldb,
                           const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chemm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zhemm(char side, char uplo, int m, int n, const void* alpha,
                           const void* a, int lda, const void* b, int ldb,
                           const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhemm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* SYRK - Symmetric rank-k update */
static void openblas_ssyrk(char uplo, char trans, int n, int k, float alpha,
                           const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyrk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_dsyrk(char uplo, char trans, int n, int k, double alpha,
                           const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyrk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_csyrk(char uplo, char trans, int n, int k, const void* alpha,
                           const void* a, int lda, const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csyrk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_zsyrk(char uplo, char trans, int n, int k, const void* alpha,
                           const void* a, int lda, const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsyrk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

/* HERK - Hermitian rank-k update */
static void openblas_cherk(char uplo, char trans, int n, int k, float alpha,
                           const void* a, int lda, float beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cherk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_zherk(char uplo, char trans, int n, int k, double alpha,
                           const void* a, int lda, double beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zherk");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc);
    }
}

/* SYR2K - Symmetric rank-2k update */
static void openblas_ssyr2k(char uplo, char trans, int n, int k, float alpha,
                            const float* a, int lda, const float* b, int ldb,
                            float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dsyr2k(char uplo, char trans, int n, int k, double alpha,
                            const double* a, int lda, const double* b, int ldb,
                            double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_csyr2k(char uplo, char trans, int n, int k, const void* alpha,
                            const void* a, int lda, const void* b, int ldb,
                            const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csyr2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zsyr2k(char uplo, char trans, int n, int k, const void* alpha,
                            const void* a, int lda, const void* b, int ldb,
                            const void* beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsyr2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* HER2K - Hermitian rank-2k update */
static void openblas_cher2k(char uplo, char trans, int n, int k, const void* alpha,
                            const void* a, int lda, const void* b, int ldb,
                            float beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zher2k(char uplo, char trans, int n, int k, const void* alpha,
                            const void* a, int lda, const void* b, int ldb,
                            double beta, void* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher2k");
    if (fn) {
        fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* TRMM - Triangular matrix-matrix multiply */
static void openblas_strmm(char side, char uplo, char transa, char diag, int m, int n,
                           float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strmm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_dtrmm(char side, char uplo, char transa, char diag, int m, int n,
                           double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrmm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ctrmm(char side, char uplo, char transa, char diag, int m, int n,
                           const void* alpha, const void* a, int lda, void* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrmm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ztrmm(char side, char uplo, char transa, char diag, int m, int n,
                           const void* alpha, const void* a, int lda, void* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrmm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

/* TRSM - Triangular solve multiple right-hand sides */
static void openblas_strsm(char side, char uplo, char transa, char diag, int m, int n,
                           float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strsm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_dtrsm(char side, char uplo, char transa, char diag, int m, int n,
                           double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrsm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ctrsm(char side, char uplo, char transa, char diag, int m, int n,
                           const void* alpha, const void* a, int lda, void* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrsm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ztrsm(char side, char uplo, char transa, char diag, int m, int n,
                           const void* alpha, const void* a, int lda, void* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrsm");
    if (fn) {
        fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag),
           m, n, alpha, a, lda, b, ldb);
    }
}

/* LAPACK operations */

static int layout_to_lapack(char layout) {
    return (layout == 'R' || layout == 'r') ? 101 : 102; /* 101=Row-major, 102=Col-major */
}

/* GESV - General linear system solve */
static void openblas_sgesv(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, int* info) {
    typedef int (*fn_t)(int, int, int, float*, int, int*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgesv");
    if (fn) {
        *info = fn(102, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_dgesv(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, int* info) {
    typedef int (*fn_t)(int, int, int, double*, int, int*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgesv");
    if (fn) {
        *info = fn(102, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_cgesv(int n, int nrhs, void* a, int lda, int* ipiv, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgesv");
    if (fn) {
        *info = fn(102, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_zgesv(int n, int nrhs, void* a, int lda, int* ipiv, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgesv");
    if (fn) {
        *info = fn(102, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

/* GETRF - LU factorization */
static void openblas_sgetrf(int m, int n, float* a, int lda, int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, int, float*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, ipiv);
    }
}

static void openblas_dgetrf(int m, int n, double* a, int lda, int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, int, double*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, ipiv);
    }
}

static void openblas_cgetrf(int m, int n, void* a, int lda, int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, ipiv);
    }
}

static void openblas_zgetrf(int m, int n, void* a, int lda, int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, ipiv);
    }
}

/* GETRS - Solve using LU factorization */
static void openblas_sgetrs(char trans, int n, int nrhs, const float* a, int lda, const int* ipiv, float* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, const int*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetrs");
    if (fn) {
        *info = fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_dgetrs(char trans, int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, const int*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetrs");
    if (fn) {
        *info = fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_cgetrs(char trans, int n, int nrhs, const void* a, int lda, const int* ipiv, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, const int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetrs");
    if (fn) {
        *info = fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

static void openblas_zgetrs(char trans, int n, int nrhs, const void* a, int lda, const int* ipiv, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, const int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetrs");
    if (fn) {
        *info = fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

/* GETRI - Matrix inversion using LU */
static void openblas_sgetri(int n, float* a, int lda, const int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, float*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetri");
    if (fn) {
        *info = fn(102, n, a, lda, ipiv);
    }
}

static void openblas_dgetri(int n, double* a, int lda, const int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, double*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetri");
    if (fn) {
        *info = fn(102, n, a, lda, ipiv);
    }
}

static void openblas_cgetri(int n, void* a, int lda, const int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, void*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetri");
    if (fn) {
        *info = fn(102, n, a, lda, ipiv);
    }
}

static void openblas_zgetri(int n, void* a, int lda, const int* ipiv, int* info) {
    typedef int (*fn_t)(int, int, void*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetri");
    if (fn) {
        *info = fn(102, n, a, lda, ipiv);
    }
}

/* POSV - Positive-definite linear system solve */
static void openblas_sposv(char uplo, int n, int nrhs, float* a, int lda, float* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sposv");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_dposv(char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dposv");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_cposv(char uplo, int n, int nrhs, void* a, int lda, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cposv");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_zposv(char uplo, int n, int nrhs, void* a, int lda, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zposv");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

/* POTRF - Cholesky factorization */
static void openblas_spotrf(char uplo, int n, float* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotrf");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_dpotrf(char uplo, int n, double* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotrf");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_cpotrf(char uplo, int n, void* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotrf");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_zpotrf(char uplo, int n, void* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotrf");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

/* POTRS - Solve using Cholesky factorization */
static void openblas_spotrs(char uplo, int n, int nrhs, const float* a, int lda, float* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotrs");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_dpotrs(char uplo, int n, int nrhs, const double* a, int lda, double* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotrs");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_cpotrs(char uplo, int n, int nrhs, const void* a, int lda, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotrs");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

static void openblas_zpotrs(char uplo, int n, int nrhs, const void* a, int lda, void* b, int ldb, int* info) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotrs");
    if (fn) {
        *info = fn(102, uplo, n, nrhs, a, lda, b, ldb);
    }
}

/* POTRI - Matrix inversion using Cholesky */
static void openblas_spotri(char uplo, int n, float* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotri");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_dpotri(char uplo, int n, double* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotri");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_cpotri(char uplo, int n, void* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotri");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

static void openblas_zpotri(char uplo, int n, void* a, int lda, int* info) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotri");
    if (fn) {
        *info = fn(102, uplo, n, a, lda);
    }
}

/* GEQRF - QR factorization */
static void openblas_sgeqrf(int m, int n, float* a, int lda, float* tau, int* info) {
    typedef int (*fn_t)(int, int, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgeqrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, tau);
    }
}

static void openblas_dgeqrf(int m, int n, double* a, int lda, double* tau, int* info) {
    typedef int (*fn_t)(int, int, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgeqrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, tau);
    }
}

static void openblas_cgeqrf(int m, int n, void* a, int lda, void* tau, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgeqrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, tau);
    }
}

static void openblas_zgeqrf(int m, int n, void* a, int lda, void* tau, int* info) {
    typedef int (*fn_t)(int, int, int, void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgeqrf");
    if (fn) {
        *info = fn(102, m, n, a, lda, tau);
    }
}

/* ORGQR/UNGQR - Generate Q from QR factorization */
static void openblas_sorgqr(int m, int n, int k, float* a, int lda, const float* tau, int* info) {
    typedef int (*fn_t)(int, int, int, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sorgqr");
    if (fn) {
        *info = fn(102, m, n, k, a, lda, tau);
    }
}

static void openblas_dorgqr(int m, int n, int k, double* a, int lda, const double* tau, int* info) {
    typedef int (*fn_t)(int, int, int, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dorgqr");
    if (fn) {
        *info = fn(102, m, n, k, a, lda, tau);
    }
}

static void openblas_cungqr(int m, int n, int k, void* a, int lda, const void* tau, int* info) {
    typedef int (*fn_t)(int, int, int, int, void*, int, const void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cungqr");
    if (fn) {
        *info = fn(102, m, n, k, a, lda, tau);
    }
}

static void openblas_zungqr(int m, int n, int k, void* a, int lda, const void* tau, int* info) {
    typedef int (*fn_t)(int, int, int, int, void*, int, const void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zungqr");
    if (fn) {
        *info = fn(102, m, n, k, a, lda, tau);
    }
}

/* GESVD - Singular value decomposition */
static void openblas_sgesvd(char jobu, char jobvt, int m, int n, float* a, int lda,
                            float* s, float* u, int ldu, float* vt, int ldvt,
                            float* superb, int* info) {
    typedef int (*fn_t)(int, char, char, int, int, float*, int, float*, float*, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgesvd");
    if (fn) {
        *info = fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
}

static void openblas_dgesvd(char jobu, char jobvt, int m, int n, double* a, int lda,
                            double* s, double* u, int ldu, double* vt, int ldvt,
                            double* superb, int* info) {
    typedef int (*fn_t)(int, char, char, int, int, double*, int, double*, double*, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgesvd");
    if (fn) {
        *info = fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
}

static void openblas_cgesvd(char jobu, char jobvt, int m, int n, void* a, int lda,
                            float* s, void* u, int ldu, void* vt, int ldvt,
                            float* superb, int* info) {
    typedef int (*fn_t)(int, char, char, int, int, void*, int, float*, void*, int, void*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgesvd");
    if (fn) {
        *info = fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
}

static void openblas_zgesvd(char jobu, char jobvt, int m, int n, void* a, int lda,
                            double* s, void* u, int ldu, void* vt, int ldvt,
                            double* superb, int* info) {
    typedef int (*fn_t)(int, char, char, int, int, void*, int, double*, void*, int, void*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgesvd");
    if (fn) {
        *info = fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
}

/* SYEV/HEEV - Eigenvalue decomposition */
static void openblas_ssyev(char jobz, char uplo, int n, float* a, int lda, float* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_ssyev");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_dsyev(char jobz, char uplo, int n, double* a, int lda, double* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dsyev");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_cheev(char jobz, char uplo, int n, void* a, int lda, float* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, void*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cheev");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_zheev(char jobz, char uplo, int n, void* a, int lda, double* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, void*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zheev");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

/* SYEVD/HEEVD - Eigenvalue decomposition (divide-and-conquer) */
static void openblas_ssyevd(char jobz, char uplo, int n, float* a, int lda, float* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_ssyevd");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_dsyevd(char jobz, char uplo, int n, double* a, int lda, double* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dsyevd");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_cheevd(char jobz, char uplo, int n, void* a, int lda, float* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, void*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cheevd");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

static void openblas_zheevd(char jobz, char uplo, int n, void* a, int lda, double* w, int* info) {
    typedef int (*fn_t)(int, char, char, int, void*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zheevd");
    if (fn) {
        *info = fn(102, jobz, uplo, n, a, lda, w);
    }
}

/* OpenBLAS vtable */
static fb_backend_vtable_t g_openblas_vtable = {
    /* Level 1 BLAS - Real */
    .sasum = openblas_sasum,
    .dasum = openblas_dasum,
    .saxpy = openblas_saxpy,
    .daxpy = openblas_daxpy,
    .sdot = openblas_sdot,
    .ddot = openblas_ddot,
    .scopy = openblas_scopy,
    .dcopy = openblas_dcopy,
    .sscal = openblas_sscal,
    .dscal = openblas_dscal,
    .snrm2 = openblas_snrm2,
    .dnrm2 = openblas_dnrm2,
    .sswap = openblas_sswap,
    .dswap = openblas_dswap,
    .isamax = openblas_isamax,
    .idamax = openblas_idamax,
    
    /* Level 1 BLAS - Complex */
    .scasum = openblas_scasum,
    .dzasum = openblas_dzasum,
    .caxpy = openblas_caxpy,
    .zaxpy = openblas_zaxpy,
    .ccopy = openblas_ccopy,
    .zcopy = openblas_zcopy,
    .cscal = openblas_cscal,
    .zscal = openblas_zscal,
    .csscal = openblas_csscal,
    .zdscal = openblas_zdscal,
    .cswap = openblas_cswap,
    .zswap = openblas_zswap,
    // TODO: These _sub variants may not be in backend_interface.h vtable
    // .cdotu_sub = openblas_cdotu_sub,
    // .zdotu_sub = openblas_zdotu_sub,
    // .cdotc_sub = openblas_cdotc_sub,
    // .zdotc_sub = openblas_zdotc_sub,
    .scnrm2 = openblas_scnrm2,
    .dznrm2 = openblas_dznrm2,
    .icamax = openblas_icamax,
    .izamax = openblas_izamax,
    
    /* Level 1 BLAS - Rotation */
    .srotg = openblas_srotg,
    .drotg = openblas_drotg,
    .srot = openblas_srot,
    .drot = openblas_drot,
    .srotm = openblas_srotm,
    .drotm = openblas_drotm,
    .srotmg = openblas_srotmg,
    .drotmg = openblas_drotmg,
    
    /* Level 2 BLAS - GEMV/GER */
    .sgemv = openblas_sgemv,
    .dgemv = openblas_dgemv,
    .cgemv = openblas_cgemv,
    .zgemv = openblas_zgemv,
    .sger = openblas_sger,
    .dger = openblas_dger,
    
    /* Level 2 BLAS - Banded */
    .sgbmv = openblas_sgbmv,
    .dgbmv = openblas_dgbmv,
    .cgbmv = openblas_cgbmv,
    .zgbmv = openblas_zgbmv,
    
    /* Level 2 BLAS - Hermitian/Symmetric */
    .chemv = openblas_chemv,
    .zhemv = openblas_zhemv,
    .chbmv = openblas_chbmv,
    .zhbmv = openblas_zhbmv,
    .chpmv = openblas_chpmv,
    .zhpmv = openblas_zhpmv,
    .ssymv = openblas_ssymv,
    .dsymv = openblas_dsymv,
    .ssbmv = openblas_ssbmv,
    .dsbmv = openblas_dsbmv,
    .sspmv = openblas_sspmv,
    .dspmv = openblas_dspmv,
    
    /* Level 2 BLAS - Triangular */
    .strmv = openblas_strmv,
    .dtrmv = openblas_dtrmv,
    .ctrmv = openblas_ctrmv,
    .ztrmv = openblas_ztrmv,
    .stbmv = openblas_stbmv,
    .dtbmv = openblas_dtbmv,
    .ctbmv = openblas_ctbmv,
    .ztbmv = openblas_ztbmv,
    .stpmv = openblas_stpmv,
    .dtpmv = openblas_dtpmv,
    .ctpmv = openblas_ctpmv,
    .ztpmv = openblas_ztpmv,
    .strsv = openblas_strsv,
    .dtrsv = openblas_dtrsv,
    .ctrsv = openblas_ctrsv,
    .ztrsv = openblas_ztrsv,
    .stbsv = openblas_stbsv,
    .dtbsv = openblas_dtbsv,
    .ctbsv = openblas_ctbsv,
    .ztbsv = openblas_ztbsv,
    .stpsv = openblas_stpsv,
    .dtpsv = openblas_dtpsv,
    .ctpsv = openblas_ctpsv,
    .ztpsv = openblas_ztpsv,
    
    /* Level 2 BLAS - Rank updates */
    .cgeru = openblas_cgeru,
    .zgeru = openblas_zgeru,
    .cgerc = openblas_cgerc,
    .zgerc = openblas_zgerc,
    .cher = openblas_cher,
    .zher = openblas_zher,
    .chpr = openblas_chpr,
    .zhpr = openblas_zhpr,
    .cher2 = openblas_cher2,
    .zher2 = openblas_zher2,
    .chpr2 = openblas_chpr2,
    .zhpr2 = openblas_zhpr2,
    .ssyr = openblas_ssyr,
    .dsyr = openblas_dsyr,
    .sspr = openblas_sspr,
    .dspr = openblas_dspr,
    .ssyr2 = openblas_ssyr2,
    .dsyr2 = openblas_dsyr2,
    .sspr2 = openblas_sspr2,
    .dspr2 = openblas_dspr2,
    
    /* Level 3 BLAS - GEMM */
    .sgemm = openblas_sgemm,
    .dgemm = openblas_dgemm,
    .cgemm = openblas_cgemm,
    .zgemm = openblas_zgemm,
    
    /* Level 3 BLAS - SYMM/HEMM */
    .ssymm = openblas_ssymm,
    .dsymm = openblas_dsymm,
    .csymm = openblas_csymm,
    .zsymm = openblas_zsymm,
    .chemm = openblas_chemm,
    .zhemm = openblas_zhemm,
    
    /* Level 3 BLAS - SYRK/HERK */
    .ssyrk = openblas_ssyrk,
    .dsyrk = openblas_dsyrk,
    .csyrk = openblas_csyrk,
    .zsyrk = openblas_zsyrk,
    .cherk = openblas_cherk,
    .zherk = openblas_zherk,
    
    /* Level 3 BLAS - SYR2K/HER2K */
    .ssyr2k = openblas_ssyr2k,
    .dsyr2k = openblas_dsyr2k,
    .csyr2k = openblas_csyr2k,
    .zsyr2k = openblas_zsyr2k,
    .cher2k = openblas_cher2k,
    .zher2k = openblas_zher2k,
    
    /* Level 3 BLAS - TRMM/TRSM */
    .strmm = openblas_strmm,
    .dtrmm = openblas_dtrmm,
    .ctrmm = openblas_ctrmm,
    .ztrmm = openblas_ztrmm,
    .strsm = openblas_strsm,
    .dtrsm = openblas_dtrsm,
    .ctrsm = openblas_ctrsm,
    .ztrsm = openblas_ztrsm,
    
    /* LAPACK - Linear system solve */
    .sgesv = openblas_sgesv,
    .dgesv = openblas_dgesv,
    .cgesv = openblas_cgesv,
    .zgesv = openblas_zgesv,
    
    /* LAPACK - LU factorization */
    .sgetrf = openblas_sgetrf,
    .dgetrf = openblas_dgetrf,
    .cgetrf = openblas_cgetrf,
    .zgetrf = openblas_zgetrf,
    .sgetrs = openblas_sgetrs,
    .dgetrs = openblas_dgetrs,
    .cgetrs = openblas_cgetrs,
    .zgetrs = openblas_zgetrs,
    .sgetri = openblas_sgetri,
    .dgetri = openblas_dgetri,
    .cgetri = openblas_cgetri,
    .zgetri = openblas_zgetri,
    
    /* LAPACK - Cholesky factorization */
    .sposv = openblas_sposv,
    .dposv = openblas_dposv,
    .cposv = openblas_cposv,
    .zposv = openblas_zposv,
    .spotrf = openblas_spotrf,
    .dpotrf = openblas_dpotrf,
    .cpotrf = openblas_cpotrf,
    .zpotrf = openblas_zpotrf,
    .spotrs = openblas_spotrs,
    .dpotrs = openblas_dpotrs,
    .cpotrs = openblas_cpotrs,
    .zpotrs = openblas_zpotrs,
    .spotri = openblas_spotri,
    .dpotri = openblas_dpotri,
    .cpotri = openblas_cpotri,
    .zpotri = openblas_zpotri,
    
    /* LAPACK - QR factorization */
    .sgeqrf = openblas_sgeqrf,
    .dgeqrf = openblas_dgeqrf,
    .cgeqrf = openblas_cgeqrf,
    .zgeqrf = openblas_zgeqrf,
    .sorgqr = openblas_sorgqr,
    .dorgqr = openblas_dorgqr,
    .cungqr = openblas_cungqr,
    .zungqr = openblas_zungqr,
    
    /* LAPACK - SVD */
    .sgesvd = openblas_sgesvd,
    .dgesvd = openblas_dgesvd,
    .cgesvd = openblas_cgesvd,
    .zgesvd = openblas_zgesvd,
    
    /* LAPACK - Eigenvalue decomposition */
    .ssyev = openblas_ssyev,
    .dsyev = openblas_dsyev,
    .cheev = openblas_cheev,
    .zheev = openblas_zheev,
    // TODO: These evd variants may not be in backend_interface.h vtable
    // .ssyevd = openblas_ssyevd,
    // .dsyevd = openblas_dsyevd,
    // .cheevd = openblas_cheevd,
    // .zheevd = openblas_zheevd,
    
    /* Unified CPU/GPU Interface - Memory Management (No-ops for CPU) */
    .mem_alloc = NULL,      /* CPU: Use system malloc */
    .mem_free = NULL,       /* CPU: Use system free */
    .mem_upload = NULL,     /* CPU: No-op, data already in RAM */
    .mem_download = NULL,   /* CPU: No-op, data already in RAM */
    .mem_copy = NULL,       /* CPU: Could use memcpy, but optional */
    
    /* Unified CPU/GPU Interface - Stream Management (No-ops for CPU) */
    .stream_create = NULL,  /* CPU: No stream needed (synchronous) */
    .stream_destroy = NULL, /* CPU: No stream needed */
    .stream_sync = NULL,    /* CPU: No-op, operations already synchronous */
    .stream_set = NULL,     /* CPU: No stream needed */
    
    /* Unified CPU/GPU Interface - Backend Properties */
    .get_capabilities = openblas_get_capabilities,
    .get_num_threads = openblas_get_num_threads_vtable,
    .set_num_threads = openblas_set_num_threads_vtable
};

const fb_backend_vtable_t* fb_openblas_get_vtable(void) {
    if (!fb_openblas_is_available()) {
        return NULL;
    }
    return &g_openblas_vtable;
}
