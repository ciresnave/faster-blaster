/**
 * @file openblas_backend.c
 * @brief OpenBLAS backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "openblas_backend.h"
#include "backend_auto_detect.h"     /* fb_auto_populate_ext_ops       */
#include "sym_tables/sym_tables.h"     /* k_lapacke_symbols, …           */
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

static float openblas_sasum(const int n, const float* x, const int incx) {
    if (g_openblas.sasum) {
        return g_openblas.sasum(n, x, incx);
    }
    return 0.0f;
}

static double openblas_dasum(const int n, const double* x, const int incx) {
    if (g_openblas.dasum) {
        return g_openblas.dasum(n, x, incx);
    }
    return 0.0;
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

static float openblas_sdot(const int n, const float* x, const int incx, const float* y, const int incy) {
    if (g_openblas.sdot) {
        return g_openblas.sdot(n, x, incx, y, incy);
    }
    return 0.0f;
}

static double openblas_ddot(const int n, const double* x, const int incx, const double* y, const int incy) {
    if (g_openblas.ddot) {
        return g_openblas.ddot(n, x, incx, y, incy);
    }
    return 0.0;
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

static float openblas_snrm2(const int n, const float* x, const int incx) {
    if (g_openblas.snrm2) {
        return g_openblas.snrm2(n, x, incx);
    }
    return 0.0f;
}

static double openblas_dnrm2(const int n, const double* x, const int incx) {
    if (g_openblas.dnrm2) {
        return g_openblas.dnrm2(n, x, incx);
    }
    return 0.0;
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

static int openblas_isamax(const int n, const float* x, const int incx) {
    if (g_openblas.isamax) {
        return g_openblas.isamax(n, x, incx);
    }
    return -1;
}

static int openblas_idamax(const int n, const double* x, const int incx) {
    if (g_openblas.idamax) {
        return g_openblas.idamax(n, x, incx);
    }
    return -1;
}

/* Complex type Level 1 BLAS */

static float openblas_scasum(const int n, const fb_complex_float_t* x, const int incx) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_scasum");
    if (fn) {
        return fn(n, x, incx);
    }
    return 0.0f;
}

static double openblas_dzasum(const int n, const fb_complex_double_t* x, const int incx) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dzasum");
    if (fn) {
        return fn(n, x, incx);
    }
    return 0.0;
}

static void openblas_caxpy(const int n, const fb_complex_float_t alpha, const fb_complex_float_t* x, const int incx, fb_complex_float_t* y, const int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_caxpy");
    if (fn) {
        fn(n, &alpha, x, incx, y, incy);
    }
}

static void openblas_zaxpy(const int n, const fb_complex_double_t alpha, const fb_complex_double_t* x, const int incx, fb_complex_double_t* y, const int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, &alpha, x, incx, y, incy);
    }
}

static void openblas_ccopy(const int n, const fb_complex_float_t* x, const int incx, fb_complex_float_t* y, const int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_zcopy(const int n, const fb_complex_double_t* x, const int incx, fb_complex_double_t* y, const int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_cscal(const int n, const fb_complex_float_t alpha, fb_complex_float_t* x, const int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cscal");
    if (fn) {
        fn(n, &alpha, x, incx);
    }
}

static void openblas_zscal(const int n, const fb_complex_double_t alpha, fb_complex_double_t* x, const int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zscal");
    if (fn) {
        fn(n, &alpha, x, incx);
    }
}

static void openblas_csscal(const int n, const float alpha, fb_complex_float_t* x, const int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_zdscal(const int n, const double alpha, fb_complex_double_t* x, const int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void openblas_cswap(const int n, fb_complex_float_t* x, const int incx, fb_complex_float_t* y, const int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void openblas_zswap(const int n, fb_complex_double_t* x, const int incx, fb_complex_double_t* y, const int incy) {
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

static float openblas_scnrm2(const int n, const fb_complex_float_t* x, const int incx) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_scnrm2");
    return fn ? fn(n, x, incx) : 0.0f;
}

static double openblas_dznrm2(const int n, const fb_complex_double_t* x, const int incx) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dznrm2");
    return fn ? fn(n, x, incx) : 0.0;
}

static int openblas_icamax(const int n, const fb_complex_float_t* x, const int incx) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_icamax");
    return fn ? fn(n, x, incx) : -1;
}

static int openblas_izamax(const int n, const fb_complex_double_t* x, const int incx) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_izamax");
    return fn ? fn(n, x, incx) : -1;
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

static void openblas_sgemv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, float alpha, 
                           const float* a, int lda, const float* x, int incx,
                           float beta, float* y, int incy) {
    if (g_openblas.sgemv) {
        g_openblas.sgemv((int)layout, (int)trans, 
                         m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dgemv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, double alpha,
                           const double* a, int lda, const double* x, int incx,
                           double beta, double* y, int incy) {
    if (g_openblas.dgemv) {
        g_openblas.dgemv((int)layout, (int)trans,
                         m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_sger(const fb_layout_t layout, int m, int n, float alpha, const float* x, int incx,
                          const float* y, int incy, float* a, int lda) {
    if (g_openblas.sger) {
        g_openblas.sger((int)layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_dger(const fb_layout_t layout, int m, int n, double alpha, const double* x, int incx,
                          const double* y, int incy, double* a, int lda) {
    if (g_openblas.dger) {
        g_openblas.dger((int)layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

/* Complex GEMV */
static void openblas_cgemv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda,
                           const fb_complex_float_t* x, int incx,
                           const fb_complex_float_t beta,
                           fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgemv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

static void openblas_zgemv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda,
                           const fb_complex_double_t* x, int incx,
                           const fb_complex_double_t beta,
                           fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgemv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/* GBMV - Banded matrix-vector multiply */
static void openblas_sgbmv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, int kl, int ku, float alpha,
                           const float* a, int lda, const float* x, int incx,
                           float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sgbmv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dgbmv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, int kl, int ku, double alpha,
                           const double* a, int lda, const double* x, int incx,
                           double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dgbmv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_cgbmv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, int kl, int ku,
                           const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda,
                           const fb_complex_float_t* x, int incx,
                           const fb_complex_float_t beta,
                           fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgbmv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

static void openblas_zgbmv(const fb_layout_t layout, const fb_transpose_t trans,
                           int m, int n, int kl, int ku,
                           const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda,
                           const fb_complex_double_t* x, int incx,
                           const fb_complex_double_t beta,
                           fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgbmv");
    if (fn) {
        fn((int)layout, (int)trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/* HEMV - Hermitian matrix-vector multiply */
static void openblas_chemv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda,
                           const fb_complex_float_t* x, int incx,
                           const fb_complex_float_t beta,
                           fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chemv");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

static void openblas_zhemv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda,
                           const fb_complex_double_t* x, int incx,
                           const fb_complex_double_t beta,
                           fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhemv");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/* HBMV - Hermitian banded matrix-vector multiply */
static void openblas_chbmv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, int k, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda,
                           const fb_complex_float_t* x, int incx,
                           const fb_complex_float_t beta,
                           fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chbmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, k, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

static void openblas_zhbmv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, int k, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda,
                           const fb_complex_double_t* x, int incx,
                           const fb_complex_double_t beta,
                           fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhbmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, k, &alpha, a, lda, x, incx, &beta, y, incy);
    }
}

/* HPMV - Hermitian packed matrix-vector multiply */
static void openblas_chpmv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, const fb_complex_float_t alpha,
                           const fb_complex_float_t* ap,
                           const fb_complex_float_t* x, int incx,
                           const fb_complex_float_t beta,
                           fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, ap, x, incx, &beta, y, incy);
    }
}

static void openblas_zhpmv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, const fb_complex_double_t alpha,
                           const fb_complex_double_t* ap,
                           const fb_complex_double_t* x, int incx,
                           const fb_complex_double_t beta,
                           fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, ap, x, incx, &beta, y, incy);
    }
}

/* SYMV - Symmetric matrix-vector multiply */
static void openblas_ssymv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, float alpha, const float* a, int lda,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssymv");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dsymv(const fb_layout_t layout, const fb_uplo_t uplo,
                           int n, double alpha, const double* a, int lda,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsymv");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* SBMV - Symmetric banded matrix-vector multiply */
static void openblas_ssbmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, int k, float alpha, const float* a, int lda,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssbmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void openblas_dsbmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, int k, double alpha, const double* a, int lda,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsbmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

/* SPMV - Symmetric packed matrix-vector multiply */
static void openblas_sspmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* ap,
                           const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(int, int, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy);
    }
}

static void openblas_dspmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* ap,
                           const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(int, int, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspmv");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy);
    }
}

/* TRMV - Triangular matrix-vector multiply */
static void openblas_strmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_dtrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_ctrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t *a, int lda, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_ztrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t *a, int lda, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

/* TBMV - Triangular banded matrix-vector multiply */
static void openblas_stbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stbmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_dtbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtbmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_ctbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_float_t *a, int lda, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctbmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_ztbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_double_t *a, int lda, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztbmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

/* TPMV - Triangular packed matrix-vector multiply */
static void openblas_stpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stpmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_dtpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtpmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_ctpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t *ap, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctpmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_ztpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t *ap, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztpmv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

/* TRSV - Triangular solve */
static void openblas_strsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_dtrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_ctrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t *a, int lda, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

static void openblas_ztrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t *a, int lda, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx);
    }
}

/* TBSV - Triangular banded solve */
static void openblas_stbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stbsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_dtbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtbsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_ctbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_float_t *a, int lda, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctbsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

static void openblas_ztbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_double_t *a, int lda, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztbsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx);
    }
}

/* TPSV - Triangular packed solve */
static void openblas_stpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_stpsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_dtpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtpsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_ctpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t *ap, fb_complex_float_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctpsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

static void openblas_ztpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t *ap, fb_complex_double_t *x, int incx) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztpsv");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx);
    }
}

/* Complex GER variants */
static void openblas_cgeru(const fb_layout_t layout, int m, int n, const fb_complex_float_t alpha, const fb_complex_float_t *x, int incx, const fb_complex_float_t *y, int incy, fb_complex_float_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgeru");
    if (fn) {
        fn((int)layout, m, n, &alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zgeru(const fb_layout_t layout, int m, int n, const fb_complex_double_t alpha, const fb_complex_double_t *x, int incx, const fb_complex_double_t *y, int incy, fb_complex_double_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgeru");
    if (fn) {
        fn((int)layout, m, n, &alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_cgerc(const fb_layout_t layout, int m, int n, const fb_complex_float_t alpha, const fb_complex_float_t *x, int incx, const fb_complex_float_t *y, int incy, fb_complex_float_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgerc");
    if (fn) {
        fn((int)layout, m, n, &alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zgerc(const fb_layout_t layout, int m, int n, const fb_complex_double_t alpha, const fb_complex_double_t *x, int incx, const fb_complex_double_t *y, int incy, fb_complex_double_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgerc");
    if (fn) {
        fn((int)layout, m, n, &alpha, x, incx, y, incy, a, lda);
    }
}

/* HER - Hermitian rank-1 update */
static void openblas_cher(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const fb_complex_float_t *x, int incx, fb_complex_float_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, a, lda);
    }
}

static void openblas_zher(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const fb_complex_double_t *x, int incx, fb_complex_double_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, a, lda);
    }
}

/* HPR - Hermitian packed rank-1 update */
static void openblas_chpr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const fb_complex_float_t *x, int incx, fb_complex_float_t *ap) {
    typedef void (*fn_t)(int, int, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, ap);
    }
}

static void openblas_zhpr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const fb_complex_double_t *x, int incx, fb_complex_double_t *ap) {
    typedef void (*fn_t)(int, int, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, ap);
    }
}

/* HER2 - Hermitian rank-2 update */
static void openblas_cher2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t alpha, const fb_complex_float_t *x, int incx, const fb_complex_float_t *y, int incy, fb_complex_float_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher2");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_zher2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t alpha, const fb_complex_double_t *x, int incx, const fb_complex_double_t *y, int incy, fb_complex_double_t *a, int lda) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher2");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, x, incx, y, incy, a, lda);
    }
}

/* HPR2 - Hermitian packed rank-2 update */
static void openblas_chpr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t alpha, const fb_complex_float_t *x, int incx, const fb_complex_float_t *y, int incy, fb_complex_float_t *ap) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chpr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, x, incx, y, incy, ap);
    }
}

static void openblas_zhpr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t alpha, const fb_complex_double_t *x, int incx, const fb_complex_double_t *y, int incy, fb_complex_double_t *ap) {
    typedef void (*fn_t)(int, int, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhpr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, &alpha, x, incx, y, incy, ap);
    }
}

/* SYR - Symmetric rank-1 update */
static void openblas_ssyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, a, lda);
    }
}

static void openblas_dsyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, a, lda);
    }
}

/* SPR - Symmetric packed rank-1 update */
static void openblas_sspr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, ap);
    }
}

static void openblas_dspr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspr");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, ap);
    }
}

/* SYR2 - Symmetric rank-2 update */
static void openblas_ssyr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void openblas_dsyr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, y, incy, a, lda);
    }
}

/* SPR2 - Symmetric packed rank-2 update */
static void openblas_sspr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(int, int, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_sspr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, y, incy, ap);
    }
}

static void openblas_dspr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(int, int, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dspr2");
    if (fn) {
        fn((int)layout, (int)uplo, n, alpha, x, incx, y, incy, ap);
    }
}

/* Wrapper functions - Level 3 BLAS */

static void openblas_sgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k,
                           float alpha, const float* a, int lda,
                           const float* b, int ldb, float beta,
                           float* c, int ldc) {
    if (g_openblas.sgemm) {
        g_openblas.sgemm((int)layout, (int)transa, (int)transb,
                         m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k,
                           double alpha, const double* a, int lda,
                           const double* b, int ldb, double beta,
                           double* c, int ldc) {
    if (g_openblas.dgemm) {
        g_openblas.dgemm((int)layout, (int)transa, (int)transb,
                         m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* Complex GEMM */
static void openblas_cgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k,
                           const fb_complex_float_t alpha, const fb_complex_float_t *a, int lda,
                           const fb_complex_float_t *b, int ldb, const fb_complex_float_t beta,
                           fb_complex_float_t *c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cgemm");
    if (fn) {
        fn((int)layout, (int)transa, (int)transb,
           m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

static void openblas_zgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k,
                           const fb_complex_double_t alpha, const fb_complex_double_t *a, int lda,
                           const fb_complex_double_t *b, int ldb, const fb_complex_double_t beta,
                           fb_complex_double_t *c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zgemm");
    if (fn) {
        fn((int)layout, (int)transa, (int)transb,
           m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

/* SYMM - Symmetric matrix-matrix multiply */
static void openblas_ssymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, float alpha,
                           const float* a, int lda, const float* b, int ldb,
                           float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssymm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dsymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, double alpha,
                           const double* a, int lda, const double* b, int ldb,
                           double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsymm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_csymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                           const fb_complex_float_t beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csymm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

static void openblas_zsymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                           const fb_complex_double_t beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsymm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

/* HEMM - Hermitian matrix-matrix multiply */
static void openblas_chemm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                           const fb_complex_float_t beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_chemm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

static void openblas_zhemm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                           const fb_complex_double_t beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zhemm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

/* SYRK - Symmetric rank-k update */
static void openblas_ssyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha,
                           const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyrk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_dsyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha,
                           const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyrk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_csyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t alpha,
                           const fb_complex_float_t* a, int lda, const fb_complex_float_t beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csyrk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, &beta, c, ldc);
    }
}

static void openblas_zsyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t alpha,
                           const fb_complex_double_t* a, int lda, const fb_complex_double_t beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsyrk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, &beta, c, ldc);
    }
}

/* HERK - Hermitian rank-k update */
static void openblas_cherk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha,
                           const fb_complex_float_t* a, int lda, float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cherk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void openblas_zherk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha,
                           const fb_complex_double_t* a, int lda, double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zherk");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

/* SYR2K - Symmetric rank-2k update */
static void openblas_ssyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha,
                            const float* a, int lda, const float* b, int ldb,
                            float beta, float* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ssyr2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_dsyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha,
                            const double* a, int lda, const double* b, int ldb,
                            double beta, double* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dsyr2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_csyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t alpha,
                            const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                            const fb_complex_float_t beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_csyr2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

static void openblas_zsyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t alpha,
                            const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                            const fb_complex_double_t beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zsyr2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }
}

/* HER2K - Hermitian rank-2k update */
static void openblas_cher2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t alpha,
                            const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                            float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_cher2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void openblas_zher2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t alpha,
                            const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                            double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(int, int, int, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_zher2k");
    if (fn) {
        fn((int)layout, (int)uplo, (int)trans, n, k, &alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* TRMM - Triangular matrix-matrix multiply */
static void openblas_strmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strmm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_dtrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrmm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ctrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           const fb_complex_float_t alpha, const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrmm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, &alpha, a, lda, b, ldb);
    }
}

static void openblas_ztrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           const fb_complex_double_t alpha, const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrmm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, &alpha, a, lda, b, ldb);
    }
}

/* TRSM - Triangular solve multiple right-hand sides */
static void openblas_strsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_strsm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_dtrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_dtrsm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, alpha, a, lda, b, ldb);
    }
}

static void openblas_ctrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           const fb_complex_float_t alpha, const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ctrsm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, &alpha, a, lda, b, ldb);
    }
}

static void openblas_ztrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n,
                           const fb_complex_double_t alpha, const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(int, int, int, int, int, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "cblas_ztrsm");
    if (fn) {
        fn((int)layout, (int)side, (int)uplo, (int)transa, (int)diag,
           m, n, &alpha, a, lda, b, ldb);
    }
}

/* LAPACK operations */

static int layout_to_lapack(char layout) {
    return (layout == 'R' || layout == 'r') ? 101 : 102; /* 101=Row-major, 102=Col-major */
}

/* GESV - General linear system solve */
static int openblas_sgesv(const fb_layout_t layout, const int n, const int nrhs, float* a, const int lda, int* ipiv, float* b, const int ldb) {
    typedef int (*fn_t)(int, int, int, float*, int, int*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgesv");
    if (fn) { return fn((int)layout, n, nrhs, a, lda, ipiv, b, ldb); }
    return -1;
}

static int openblas_dgesv(const fb_layout_t layout, const int n, const int nrhs, double* a, const int lda, int* ipiv, double* b, const int ldb) {
    typedef int (*fn_t)(int, int, int, double*, int, int*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgesv");
    if (fn) { return fn((int)layout, n, nrhs, a, lda, ipiv, b, ldb); }
    return -1;
}

static int openblas_cgesv(const fb_layout_t layout, const int n, const int nrhs, fb_complex_float_t* a, const int lda, int* ipiv, fb_complex_float_t* b, const int ldb) {
    typedef int (*fn_t)(int, int, int, void*, int, int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgesv");
    if (fn) { return fn((int)layout, n, nrhs, a, lda, ipiv, b, ldb); }
    return -1;
}

static int openblas_zgesv(const fb_layout_t layout, const int n, const int nrhs, fb_complex_double_t* a, const int lda, int* ipiv, fb_complex_double_t* b, const int ldb) {
    typedef int (*fn_t)(int, int, int, void*, int, int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgesv");
    if (fn) { return fn((int)layout, n, nrhs, a, lda, ipiv, b, ldb); }
    return -1;
}

/* GETRF - LU factorization */
static int openblas_sgetrf(const fb_layout_t layout, const int m, const int n, float* a, const int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, float*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetrf");
    if (fn) { return fn((int)layout, m, n, a, lda, ipiv); }
    return -1;
}

static int openblas_dgetrf(const fb_layout_t layout, const int m, const int n, double* a, const int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, double*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetrf");
    if (fn) { return fn((int)layout, m, n, a, lda, ipiv); }
    return -1;
}

static int openblas_cgetrf(const fb_layout_t layout, const int m, const int n, fb_complex_float_t* a, const int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, void*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetrf");
    if (fn) { return fn((int)layout, m, n, a, lda, ipiv); }
    return -1;
}

static int openblas_zgetrf(const fb_layout_t layout, const int m, const int n, fb_complex_double_t* a, const int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, void*, int, int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetrf");
    if (fn) { return fn((int)layout, m, n, a, lda, ipiv); }
    return -1;
}

/* GETRS - Solve using LU factorization */
static int openblas_sgetrs(const fb_layout_t layout, const fb_transpose_t trans, const int n, const int nrhs, const float* a, const int lda, const int* ipiv, float* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, const int*, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetrs");
    if (fn) {
        char tc = ((int)trans == 111) ? 'N' : ((int)trans == 112) ? 'T' : 'C';
        return fn((int)layout, tc, n, nrhs, a, lda, ipiv, b, ldb);
    }
    return -1;
}

static int openblas_dgetrs(const fb_layout_t layout, const fb_transpose_t trans, const int n, const int nrhs, const double* a, const int lda, const int* ipiv, double* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, const int*, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetrs");
    if (fn) {
        char tc = ((int)trans == 111) ? 'N' : ((int)trans == 112) ? 'T' : 'C';
        return fn((int)layout, tc, n, nrhs, a, lda, ipiv, b, ldb);
    }
    return -1;
}

static int openblas_cgetrs(const fb_layout_t layout, const fb_transpose_t trans, const int n, const int nrhs, const fb_complex_float_t* a, const int lda, const int* ipiv, fb_complex_float_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, const int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetrs");
    if (fn) {
        char tc = ((int)trans == 111) ? 'N' : ((int)trans == 112) ? 'T' : 'C';
        return fn((int)layout, tc, n, nrhs, a, lda, ipiv, b, ldb);
    }
    return -1;
}

static int openblas_zgetrs(const fb_layout_t layout, const fb_transpose_t trans, const int n, const int nrhs, const fb_complex_double_t* a, const int lda, const int* ipiv, fb_complex_double_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, const int*, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetrs");
    if (fn) {
        char tc = ((int)trans == 111) ? 'N' : ((int)trans == 112) ? 'T' : 'C';
        return fn((int)layout, tc, n, nrhs, a, lda, ipiv, b, ldb);
    }
    return -1;
}

/* GETRI - Matrix inversion using LU */
static int openblas_sgetri(const fb_layout_t layout, const int n, float* a, const int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, float*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgetri");
    if (fn) { return fn((int)layout, n, a, lda, ipiv); }
    return -1;
}

static int openblas_dgetri(const fb_layout_t layout, const int n, double* a, const int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, double*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgetri");
    if (fn) { return fn((int)layout, n, a, lda, ipiv); }
    return -1;
}

static int openblas_cgetri(const fb_layout_t layout, const int n, fb_complex_float_t* a, const int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, void*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgetri");
    if (fn) { return fn((int)layout, n, a, lda, ipiv); }
    return -1;
}

static int openblas_zgetri(const fb_layout_t layout, const int n, fb_complex_double_t* a, const int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, void*, int, const int*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgetri");
    if (fn) { return fn((int)layout, n, a, lda, ipiv); }
    return -1;
}

/* POSV - Positive-definite linear system solve */
static int openblas_sposv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, float* a, const int lda, float* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sposv");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_dposv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, double* a, const int lda, double* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dposv");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_cposv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, fb_complex_float_t* a, const int lda, fb_complex_float_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cposv");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_zposv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, fb_complex_double_t* a, const int lda, fb_complex_double_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zposv");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

/* POTRF - Cholesky factorization */
static int openblas_spotrf(const fb_layout_t layout, const fb_uplo_t uplo, const int n, float* a, const int lda) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotrf");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_dpotrf(const fb_layout_t layout, const fb_uplo_t uplo, const int n, double* a, const int lda) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotrf");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_cpotrf(const fb_layout_t layout, const fb_uplo_t uplo, const int n, fb_complex_float_t* a, const int lda) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotrf");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_zpotrf(const fb_layout_t layout, const fb_uplo_t uplo, const int n, fb_complex_double_t* a, const int lda) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotrf");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

/* POTRS - Solve using Cholesky factorization */
static int openblas_spotrs(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, const float* a, const int lda, float* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotrs");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_dpotrs(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, const double* a, const int lda, double* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotrs");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_cpotrs(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, const fb_complex_float_t* a, const int lda, fb_complex_float_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotrs");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

static int openblas_zpotrs(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int nrhs, const fb_complex_double_t* a, const int lda, fb_complex_double_t* b, const int ldb) {
    typedef int (*fn_t)(int, char, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotrs");
    if (fn) {
        char uc = ((int)uplo == 121) ? 'U' : 'L';
        return fn((int)layout, uc, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}

/* POTRI - Matrix inversion using Cholesky */
static int openblas_spotri(const fb_layout_t layout, const fb_uplo_t uplo, const int n, float* a, const int lda) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_spotri");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_dpotri(const fb_layout_t layout, const fb_uplo_t uplo, const int n, double* a, const int lda) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dpotri");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_cpotri(const fb_layout_t layout, const fb_uplo_t uplo, const int n, fb_complex_float_t* a, const int lda) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cpotri");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

static int openblas_zpotri(const fb_layout_t layout, const fb_uplo_t uplo, const int n, fb_complex_double_t* a, const int lda) {
    typedef int (*fn_t)(int, char, int, void*, int);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zpotri");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, uc, n, a, lda); }
    return -1;
}

/* GEQRF - QR factorization */
static int openblas_sgeqrf(const fb_layout_t layout, const int m, const int n, float* a, const int lda, float* tau) {
    typedef int (*fn_t)(int, int, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgeqrf");
    if (fn) { return fn((int)layout, m, n, a, lda, tau); }
    return -1;
}

static int openblas_dgeqrf(const fb_layout_t layout, const int m, const int n, double* a, const int lda, double* tau) {
    typedef int (*fn_t)(int, int, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgeqrf");
    if (fn) { return fn((int)layout, m, n, a, lda, tau); }
    return -1;
}

static int openblas_cgeqrf(const fb_layout_t layout, const int m, const int n, fb_complex_float_t* a, const int lda, fb_complex_float_t* tau) {
    typedef int (*fn_t)(int, int, int, void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgeqrf");
    if (fn) { return fn((int)layout, m, n, a, lda, tau); }
    return -1;
}

static int openblas_zgeqrf(const fb_layout_t layout, const int m, const int n, fb_complex_double_t* a, const int lda, fb_complex_double_t* tau) {
    typedef int (*fn_t)(int, int, int, void*, int, void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgeqrf");
    if (fn) { return fn((int)layout, m, n, a, lda, tau); }
    return -1;
}

/* ORGQR/UNGQR - Generate Q from QR factorization */
static int openblas_sorgqr(const fb_layout_t layout, const int m, const int n, const int k, float* a, const int lda, const float* tau) {
    typedef int (*fn_t)(int, int, int, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sorgqr");
    if (fn) { return fn((int)layout, m, n, k, a, lda, tau); }
    return -1;
}

static int openblas_dorgqr(const fb_layout_t layout, const int m, const int n, const int k, double* a, const int lda, const double* tau) {
    typedef int (*fn_t)(int, int, int, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dorgqr");
    if (fn) { return fn((int)layout, m, n, k, a, lda, tau); }
    return -1;
}

static int openblas_cungqr(const fb_layout_t layout, const int m, const int n, const int k, fb_complex_float_t* a, const int lda, const fb_complex_float_t* tau) {
    typedef int (*fn_t)(int, int, int, int, void*, int, const void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cungqr");
    if (fn) { return fn((int)layout, m, n, k, a, lda, tau); }
    return -1;
}

static int openblas_zungqr(const fb_layout_t layout, const int m, const int n, const int k, fb_complex_double_t* a, const int lda, const fb_complex_double_t* tau) {
    typedef int (*fn_t)(int, int, int, int, void*, int, const void*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zungqr");
    if (fn) { return fn((int)layout, m, n, k, a, lda, tau); }
    return -1;
}

/* GESVD - Singular value decomposition */
static int openblas_sgesvd(const fb_layout_t layout, const char jobu, const char jobvt,
                           const int m, const int n, float* a, const int lda,
                           float* s, float* u, const int ldu, float* vt, const int ldvt, float* superb) {
    typedef int (*fn_t)(int, char, char, int, int, float*, int, float*, float*, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_sgesvd");
    if (fn) { return fn((int)layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb); }
    return -1;
}

static int openblas_dgesvd(const fb_layout_t layout, const char jobu, const char jobvt,
                           const int m, const int n, double* a, const int lda,
                           double* s, double* u, const int ldu, double* vt, const int ldvt, double* superb) {
    typedef int (*fn_t)(int, char, char, int, int, double*, int, double*, double*, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dgesvd");
    if (fn) { return fn((int)layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb); }
    return -1;
}

static int openblas_cgesvd(const fb_layout_t layout, const char jobu, const char jobvt,
                           const int m, const int n, fb_complex_float_t* a, const int lda,
                           float* s, fb_complex_float_t* u, const int ldu,
                           fb_complex_float_t* vt, const int ldvt, float* superb) {
    typedef int (*fn_t)(int, char, char, int, int, void*, int, float*, void*, int, void*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cgesvd");
    if (fn) { return fn((int)layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb); }
    return -1;
}

static int openblas_zgesvd(const fb_layout_t layout, const char jobu, const char jobvt,
                           const int m, const int n, fb_complex_double_t* a, const int lda,
                           double* s, fb_complex_double_t* u, const int ldu,
                           fb_complex_double_t* vt, const int ldvt, double* superb) {
    typedef int (*fn_t)(int, char, char, int, int, void*, int, double*, void*, int, void*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zgesvd");
    if (fn) { return fn((int)layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb); }
    return -1;
}

/* SYEV/HEEV - Eigenvalue decomposition */
static int openblas_ssyev(const fb_layout_t layout, const char jobz, const fb_uplo_t uplo, const int n, float* a, const int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, float*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_ssyev");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, jobz, uc, n, a, lda, w); }
    return -1;
}

static int openblas_dsyev(const fb_layout_t layout, const char jobz, const fb_uplo_t uplo, const int n, double* a, const int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, double*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_dsyev");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, jobz, uc, n, a, lda, w); }
    return -1;
}

static int openblas_cheev(const fb_layout_t layout, const char jobz, const fb_uplo_t uplo, const int n, fb_complex_float_t* a, const int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, void*, int, float*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_cheev");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, jobz, uc, n, a, lda, w); }
    return -1;
}

static int openblas_zheev(const fb_layout_t layout, const char jobz, const fb_uplo_t uplo, const int n, fb_complex_double_t* a, const int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, void*, int, double*);
    fn_t fn = (fn_t)FB_GET_PROC_ADDRESS(g_openblas.handle, "LAPACKE_zheev");
    if (fn) { char uc = ((int)uplo == 121) ? 'U' : 'L'; return fn((int)layout, jobz, uc, n, a, lda, w); }
    return -1;
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
    /* Lazily populate ext_ops[] via dlsym for LAPACK and extended BLAS.
     * Only NULL slots are filled — typed wrappers (BLAS L1/L2/L3) win.
     * Guard is benign for single-threaded init; backends are not called
     * concurrently until after all initialization completes. */
    static bool g_ext_ops_populated = false;
    if (!g_ext_ops_populated) {
        fb_auto_populate_ext_ops(&g_openblas_vtable, g_openblas.handle,
                                 k_lapacke_symbols, k_lapacke_symbols_count);
        g_ext_ops_populated = true;
    }
    return &g_openblas_vtable;
}
