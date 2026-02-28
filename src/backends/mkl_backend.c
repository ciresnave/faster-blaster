/**
 * @file mkl_backend.c
 * @brief Intel MKL backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "mkl_backend.h"
#include "backend_auto_detect.h"     /* fb_auto_populate_ext_ops       */
#include "sym_tables/sym_tables.h"     /* k_lapacke_symbols, k_cblas_ext_symbols, k_mkl_ext_symbols */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Define CBLAS enums only if not already defined by MKL headers */
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

/* MKL uses CBLAS interface - same signatures as OpenBLAS */
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

typedef void (*cblas_sgemv_t)(const int Order, const int TransA, const int M, const int N,
                               const float alpha, const float* A, const int lda,
                               const float* X, const int incX, const float beta,
                               float* Y, const int incY);
typedef void (*cblas_dgemv_t)(const int Order, const int TransA, const int M, const int N,
                               const double alpha, const double* A, const int lda,
                               const double* X, const int incX, const double beta,
                               double* Y, const int incY);
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

/* MKL-specific functions */
typedef void (*mkl_set_num_threads_t)(int num_threads);
typedef int (*mkl_get_max_threads_t)(void);
typedef void (*mkl_get_version_t)(int* major, int* minor, int* update);
typedef int (*mkl_set_threading_layer_t)(int layer);
typedef void (*mkl_verbose_t)(int enable);

/* Global MKL API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    
    /* MKL control functions */
    mkl_set_num_threads_t set_num_threads;
    mkl_get_max_threads_t get_max_threads;
    mkl_get_version_t get_version;
    mkl_set_threading_layer_t set_threading_layer;
    mkl_verbose_t verbose;
    
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
    
    /* CBLAS Level 3 */
    cblas_sgemm_t sgemm;
    cblas_dgemm_t dgemm;
    
} g_mkl;

/* MKL threading layer constants */
enum MKL_THREADING_LAYER {
    MKL_THREADING_INTEL = 0,
    MKL_THREADING_SEQUENTIAL = 1,
    MKL_THREADING_TBB = 2,
    MKL_THREADING_GNU = 3
};

static int load_mkl_library(void) {
    if (g_mkl.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        "mkl_rt.2.dll",
        "mkl_rt.dll"
#elif defined(__APPLE__)
        "libmkl_rt.dylib",
        "libmkl_rt.2.dylib",
        "/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.dylib"
#else
        "libmkl_rt.so",
        "libmkl_rt.so.2",
        "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_mkl.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_mkl.handle) {
            break;
        }
    }
    
    if (!g_mkl.handle) {
        return -1;
    }
    
    /* Load MKL control functions */
    g_mkl.set_num_threads = (mkl_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_mkl.handle, "mkl_set_num_threads");
    g_mkl.get_max_threads = (mkl_get_max_threads_t)
        FB_GET_PROC_ADDRESS(g_mkl.handle, "mkl_get_max_threads");
    g_mkl.get_version = (mkl_get_version_t)
        FB_GET_PROC_ADDRESS(g_mkl.handle, "mkl_get_version");
    g_mkl.set_threading_layer = (mkl_set_threading_layer_t)
        FB_GET_PROC_ADDRESS(g_mkl.handle, "mkl_set_threading_layer");
    g_mkl.verbose = (mkl_verbose_t)
        FB_GET_PROC_ADDRESS(g_mkl.handle, "mkl_verbose");
    
    /* Load CBLAS functions (same names as OpenBLAS) */
    g_mkl.sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sasum");
    g_mkl.dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dasum");
    g_mkl.saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_saxpy");
    g_mkl.daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_daxpy");
    g_mkl.sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sdot");
    g_mkl.ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_ddot");
    g_mkl.scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_scopy");
    g_mkl.dcopy = (cblas_dcopy_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dcopy");
    g_mkl.sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sscal");
    g_mkl.dscal = (cblas_dscal_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dscal");
    g_mkl.snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_snrm2");
    g_mkl.dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dnrm2");
    g_mkl.sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sswap");
    g_mkl.dswap = (cblas_dswap_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dswap");
    g_mkl.isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_isamax");
    g_mkl.idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_idamax");
    
    g_mkl.sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sgemv");
    g_mkl.dgemv = (cblas_dgemv_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dgemv");
    
    g_mkl.sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_sgemm");
    g_mkl.dgemm = (cblas_dgemm_t)FB_GET_PROC_ADDRESS(g_mkl.handle, "cblas_dgemm");
    
    g_mkl.initialized = true;
    return 0;
}

bool fb_mkl_is_available(void) {
    return load_mkl_library() == 0;
}

int fb_mkl_init(void) {
    return load_mkl_library();
}

void fb_mkl_shutdown(void) {
    if (g_mkl.initialized && g_mkl.handle) {
        FB_FREE_LIBRARY(g_mkl.handle);
        g_mkl.handle = NULL;
        g_mkl.initialized = false;
    }
}

int fb_mkl_get_version(int* major, int* minor, int* update) {
    if (!g_mkl.initialized || !g_mkl.get_version) {
        return -1;
    }
    g_mkl.get_version(major, minor, update);
    return 0;
}

void fb_mkl_set_num_threads(int num_threads) {
    if (g_mkl.set_num_threads) {
        g_mkl.set_num_threads(num_threads);
    }
}

int fb_mkl_get_num_threads(void) {
    if (g_mkl.get_max_threads) {
        return g_mkl.get_max_threads();
    }
    return 1;
}

int fb_mkl_set_threading_layer(const char* layer) {
    if (!g_mkl.set_threading_layer) {
        return -1;
    }
    
    int mkl_layer = MKL_THREADING_INTEL;
    if (strcmp(layer, "sequential") == 0) {
        mkl_layer = MKL_THREADING_SEQUENTIAL;
    } else if (strcmp(layer, "tbb") == 0) {
        mkl_layer = MKL_THREADING_TBB;
    } else if (strcmp(layer, "gnu") == 0) {
        mkl_layer = MKL_THREADING_GNU;
    }
    
    return g_mkl.set_threading_layer(mkl_layer);
}

void fb_mkl_set_verbose(bool enable) {
    if (g_mkl.verbose) {
        g_mkl.verbose(enable ? 1 : 0);
    }
}

/* Backend property functions for unified interface */

static uint32_t mkl_get_capabilities(void* handle) {
    (void)handle;
    return FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

static int mkl_get_num_threads_vtable(void* handle) {
    (void)handle;
    return fb_mkl_get_num_threads();
}

static void mkl_set_num_threads_vtable(void* handle, int num_threads) {
    (void)handle;
    fb_mkl_set_num_threads(num_threads);
}

/* Wrapper functions (nearly identical to OpenBLAS wrappers) */

static float mkl_sasum(const int n, const float* x, const int incx) {
    if (g_mkl.sasum) {
        return g_mkl.sasum(n, x, incx);
    }
    return 0.0f;
}

static double mkl_dasum(const int n, const double* x, const int incx) {
    if (g_mkl.dasum) {
        return g_mkl.dasum(n, x, incx);
    }
    return 0.0;
}

static void mkl_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_mkl.saxpy) {
        g_mkl.saxpy(n, alpha, x, incx, y, incy);
    }
}

static void mkl_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_mkl.daxpy) {
        g_mkl.daxpy(n, alpha, x, incx, y, incy);
    }
}

static float mkl_sdot(const int n, const float* x, const int incx, const float* y, const int incy) {
    if (g_mkl.sdot) {
        return g_mkl.sdot(n, x, incx, y, incy);
    }
    return 0.0f;
}

static double mkl_ddot(const int n, const double* x, const int incx, const double* y, const int incy) {
    if (g_mkl.ddot) {
        return g_mkl.ddot(n, x, incx, y, incy);
    }
    return 0.0;
}

static void mkl_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_mkl.scopy) {
        g_mkl.scopy(n, x, incx, y, incy);
    }
}

static void mkl_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_mkl.dcopy) {
        g_mkl.dcopy(n, x, incx, y, incy);
    }
}

static void mkl_sscal(int n, float alpha, float* x, int incx) {
    if (g_mkl.sscal) {
        g_mkl.sscal(n, alpha, x, incx);
    }
}

static void mkl_dscal(int n, double alpha, double* x, int incx) {
    if (g_mkl.dscal) {
        g_mkl.dscal(n, alpha, x, incx);
    }
}

static float mkl_snrm2(const int n, const float* x, const int incx) {
    if (g_mkl.snrm2) {
        return g_mkl.snrm2(n, x, incx);
    }
    return 0.0f;
}

static double mkl_dnrm2(const int n, const double* x, const int incx) {
    if (g_mkl.dnrm2) {
        return g_mkl.dnrm2(n, x, incx);
    }
    return 0.0;
}

static void mkl_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_mkl.sswap) {
        g_mkl.sswap(n, x, incx, y, incy);
    }
}

static void mkl_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_mkl.dswap) {
        g_mkl.dswap(n, x, incx, y, incy);
    }
}

static int mkl_isamax(const int n, const float* x, const int incx) {
    if (g_mkl.isamax) {
        return g_mkl.isamax(n, x, incx);
    }
    return -1;
}

static int mkl_idamax(const int n, const double* x, const int incx) {
    if (g_mkl.idamax) {
        return g_mkl.idamax(n, x, incx);
    }
    return -1;
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

static void mkl_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void mkl_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void mkl_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void mkl_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void mkl_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void mkl_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void mkl_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void mkl_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void mkl_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void mkl_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void mkl_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void mkl_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void mkl_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void mkl_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void mkl_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void mkl_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void mkl_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void mkl_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void mkl_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void mkl_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void mkl_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void mkl_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void mkl_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void mkl_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void mkl_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void mkl_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void mkl_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void mkl_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Level 2 BLAS operations */

static void mkl_sgemv(const fb_layout_t layout, const fb_transpose_t trans,
                      const int m, const int n, const float alpha,
                      const float* a, const int lda, const float* x, const int incx,
                      const float beta, float* y, const int incy) {
    if (g_mkl.sgemv) {
        g_mkl.sgemv((int)layout, (int)trans,
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void mkl_dgemv(const fb_layout_t layout, const fb_transpose_t trans,
                      const int m, const int n, const double alpha,
                      const double* a, const int lda, const double* x, const int incx,
                      const double beta, double* y, const int incy) {
    if (g_mkl.dgemv) {
        g_mkl.dgemv((int)layout, (int)trans,
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void mkl_sgemm(const fb_layout_t layout,
                      const fb_transpose_t transa, const fb_transpose_t transb,
                      const int m, const int n, const int k,
                      const float alpha, const float* a, const int lda,
                      const float* b, const int ldb, const float beta,
                      float* c, const int ldc) {
    if (g_mkl.sgemm) {
        g_mkl.sgemm((int)layout, (int)transa, (int)transb,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void mkl_dgemm(const fb_layout_t layout,
                      const fb_transpose_t transa, const fb_transpose_t transb,
                      const int m, const int n, const int k,
                      const double alpha, const double* a, const int lda,
                      const double* b, const int ldb, const double beta,
                      double* c, const int ldc) {
    if (g_mkl.dgemm) {
        g_mkl.dgemm((int)layout, (int)transa, (int)transb,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* Level 2 BLAS - Complex GEMV */
static void mkl_cgemv(char trans, int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_zgemv(char trans, int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - GBMV (banded matrix-vector) */
static void mkl_sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_sgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_cgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_zgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HEMV (Hermitian matrix-vector) */
static void mkl_chemv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_zhemv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HBMV (Hermitian banded matrix-vector) */
static void mkl_chbmv(char uplo, int n, int k, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_zhbmv(char uplo, int n, int k, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HPMV (Hermitian packed matrix-vector) */
static void mkl_chpmv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* ap,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void mkl_zhpmv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* ap,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SYMV (symmetric matrix-vector) */
static void mkl_ssymv(char uplo, int n, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_dsymv(char uplo, int n, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_csymv(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_zsymv(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SBMV (symmetric banded matrix-vector) */
static void mkl_ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void mkl_dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SPMV (symmetric packed matrix-vector) */
static void mkl_sspmv(char uplo, int n, float alpha, const float* ap,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_sspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void mkl_dspmv(char uplo, int n, double alpha, const double* ap,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - TRMV (triangular matrix-vector) */
static void mkl_strmv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_strmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_ctrmv(char uplo, char trans, char diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_ztrmv(char uplo, char trans, char diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBMV (triangular banded matrix-vector) */
static void mkl_stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_stbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_ctbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_ztbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPMV (triangular packed matrix-vector) */
static void mkl_stpmv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_stpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_dtpmv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_ctpmv(char uplo, char trans, char diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_ztpmv(char uplo, char trans, char diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - TRSV (triangular solve) */
static void mkl_strsv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_strsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_ctrsv(char uplo, char trans, char diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void mkl_ztrsv(char uplo, char trans, char diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBSV (triangular banded solve) */
static void mkl_stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_stbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_ctbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void mkl_ztbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPSV (triangular packed solve) */
static void mkl_stpsv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_stpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_dtpsv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_ctpsv(char uplo, char trans, char diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void mkl_ztpsv(char uplo, char trans, char diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - GER (rank-1 update) */
static void mkl_sger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_sger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_cgeru(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_zgeru(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_cgerc(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_zgerc(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HER (Hermitian rank-1 update) */
static void mkl_cher(char uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void mkl_zher(char uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - HPR (Hermitian packed rank-1 update) */
static void mkl_chpr(char uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void mkl_zhpr(char uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - HER2 (Hermitian rank-2 update) */
static void mkl_cher2(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_zher2(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HPR2 (Hermitian packed rank-2 update) */
static void mkl_chpr2(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void mkl_zhpr2(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 2 BLAS - SYR (symmetric rank-1 update) */
static void mkl_ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void mkl_dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void mkl_csyr(char uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void mkl_zsyr(char uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - SPR (symmetric packed rank-1 update) */
static void mkl_sspr(char uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_sspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void mkl_dspr(char uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - SYR2 (symmetric rank-2 update) */
static void mkl_ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void mkl_dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - SPR2 (symmetric packed rank-2 update) */
static void mkl_sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_sspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void mkl_dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 3 BLAS - GEMM (complex) */
static void mkl_cgemm(char transa, char transb, int m, int n, int k, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_zgemm(char transa, char transb, int m, int n, int k, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYMM (symmetric matrix-matrix) */
static void mkl_ssymm(char side, char uplo, int m, int n, float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, float,
                         const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_dsymm(char side, char uplo, int m, int n, double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, double,
                         const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_csymm(char side, char uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_zsymm(char side, char uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HEMM (Hermitian matrix-matrix) */
static void mkl_chemm(char side, char uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_chemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_zhemm(char side, char uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zhemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYRK (symmetric rank-k update) */
static void mkl_ssyrk(char uplo, char trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void mkl_dsyrk(char uplo, char trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void mkl_csyrk(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void mkl_zsyrk(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - HERK (Hermitian rank-k update) */
static void mkl_cherk(char uplo, char trans, int n, int k, float alpha, const fb_complex_float_t* a, int lda, float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void mkl_zherk(char uplo, char trans, int n, int k, double alpha, const fb_complex_double_t* a, int lda, double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - SYR2K (symmetric rank-2k update) */
static void mkl_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float* a, int lda,
                       const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ssyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_dsyr2k(char uplo, char trans, int n, int k, double alpha, const double* a, int lda,
                       const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_csyr2k(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_csyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_zsyr2k(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HER2K (Hermitian rank-2k update) */
static void mkl_cher2k(char uplo, char trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_cher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void mkl_zher2k(char uplo, char trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_zher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - TRMM (triangular matrix-matrix) */
static void mkl_strmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_strmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_ctrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_ztrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* Level 3 BLAS - TRSM (triangular solve matrix) */
static void mkl_strsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_strsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_dtrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_ctrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ctrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void mkl_ztrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "cblas_ztrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* LAPACK - Linear system solve (GESV) */
static int mkl_sgesv(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb) {
    typedef int (*fn_t)(int, int, int, float*, int, int*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgesv");
    return fn ? fn(102, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_dgesv(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb) {
    typedef int (*fn_t)(int, int, int, double*, int, int*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgesv");
    return fn ? fn(102, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_cgesv(int n, int nrhs, fb_complex_float_t* a, int lda, int* ipiv, fb_complex_float_t* b, int ldb) {
    typedef int (*fn_t)(int, int, int, fb_complex_float_t*, int, int*, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgesv");
    return fn ? fn(102, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_zgesv(int n, int nrhs, fb_complex_double_t* a, int lda, int* ipiv, fb_complex_double_t* b, int ldb) {
    typedef int (*fn_t)(int, int, int, fb_complex_double_t*, int, int*, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgesv");
    return fn ? fn(102, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

/* LAPACK - LU factorization (GETRF) */
static int mkl_sgetrf(int m, int n, float* a, int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, float*, int, int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgetrf");
    return fn ? fn(102, m, n, a, lda, ipiv) : -1;
}

static int mkl_dgetrf(int m, int n, double* a, int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, double*, int, int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgetrf");
    return fn ? fn(102, m, n, a, lda, ipiv) : -1;
}

static int mkl_cgetrf(int m, int n, fb_complex_float_t* a, int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, fb_complex_float_t*, int, int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgetrf");
    return fn ? fn(102, m, n, a, lda, ipiv) : -1;
}

static int mkl_zgetrf(int m, int n, fb_complex_double_t* a, int lda, int* ipiv) {
    typedef int (*fn_t)(int, int, int, fb_complex_double_t*, int, int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgetrf");
    return fn ? fn(102, m, n, a, lda, ipiv) : -1;
}

/* LAPACK - Solve with LU factorization (GETRS) */
static int mkl_sgetrs(char trans, int n, int nrhs, const float* a, int lda, const int* ipiv, float* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, const int*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgetrs");
    return fn ? fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_dgetrs(char trans, int n, int nrhs, const double* a, int lda, const int* ipiv, double* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, const int*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgetrs");
    return fn ? fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_cgetrs(char trans, int n, int nrhs, const fb_complex_float_t* a, int lda, const int* ipiv, fb_complex_float_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const fb_complex_float_t*, int, const int*, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgetrs");
    return fn ? fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

static int mkl_zgetrs(char trans, int n, int nrhs, const fb_complex_double_t* a, int lda, const int* ipiv, fb_complex_double_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const fb_complex_double_t*, int, const int*, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgetrs");
    return fn ? fn(102, trans, n, nrhs, a, lda, ipiv, b, ldb) : -1;
}

/* LAPACK - Matrix inverse (GETRI) */
static int mkl_sgetri(int n, float* a, int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, float*, int, const int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgetri");
    return fn ? fn(102, n, a, lda, ipiv) : -1;
}

static int mkl_dgetri(int n, double* a, int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, double*, int, const int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgetri");
    return fn ? fn(102, n, a, lda, ipiv) : -1;
}

static int mkl_cgetri(int n, fb_complex_float_t* a, int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, fb_complex_float_t*, int, const int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgetri");
    return fn ? fn(102, n, a, lda, ipiv) : -1;
}

static int mkl_zgetri(int n, fb_complex_double_t* a, int lda, const int* ipiv) {
    typedef int (*fn_t)(int, int, fb_complex_double_t*, int, const int*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgetri");
    return fn ? fn(102, n, a, lda, ipiv) : -1;
}

/* LAPACK - Cholesky solve (POSV) */
static int mkl_sposv(char uplo, int n, int nrhs, float* a, int lda, float* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sposv");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_dposv(char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dposv");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_cposv(char uplo, int n, int nrhs, fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, fb_complex_float_t*, int, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cposv");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_zposv(char uplo, int n, int nrhs, fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, fb_complex_double_t*, int, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zposv");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

/* LAPACK - Cholesky factorization (POTRF) */
static int mkl_spotrf(char uplo, int n, float* a, int lda) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_spotrf");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_dpotrf(char uplo, int n, double* a, int lda) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dpotrf");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_cpotrf(char uplo, int n, fb_complex_float_t* a, int lda) {
    typedef int (*fn_t)(int, char, int, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cpotrf");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_zpotrf(char uplo, int n, fb_complex_double_t* a, int lda) {
    typedef int (*fn_t)(int, char, int, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zpotrf");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

/* LAPACK - Solve with Cholesky factorization (POTRS) */
static int mkl_spotrs(char uplo, int n, int nrhs, const float* a, int lda, float* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_spotrs");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_dpotrs(char uplo, int n, int nrhs, const double* a, int lda, double* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dpotrs");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_cpotrs(char uplo, int n, int nrhs, const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const fb_complex_float_t*, int, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cpotrs");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

static int mkl_zpotrs(char uplo, int n, int nrhs, const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef int (*fn_t)(int, char, int, int, const fb_complex_double_t*, int, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zpotrs");
    return fn ? fn(102, uplo, n, nrhs, a, lda, b, ldb) : -1;
}

/* LAPACK - Cholesky inverse (POTRI) */
static int mkl_spotri(char uplo, int n, float* a, int lda) {
    typedef int (*fn_t)(int, char, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_spotri");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_dpotri(char uplo, int n, double* a, int lda) {
    typedef int (*fn_t)(int, char, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dpotri");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_cpotri(char uplo, int n, fb_complex_float_t* a, int lda) {
    typedef int (*fn_t)(int, char, int, fb_complex_float_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cpotri");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

static int mkl_zpotri(char uplo, int n, fb_complex_double_t* a, int lda) {
    typedef int (*fn_t)(int, char, int, fb_complex_double_t*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zpotri");
    return fn ? fn(102, uplo, n, a, lda) : -1;
}

/* LAPACK - QR factorization (GEQRF) */
static int mkl_sgeqrf(int m, int n, float* a, int lda, float* tau) {
    typedef int (*fn_t)(int, int, int, float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgeqrf");
    return fn ? fn(102, m, n, a, lda, tau) : -1;
}

static int mkl_dgeqrf(int m, int n, double* a, int lda, double* tau) {
    typedef int (*fn_t)(int, int, int, double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgeqrf");
    return fn ? fn(102, m, n, a, lda, tau) : -1;
}

static int mkl_cgeqrf(int m, int n, fb_complex_float_t* a, int lda, fb_complex_float_t* tau) {
    typedef int (*fn_t)(int, int, int, fb_complex_float_t*, int, fb_complex_float_t*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgeqrf");
    return fn ? fn(102, m, n, a, lda, tau) : -1;
}

static int mkl_zgeqrf(int m, int n, fb_complex_double_t* a, int lda, fb_complex_double_t* tau) {
    typedef int (*fn_t)(int, int, int, fb_complex_double_t*, int, fb_complex_double_t*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgeqrf");
    return fn ? fn(102, m, n, a, lda, tau) : -1;
}

/* LAPACK - Generate Q from QR (ORGQR/UNGQR) */
static int mkl_sorgqr(int m, int n, int k, float* a, int lda, const float* tau) {
    typedef int (*fn_t)(int, int, int, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sorgqr");
    return fn ? fn(102, m, n, k, a, lda, tau) : -1;
}

static int mkl_dorgqr(int m, int n, int k, double* a, int lda, const double* tau) {
    typedef int (*fn_t)(int, int, int, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dorgqr");
    return fn ? fn(102, m, n, k, a, lda, tau) : -1;
}

static int mkl_cungqr(int m, int n, int k, fb_complex_float_t* a, int lda, const fb_complex_float_t* tau) {
    typedef int (*fn_t)(int, int, int, int, fb_complex_float_t*, int, const fb_complex_float_t*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cungqr");
    return fn ? fn(102, m, n, k, a, lda, tau) : -1;
}

static int mkl_zungqr(int m, int n, int k, fb_complex_double_t* a, int lda, const fb_complex_double_t* tau) {
    typedef int (*fn_t)(int, int, int, int, fb_complex_double_t*, int, const fb_complex_double_t*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zungqr");
    return fn ? fn(102, m, n, k, a, lda, tau) : -1;
}

/* LAPACK - SVD (GESVD) */
static int mkl_sgesvd(char jobu, char jobvt, int m, int n, float* a, int lda, float* s, float* u, int ldu, float* vt, int ldvt, float* superb) {
    typedef int (*fn_t)(int, char, char, int, int, float*, int, float*, float*, int, float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_sgesvd");
    return fn ? fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb) : -1;
}

static int mkl_dgesvd(char jobu, char jobvt, int m, int n, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* superb) {
    typedef int (*fn_t)(int, char, char, int, int, double*, int, double*, double*, int, double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dgesvd");
    return fn ? fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb) : -1;
}

static int mkl_cgesvd(char jobu, char jobvt, int m, int n, fb_complex_float_t* a, int lda, float* s,
                      fb_complex_float_t* u, int ldu, fb_complex_float_t* vt, int ldvt, float* superb) {
    typedef int (*fn_t)(int, char, char, int, int, fb_complex_float_t*, int, float*, fb_complex_float_t*, int, fb_complex_float_t*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cgesvd");
    return fn ? fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb) : -1;
}

static int mkl_zgesvd(char jobu, char jobvt, int m, int n, fb_complex_double_t* a, int lda, double* s,
                      fb_complex_double_t* u, int ldu, fb_complex_double_t* vt, int ldvt, double* superb) {
    typedef int (*fn_t)(int, char, char, int, int, fb_complex_double_t*, int, double*, fb_complex_double_t*, int, fb_complex_double_t*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zgesvd");
    return fn ? fn(102, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb) : -1;
}

/* LAPACK - Eigenvalues (SYEV/HEEV) */
static int mkl_ssyev(char jobz, char uplo, int n, float* a, int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_ssyev");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_dsyev(char jobz, char uplo, int n, double* a, int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dsyev");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_cheev(char jobz, char uplo, int n, fb_complex_float_t* a, int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, fb_complex_float_t*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cheev");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_zheev(char jobz, char uplo, int n, fb_complex_double_t* a, int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, fb_complex_double_t*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zheev");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

/* LAPACK - Eigenvalues divide-and-conquer (SYEVD/HEEVD) */
static int mkl_ssyevd(char jobz, char uplo, int n, float* a, int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_ssyevd");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_dsyevd(char jobz, char uplo, int n, double* a, int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_dsyevd");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_cheevd(char jobz, char uplo, int n, fb_complex_float_t* a, int lda, float* w) {
    typedef int (*fn_t)(int, char, char, int, fb_complex_float_t*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_cheevd");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

static int mkl_zheevd(char jobz, char uplo, int n, fb_complex_double_t* a, int lda, double* w) {
    typedef int (*fn_t)(int, char, char, int, fb_complex_double_t*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_mkl.handle, "LAPACKE_zheevd");
    return fn ? fn(102, jobz, uplo, n, a, lda, w) : -1;
}

/* MKL vtable */
static fb_backend_vtable_t g_mkl_vtable = {
    /* Level 1 BLAS */
    .sasum = mkl_sasum,
    .dasum = mkl_dasum,
    .saxpy = mkl_saxpy,
    .daxpy = mkl_daxpy,
    .sdot = mkl_sdot,
    .ddot = mkl_ddot,
    .scopy = mkl_scopy,
    .dcopy = mkl_dcopy,
    .sscal = mkl_sscal,
    .dscal = mkl_dscal,
    .snrm2 = mkl_snrm2,
    .dnrm2 = mkl_dnrm2,
    .sswap = mkl_sswap,
    .dswap = mkl_dswap,
    .isamax = mkl_isamax,
    .idamax = mkl_idamax,
    
    /* Level 2 BLAS */
    .sgemv = mkl_sgemv,
    .dgemv = mkl_dgemv,
    
    /* Level 3 BLAS */
    .sgemm = mkl_sgemm,
    .dgemm = mkl_dgemm,
    
    /* TODO: Add remaining operations */
    
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
    .get_capabilities = mkl_get_capabilities,
    .get_num_threads = mkl_get_num_threads_vtable,
    .set_num_threads = mkl_set_num_threads_vtable
};

const fb_backend_vtable_t* fb_mkl_get_vtable(void) {
    if (!fb_mkl_is_available()) {
        return NULL;
    }
    /* Lazily dlsym-populate ext_ops[] for all three MKL symbol ranges:
     *   LAPACKE_*   (LAPACK ops 157-1321)
     *   cblas_*     (extended batch ops 1634-1807)
     *   mkl_*       (MKL-specific extensions 1808-1850)
     * NULL-check inside fb_auto_populate_ext_ops ensures typed wrappers win. */
    static bool g_ext_ops_populated = false;
    if (!g_ext_ops_populated) {
        fb_auto_populate_ext_ops(&g_mkl_vtable, g_mkl.handle,
                                 k_lapacke_symbols,  k_lapacke_symbols_count);
        fb_auto_populate_ext_ops(&g_mkl_vtable, g_mkl.handle,
                                 k_cblas_ext_symbols, k_cblas_ext_symbols_count);
        fb_auto_populate_ext_ops(&g_mkl_vtable, g_mkl.handle,
                                 k_mkl_ext_symbols,  k_mkl_ext_symbols_count);
        g_ext_ops_populated = true;
    }
    return &g_mkl_vtable;
}
