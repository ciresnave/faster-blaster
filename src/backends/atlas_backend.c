/**
 * @file atlas_backend.c
 * @brief Intel atlas backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "atlas_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* atlas uses CBLAS interface - same signatures as OpenBLAS */
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

/* atlas-specific functions */
typedef void (*atlas_set_num_threads_t)(int num_threads);
typedef int (*atlas_get_max_threads_t)(void);
typedef void (*atlas_get_version_t)(int* major, int* minor, int* update);
typedef int (*atlas_set_threading_layer_t)(int layer);
typedef void (*atlas_verbose_t)(int enable);

/* Global atlas API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    
    /* atlas control functions */
    atlas_set_num_threads_t set_num_threads;
    atlas_get_max_threads_t get_max_threads;
    atlas_get_version_t get_version;
    atlas_set_threading_layer_t set_threading_layer;
    atlas_verbose_t verbose;
    
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
    
} g_atlas;

/* CBLAS constants (same as OpenBLAS) */
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

/* atlas threading layer constants */
enum atlas_THREADING_LAYER {
    atlas_THREADING_INTEL = 0,
    atlas_THREADING_SEQUENTIAL = 1,
    atlas_THREADING_TBB = 2,
    atlas_THREADING_GNU = 3
};

static int load_atlas_library(void) {
    if (g_atlas.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        "atlas_rt.2.dll",
        "atlas_rt.dll"
#elif defined(__APPLE__)
        "libatlas_rt.dylib",
        "libatlas_rt.2.dylib",
        "/opt/intel/oneapi/atlas/latest/lib/libatlas_rt.dylib"
#else
        "libatlas_rt.so",
        "libatlas_rt.so.2",
        "/opt/intel/oneapi/atlas/latest/lib/intel64/libatlas_rt.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_atlas.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_atlas.handle) {
            break;
        }
    }
    
    if (!g_atlas.handle) {
        return -1;
    }
    
    /* Load atlas control functions */
    g_atlas.set_num_threads = (atlas_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_atlas.handle, "atlas_set_num_threads");
    g_atlas.get_max_threads = (atlas_get_max_threads_t)
        FB_GET_PROC_ADDRESS(g_atlas.handle, "atlas_get_max_threads");
    g_atlas.get_version = (atlas_get_version_t)
        FB_GET_PROC_ADDRESS(g_atlas.handle, "atlas_get_version");
    g_atlas.set_threading_layer = (atlas_set_threading_layer_t)
        FB_GET_PROC_ADDRESS(g_atlas.handle, "atlas_set_threading_layer");
    g_atlas.verbose = (atlas_verbose_t)
        FB_GET_PROC_ADDRESS(g_atlas.handle, "atlas_verbose");
    
    /* Load CBLAS functions (same names as OpenBLAS) */
    g_atlas.sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sasum");
    g_atlas.dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dasum");
    g_atlas.saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_saxpy");
    g_atlas.daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_daxpy");
    g_atlas.sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sdot");
    g_atlas.ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_ddot");
    g_atlas.scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_scopy");
    g_atlas.dcopy = (cblas_dcopy_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dcopy");
    g_atlas.sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sscal");
    g_atlas.dscal = (cblas_dscal_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dscal");
    g_atlas.snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_snrm2");
    g_atlas.dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dnrm2");
    g_atlas.sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sswap");
    g_atlas.dswap = (cblas_dswap_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dswap");
    g_atlas.isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_isamax");
    g_atlas.idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_idamax");
    
    g_atlas.sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sgemv");
    g_atlas.dgemv = (cblas_dgemv_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dgemv");
    
    g_atlas.sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_sgemm");
    g_atlas.dgemm = (cblas_dgemm_t)FB_GET_PROC_ADDRESS(g_atlas.handle, "cblas_dgemm");
    
    g_atlas.initialized = true;
    return 0;
}

bool fb_atlas_is_available(void) {
    return load_atlas_library() == 0;
}

int fb_atlas_init(void) {
    return load_atlas_library();
}

void fb_atlas_shutdown(void) {
    if (g_atlas.initialized && g_atlas.handle) {
        FB_FREE_LIBRARY(g_atlas.handle);
        g_atlas.handle = NULL;
        g_atlas.initialized = false;
    }
}

int fb_atlas_get_version(int* major, int* minor, int* update) {
    if (!g_atlas.initialized || !g_atlas.get_version) {
        return -1;
    }
    g_atlas.get_version(major, minor, update);
    return 0;
}

void fb_atlas_set_num_threads(int num_threads) {
    if (g_atlas.set_num_threads) {
        g_atlas.set_num_threads(num_threads);
    }
}

int fb_atlas_get_num_threads(void) {
    if (g_atlas.get_max_threads) {
        return g_atlas.get_max_threads();
    }
    return 1;
}

int fb_atlas_set_threading_layer(const char* layer) {
    if (!g_atlas.set_threading_layer) {
        return -1;
    }
    
    int atlas_layer = atlas_THREADING_INTEL;
    if (strcmp(layer, "sequential") == 0) {
        atlas_layer = atlas_THREADING_SEQUENTIAL;
    } else if (strcmp(layer, "tbb") == 0) {
        atlas_layer = atlas_THREADING_TBB;
    } else if (strcmp(layer, "gnu") == 0) {
        atlas_layer = atlas_THREADING_GNU;
    }
    
    return g_atlas.set_threading_layer(atlas_layer);
}

void fb_atlas_set_verbose(bool enable) {
    if (g_atlas.verbose) {
        g_atlas.verbose(enable ? 1 : 0);
    }
}

/* Wrapper functions (nearly identical to OpenBLAS wrappers) */

static void atlas_sasum(int n, const float* x, int incx, float* result) {
    if (g_atlas.sasum) {
        *result = g_atlas.sasum(n, x, incx);
    }
}

static void atlas_dasum(int n, const double* x, int incx, double* result) {
    if (g_atlas.dasum) {
        *result = g_atlas.dasum(n, x, incx);
    }
}

static void atlas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_atlas.saxpy) {
        g_atlas.saxpy(n, alpha, x, incx, y, incy);
    }
}

static void atlas_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_atlas.daxpy) {
        g_atlas.daxpy(n, alpha, x, incx, y, incy);
    }
}

static void atlas_sdot(int n, const float* x, int incx, const float* y, int incy, float* result) {
    if (g_atlas.sdot) {
        *result = g_atlas.sdot(n, x, incx, y, incy);
    }
}

static void atlas_ddot(int n, const double* x, int incx, const double* y, int incy, double* result) {
    if (g_atlas.ddot) {
        *result = g_atlas.ddot(n, x, incx, y, incy);
    }
}

static void atlas_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_atlas.scopy) {
        g_atlas.scopy(n, x, incx, y, incy);
    }
}

static void atlas_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_atlas.dcopy) {
        g_atlas.dcopy(n, x, incx, y, incy);
    }
}

static void atlas_sscal(int n, float alpha, float* x, int incx) {
    if (g_atlas.sscal) {
        g_atlas.sscal(n, alpha, x, incx);
    }
}

static void atlas_dscal(int n, double alpha, double* x, int incx) {
    if (g_atlas.dscal) {
        g_atlas.dscal(n, alpha, x, incx);
    }
}

static void atlas_snrm2(int n, const float* x, int incx, float* result) {
    if (g_atlas.snrm2) {
        *result = g_atlas.snrm2(n, x, incx);
    }
}

static void atlas_dnrm2(int n, const double* x, int incx, double* result) {
    if (g_atlas.dnrm2) {
        *result = g_atlas.dnrm2(n, x, incx);
    }
}

static void atlas_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_atlas.sswap) {
        g_atlas.sswap(n, x, incx, y, incy);
    }
}

static void atlas_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_atlas.dswap) {
        g_atlas.dswap(n, x, incx, y, incy);
    }
}

static void atlas_isamax(int n, const float* x, int incx, int* result) {
    if (g_atlas.isamax) {
        *result = g_atlas.isamax(n, x, incx);
    }
}

static void atlas_idamax(int n, const double* x, int incx, int* result) {
    if (g_atlas.idamax) {
        *result = g_atlas.idamax(n, x, incx);
    }
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

static void atlas_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void atlas_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void atlas_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void atlas_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void atlas_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void atlas_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void atlas_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void atlas_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void atlas_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void atlas_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void atlas_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void atlas_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void atlas_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void atlas_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void atlas_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void atlas_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void atlas_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void atlas_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void atlas_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void atlas_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void atlas_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void atlas_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void atlas_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void atlas_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void atlas_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void atlas_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void atlas_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void atlas_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Level 2 BLAS operations */

static void atlas_sgemv(char trans, int m, int n, float alpha,
                      const float* a, int lda, const float* x, int incx,
                      float beta, float* y, int incy) {
    if (g_atlas.sgemv) {
        g_atlas.sgemv(CblasColMajor, transpose_to_cblas(trans),
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void atlas_dgemv(char trans, int m, int n, double alpha,
                      const double* a, int lda, const double* x, int incx,
                      double beta, double* y, int incy) {
    if (g_atlas.dgemv) {
        g_atlas.dgemv(CblasColMajor, transpose_to_cblas(trans),
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void atlas_sgemm(char transa, char transb, int m, int n, int k,
                      float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta,
                      float* c, int ldc) {
    if (g_atlas.sgemm) {
        g_atlas.sgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void atlas_dgemm(char transa, char transb, int m, int n, int k,
                      double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta,
                      double* c, int ldc) {
    if (g_atlas.dgemm) {
        g_atlas.dgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* Level 2 BLAS - Complex GEMV */
static void atlas_cgemv(char trans, int m, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_zgemv(char trans, int m, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - GBMV (banded matrix-vector) */
static void atlas_sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_sgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_cgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_zgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HEMV (Hermitian matrix-vector) */
static void atlas_chemv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_zhemv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HBMV (Hermitian banded matrix-vector) */
static void atlas_chbmv(char uplo, int n, int k, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_zhbmv(char uplo, int n, int k, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HPMV (Hermitian packed matrix-vector) */
static void atlas_chpmv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* ap,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void atlas_zhpmv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* ap,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SYMV (symmetric matrix-vector) */
static void atlas_ssymv(char uplo, int n, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_dsymv(char uplo, int n, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_csymv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_zsymv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SBMV (symmetric banded matrix-vector) */
static void atlas_ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void atlas_dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SPMV (symmetric packed matrix-vector) */
static void atlas_sspmv(char uplo, int n, float alpha, const float* ap,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_sspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void atlas_dspmv(char uplo, int n, double alpha, const double* ap,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - TRMV (triangular matrix-vector) */
static void atlas_strmv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_strmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_ctrmv(char uplo, char trans, char diag, int n, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_ztrmv(char uplo, char trans, char diag, int n, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBMV (triangular banded matrix-vector) */
static void atlas_stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_stbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_ctbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_ztbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPMV (triangular packed matrix-vector) */
static void atlas_stpmv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_stpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_dtpmv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_ctpmv(char uplo, char trans, char diag, int n, const fb_complex_float* ap, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_ztpmv(char uplo, char trans, char diag, int n, const fb_complex_double* ap, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - TRSV (triangular solve) */
static void atlas_strsv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_strsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_ctrsv(char uplo, char trans, char diag, int n, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void atlas_ztrsv(char uplo, char trans, char diag, int n, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBSV (triangular banded solve) */
static void atlas_stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_stbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_ctbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void atlas_ztbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPSV (triangular packed solve) */
static void atlas_stpsv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_stpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_dtpsv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_ctpsv(char uplo, char trans, char diag, int n, const fb_complex_float* ap, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void atlas_ztpsv(char uplo, char trans, char diag, int n, const fb_complex_double* ap, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - GER (rank-1 update) */
static void atlas_sger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_sger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_cgeru(int m, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_zgeru(int m, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_cgerc(int m, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_zgerc(int m, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HER (Hermitian rank-1 update) */
static void atlas_cher(char uplo, int n, float alpha, const fb_complex_float* x, int incx, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void atlas_zher(char uplo, int n, double alpha, const fb_complex_double* x, int incx, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - HPR (Hermitian packed rank-1 update) */
static void atlas_chpr(char uplo, int n, float alpha, const fb_complex_float* x, int incx, fb_complex_float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void atlas_zhpr(char uplo, int n, double alpha, const fb_complex_double* x, int incx, fb_complex_double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - HER2 (Hermitian rank-2 update) */
static void atlas_cher2(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_zher2(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HPR2 (Hermitian packed rank-2 update) */
static void atlas_chpr2(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void atlas_zhpr2(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 2 BLAS - SYR (symmetric rank-1 update) */
static void atlas_ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void atlas_dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void atlas_csyr(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void atlas_zsyr(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - SPR (symmetric packed rank-1 update) */
static void atlas_sspr(char uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_sspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void atlas_dspr(char uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - SYR2 (symmetric rank-2 update) */
static void atlas_ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void atlas_dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - SPR2 (symmetric packed rank-2 update) */
static void atlas_sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_sspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void atlas_dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 3 BLAS - GEMM (complex) */
static void atlas_cgemm(char transa, char transb, int m, int n, int k, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_zgemm(char transa, char transb, int m, int n, int k, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYMM (symmetric matrix-matrix) */
static void atlas_ssymm(char side, char uplo, int m, int n, float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, float,
                         const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_dsymm(char side, char uplo, int m, int n, double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, double,
                         const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_csymm(char side, char uplo, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_zsymm(char side, char uplo, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HEMM (Hermitian matrix-matrix) */
static void atlas_chemm(char side, char uplo, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_chemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_zhemm(char side, char uplo, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zhemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYRK (symmetric rank-k update) */
static void atlas_ssyrk(char uplo, char trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void atlas_dsyrk(char uplo, char trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void atlas_csyrk(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void atlas_zsyrk(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - HERK (Hermitian rank-k update) */
static void atlas_cherk(char uplo, char trans, int n, int k, float alpha, const fb_complex_float* a, int lda, float beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void atlas_zherk(char uplo, char trans, int n, int k, double alpha, const fb_complex_double* a, int lda, double beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - SYR2K (symmetric rank-2k update) */
static void atlas_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float* a, int lda,
                       const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ssyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_dsyr2k(char uplo, char trans, int n, int k, double alpha, const double* a, int lda,
                       const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_csyr2k(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                       const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                       const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_csyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_zsyr2k(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                       const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                       const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HER2K (Hermitian rank-2k update) */
static void atlas_cher2k(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                       const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                       float beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_cher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void atlas_zher2k(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                       const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                       double beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_zher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - TRMM (triangular matrix-matrix) */
static void atlas_strmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_strmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_ctrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, fb_complex_float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_ztrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, fb_complex_double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* Level 3 BLAS - TRSM (triangular solve matrix) */
static void atlas_strsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_strsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_dtrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_ctrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, fb_complex_float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ctrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void atlas_ztrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, fb_complex_double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_atlas.handle, "cblas_ztrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* atlas vtable */
static fb_backend_vtable_t g_atlas_vtable = {
    /* Level 1 BLAS */
    .sasum = atlas_sasum,
    .dasum = atlas_dasum,
    .saxpy = atlas_saxpy,
    .daxpy = atlas_daxpy,
    .sdot = atlas_sdot,
    .ddot = atlas_ddot,
    .scopy = atlas_scopy,
    .dcopy = atlas_dcopy,
    .sscal = atlas_sscal,
    .dscal = atlas_dscal,
    .snrm2 = atlas_snrm2,
    .dnrm2 = atlas_dnrm2,
    .sswap = atlas_sswap,
    .dswap = atlas_dswap,
    .isamax = atlas_isamax,
    .idamax = atlas_idamax,
    
    /* Level 2 BLAS */
    .sgemv = atlas_sgemv,
    .dgemv = atlas_dgemv,
    
    /* Level 3 BLAS */
    .sgemm = atlas_sgemm,
    .dgemm = atlas_dgemm,
    
    /* TODO: Add remaining operations */
};

const fb_backend_vtable_t* fb_atlas_get_vtable(void) {
    if (!fb_atlas_is_available()) {
        return NULL;
    }
    return &g_atlas_vtable;
}
