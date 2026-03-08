/**
 * @file accelerate_backend.c
 * @brief Intel accelerate backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "accelerate_backend.h"
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

/* accelerate uses CBLAS interface - same signatures as OpenBLAS */
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

/* accelerate-specific functions */
typedef void (*accelerate_set_num_threads_t)(int num_threads);
typedef int (*accelerate_get_max_threads_t)(void);
typedef void (*accelerate_get_version_t)(int* major, int* minor, int* update);
typedef int (*accelerate_set_threading_layer_t)(int layer);
typedef void (*accelerate_verbose_t)(int enable);

/* Global accelerate API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    
    /* accelerate control functions */
    accelerate_set_num_threads_t set_num_threads;
    accelerate_get_max_threads_t get_max_threads;
    accelerate_get_version_t get_version;
    accelerate_set_threading_layer_t set_threading_layer;
    accelerate_verbose_t verbose;
    
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
    
} g_accelerate;

/* CBLAS constants (same as OpenBLAS) */
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

/* accelerate threading layer constants */
enum accelerate_THREADING_LAYER {
    accelerate_THREADING_INTEL = 0,
    accelerate_THREADING_SEQUENTIAL = 1,
    accelerate_THREADING_TBB = 2,
    accelerate_THREADING_GNU = 3
};

static int load_accelerate_library(void) {
    if (g_accelerate.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        "accelerate_rt.2.dll",
        "accelerate_rt.dll"
#elif defined(__APPLE__)
        "libaccelerate_rt.dylib",
        "libaccelerate_rt.2.dylib",
        "/opt/intel/oneapi/accelerate/latest/lib/libaccelerate_rt.dylib"
#else
        "libaccelerate_rt.so",
        "libaccelerate_rt.so.2",
        "/opt/intel/oneapi/accelerate/latest/lib/intel64/libaccelerate_rt.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_accelerate.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_accelerate.handle) {
            break;
        }
    }
    
    if (!g_accelerate.handle) {
        return -1;
    }
    
    /* Load accelerate control functions */
    g_accelerate.set_num_threads = (accelerate_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_accelerate.handle, "accelerate_set_num_threads");
    g_accelerate.get_max_threads = (accelerate_get_max_threads_t)
        FB_GET_PROC_ADDRESS(g_accelerate.handle, "accelerate_get_max_threads");
    g_accelerate.get_version = (accelerate_get_version_t)
        FB_GET_PROC_ADDRESS(g_accelerate.handle, "accelerate_get_version");
    g_accelerate.set_threading_layer = (accelerate_set_threading_layer_t)
        FB_GET_PROC_ADDRESS(g_accelerate.handle, "accelerate_set_threading_layer");
    g_accelerate.verbose = (accelerate_verbose_t)
        FB_GET_PROC_ADDRESS(g_accelerate.handle, "accelerate_verbose");
    
    /* Load CBLAS functions (same names as OpenBLAS) */
    g_accelerate.sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sasum");
    g_accelerate.dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dasum");
    g_accelerate.saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_saxpy");
    g_accelerate.daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_daxpy");
    g_accelerate.sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sdot");
    g_accelerate.ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_ddot");
    g_accelerate.scopy = (cblas_scopy_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_scopy");
    g_accelerate.dcopy = (cblas_dcopy_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dcopy");
    g_accelerate.sscal = (cblas_sscal_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sscal");
    g_accelerate.dscal = (cblas_dscal_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dscal");
    g_accelerate.snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_snrm2");
    g_accelerate.dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dnrm2");
    g_accelerate.sswap = (cblas_sswap_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sswap");
    g_accelerate.dswap = (cblas_dswap_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dswap");
    g_accelerate.isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_isamax");
    g_accelerate.idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_idamax");
    
    g_accelerate.sgemv = (cblas_sgemv_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sgemv");
    g_accelerate.dgemv = (cblas_dgemv_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dgemv");
    
    g_accelerate.sgemm = (cblas_sgemm_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_sgemm");
    g_accelerate.dgemm = (cblas_dgemm_t)FB_GET_PROC_ADDRESS(g_accelerate.handle, "cblas_dgemm");
    
    g_accelerate.initialized = true;
    return 0;
}

bool fb_accelerate_is_available(void) {
    return load_accelerate_library() == 0;
}

int fb_accelerate_init(void) {
    return load_accelerate_library();
}

void fb_accelerate_shutdown(void) {
    if (g_accelerate.initialized && g_accelerate.handle) {
        FB_FREE_LIBRARY(g_accelerate.handle);
        g_accelerate.handle = NULL;
        g_accelerate.initialized = false;
    }
}

int fb_accelerate_get_version(int* major, int* minor, int* update) {
    if (!g_accelerate.initialized || !g_accelerate.get_version) {
        return -1;
    }
    g_accelerate.get_version(major, minor, update);
    return 0;
}

void fb_accelerate_set_num_threads(int num_threads) {
    if (g_accelerate.set_num_threads) {
        g_accelerate.set_num_threads(num_threads);
    }
}

int fb_accelerate_get_num_threads(void) {
    if (g_accelerate.get_max_threads) {
        return g_accelerate.get_max_threads();
    }
    return 1;
}

int fb_accelerate_set_threading_layer(const char* layer) {
    if (!g_accelerate.set_threading_layer) {
        return -1;
    }
    
    int accelerate_layer = accelerate_THREADING_INTEL;
    if (strcmp(layer, "sequential") == 0) {
        accelerate_layer = accelerate_THREADING_SEQUENTIAL;
    } else if (strcmp(layer, "tbb") == 0) {
        accelerate_layer = accelerate_THREADING_TBB;
    } else if (strcmp(layer, "gnu") == 0) {
        accelerate_layer = accelerate_THREADING_GNU;
    }
    
    return g_accelerate.set_threading_layer(accelerate_layer);
}

void fb_accelerate_set_verbose(bool enable) {
    if (g_accelerate.verbose) {
        g_accelerate.verbose(enable ? 1 : 0);
    }
}

/* Wrapper functions (nearly identical to OpenBLAS wrappers) */

static void accelerate_sasum(int n, const float* x, int incx, float* result) {
    if (g_accelerate.sasum) {
        *result = g_accelerate.sasum(n, x, incx);
    }
}

static void accelerate_dasum(int n, const double* x, int incx, double* result) {
    if (g_accelerate.dasum) {
        *result = g_accelerate.dasum(n, x, incx);
    }
}

static void accelerate_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (g_accelerate.saxpy) {
        g_accelerate.saxpy(n, alpha, x, incx, y, incy);
    }
}

static void accelerate_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_accelerate.daxpy) {
        g_accelerate.daxpy(n, alpha, x, incx, y, incy);
    }
}

static void accelerate_sdot(int n, const float* x, int incx, const float* y, int incy, float* result) {
    if (g_accelerate.sdot) {
        *result = g_accelerate.sdot(n, x, incx, y, incy);
    }
}

static void accelerate_ddot(int n, const double* x, int incx, const double* y, int incy, double* result) {
    if (g_accelerate.ddot) {
        *result = g_accelerate.ddot(n, x, incx, y, incy);
    }
}

static void accelerate_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_accelerate.scopy) {
        g_accelerate.scopy(n, x, incx, y, incy);
    }
}

static void accelerate_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_accelerate.dcopy) {
        g_accelerate.dcopy(n, x, incx, y, incy);
    }
}

static void accelerate_sscal(int n, float alpha, float* x, int incx) {
    if (g_accelerate.sscal) {
        g_accelerate.sscal(n, alpha, x, incx);
    }
}

static void accelerate_dscal(int n, double alpha, double* x, int incx) {
    if (g_accelerate.dscal) {
        g_accelerate.dscal(n, alpha, x, incx);
    }
}

static void accelerate_snrm2(int n, const float* x, int incx, float* result) {
    if (g_accelerate.snrm2) {
        *result = g_accelerate.snrm2(n, x, incx);
    }
}

static void accelerate_dnrm2(int n, const double* x, int incx, double* result) {
    if (g_accelerate.dnrm2) {
        *result = g_accelerate.dnrm2(n, x, incx);
    }
}

static void accelerate_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_accelerate.sswap) {
        g_accelerate.sswap(n, x, incx, y, incy);
    }
}

static void accelerate_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_accelerate.dswap) {
        g_accelerate.dswap(n, x, incx, y, incy);
    }
}

static void accelerate_isamax(int n, const float* x, int incx, int* result) {
    if (g_accelerate.isamax) {
        *result = g_accelerate.isamax(n, x, incx);
    }
}

static void accelerate_idamax(int n, const double* x, int incx, int* result) {
    if (g_accelerate.idamax) {
        *result = g_accelerate.idamax(n, x, incx);
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

static void accelerate_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void accelerate_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void accelerate_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void accelerate_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void accelerate_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void accelerate_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void accelerate_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void accelerate_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void accelerate_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void accelerate_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void accelerate_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void accelerate_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void accelerate_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void accelerate_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void accelerate_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void accelerate_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void accelerate_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void accelerate_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void accelerate_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void accelerate_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void accelerate_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void accelerate_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void accelerate_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void accelerate_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    }
}

static void accelerate_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void accelerate_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void accelerate_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void accelerate_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Level 2 BLAS operations */

static void accelerate_sgemv(char trans, int m, int n, float alpha,
                      const float* a, int lda, const float* x, int incx,
                      float beta, float* y, int incy) {
    if (g_accelerate.sgemv) {
        g_accelerate.sgemv(CblasColMajor, transpose_to_cblas(trans),
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void accelerate_dgemv(char trans, int m, int n, double alpha,
                      const double* a, int lda, const double* x, int incx,
                      double beta, double* y, int incy) {
    if (g_accelerate.dgemv) {
        g_accelerate.dgemv(CblasColMajor, transpose_to_cblas(trans),
                    m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void accelerate_sgemm(char transa, char transb, int m, int n, int k,
                      float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta,
                      float* c, int ldc) {
    if (g_accelerate.sgemm) {
        g_accelerate.sgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void accelerate_dgemm(char transa, char transb, int m, int n, int k,
                      double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta,
                      double* c, int ldc) {
    if (g_accelerate.dgemm) {
        g_accelerate.dgemm(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/* Level 2 BLAS - Complex GEMV */
static void accelerate_cgemv(char trans, int m, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_zgemv(char trans, int m, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zgemv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - GBMV (banded matrix-vector) */
static void accelerate_sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_sgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_cgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_zgbmv(char trans, int m, int n, int kl, int ku, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zgbmv");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HEMV (Hermitian matrix-vector) */
static void accelerate_chemv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_zhemv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhemv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HBMV (Hermitian banded matrix-vector) */
static void accelerate_chbmv(char uplo, int n, int k, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_zhbmv(char uplo, int n, int k, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HPMV (Hermitian packed matrix-vector) */
static void accelerate_chpmv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* ap,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void accelerate_zhpmv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* ap,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SYMV (symmetric matrix-vector) */
static void accelerate_ssymv(char uplo, int n, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_dsymv(char uplo, int n, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_csymv(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* a, int lda,
                      const fb_complex_float* x, int incx, const fb_complex_float* beta, fb_complex_float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_zsymv(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* a, int lda,
                      const fb_complex_double* x, int incx, const fb_complex_double* beta, fb_complex_double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zsymv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SBMV (symmetric banded matrix-vector) */
static void accelerate_ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void accelerate_dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SPMV (symmetric packed matrix-vector) */
static void accelerate_sspmv(char uplo, int n, float alpha, const float* ap,
                      const float* x, int incx, float beta, float* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_sspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

static void accelerate_dspmv(char uplo, int n, double alpha, const double* ap,
                      const double* x, int incx, double beta, double* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dspmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - TRMV (triangular matrix-vector) */
static void accelerate_strmv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_strmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_ctrmv(char uplo, char trans, char diag, int n, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_ztrmv(char uplo, char trans, char diag, int n, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztrmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBMV (triangular banded matrix-vector) */
static void accelerate_stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_stbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_ctbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_ztbmv(char uplo, char trans, char diag, int n, int k, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztbmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPMV (triangular packed matrix-vector) */
static void accelerate_stpmv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_stpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_dtpmv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_ctpmv(char uplo, char trans, char diag, int n, const fb_complex_float* ap, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_ztpmv(char uplo, char trans, char diag, int n, const fb_complex_double* ap, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztpmv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - TRSV (triangular solve) */
static void accelerate_strsv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_strsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_ctrsv(char uplo, char trans, char diag, int n, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

static void accelerate_ztrsv(char uplo, char trans, char diag, int n, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztrsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBSV (triangular banded solve) */
static void accelerate_stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_stbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_ctbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_float* a, int lda, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

static void accelerate_ztbsv(char uplo, char trans, char diag, int n, int k, const fb_complex_double* a, int lda, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztbsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPSV (triangular packed solve) */
static void accelerate_stpsv(char uplo, char trans, char diag, int n, const float* ap, float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_stpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_dtpsv(char uplo, char trans, char diag, int n, const double* ap, double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_ctpsv(char uplo, char trans, char diag, int n, const fb_complex_float* ap, fb_complex_float* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

static void accelerate_ztpsv(char uplo, char trans, char diag, int n, const fb_complex_double* ap, fb_complex_double* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztpsv");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), diag_to_cblas(diag), n, ap, x, incx); }
}

/* Level 2 BLAS - GER (rank-1 update) */
static void accelerate_sger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_sger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dger");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_cgeru(int m, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_zgeru(int m, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_cgerc(int m, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_zgerc(int m, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HER (Hermitian rank-1 update) */
static void accelerate_cher(char uplo, int n, float alpha, const fb_complex_float* x, int incx, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void accelerate_zher(char uplo, int n, double alpha, const fb_complex_double* x, int incx, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zher");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - HPR (Hermitian packed rank-1 update) */
static void accelerate_chpr(char uplo, int n, float alpha, const fb_complex_float* x, int incx, fb_complex_float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void accelerate_zhpr(char uplo, int n, double alpha, const fb_complex_double* x, int incx, fb_complex_double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhpr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - HER2 (Hermitian rank-2 update) */
static void accelerate_cher2(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_zher2(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zher2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HPR2 (Hermitian packed rank-2 update) */
static void accelerate_chpr2(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx,
                      const fb_complex_float* y, int incy, fb_complex_float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void accelerate_zhpr2(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx,
                      const fb_complex_double* y, int incy, fb_complex_double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhpr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 2 BLAS - SYR (symmetric rank-1 update) */
static void accelerate_ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void accelerate_dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void accelerate_csyr(char uplo, int n, const fb_complex_float* alpha, const fb_complex_float* x, int incx, fb_complex_float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

static void accelerate_zsyr(char uplo, int n, const fb_complex_double* alpha, const fb_complex_double* x, int incx, fb_complex_double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zsyr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - SPR (symmetric packed rank-1 update) */
static void accelerate_sspr(char uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_sspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

static void accelerate_dspr(char uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dspr");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - SYR2 (symmetric rank-2 update) */
static void accelerate_ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

static void accelerate_dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsyr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - SPR2 (symmetric packed rank-2 update) */
static void accelerate_sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_sspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

static void accelerate_dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dspr2");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), n, alpha, x, incx, y, incy, ap); }
}

/* Level 3 BLAS - GEMM (complex) */
static void accelerate_cgemm(char transa, char transb, int m, int n, int k, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_zgemm(char transa, char transb, int m, int n, int k, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zgemm");
    if (fn) { fn(CblasColMajor, transpose_to_cblas(transa), transpose_to_cblas(transb),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYMM (symmetric matrix-matrix) */
static void accelerate_ssymm(char side, char uplo, int m, int n, float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, float,
                         const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_dsymm(char side, char uplo, int m, int n, double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, double,
                         const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_csymm(char side, char uplo, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_zsymm(char side, char uplo, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zsymm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HEMM (Hermitian matrix-matrix) */
static void accelerate_chemm(char side, char uplo, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                      const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_chemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_zhemm(char side, char uplo, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                      const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zhemm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYRK (symmetric rank-k update) */
static void accelerate_ssyrk(char uplo, char trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void accelerate_dsyrk(char uplo, char trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void accelerate_csyrk(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void accelerate_zsyrk(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zsyrk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - HERK (Hermitian rank-k update) */
static void accelerate_cherk(char uplo, char trans, int n, int k, float alpha, const fb_complex_float* a, int lda, float beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

static void accelerate_zherk(char uplo, char trans, int n, int k, double alpha, const fb_complex_double* a, int lda, double beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zherk");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - SYR2K (symmetric rank-2k update) */
static void accelerate_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float* a, int lda,
                       const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ssyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_dsyr2k(char uplo, char trans, int n, int k, double alpha, const double* a, int lda,
                       const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_csyr2k(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                       const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                       const fb_complex_float* beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_csyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_zsyr2k(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                       const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                       const fb_complex_double* beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zsyr2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HER2K (Hermitian rank-2k update) */
static void accelerate_cher2k(char uplo, char trans, int n, int k, const fb_complex_float* alpha,
                       const fb_complex_float* a, int lda, const fb_complex_float* b, int ldb,
                       float beta, fb_complex_float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_cher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void accelerate_zher2k(char uplo, char trans, int n, int k, const fb_complex_double* alpha,
                       const fb_complex_double* a, int lda, const fb_complex_double* b, int ldb,
                       double beta, fb_complex_double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_zher2k");
    if (fn) { fn(CblasColMajor, uplo_to_cblas(uplo), transpose_to_cblas(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - TRMM (triangular matrix-matrix) */
static void accelerate_strmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_strmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_dtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_ctrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, fb_complex_float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_ztrmm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, fb_complex_double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztrmm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* Level 3 BLAS - TRSM (triangular solve matrix) */
static void accelerate_strsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_strsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_dtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_dtrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_ctrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_float* alpha,
                      const fb_complex_float* a, int lda, fb_complex_float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ctrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

static void accelerate_ztrsm(char side, char uplo, char transa, char diag, int m, int n, const fb_complex_double* alpha,
                      const fb_complex_double* a, int lda, fb_complex_double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_accelerate.handle, "cblas_ztrsm");
    if (fn) { fn(CblasColMajor, side_to_cblas(side), uplo_to_cblas(uplo), transpose_to_cblas(transa), diag_to_cblas(diag), m, n, alpha, a, lda, b, ldb); }
}

/* accelerate vtable */
static fb_backend_vtable_t g_accelerate_vtable = {
    /* Level 1 BLAS */
    .sasum = accelerate_sasum,
    .dasum = accelerate_dasum,
    .saxpy = accelerate_saxpy,
    .daxpy = accelerate_daxpy,
    .sdot = accelerate_sdot,
    .ddot = accelerate_ddot,
    .scopy = accelerate_scopy,
    .dcopy = accelerate_dcopy,
    .sscal = accelerate_sscal,
    .dscal = accelerate_dscal,
    .snrm2 = accelerate_snrm2,
    .dnrm2 = accelerate_dnrm2,
    .sswap = accelerate_sswap,
    .dswap = accelerate_dswap,
    .isamax = accelerate_isamax,
    .idamax = accelerate_idamax,
    
    /* Level 2 BLAS */
    .sgemv = accelerate_sgemv,
    .dgemv = accelerate_dgemv,
    
    /* Level 3 BLAS */
    .sgemm = accelerate_sgemm,
    .dgemm = accelerate_dgemm,
    
    /* TODO: Add remaining operations */
};

const fb_backend_vtable_t* fb_accelerate_get_vtable(void) {
    if (!fb_accelerate_is_available()) {
        return NULL;
    }
    return &g_accelerate_vtable;
}
